"""Inference and named-joint PD control for library policies."""

import importlib
import math
from pathlib import Path
from typing import Mapping, Protocol

import mujoco
import numpy as np

from .library.cards import ObservationBlock, PolicyCard

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferenceModel(Protocol):
    input_size: int
    output_size: int

    def run(self, observation: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


class _OnnxModel:
    def __init__(self, path: Path, recurrent_state_size: int | None) -> None:
        ort = importlib.import_module("onnxruntime")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        is_recurrent = recurrent_state_size is not None
        expected_count = 3 if is_recurrent else 1
        if len(inputs) != expected_count or len(outputs) != expected_count:
            raise ValueError(
                f"policy ONNX model must have {expected_count} input(s) and output(s) "
                "according to its recurrent_state_size"
            )
        input_shape, output_shape = inputs[0].shape, outputs[0].shape
        if len(input_shape) != 2 or len(output_shape) != 2:
            raise ValueError("policy ONNX input and output must be rank-2 batches")
        if not isinstance(input_shape[1], int) or not isinstance(output_shape[1], int):
            raise ValueError("policy ONNX observation and action dimensions must be fixed")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_size = input_shape[1]
        self.output_size = output_shape[1]
        self.state_inputs = [item.name for item in inputs[1:]]
        self.state_outputs = [item.name for item in outputs[1:]]
        self.state_size = recurrent_state_size
        if is_recurrent:
            for item in inputs[1:]:
                if item.shape != [1, 1, recurrent_state_size]:
                    raise ValueError(f"invalid recurrent state shape for ONNX input {item.name}")
            for item in outputs[1:]:
                if item.shape != [1, 1, recurrent_state_size]:
                    raise ValueError(f"invalid recurrent state shape for ONNX output {item.name}")
        self.reset()

    def run(self, observation: np.ndarray) -> np.ndarray:
        feeds = {self.input_name: observation}
        feeds.update(
            {name: state for name, state in zip(self.state_inputs, self.states)}
        )
        outputs = self.session.run(
            [self.output_name, *self.state_outputs],
            feeds,
        )
        self.states = [np.asarray(state, dtype=np.float32) for state in outputs[1:]]
        if any(not np.isfinite(state).all() for state in self.states):
            raise ValueError("policy recurrent state contains non-finite values")
        return np.asarray(outputs[0], dtype=np.float64)

    def reset(self) -> None:
        if self.state_size is None:
            self.states = []
        else:
            self.states = [
                np.zeros((1, 1, self.state_size), dtype=np.float32)
                for _ in self.state_inputs
            ]


class _TorchScriptModel:
    def __init__(self, path: Path, input_size: int) -> None:
        torch = importlib.import_module("torch")
        self.torch = torch
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.input_size = input_size
        self.is_recurrent = hasattr(self.model, "hidden_state") and hasattr(self.model, "cell_state")
        self.reset()
        try:
            output = self.model(torch.zeros((1, input_size), dtype=torch.float32))
        except RuntimeError as exc:
            raise ValueError(f"TorchScript policy rejected observation size {input_size}: {exc}") from exc
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("TorchScript policy must return one rank-2 tensor")
        self.output_size = int(output.shape[1])
        self.reset()

    def run(self, observation: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(observation.astype(np.float32, copy=False))
        output = self.model(tensor)
        if not isinstance(output, self.torch.Tensor) or output.ndim != 2:
            raise ValueError("TorchScript policy must return one rank-2 tensor")
        return output.detach().cpu().numpy().astype(np.float64, copy=False)

    def reset(self) -> None:
        if self.is_recurrent:
            self.model.hidden_state.zero_()
            self.model.cell_state.zero_()


class PolicyRunner:
    """Runs a policy at its control interval and applies PD torque each physics step."""

    def __init__(
        self,
        card: PolicyCard,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        inference_model: InferenceModel | None = None,
        name_prefix: str = "",
    ) -> None:
        if card.format == "builtin":
            raise ValueError("built-in controllers are managed by the simulator, not PolicyRunner")
        if not card.joint_order:
            raise ValueError("policy card must specify at least one joint")
        if not card.obs_layout:
            raise ValueError("policy card must specify an observation layout")
        self.card = card
        self.model = model
        self.data = data
        self.name_prefix = name_prefix
        self.policy = inference_model or self._load_model(card)
        if self.policy.input_size != card.observation_size:
            raise ValueError(
                f"policy expects {self.policy.input_size} observations, "
                f"but the card defines {card.observation_size}"
            )
        if self.policy.output_size != len(card.joint_order):
            raise ValueError(
                f"policy returns {self.policy.output_size} actions, "
                f"but the card defines {len(card.joint_order)} joints"
            )
        self.joint_ids, self.qpos_addresses, self.dof_addresses, self.actuator_ids = self._map_joints()
        self.root_qpos_address, self.root_dof_address = self._root_addresses()
        self._default_angles = np.asarray(card.default_angles, dtype=np.float64)
        self._kps = np.asarray(card.kps, dtype=np.float64)
        self._kds = np.asarray(card.kds, dtype=np.float64)
        self._action_scale = np.asarray(card.action_scale, dtype=np.float64)
        if self._action_scale.ndim == 0:
            self._action_scale = np.full(len(card.joint_order), float(self._action_scale))
        self._command = np.asarray(card.command.default if card.command else [], dtype=np.float64)
        self._target = self._default_angles.copy()
        self._last_action = np.zeros(len(card.joint_order), dtype=np.float64)
        self._next_control_time = 0.0
        self._control_steps = card.control_dt / float(model.opt.timestep)
        if not math.isclose(self._control_steps, round(self._control_steps), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("policy control_dt must be an integer multiple of the MuJoCo timestep")
        self._validate_joint_targets(self._target)
        zero_observation = np.zeros((1, card.observation_size), dtype=np.float32)
        initial_action = self.policy.run(zero_observation)
        self._validate_action(initial_action)
        self._validate_joint_targets(self._targets_for(initial_action[0]))
        self.policy.reset()

    @staticmethod
    def _load_model(card: PolicyCard) -> InferenceModel:
        path = (PROJECT_ROOT / str(card.file)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy model file does not exist: {path}")
        if card.format == "torchscript":
            return _TorchScriptModel(path, card.observation_size)
        try:
            return _OnnxModel(path, card.recurrent_state_size)
        except ImportError as exc:
            torchscript_path = path.with_suffix(".pt")
            if not torchscript_path.is_file():
                raise RuntimeError(
                    "onnxruntime is required for ONNX policies; the TorchScript fallback file is missing"
                ) from exc
            try:
                return _TorchScriptModel(torchscript_path, card.observation_size)
            except ImportError as torch_exc:
                raise RuntimeError(
                    "onnxruntime is required to run this policy; install requirements.txt"
                ) from torch_exc

    def set_command(self, command: Mapping[str, float]) -> None:
        if self.card.command is None:
            raise ValueError("policy card does not define command axes")
        unknown = set(command) - set(self.card.command.names)
        if unknown:
            raise ValueError(f"unknown policy command axes: {', '.join(sorted(unknown))}")
        values = np.asarray(
            [
                float(command.get(name, default))
                for name, default in zip(self.card.command.names, self.card.command.default)
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("policy command values must be finite")
        for value, (minimum, maximum), name in zip(
            values, self.card.command.ranges, self.card.command.names
        ):
            if not minimum <= value <= maximum:
                raise ValueError(f"policy command {name} must be between {minimum} and {maximum}")
        self._command = values

    def observation(self, sim_time: float) -> np.ndarray:
        blocks = {
            "ang_vel": self.data.qvel[self.root_dof_address + 3 : self.root_dof_address + 6],
            "gravity_orientation": self._gravity_orientation(),
            "command": self._command,
            "joint_position_delta": (
                self.data.qpos[self.qpos_addresses] - self._default_angles
            ),
            "joint_velocity": self.data.qvel[self.dof_addresses],
            "last_action": self._last_action,
            "gait_phase": self._gait_phase(sim_time),
        }
        parts = [
            self._scaled_block(block, self._block_value(block, blocks))
            for block in self.card.obs_layout
        ]
        result = np.concatenate(parts).astype(np.float32, copy=False)
        if result.size != self.card.observation_size or not np.isfinite(result).all():
            raise ValueError("policy observation has an invalid size or contains non-finite values")
        return result

    def step(self, sim_time: float) -> np.ndarray:
        if sim_time + 1e-10 >= self._next_control_time:
            observation = self.observation(sim_time).reshape(1, -1)
            action = self.policy.run(observation)
            self._validate_action(action)
            self._last_action = action[0].copy()
            self._target = self._targets_for(self._last_action)
            self._validate_joint_targets(self._target)
            self._next_control_time = sim_time + self.card.control_dt
        torque = self._kps * (self._target - self.data.qpos[self.qpos_addresses])
        torque -= self._kds * self.data.qvel[self.dof_addresses]
        if not np.isfinite(torque).all():
            raise ValueError("policy produced non-finite PD torques")
        self.data.ctrl[self.actuator_ids] = torque
        return torque.copy()

    def hold_default(self) -> np.ndarray:
        self._target = self._default_angles.copy()
        torque = self._kps * (self._target - self.data.qpos[self.qpos_addresses])
        torque -= self._kds * self.data.qvel[self.dof_addresses]
        if not np.isfinite(torque).all():
            raise ValueError("policy produced non-finite standing torques")
        self.data.ctrl[self.actuator_ids] = torque
        return torque.copy()

    def _map_joints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        joint_ids, qpos_addresses, dof_addresses, actuator_ids = [], [], [], []
        for name in self.card.joint_order:
            joint_name = f"{self.name_prefix}{name}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"policy joint does not exist in MuJoCo model: {joint_name}")
            if int(self.model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise ValueError(f"policy joint must be scalar: {name}")
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                raise ValueError(f"policy actuator does not exist in MuJoCo model: {joint_name}")
            if int(self.model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ValueError(f"policy actuator {joint_name} is not attached to its named joint")
            joint_ids.append(joint_id)
            qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
            dof_addresses.append(int(self.model.jnt_dofadr[joint_id]))
            actuator_ids.append(actuator_id)
        return tuple(np.asarray(values, dtype=np.int32) for values in (
            joint_ids, qpos_addresses, dof_addresses, actuator_ids
        ))

    def _root_addresses(self) -> tuple[int, int]:
        joint_bodies = [int(self.model.jnt_bodyid[joint_id]) for joint_id in self.joint_ids]
        first_ancestors = []
        body_id = joint_bodies[0]
        while body_id > 0:
            first_ancestors.append(body_id)
            body_id = int(self.model.body_parentid[body_id])
        root_body = next(
            (
                candidate
                for candidate in first_ancestors
                if all(self._is_descendant(body, candidate) for body in joint_bodies)
            ),
            None,
        )
        if root_body is None:
            raise ValueError("policy joints do not share a robot root body")
        free_joint_ids = [
            joint_id
            for joint_id in range(self.model.njnt)
            if int(self.model.jnt_bodyid[joint_id]) == root_body
            and int(self.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free_joint_ids) != 1:
            raise ValueError("policy robot root must have exactly one free joint")
        free_joint = free_joint_ids[0]
        return int(self.model.jnt_qposadr[free_joint]), int(self.model.jnt_dofadr[free_joint])

    def _is_descendant(self, body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _gravity_orientation(self) -> np.ndarray:
        quaternion = self.data.qpos[self.root_qpos_address + 3 : self.root_qpos_address + 7]
        qw, qx, qy, qz = quaternion
        return np.array(
            [
                2 * (-qz * qx + qw * qy),
                -2 * (qz * qy + qw * qx),
                1 - 2 * (qw * qw + qz * qz),
            ],
            dtype=np.float64,
        )

    def _gait_phase(self, sim_time: float) -> np.ndarray:
        if self.card.clock_period_s is None:
            raise ValueError("gait_phase observation requires clock_period_s on the policy card")
        phase = sim_time % self.card.clock_period_s / self.card.clock_period_s
        return np.array([math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)])

    @staticmethod
    def _block_value(block: ObservationBlock, values: dict[str, np.ndarray]) -> np.ndarray:
        try:
            value = np.asarray(values[block.name], dtype=np.float64).reshape(-1)
        except KeyError as exc:
            raise ValueError(f"unsupported policy observation block: {block.name}") from exc
        if value.size != block.size:
            raise ValueError(f"observation block {block.name} expected {block.size} values, got {value.size}")
        return value

    @staticmethod
    def _scaled_block(block: ObservationBlock, values: np.ndarray) -> np.ndarray:
        scale = np.asarray(block.scale, dtype=np.float64)
        return values * scale

    def _targets_for(self, action: np.ndarray) -> np.ndarray:
        targets = self._default_angles + action * self._action_scale
        for index, joint_id in enumerate(self.joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                minimum, maximum = self.model.jnt_range[joint_id]
                targets[index] = np.clip(targets[index], minimum, maximum)
        return targets

    def _validate_action(self, action: np.ndarray) -> None:
        if action.shape != (1, len(self.card.joint_order)):
            raise ValueError("policy action has an invalid shape")
        if not np.isfinite(action).all():
            raise ValueError("policy action contains non-finite values")

    def _validate_joint_targets(self, targets: np.ndarray) -> None:
        for joint_id, target, name in zip(self.joint_ids, targets, self.card.joint_order):
            if bool(self.model.jnt_limited[joint_id]):
                minimum, maximum = self.model.jnt_range[joint_id]
                if not minimum <= target <= maximum:
                    raise ValueError(
                        f"policy target for {name} ({target:.4f}) is outside "
                        f"the joint range [{minimum:.4f}, {maximum:.4f}]"
                    )
