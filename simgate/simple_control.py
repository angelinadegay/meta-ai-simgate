"""Built-in controllers for uploaded robots that have no trained policy: a wheel controller, a baseline gait
for legged robots, and holding the default pose. Joints and actuators are found by name (with the robot's
prefix in a shared scene), and each actuator is driven the way its type expects: a position actuator gets
the target angle, a velocity actuator the target speed, a motor a PD torque.

It has the PolicyRunner interface the scene loop uses: card.command, set_command, step and hold_default.
"""
import math
from typing import Mapping

import mujoco
import numpy as np

from .library.cards import PolicyCard, RobotCard

GAIT_HZ = 1.5
HIP_SWING_RAD = 0.5      # full stride from FULL_STRIDE_M_S up
KNEE_LIFT_RAD = 0.6
FULL_STRIDE_M_S = 0.6    # the default walking speed asked of library robots


class SimpleController:
    def __init__(self, robot: RobotCard, card: PolicyCard, model: mujoco.MjModel, data: mujoco.MjData,
                 name_prefix: str = "") -> None:
        if robot.controller not in ("wheels", "gait", "hold"):
            raise ValueError(f"{robot.id} has no built-in controller (controller={robot.controller!r}).")
        self.robot, self.card, self.model, self.data = robot, card, model, data
        self.mode = robot.controller
        self._joint = {}
        for index, name in enumerate(card.joint_order):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name_prefix}{name}")
            if joint_id < 0:
                raise ValueError(f"controller joint does not exist in MuJoCo model: {name_prefix}{name}")
            actuators = [a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == joint_id]
            if not actuators:
                raise ValueError(f"no actuator drives joint {name_prefix}{name}")
            self._joint[name] = (int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]), actuators[0],
                                 card.default_angles[index], card.kps[index], card.kds[index])
        self._command = dict(zip(card.command.names, card.command.default))

    def set_command(self, command: Mapping[str, float]) -> None:
        unknown = set(command) - set(self.card.command.names)
        if unknown:
            raise ValueError(f"unknown controller command axes: {', '.join(sorted(unknown))}")
        for name, (low, high) in zip(self.card.command.names, self.card.command.ranges):
            value = float(command.get(name, self._command[name]))
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"controller command {name} must be between {low} and {high}")
            self._command[name] = value

    def hold_default(self) -> None:
        for name in self._joint:
            self._drive(name, position=self._joint[name][3])

    def step(self, sim_time: float) -> None:
        vx, yaw = self._command.get("vx", 0.0), self._command.get("yaw_rate", 0.0)
        if self.mode == "wheels":
            half_track = (self.robot.track_m or 0.0) / 2
            for wheel in self.robot.wheels:
                speed = vx + (-yaw if wheel.side == "left" else yaw) * half_track   # turning left: right side faster
                self._drive(wheel.joint, velocity=wheel.spin_sign * speed / wheel.radius_m)
            wheels = {wheel.joint for wheel in self.robot.wheels}
            for name in self._joint:
                if name not in wheels:
                    self._drive(name, position=self._joint[name][3])
            return
        if self.mode == "hold":
            self.hold_default()
            return
        stride = HIP_SWING_RAD * float(np.clip(vx / FULL_STRIDE_M_S, -1.0, 1.0))
        moving = set()
        for leg in self.robot.legs:
            side = 1.0 if leg.side == "left" else -1.0
            swing = stride * float(np.clip(1.0 - side * yaw, 0.0, 2.0))   # turning left: shorter left steps
            angle = 2 * math.pi * GAIT_HZ * sim_time + leg.phase
            hip_default = self._joint[leg.hip][3]
            self._drive(leg.hip, position=hip_default + leg.hip_sign * swing * math.sin(angle))
            moving.add(leg.hip)
            if leg.knee:   # lift the foot while it swings forward (hip moving back toward the front)
                lift = KNEE_LIFT_RAD * max(0.0, -math.cos(angle)) if abs(swing) > 1e-6 else 0.0
                self._drive(leg.knee, position=self._joint[leg.knee][3] + leg.knee_sign * lift)
                moving.add(leg.knee)
        for name in self._joint:
            if name not in moving:
                self._drive(name, position=self._joint[name][3])

    def _drive(self, name: str, position: float | None = None, velocity: float | None = None) -> None:
        qpos_address, dof_address, actuator, _, kp, kd = self._joint[name]
        model = self.model
        bias = model.actuator_biasprm[actuator]
        is_affine = int(model.actuator_biastype[actuator]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        if is_affine and bias[1] < 0 and position is not None:          # position actuator
            ctrl = position
        elif is_affine and bias[1] == 0 and bias[2] < 0:                # velocity actuator
            ctrl = velocity if velocity is not None else 0.0
        else:                                                           # motor: PD torque through its gear
            q, qd = self.data.qpos[qpos_address], self.data.qvel[dof_address]
            torque = kd * ((velocity or 0.0) - qd) if velocity is not None else kp * (position - q) - kd * qd
            ctrl = torque / (model.actuator_gear[actuator, 0] or 1.0)
        if model.actuator_ctrllimited[actuator]:
            ctrl = float(np.clip(ctrl, *model.actuator_ctrlrange[actuator]))
        if not math.isfinite(ctrl):
            raise ValueError(f"controller produced a non-finite command for {name}")
        self.data.ctrl[actuator] = ctrl
