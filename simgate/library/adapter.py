"""The Adapter: turns an uploaded ONNX policy into a Library PolicyCard for a chosen robot.

1. FACTS, read in code: the ONNX inputs and outputs (names, shapes; an LSTM's state in and out), the config's
   keys and values, the robot's joint names and ranges, and any README.
2. A draft from the Adapter agent (PLANNER_MODEL; rules in mock mode) in a FIXED vocabulary of observation
   blocks, checked against FACTS with one retry. Uncertain items become questions.
3. Checks: block sizes sum to the ONNX input, actions = ONNX output = joints mapped by name, the policy loads,
   then on flat ground: stands 2 s at command 0 and walks >= 0.5 m forward in 4 s. A failure says what is
   likely wrong. With no open questions and passing checks the card is listed (policy_card.json).
"""
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import mujoco
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import config, llm
from .cards import CommandSpec, ObservationBlock, PolicyCard
from .registry import LibraryRegistry
from .uploads import UploadError, UploadRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# the Adapter's words for observation blocks -> the policy runtime's; None = not supported in SimGate
VOCABULARY = {"base_ang_vel": "ang_vel", "base_lin_vel": "lin_vel", "projected_gravity": "gravity_orientation",
              "command": "command", "joint_pos": "joint_position_delta", "joint_vel": "joint_velocity",
              "last_action": "last_action", "clock": "gait_phase", "height_scan": None}
BlockName = Literal["base_ang_vel", "base_lin_vel", "projected_gravity", "command", "joint_pos", "joint_vel",
                    "last_action", "clock", "height_scan"]
NUMBER_QUESTIONS = {"clock_period_s": (0.1, 5.0), "action_scale": (0.001, 10.0), "control_dt": (0.001, 0.2)}
CHOICE_QUESTIONS = {"default_angles": ("robot pose", "zeros")}   # the robot's own default pose, or all zero
STAND_S, WALK_S, WALK_GOAL_M, WALK_COMMAND = 2.0, 4.0, 0.5, 0.5


class BlockDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: BlockName
    size: int = Field(gt=0)
    scale: float | list[float] = 1.0


class PolicyQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["clock_period_s", "action_scale", "control_dt", "default_angles"]
    question: str = Field(min_length=1, max_length=200)
    choices: list[str] = []          # suggestions; any number in range is accepted

    @field_validator("choices", mode="before")
    @classmethod
    def numbers_as_text(cls, value):
        return [str(choice) for choice in value] if isinstance(value, list) else value


class PolicyDraft(BaseModel):
    """A policy card in the Adapter's vocabulary; observation blocks in the order the policy reads them."""
    model_config = ConfigDict(extra="forbid")
    observation: list[BlockDraft] = Field(min_length=1)
    clock_period_s: float | None = Field(default=None, gt=0)
    action_scale: float | list[float]
    default_angles: list[float] | None = None   # None only while a question asks for it
    kps: list[float]
    kds: list[float]
    control_dt: float | None = Field(default=None, gt=0)   # None only while a question asks for it
    joint_order: list[str]           # the robot's joint names, in the policy's action order
    command_names: list[str] = ["vx", "vy", "yaw_rate"]
    command_ranges: list[tuple[float, float]] = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    command_default: list[float] = [0.5, 0.0, 0.0]
    questions: list[PolicyQuestion] = []
    notes: str = Field(default="", max_length=800)


ADAPTER_SYSTEM = """You are the Adapter in a robot simulation sandbox. A user uploaded a trained ONNX policy for a
robot. Read the FACTS (measured in code; always right), the config and README, and write a draft policy card as
one JSON object with exactly these fields: observation (list of {name, size, scale}), clock_period_s, action_scale,
default_angles, kps, kds, control_dt, joint_order, command_names, command_ranges, command_default, questions (list
of {field, question, choices}), notes. An illustrative example for a made-up 2-joint robot (not real values):
{"observation": [{"name": "projected_gravity", "size": 3, "scale": 1.0}, {"name": "joint_pos", "size": 2, "scale": 1.0},
 {"name": "joint_vel", "size": 2, "scale": 0.1}], "clock_period_s": null, "action_scale": 0.5,
 "default_angles": [0.0, 0.1], "kps": [20, 20], "kds": [0.5, 0.5], "control_dt": 0.01, "joint_order": ["a", "b"],
 "command_names": ["vx", "vy", "yaw_rate"], "command_ranges": [[-1, 1], [-1, 1], [-1, 1]],
 "command_default": [0.5, 0, 0], "questions": [], "notes": "..."}
Rules:
- observation lists the blocks in the order the policy reads them, using ONLY these names: base_ang_vel (3),
  base_lin_vel (3), projected_gravity (3), command (one per command axis), joint_pos (joint angle minus its
  default, one per joint), joint_vel (one per joint), last_action (one per joint), clock (sin and cos of the gait
  phase: size 2, needs clock_period_s), height_scan (not supported in SimGate). List ONLY the blocks this policy
  reads, never one of size 0; list height_scan only if the config or README says the policy uses a height scan.
  Each block has a size and a scale (a number, or one number per value).
- The sizes must add up to FACTS.onnx.observation_size; joint_order must have FACTS.onnx.action_size names, all from
  FACTS.robot.joint_names, in the policy's action order; default_angles, kps and kds follow joint_order.
- control_dt is the policy's step in seconds: the config's simulation dt multiplied by its control decimation
  (e.g. 0.005 x 4 = 0.02). Leave it null only if the files don't say, and then ask a control_dt question.
- Take numbers from the config. When something isn't in the files (e.g. the clock period), add a question for
  clock_period_s, action_scale or control_dt (with suggested choices) instead of guessing. With no default joint
  angles in the files, set default_angles to null and ask a default_angles question with choices
  ["robot pose", "zeros"].
Return only the JSON object."""


# ── FACTS ───────────────────────────────────────────────────────────────────

def onnx_facts(path: Path) -> dict:
    """Inputs and outputs of an ONNX model. Loading it with onnxruntime parses the graph; no user code runs."""
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 - any parse failure is the file's
        raise UploadError(f"onnxruntime can't load {path.name}: {exc}") from exc
    inputs = [{"name": i.name, "shape": list(i.shape), "type": i.type} for i in session.get_inputs()]
    outputs = [{"name": o.name, "shape": list(o.shape), "type": o.type} for o in session.get_outputs()]
    size = lambda shape: shape[-1] if shape and isinstance(shape[-1], int) else None   # noqa: E731
    states = [i for i in inputs[1:] if len(i["shape"]) == 3 and isinstance(i["shape"][-1], int)]
    return {"inputs": inputs, "outputs": outputs, "observation_size": size(inputs[0]["shape"]),
            "action_size": size(outputs[0]["shape"]),
            "recurrent_state_size": states[0]["shape"][-1] if states and len(states) == len(inputs) - 1 else None}


def config_facts(folder: Path, files: list[str]) -> dict:
    """Every yaml/json config, parsed safely, and the README text."""
    configs = {}
    for path in files:
        suffix = PurePosixPath(path).suffix.lower()
        text = (folder / path).read_text(encoding="utf-8", errors="replace")
        if suffix in (".yaml", ".yml"):
            configs[path] = yaml.safe_load(text)
        elif suffix == ".json":
            configs[path] = json.loads(text)
    readme = "\n\n".join((folder / p).read_text(encoding="utf-8", errors="replace") for p in files
                         if p.lower().endswith((".md", ".txt")))
    return {"configs": configs, "readme": readme[:6000]}


def robot_facts(robot_id: str) -> dict:
    robot = LibraryRegistry().robot(robot_id)
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / robot.mjcf_path)) if robot.mjcf_path else None
    ranges, pose = {}, {}
    if model is not None:
        for name in robot.joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0 and model.jnt_limited[joint]:
                ranges[name] = [round(float(v), 4) for v in model.jnt_range[joint]]
            if joint >= 0:   # the angle each joint has in the model's own default pose
                pose[name] = round(float(model.qpos0[model.jnt_qposadr[joint]]), 6)
    return {"id": robot.id, "name": robot.name, "kind": robot.kind, "joint_names": robot.joint_names,
            "joint_ranges": ranges, "default_pose": pose, "mass_kg": robot.mass_kg,
            "standing_height_m": robot.standing_height_m}


# ── The draft: rules, the agent, and the check against FACTS ──────────────────

def _flat(configs: dict) -> dict[str, Any]:
    """The config keys from every file, top level and one level down (later files win)."""
    flat: dict[str, Any] = {}
    for content in configs.values():
        if isinstance(content, dict):
            for key, value in content.items():
                flat[str(key).lower()] = value
                if isinstance(value, dict):
                    flat.update({str(k).lower(): v for k, v in value.items()})
    return flat


def _first(flat: dict, *keys: str, default=None):
    return next((flat[key] for key in keys if key in flat and flat[key] is not None), default)


def rules_draft(found: dict) -> PolicyDraft:
    """A legged_gym/unitree_rl_gym deploy config: observations [ang_vel, gravity, command, joint_pos, joint_vel,
    last_action, (clock)], scales and gains from the config, joints in the robot's order."""
    flat, onnx, robot = _flat(found["config"]["configs"]), found["onnx"], found["robot"]
    actions = onnx["action_size"] or int(_first(flat, "num_actions", default=len(robot["joint_names"])))
    observations = onnx["observation_size"] or int(_first(flat, "num_obs", default=0))
    command_scale = _first(flat, "cmd_scale", "command_scale", default=[1.0, 1.0, 1.0])
    commands = len(command_scale) if isinstance(command_scale, list) else 3
    blocks = [BlockDraft(name="base_ang_vel", size=3, scale=float(_first(flat, "ang_vel_scale", default=1.0))),
              BlockDraft(name="projected_gravity", size=3, scale=1.0),
              BlockDraft(name="command", size=commands, scale=command_scale),
              BlockDraft(name="joint_pos", size=actions, scale=float(_first(flat, "dof_pos_scale", default=1.0))),
              BlockDraft(name="joint_vel", size=actions, scale=float(_first(flat, "dof_vel_scale", default=1.0))),
              BlockDraft(name="last_action", size=actions, scale=1.0)]
    questions, notes = [], ["Observation order assumed from the legged_gym deploy convention."]
    if "lin_vel_scale" in flat:   # legged_gym's full observation starts with the base linear velocity
        blocks.insert(0, BlockDraft(name="base_lin_vel", size=3, scale=float(flat["lin_vel_scale"])))
    period = _period(found)
    remaining = observations - sum(block.size for block in blocks)
    if remaining == 2:
        blocks.append(BlockDraft(name="clock", size=2, scale=1.0))
        if period is None:
            questions.append(PolicyQuestion(field="clock_period_s", question="The policy reads a gait clock (sin/cos). "
                             "How long is one gait cycle, in seconds? (unitree_rl_gym uses 0.8 s)", choices=["0.8", "1.0", "0.6"]))
    elif remaining:   # left as a size mismatch for the check to report; not guessed as a height scan
        notes.append(f"The config explains {observations - remaining} of the ONNX's {observations} observation values.")
    dt = _first(flat, "simulation_dt", "sim_dt", "dt")
    decimation = _first(flat, "control_decimation", "decimation")
    control_dt = float(dt) * int(decimation) if dt and decimation else float(_first(flat, "control_dt", default=0.02))
    joint_order = _first(flat, "joint_names", "joint_order", "dof_names", default=None) or robot["joint_names"][:actions]
    if not _first(flat, "joint_names", "joint_order", "dof_names"):
        notes.append("Joint order assumed to be the robot's own joint order (the config names no joints).")
    zeros = [0.0] * actions
    default_angles = _first(flat, "default_angles", "default_joint_angles")
    if default_angles is None:
        questions.append(PolicyQuestion(field="default_angles", question="The config gives no default joint angles "
                         "(the pose the actions are added to). Use the robot's own default pose, or zeros?",
                         choices=list(CHOICE_QUESTIONS["default_angles"])))
    return PolicyDraft(observation=blocks, clock_period_s=period if remaining == 2 else None,
                       action_scale=_first(flat, "action_scale", default=1.0),
                       default_angles=default_angles,
                       kps=_first(flat, "kps", "stiffness", default=zeros), kds=_first(flat, "kds", "damping", default=zeros),
                       control_dt=round(control_dt, 6), joint_order=list(joint_order),
                       command_default=list(_first(flat, "cmd_init", "command_init", default=[0.5] + [0.0] * (commands - 1))),
                       command_names=["vx", "vy", "yaw_rate"][:commands] if commands <= 3 else [f"c{i}" for i in range(commands)],
                       command_ranges=[(-1.0, 1.0)] * commands, questions=questions, notes=" ".join(notes))


def _period(found: dict) -> float | None:
    flat = _flat(found["config"]["configs"])
    value = _first(flat, "period", "gait_period", "clock_period", "cycle_time")
    if value is not None:
        return float(value)
    match = re.search(r"(?:gait\s+)?(?:period|cycle)\D{0,25}?(\d+(?:\.\d+)?)\s*s\b", found["config"]["readme"], re.I)
    return float(match.group(1)) if match else None


def check_draft(draft: PolicyDraft, found: dict) -> list[str]:
    """Where the draft disagrees with FACTS; empty when it can be tried."""
    onnx, robot = found["onnx"], found["robot"]
    errors = []
    unsupported = [block.name for block in draft.observation if VOCABULARY[block.name] is None]
    if unsupported:
        errors.append(f"{', '.join(unsupported)} isn't supported in SimGate (no height scanner), so this policy can't run here")
    total = sum(block.size for block in draft.observation)
    if total != onnx["observation_size"]:
        errors.append(f"observation block sizes add up to {total}, but the ONNX input has {onnx['observation_size']} values")
    for block in draft.observation:
        if isinstance(block.scale, list) and len(block.scale) != block.size:
            errors.append(f"{block.name} has {len(block.scale)} scales for {block.size} values")
        fixed = {"base_ang_vel": 3, "base_lin_vel": 3, "projected_gravity": 3, "clock": 2,
                 "command": len(draft.command_names), "joint_pos": len(draft.joint_order),
                 "joint_vel": len(draft.joint_order), "last_action": len(draft.joint_order)}.get(block.name)
        if fixed is not None and block.size != fixed:
            errors.append(f"{block.name} must have {fixed} values, not {block.size}")
    if len(draft.joint_order) != onnx["action_size"]:
        errors.append(f"joint_order has {len(draft.joint_order)} joints, but the ONNX output has {onnx['action_size']} actions")
    unknown = [name for name in draft.joint_order if name not in robot["joint_names"]]
    if unknown:
        errors.append(f"joints {unknown} aren't joints of robot {robot['id']} (it has {robot['joint_names']})")
    if len(set(draft.joint_order)) != len(draft.joint_order):
        errors.append("joint_order names a joint twice")
    if draft.default_angles is None and not any(q.field == "default_angles" for q in draft.questions):
        errors.append("default_angles is missing: take them from the config or ask a default_angles question")
    for field in ("default_angles", "kps", "kds"):
        if getattr(draft, field) is not None and len(getattr(draft, field)) != len(draft.joint_order):
            errors.append(f"{field} has {len(getattr(draft, field))} values for {len(draft.joint_order)} joints")
    if isinstance(draft.action_scale, list) and len(draft.action_scale) != len(draft.joint_order):
        errors.append("action_scale must be one number or one per joint")
    if any(block.name == "clock" for block in draft.observation) and draft.clock_period_s is None \
            and not any(q.field == "clock_period_s" for q in draft.questions):
        errors.append("a clock block needs clock_period_s, or a question asking for it")
    errors += _config_conflicts(draft, _flat(found["config"]["configs"]))
    if draft.control_dt is None and not any(q.field == "control_dt" for q in draft.questions):
        errors.append("control_dt is missing: work it out from the config (dt x decimation) or ask a control_dt question")
    if not len(draft.command_names) == len(draft.command_ranges) == len(draft.command_default):
        errors.append("command_names, command_ranges and command_default must have the same length")
    return errors


CONFIG_SCALES = {"ang_vel_scale": "base_ang_vel", "lin_vel_scale": "base_lin_vel", "dof_pos_scale": "joint_pos",
                 "dof_vel_scale": "joint_vel", "cmd_scale": "command"}


def _config_conflicts(draft: PolicyDraft, flat: dict) -> list[str]:
    """Where the draft contradicts numbers the uploaded config states (the config is FACTS too)."""
    def differs(a, b) -> bool:
        a, b = np.atleast_1d(np.asarray(a, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
        return a.shape != b.shape and a.size != 1 and b.size != 1 or not np.allclose(a, b, atol=1e-6)

    errors = []
    blocks = {block.name: block for block in draft.observation}
    for key, name in CONFIG_SCALES.items():
        if key in flat and name not in blocks:
            errors.append(f"the config gives {key}, but there is no {name} block")
        elif key in flat and differs(blocks[name].scale, flat[key]):
            errors.append(f"{name} scale is {blocks[name].scale}, but the config's {key} is {flat[key]}")
    scaled = [key for key in CONFIG_SCALES if key in flat]
    if scaled and "lin_vel_scale" not in flat and "base_lin_vel" in blocks:
        errors.append("base_lin_vel is listed, but the config scales the other blocks and not lin_vel: "
                      "the policy probably doesn't read it")
    for key, field in (("action_scale", "action_scale"), ("kps", "kps"), ("kds", "kds"), ("default_angles", "default_angles")):
        if key in flat and getattr(draft, field) is not None and differs(getattr(draft, field), flat[key]):
            errors.append(f"{field} is {getattr(draft, field)}, but the config says {flat[key]}")
    dt, decimation = _first(flat, "simulation_dt", "sim_dt"), _first(flat, "control_decimation", "decimation")
    if dt and decimation and draft.control_dt is not None and abs(draft.control_dt - float(dt) * int(decimation)) > 1e-6:
        errors.append(f"control_dt is {draft.control_dt}, but the config's {dt} s x decimation {decimation} = "
                      f"{float(dt) * int(decimation):g} s")
    return errors


def draft_policy(found: dict) -> tuple[PolicyDraft, str, list[dict]]:
    rules = rules_draft(found)
    if config.MOCK_MODE:
        return rules, "rules", []

    def parse(reply: dict) -> PolicyDraft:
        draft = PolicyDraft.model_validate(reply)
        errors = check_draft(draft, found)
        if errors:
            raise ValueError("The draft disagrees with FACTS: " + "; ".join(errors))
        return draft

    messages = [{"role": "system", "content": ADAPTER_SYSTEM},
                {"role": "user", "content": "FACTS:\n" + json.dumps({"onnx": found["onnx"], "robot": found["robot"],
                                                                     "configs": found["config"]["configs"]}, default=str)
                 + f"\n\nFiles: {', '.join(found['files'])}\n\nREADME:\n{found['config']['readme'] or '(none)'}"}]
    try:
        draft, calls = llm.chat_json(messages, config.PLANNER_MODEL, parse)
    except Exception as exc:  # noqa: BLE001 - a model outage falls back to the rules
        return rules, "rules", [{"error": f"{type(exc).__name__}: {exc}"[:300]}]
    return (draft, "model", calls) if draft is not None else (rules, "rules", calls)


# ── Card, checks, listing ─────────────────────────────────────────────────────

def to_card(record: UploadRecord, robot_id: str, onnx_path: str, draft: PolicyDraft, found: dict) -> PolicyCard:
    return PolicyCard(
        id=record.id, name=record.name, license=record.license, credit=record.credit, robot_id=robot_id, format="onnx",
        file=onnx_path, control_dt=draft.control_dt, joint_order=draft.joint_order,
        obs_layout=[ObservationBlock(name=VOCABULARY[block.name] or block.name, size=block.size, scale=block.scale)
                    for block in draft.observation],
        action_scale=draft.action_scale, default_angles=draft.default_angles, kps=draft.kps, kds=draft.kds,
        command=CommandSpec(names=draft.command_names, ranges=draft.command_ranges, default=draft.command_default),
        clock_period_s=draft.clock_period_s if any(b.name == "clock" for b in draft.observation) else None,
        recurrent_state_size=found["onnx"]["recurrent_state_size"],
        training_notes=[note for note in (draft.notes,) if note] + ["Uploaded policy, adapted by SimGate's Adapter."])


def run_checks(card: PolicyCard, draft: PolicyDraft, found: dict) -> list[dict]:
    """Sizes, the joint mapping, loading, then standing and walking on flat Earth ground."""
    from ..backends.mujoco_backend import MuJoCoSimulator
    from ..policy_runtime import PolicyRunner
    from ..spec import Robot, SceneSpec, Terrain

    onnx, robot = found["onnx"], found["robot"]
    total = sum(block.size for block in draft.observation)
    checks = [{"name": "Observation blocks add up to the ONNX input", "passed": total == onnx["observation_size"],
               "detail": f"{' + '.join(f'{b.name} {b.size}' for b in draft.observation)} = {total}; the ONNX input has "
                         f"{onnx['observation_size']}."},
              {"name": "Actions match the ONNX output and the mapped joints",
               "passed": len(card.joint_order) == onnx["action_size"],
               "detail": f"{onnx['action_size']} ONNX actions, {len(card.joint_order)} joints mapped."}]
    missing = [name for name in card.joint_order if name not in robot["joint_names"]]
    checks.append({"name": "Joints map to the robot by name", "passed": not missing,
                   "detail": f"Not joints of {robot['id']}: {missing}." if missing else
                             f"All {len(card.joint_order)} joints found on {robot['id']} by name."})
    if not all(check["passed"] for check in checks):
        return checks
    registry = LibraryRegistry()
    holder = next(iter(registry.compatible_policies(robot["id"])), None)   # any policy of this robot, only to build it
    if holder is None:
        return checks + [{"name": "Policy loads", "passed": False, "detail": f"{robot['id']} can't be placed in a scene."}]
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0),
                     robot=Robot(kind="library", robot_id=robot["id"], policy_id=holder.id)).fill_defaults()
    sim = MuJoCoSimulator()
    build = sim.build(spec)

    def simulate(seconds: float, command: dict) -> dict:
        model, data = sim._make(build)
        mujoco.mj_forward(model, data)
        runner = PolicyRunner(card, model, data)
        runner.set_command(command)
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        for _ in range(int(0.5 / model.opt.timestep)):
            runner.hold_default()
            mujoco.mj_step(model, data)
        start, start_z, fell_at = data.xpos[root].copy(), float(data.xpos[root][2]), None
        for step in range(int(seconds / model.opt.timestep)):
            runner.step(step * model.opt.timestep)
            mujoco.mj_step(model, data)
            tilt = math.degrees(math.acos(float(np.clip(data.xmat[root][8], -1, 1))))
            if fell_at is None and (tilt > 45 or not np.isfinite(data.qpos).all()):
                fell_at = round(step * model.opt.timestep, 2)
                break
        return {"fell_at": fell_at, "dz": float(data.xpos[root][2]) - start_z, "dx": float(data.xpos[root][0] - start[0]),
                "drift": float(np.linalg.norm(data.xpos[root][:2] - start[:2]))}

    try:
        stand = simulate(STAND_S, {name: 0.0 for name in card.command.names})
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        return checks + [{"name": "Policy loads", "passed": False, "detail": str(exc)[:300]}]
    checks.append({"name": "Policy loads", "passed": True, "detail": "onnxruntime ran it on the robot's named joints."})
    height = robot["standing_height_m"]
    stood = stand["fell_at"] is None and abs(stand["dz"]) <= 0.1 * height
    checks.append({"name": "Stands for 2 s at command 0", "passed": stood,
                   "detail": (f"Upright for {STAND_S:g} s; root {stand['dz']:+.3f} m, drifted {stand['drift']:.2f} m." if stood else
                              f"It fell after {stand['fell_at']} s." if stand["fell_at"] is not None else
                              f"Its root sank {stand['dz']:+.3f} m.")
                             + ("" if stood else " " + _likely_wrong("stand", stand))})
    forward = {name: 0.0 for name in card.command.names}
    forward[card.command.names[0]] = min(WALK_COMMAND, card.command.ranges[0][1])
    walk = simulate(WALK_S, forward)
    walked = walk["fell_at"] is None and walk["dx"] >= WALK_GOAL_M
    checks.append({"name": f"Walks forward {WALK_GOAL_M:g} m in {WALK_S:g} s", "passed": walked,
                   "detail": f"At {card.command.names[0]}={forward[card.command.names[0]]:g} it moved {walk['dx']:+.2f} m"
                             + (f" and fell after {walk['fell_at']} s." if walk["fell_at"] is not None else ".")
                             + ("" if walked else " " + _likely_wrong("walk", walk))})
    return checks


def _likely_wrong(test: str, result: dict) -> str:
    if result["fell_at"] is not None and result["fell_at"] < 0.5:
        return "Likely wrong: the observation order or scales (the robot falls immediately), or the joint order."
    if result["fell_at"] is not None:
        return "Likely wrong: the gains (kps/kds), the action scale, or the default angles."
    if test == "walk" and result["dx"] < -0.1:
        return "Likely wrong: the command's sign or axis order (it walks backward)."
    if test == "walk":
        return "Likely wrong: the command scale or names, or the clock period (it stays upright but doesn't walk)."
    return "Likely wrong: the default angles or the gains (it can't hold its height)."


def _state_path(record: UploadRecord) -> Path:
    return Path(record.folder) / "adapter.json"


def adapt_policy(record: UploadRecord, robot_id: str) -> dict:
    """Read FACTS, draft, check and (if nothing is open) list an uploaded ONNX policy for robot_id."""
    robot = LibraryRegistry().robot(robot_id)   # raises for an unknown robot
    if not robot.joint_names:
        raise UploadError(f"{robot.name} has no named joints for a policy to drive.")
    folder = Path(record.folder)
    onnx_files = [path for path in record.files if path.lower().endswith(".onnx")]
    found = {"onnx": onnx_facts(folder / onnx_files[0]), "config": config_facts(folder, record.files),
             "robot": robot_facts(robot_id), "files": record.files}
    if found["onnx"]["observation_size"] is None or found["onnx"]["action_size"] is None:
        raise UploadError("The ONNX model's observation and action sizes must be fixed numbers.")
    draft, author, calls = draft_policy(found)
    onnx_path = (folder / onnx_files[0]).resolve().relative_to(PROJECT_ROOT).as_posix()
    state = {"upload_id": record.id, "robot_id": robot_id, "adapter": author, "model_calls": calls,
             "facts": found, "onnx_path": onnx_path, "draft": draft.model_dump(),
             "questions": [question.model_dump() for question in draft.questions], "answers": {}}
    return _finish(record, state)


def answer_policy(record: UploadRecord, answers: dict) -> dict:
    """Apply answers (clock_period_s, action_scale, control_dt numbers; default_angles "robot pose" or "zeros"),
    re-check, and list when nothing is open."""
    path = _state_path(record)
    if not path.is_file():
        raise UploadError(f"No adaptation found for upload {record.id}.")
    state = json.loads(path.read_text(encoding="utf-8"))
    for field, value in answers.items():
        if field in CHOICE_QUESTIONS:
            if value not in CHOICE_QUESTIONS[field]:
                raise UploadError(f"{field} must be one of: {', '.join(CHOICE_QUESTIONS[field])}.")
            pose = state["facts"]["robot"]["default_pose"]
            state["draft"][field] = [pose.get(name, 0.0) if value == "robot pose" else 0.0 for name in state["draft"]["joint_order"]]
            state["answers"][field] = value
            state["questions"] = [q for q in state["questions"] if q["field"] != field]
            continue
        if field not in NUMBER_QUESTIONS:
            raise UploadError(f"Can't set {field!r}.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise UploadError(f"{field} must be a number, not {value!r}.") from None
        low, high = NUMBER_QUESTIONS[field]
        if not low <= number <= high:
            raise UploadError(f"{field} must be between {low:g} and {high:g}.")
        state["draft"][field] = number
        state["answers"][field] = number
        state["questions"] = [q for q in state["questions"] if q["field"] != field]
    return _finish(record, state)


def _finish(record: UploadRecord, state: dict) -> dict:
    folder = Path(record.folder)
    draft = PolicyDraft.model_validate({**state["draft"], "questions": state["questions"]})
    found = state["facts"]
    errors = check_draft(draft, found)   # an open question (e.g. the clock period) is not an error
    (folder / "policy_card.json").unlink(missing_ok=True)
    if errors:
        state.update(status="unusable", errors=errors, checks=[], policy_card=None)
    elif state["questions"]:
        state.update(status="needs_answers", errors=[], checks=[], policy_card=None)
    else:
        card = to_card(record, state["robot_id"], state["onnx_path"], draft, found)
        checks = run_checks(card, draft, found)
        passed = all(check["passed"] for check in checks)
        state.update(status="listed" if passed else "checks_failed", errors=[], checks=checks, policy_card=card.model_dump())
        if passed:
            (folder / "policy_card.json").write_text(card.model_dump_json(indent=2), encoding="utf-8")
    _state_path(record).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    return state
