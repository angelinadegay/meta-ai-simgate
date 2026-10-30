"""Validated metadata models for library robots, policies, and worlds."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Card(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str
    license: str
    credit: str


class WheelSpec(BaseModel):
    """A driven wheel of an uploaded robot; side, radius and spin sign are measured in code."""
    model_config = ConfigDict(extra="forbid")

    joint: str
    side: Literal["left", "right"]
    radius_m: float = Field(gt=0)
    spin_sign: Literal[-1, 1] = 1   # joint velocity sign that rolls the robot forward (+x)


class LegSpec(BaseModel):
    """A leg of an uploaded robot for the baseline gait: hip swings fore-aft, knee lifts the foot."""
    model_config = ConfigDict(extra="forbid")

    name: str
    hip: str
    knee: str | None = None
    foot: str | None = None
    side: Literal["left", "right"] = "left"
    phase: float = 0.0              # gait phase offset, radians
    hip_sign: Literal[-1, 1] = 1    # hip direction that moves the foot backward on the ground
    knee_sign: Literal[-1, 1] = 1   # knee direction that lifts the foot


class RobotCard(Card):
    kind: Literal["wheeled", "legged", "humanoid", "other"]
    source: Literal["builtin", "preset", "upload"]
    mjcf_path: str | None = None
    root_body: str
    joint_names: list[str] = []
    actuator_type: Literal["torque", "position", "velocity"]
    standing_height_m: float = Field(gt=0)
    mass_kg: float = Field(gt=0)
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # uploaded robots without a trained policy: which built-in controller drives them, and its parts
    controller: Literal["policy", "wheels", "gait", "hold"] = "policy"
    wheels: list[WheelSpec] = []
    track_m: float | None = None    # distance between left and right wheels
    legs: list[LegSpec] = []
    feet: list[str] = []

    @model_validator(mode="after")
    def validate_mjcf_path(self) -> "RobotCard":
        if self.mjcf_path is not None:
            _validate_relative_asset_path(self.mjcf_path)
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("robot joint_names must be unique")
        return self


class ObservationBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    size: int = Field(gt=0)
    scale: float | list[float] = 1.0

    @model_validator(mode="after")
    def validate_scale(self) -> "ObservationBlock":
        if isinstance(self.scale, list) and len(self.scale) != self.size:
            raise ValueError(f"{self.name} scale must have {self.size} values")
        return self


class CommandSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str]
    ranges: list[tuple[float, float]]
    default: list[float]

    @model_validator(mode="after")
    def validate_axes(self) -> "CommandSpec":
        count = len(self.names)
        if not count or any(len(values) != count for values in (self.ranges, self.default)):
            raise ValueError("command names, ranges, and default must have equal non-zero lengths")
        if len(set(self.names)) != count:
            raise ValueError("command names must be unique")
        for name, (minimum, maximum), default in zip(self.names, self.ranges, self.default):
            if minimum >= maximum or not minimum <= default <= maximum:
                raise ValueError(f"invalid range or default for command {name}")
        return self


class PolicyCard(Card):
    robot_id: str
    format: Literal["onnx", "torchscript", "builtin"]
    file: str | None = None
    control_dt: float = Field(gt=0)
    joint_order: list[str] = []
    obs_layout: list[ObservationBlock] = []
    action_scale: float | list[float] = 1.0
    default_angles: list[float] = []
    kps: list[float] = []
    kds: list[float] = []
    command: CommandSpec | None = None
    clock_period_s: float | None = Field(None, gt=0)
    recurrent_state_size: int | None = Field(None, gt=0)
    training_notes: list[str] = []

    @model_validator(mode="after")
    def validate_policy_layout(self) -> "PolicyCard":
        if self.format == "builtin" and self.file is not None:
            raise ValueError("built-in policies do not have a model file")
        if self.format != "builtin" and self.file is None:
            raise ValueError(f"{self.format} policies require a model file")
        if self.file is not None:
            _validate_relative_asset_path(self.file)
        if len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("policy joint_order must be unique")
        if len({block.name for block in self.obs_layout}) != len(self.obs_layout):
            raise ValueError("observation block names must be unique")
        sizes = (len(self.default_angles), len(self.kps), len(self.kds))
        if any(sizes) and any(size != len(self.joint_order) for size in sizes):
            raise ValueError("default_angles, kps, and kds must match the policy joint count")
        if isinstance(self.action_scale, list) and len(self.action_scale) != len(self.joint_order):
            raise ValueError("action_scale must match the policy joint count")
        return self

    @property
    def observation_size(self) -> int:
        return sum(block.size for block in self.obs_layout)


class WorldCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    body: str
    gravity_m_s2: float
    surface: str
    friction: float
    terrain_kind: Literal["generated", "real"] = "generated"
    terrain_preset: str | None = None
    source: Literal["builtin", "upload"] = "builtin"
    license: str
    credit: str


def _validate_relative_asset_path(value: str) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0].endswith(":")):
        raise ValueError("library asset paths must be relative and cannot escape the project")
