"""Validated metadata models for library robots, their plain-code controllers, and worlds."""

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
    footprint_length_m: float | None = Field(None, gt=0)
    footprint_width_m: float | None = Field(None, gt=0)
    forward_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # which plain-code controller drives the robot ("builtin": SimGate's procedural rover/walker code), and its parts
    controller: Literal["builtin", "wheels", "gait", "hold", "stand"] = "builtin"
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
        if (self.footprint_length_m is None) != (self.footprint_width_m is None):
            raise ValueError("robot footprint length and width must be provided together")
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


class ControllerCard(Card):
    """A controller written as plain code (wheel speeds, a rhythmic gait, PD hold of a pose); no learned model."""
    robot_id: str
    control_dt: float = Field(gt=0)
    joint_order: list[str] = []
    default_angles: list[float] = []
    kps: list[float] = []
    kds: list[float] = []
    command: CommandSpec | None = None
    notes: list[str] = []

    @model_validator(mode="after")
    def validate_joint_gains(self) -> "ControllerCard":
        if len(set(self.joint_order)) != len(self.joint_order):
            raise ValueError("controller joint_order must be unique")
        sizes = (len(self.default_angles), len(self.kps), len(self.kds))
        if any(sizes) and any(size != len(self.joint_order) for size in sizes):
            raise ValueError("default_angles, kps, and kds must match the controller joint count")
        return self


class WorldCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    body: str
    gravity_m_s2: float
    surface: str
    friction: float
    terrain_kind: Literal["generated", "real", "indoor"] = "generated"
    terrain_preset: str | None = None
    source: Literal["builtin", "upload"] = "builtin"
    license: str
    credit: str


def _validate_relative_asset_path(value: str) -> None:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0].endswith(":")):
        raise ValueError("library asset paths must be relative and cannot escape the project")
