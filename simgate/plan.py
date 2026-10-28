"""Validated single-response plans produced by the Director."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import presets
from .batch import Variation
from .library import LibraryRegistry
from .spec import Load, Mission, Obstacle, Place, Robot, Structure
from .terrain_data import list_presets

Workflow = Literal["build", "race", "compare", "stress_test", "edit", "custom", "answer"]
TerrainKind = Literal["flat", "hills", "rocky", "slope", "craters", "real"]
Surface = Literal["rock", "regolith", "gravel", "sand", "mud", "ice"]
RobotKind = Literal["rover", "walker"]
SceneDomain = Literal["robot_terrain", "structure"]


class WorldPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str | None = None
    surface: Surface | None = None
    terrain_kind: TerrainKind | None = None
    preset: str | None = None
    slope_deg: float | None = Field(default=None, ge=-89, le=89)
    roughness: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_registered_values(self) -> "WorldPlan":
        if self.body is not None and self.body not in presets.GRAVITY:
            raise ValueError(f"Unknown body {self.body!r}; available: {', '.join(presets.GRAVITY)}")
        if self.surface is not None and self.surface not in presets.SURFACE_FRICTION:
            raise ValueError(
                f"Unknown surface {self.surface!r}; available: {', '.join(presets.SURFACE_FRICTION)}"
            )
        if self.preset is not None:
            registry = LibraryRegistry()
            preset_ids = {world.id for world in registry.worlds()}
            terrain_ids = set(list_presets())
            if self.preset not in preset_ids and self.preset not in terrain_ids:
                available = sorted(preset_ids | terrain_ids)
                raise ValueError(f"Unknown preset {self.preset!r}; available: {', '.join(available)}")
        if self.terrain_kind == "real" and self.preset is None:
            raise ValueError("terrain_kind 'real' requires a registered terrain preset.")
        if self.slope_deg is not None and self.terrain_kind not in (None, "slope"):
            raise ValueError("slope_deg can only be used with terrain_kind 'slope'.")
        return self


class RobotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1, max_length=80)
    kind: RobotKind | None = None
    robot_id: str | None = None
    policy_id: str | None = None
    count: int = Field(default=1, ge=1, le=6)
    legs: Literal[4, 6] | None = None
    wheels: Literal[4, 6] | None = None
    start_place: Place | None = None
    goal_place: Place | None = None

    @model_validator(mode="after")
    def validate_library_selection(self) -> "RobotPlan":
        if (self.kind is None) == (self.robot_id is None):
            raise ValueError("Provide exactly one of kind or robot_id.")
        if self.robot_id is None:
            if self.policy_id is not None:
                raise ValueError("policy_id requires robot_id.")
            return self
        registry = LibraryRegistry()
        try:
            robot = registry.robot(self.robot_id)
        except ValueError as exc:
            available = ", ".join(card.id for card in registry.robots())
            raise ValueError(f"robot_id {self.robot_id!r} doesn't exist; available: {available}") from exc
        if self.policy_id is not None:
            try:
                policy = registry.policy(self.policy_id)
            except ValueError as exc:
                available = ", ".join(card.id for card in registry.compatible_policies(robot.id))
                raise ValueError(
                    f"policy_id {self.policy_id!r} doesn't exist for robot {robot.id!r}; available: {available}"
                ) from exc
            if policy not in registry.compatible_policies(robot.id):
                available = ", ".join(card.id for card in registry.compatible_policies(robot.id))
                raise ValueError(
                    f"policy_id {self.policy_id!r} is incompatible with {robot.id!r}; available: {available}"
                )
        return self


class GoalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    place: Place | None = None
    xy: list[float] | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "GoalPlan":
        if (self.place is None) == (self.xy is None):
            raise ValueError("Provide exactly one of place or xy for the shared goal.")
        if self.xy is not None and len(self.xy) != 2:
            raise ValueError("Goal coordinates must contain exactly two values.")
        return self


class EditChanges(BaseModel):
    """Changes to an existing scene; only fields given here change. Checked against the scene in Director.plan."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=120)
    world: WorldPlan | None = None
    robot: Robot | None = None
    robots: list[RobotPlan] | None = Field(default=None, min_length=1, max_length=6)
    obstacles: list[Obstacle] | None = Field(default=None, max_length=30)
    goal: GoalPlan | None = None
    mission: Mission | None = None
    duration_s: float | None = Field(default=None, gt=0, le=120)
    structure: Structure | None = None
    load: Load | None = None

    @model_validator(mode="before")
    @classmethod
    def merge_world_aliases(cls, data: Any) -> Any:
        """Accept {"terrain": {...}} and {"environment": {...}} as parts of world."""
        if not isinstance(data, dict) or not ({"terrain", "environment"} & set(data)):
            return data
        data = dict(data)
        world = dict(data.get("world") or {})
        terrain = data.pop("terrain", None) or {}
        environment = data.pop("environment", None) or {}
        if not isinstance(terrain, dict) or not isinstance(environment, dict):
            raise ValueError("terrain and environment changes must be objects.")
        if "kind" in terrain:
            terrain = {**terrain, "terrain_kind": terrain["kind"]}
            del terrain["kind"]
        data["world"] = {**environment, **terrain, **world}
        return data

    def given(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: Workflow
    summary: str = Field(min_length=1, max_length=400)
    domain: SceneDomain = "robot_terrain"
    world: WorldPlan = Field(default_factory=WorldPlan)
    robots: list[RobotPlan] = Field(default_factory=list, max_length=6)
    structure: Structure | None = None
    load: Load | None = None
    obstacles: list[Obstacle] = Field(default_factory=list, max_length=30)
    goal: GoalPlan | None = None
    variants: list[Variation] = Field(default_factory=list, max_length=23)
    question: str | None = Field(default=None, max_length=1000)
    changes: EditChanges = Field(default_factory=EditChanges)
    custom_request: str | None = Field(default=None, max_length=2000)
    run_now: bool = False
    reply: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_workflow_payload(self) -> "Plan":
        if sum(robot.count for robot in self.robots) > 6:
            raise ValueError("A plan can expand to at most six robots.")
        if self.workflow == "race" and sum(robot.count for robot in self.robots) < 2:
            raise ValueError("workflow 'race' requires at least two robots after count expansion.")
        if self.workflow == "compare" and not self.variants:
            raise ValueError("workflow 'compare' requires at least one variant.")
        if self.workflow == "stress_test" and not self.question:
            raise ValueError("workflow 'stress_test' requires a question.")
        if self.workflow == "edit" and not self.changes.given():
            raise ValueError("workflow 'edit' requires at least one change.")
        if self.workflow == "custom" and not self.custom_request:
            raise ValueError("workflow 'custom' requires custom_request.")
        if self.workflow == "answer" and not self.reply:
            raise ValueError("workflow 'answer' requires reply.")
        if self.domain == "structure" and self.workflow in {"race", "compare", "stress_test", "custom"}:
            raise ValueError(f"workflow {self.workflow!r} is not available for structure scenes.")
        if self.world.slope_deg is not None and self.world.terrain_kind not in (None, "slope"):
            raise ValueError("slope_deg can only be used with terrain_kind 'slope'.")
        return self
