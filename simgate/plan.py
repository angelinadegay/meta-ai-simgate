"""Validated single-response plans produced by the Director."""

from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr, model_validator

from . import presets
from .batch import Variation
from .library import LibraryRegistry
from .spec import Load, Mission, Obstacle, Place, Robot, Structure
from .terrain_data import list_presets


def _far_side_to_place(value: Any) -> Any:
    """Models often say "far side"; the far side of the default course is the east edge."""
    if isinstance(value, str) and value.strip().lower().replace("_", " ") in ("far side", "the far side"):
        return "east"
    return value


PlanPlace = Annotated[Place, BeforeValidator(_far_side_to_place)]

Workflow = Literal["build", "race", "compare", "stress_test", "edit", "custom", "answer", "diagnose", "redesign"]
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
    # an uploaded heightmap's width and height range for this scene, in metres; only numbers the user gave
    size_m: float | None = Field(default=None, ge=1, le=1000)
    relief_m: float | None = Field(default=None, ge=0, le=500)

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


def _robot_choices() -> str:
    """What a robot entry may name, for validation errors the planner sees on its retry."""
    library = [card.id for card in LibraryRegistry().robots() if card.source != "builtin"]
    return (f"Built-in kinds: {', '.join(get_args(RobotKind))}. Library robot_ids: {', '.join(library) or 'none'}. "
            "For library robots, give robot_id and omit kind.")


class RobotPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1, max_length=80)
    kind: RobotKind | None = None
    robot_id: str | None = None
    policy_id: str | None = None
    count: int = Field(default=1, ge=1, le=6)
    legs: Literal[4, 6] | None = None
    wheels: Literal[4, 6] | None = None
    start_place: PlanPlace | None = None
    goal_place: PlanPlace | None = None
    _notes: list[str] = PrivateAttr(default_factory=list)   # assumptions to show the user

    @model_validator(mode="before")
    @classmethod
    def check_kind(cls, data: Any) -> Any:
        """Models often add a kind ("library", "humanoid", "walker") to a library robot_id; the robot_id decides.
        Only a built-in kind of the other sort (a rover for a legged robot) is a real conflict. Without a
        robot_id, an unknown kind gets an error that lists what can be used."""
        if not isinstance(data, dict) or data.get("kind") is None:
            return data
        if data.get("robot_id") is None:
            if data["kind"] not in get_args(RobotKind):
                raise ValueError(f"kind {data['kind']!r} is not a built-in kind. {_robot_choices()}")
            return data
        data = dict(data)
        kind = data.pop("kind")
        try:
            card = LibraryRegistry().robot(data["robot_id"])
        except ValueError:
            return data   # the robot_id check below names the problem
        if kind in ("rover", "walker") and (kind == "rover") != (card.kind == "wheeled"):
            raise ValueError(f"kind {kind!r} conflicts with robot_id {card.id!r} ({card.kind}): omit kind when you "
                             f"give robot_id. {_robot_choices()}")
        return data

    @model_validator(mode="after")
    def validate_library_selection(self) -> "RobotPlan":
        if (self.kind is None) == (self.robot_id is None):
            raise ValueError(f"Provide exactly one of kind or robot_id. {_robot_choices()}")
        if self.robot_id is None:
            if self.policy_id is not None:
                raise ValueError("policy_id requires robot_id.")
            return self
        registry = LibraryRegistry()
        try:
            robot = registry.robot(self.robot_id)
        except ValueError as exc:
            raise ValueError(f"robot_id {self.robot_id!r} doesn't exist. {_robot_choices()}") from exc
        compatible = registry.compatible_policies(robot.id)
        if self.policy_id is None:   # the robot's default policy, as spawn_robot would pick
            if not compatible:
                raise ValueError(f"robot_id {robot.id!r} has no compatible policy installed.")
            self.policy_id = compatible[0].id
        else:
            try:
                policy = registry.policy(self.policy_id)
            except ValueError as exc:
                if len(compatible) == 1:
                    self._notes.append(f"There is no policy {self.policy_id!r}, so {robot.id} uses its only "
                                       f"policy, {compatible[0].id}.")
                    self.policy_id = compatible[0].id
                    return self
                available = ", ".join(card.id for card in compatible)
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

    place: PlanPlace | None = None
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
    robot_id: str | None = None     # a robot change applies to this robot only (default: every robot)
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
        return self.model_dump(exclude_none=True, exclude={"robot_id"})


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow: Workflow
    summary: str = Field(
        min_length=1, max_length=400,
        description="A short name for the scene, used as its title, e.g. \"Three rovers race on Mars hills\".",
    )
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
    robot_id: str | None = Field(default=None, max_length=40)        # diagnose/redesign: which robot (default: the first that failed)
    constraints: str | None = Field(default=None, max_length=500)    # redesign: the user's limits, in their words
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
        if self.domain == "structure" and self.workflow in {"race", "compare", "stress_test", "custom", "diagnose", "redesign"}:
            raise ValueError(f"workflow {self.workflow!r} is not available for structure scenes.")
        if self.world.slope_deg is not None and self.world.terrain_kind not in (None, "slope"):
            raise ValueError("slope_deg can only be used with terrain_kind 'slope'.")
        return self
