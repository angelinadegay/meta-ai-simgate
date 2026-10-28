"""Validated tools that let the Director operate on one mutable world draft."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import agents, config, llm, presets
from .backends.base import BuildResult, Check, Simulator
from .library import LibraryRegistry
from .spec import Mission, Obstacle, Robot, RobotEntry, SceneSpec, Terrain


Place = Literal["start_area", "center", "lowest", "highest", "north", "south", "east", "west"]
TerrainKind = Literal["flat", "hills", "rocky", "slope", "craters", "real"]
RobotKind = Literal["rover", "walker"]
Surface = Literal["rock", "regolith", "gravel", "sand", "mud", "ice"]


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetWorldArgs(ToolArgs):
    body: str | None = None
    surface: Surface | None = None
    terrain_kind: TerrainKind | None = None
    preset: str | None = None
    slope_deg: float | None = Field(default=None, ge=-89, le=89)
    roughness: float | None = Field(default=None, ge=0, le=1)


class SpawnRobotArgs(ToolArgs):
    kind: RobotKind | None = None
    robot_id: str | None = None
    policy_id: str | None = None
    label: str | None = Field(default=None, min_length=1, max_length=80)
    start: Place | list[float] | None = None
    goal: Place | list[float] | None = None

    @model_validator(mode="after")
    def validate_choice(self) -> "SpawnRobotArgs":
        if (self.kind is None) == (self.robot_id is None):
            raise ValueError("Provide exactly one of kind or robot_id.")
        if self.policy_id is not None and self.robot_id is None:
            raise ValueError("policy_id applies only to a library robot_id.")
        if any(value is not None and isinstance(value, list) and len(value) != 2 for value in (self.start, self.goal)):
            raise ValueError("Start and goal coordinates must contain exactly two values.")
        return self


class RemoveRobotArgs(ToolArgs):
    id: str


class AddObstaclesArgs(ToolArgs):
    obstacles: list[Obstacle] = Field(min_length=1, max_length=30)


class SetGoalArgs(ToolArgs):
    robot_id: str = "all"
    place: Place | None = None
    xy: list[float] | None = None

    @model_validator(mode="after")
    def validate_target(self) -> "SetGoalArgs":
        if (self.place is None) == (self.xy is None):
            raise ValueError("Provide exactly one of place or xy.")
        if self.xy is not None and len(self.xy) != 2:
            raise ValueError("Goal coordinates must contain exactly two values.")
        return self


class CompareArgs(ToolArgs):
    variants: list[dict[str, Any]] = Field(min_length=1, max_length=23)


class StressTestArgs(ToolArgs):
    question: str = Field(min_length=1, max_length=1000)


class BuildCustomArgs(ToolArgs):
    request: str = Field(min_length=1, max_length=2000)


class AskUserArgs(ToolArgs):
    question: str = Field(min_length=1, max_length=1000)


class ExplainArgs(ToolArgs):
    focus: str | None = Field(default=None, max_length=500)


@dataclass
class ToolResult:
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    """Mutable Director state; tool outputs remain plain data for the UI."""

    spec: SceneSpec
    simulator: Simulator | None = None
    build: BuildResult | None = None
    checks: list[Check] = field(default_factory=list)
    results: dict[str, Any] | None = None
    question: str | None = None
    emit: Any = None

    def state(self) -> dict[str, Any]:
        return {
            "spec": self.spec.model_dump(mode="json"),
            "built": self.build is not None,
            "checks": [check.__dict__ for check in self.checks],
            "has_results": self.results is not None,
        }


def _validate(model: type[BaseModel], args: dict[str, Any]) -> BaseModel:
    return model.model_validate(args)


def _result(text: str, **data: Any) -> ToolResult:
    return ToolResult(text, data)


def _apply_place(entry: RobotEntry, field_name: Literal["start", "goal"], value: Place | list[float]) -> None:
    if field_name == "start":
        entry.start_xy, entry.start_place = (list(value), None) if isinstance(value, list) else (None, value)
    else:
        entry.goal_xy, entry.goal_place = (list(value), None) if isinstance(value, list) else (None, value)


def _refresh_spec(spec: SceneSpec) -> None:
    spec.assumptions = []
    spec.fill_defaults()


def set_world(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(SetWorldArgs, args)
    assert isinstance(parsed, SetWorldArgs)
    spec = world.spec.model_copy(deep=True)
    if spec.domain != "robot_terrain":
        raise ValueError("set_world currently supports robot scenes only.")
    registry = LibraryRegistry()
    spec.terrain = spec.terrain or Terrain()
    if parsed.preset is not None:
        world_card = next((item for item in registry.worlds() if item.id == parsed.preset), None)
        if world_card is not None:
            spec.environment.body = world_card.body
            spec.environment.gravity = world_card.gravity_m_s2
            spec.environment.surface = world_card.surface
            spec.environment.friction = world_card.friction
            spec.terrain = Terrain(
                kind=world_card.terrain_kind if world_card.terrain_kind == "real" else parsed.terrain_kind or (spec.terrain.kind if spec.terrain else "hills"),
                preset=world_card.terrain_preset,
            ) if world_card.terrain_kind == "real" else (spec.terrain or Terrain())
        else:
            available = {name for name in registry.worlds() if name.terrain_kind == "real"}
            real = next((item for item in available if item.terrain_preset == parsed.preset), None)
            if real is None:
                raise ValueError(f"Unknown world or terrain preset: {parsed.preset}")
            spec.environment.body = real.body
            spec.environment.gravity = real.gravity_m_s2
            spec.environment.surface = real.surface
            spec.environment.friction = real.friction
            spec.terrain = Terrain(kind="real", preset=real.terrain_preset)
    if parsed.body is not None:
        if parsed.body not in presets.GRAVITY:
            raise ValueError(f"Unknown body {parsed.body!r}; choose a registered world.")
        spec.environment.body = parsed.body
        spec.environment.gravity = presets.gravity_for(parsed.body)
    if parsed.surface is not None:
        spec.environment.surface = parsed.surface
        spec.environment.friction = presets.friction_for(parsed.surface)
    if parsed.terrain_kind is not None:
        spec.terrain.kind = parsed.terrain_kind
        if parsed.terrain_kind != "real":
            spec.terrain.preset = None
    if parsed.slope_deg is not None:
        if spec.terrain.kind != "slope":
            raise ValueError("slope_deg can only be set for slope terrain.")
        spec.terrain.slope_deg = parsed.slope_deg
    if parsed.roughness is not None:
        spec.terrain.roughness = parsed.roughness
    _refresh_spec(spec)
    world.spec = spec
    return _result(
        f"World set to {spec.environment.body} with {spec.environment.surface} on {spec.terrain.kind} terrain.",
        spec=spec.model_dump(mode="json"),
    )


def spawn_robot(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(SpawnRobotArgs, args)
    assert isinstance(parsed, SpawnRobotArgs)
    spec = world.spec.model_copy(deep=True)
    if spec.domain != "robot_terrain":
        raise ValueError("Robots can only be added to robot scenes.")
    if len(spec.robots) >= 6:
        raise ValueError("A world can contain at most six robots.")
    registry = LibraryRegistry()
    if parsed.robot_id is not None:
        card = registry.robot(parsed.robot_id)
        if card.source == "builtin":
            robot = Robot(kind="walker" if card.kind == "legged" else "rover", mass_kg=card.mass_kg)
        else:
            policies = registry.compatible_policies(card.id)
            policy_id = parsed.policy_id or (policies[0].id if policies else None)
            if policy_id is None or policy_id not in {policy.id for policy in policies}:
                raise ValueError(f"Robot {card.id!r} has no compatible selected policy.")
            robot = Robot(kind="library", robot_id=card.id, policy_id=policy_id, mass_kg=card.mass_kg)
        default_label = card.name
    else:
        robot = Robot(kind=parsed.kind)
        default_label = parsed.kind.title()
    index = 1
    existing = {entry.id for entry in spec.robots}
    while f"r{index}" in existing:
        index += 1
    entry = RobotEntry(id=f"r{index}", label=parsed.label or f"{default_label} {index}", robot=robot)
    if parsed.start is not None:
        _apply_place(entry, "start", parsed.start)
    if parsed.goal is not None:
        _apply_place(entry, "goal", parsed.goal)
    spec.robots.append(entry)
    _refresh_spec(spec)
    world.spec = spec
    return _result(f"Spawned {entry.label} as {entry.id}.", robot=entry.model_dump(mode="json"))


def remove_robot(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(RemoveRobotArgs, args)
    assert isinstance(parsed, RemoveRobotArgs)
    spec = world.spec.model_copy(deep=True)
    if len(spec.robots) <= 1:
        raise ValueError("A world must keep at least one robot.")
    remaining = [entry for entry in spec.robots if entry.id != parsed.id]
    if len(remaining) == len(spec.robots):
        raise ValueError(f"Unknown robot id: {parsed.id}")
    spec.robots = remaining
    _refresh_spec(spec)
    world.spec = spec
    return _result(f"Removed robot {parsed.id}.", robots=[entry.model_dump(mode="json") for entry in remaining])


def list_library(world: World, args: dict[str, Any] | None = None) -> ToolResult:
    if args:
        raise ValueError("list_library takes no arguments.")
    registry = LibraryRegistry()
    data = {
        "robots": [card.model_dump(mode="json") for card in registry.robots()],
        "policies": [card.model_dump(mode="json") for card in registry.policies()],
        "worlds": [card.model_dump(mode="json") for card in registry.worlds()],
    }
    return _result(
        f"Library has {len(data['robots'])} robots, {len(data['policies'])} policies, and {len(data['worlds'])} worlds.",
        **data,
    )


def add_obstacles(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(AddObstaclesArgs, args)
    assert isinstance(parsed, AddObstaclesArgs)
    spec = world.spec.model_copy(deep=True)
    if spec.domain != "robot_terrain":
        raise ValueError("Obstacles can only be added to robot scenes.")
    spec.obstacles.extend(parsed.obstacles)
    _refresh_spec(spec)
    world.spec = spec
    return _result(f"Added {len(parsed.obstacles)} obstacle entries.", obstacles=[o.model_dump() for o in spec.obstacles])


def clear_obstacles(world: World, args: dict[str, Any] | None = None) -> ToolResult:
    if args:
        raise ValueError("clear_obstacles takes no arguments.")
    spec = world.spec.model_copy(deep=True)
    spec.obstacles = []
    _refresh_spec(spec)
    world.spec = spec
    return _result("Cleared all obstacles.")


def set_goal(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(SetGoalArgs, args)
    assert isinstance(parsed, SetGoalArgs)
    spec = world.spec.model_copy(deep=True)
    if not spec.robots:
        raise ValueError("There is no robot scene to assign a goal to.")
    entries = spec.robots if parsed.robot_id == "all" else [entry for entry in spec.robots if entry.id == parsed.robot_id]
    if not entries:
        raise ValueError(f"Unknown robot id: {parsed.robot_id}")
    target: Place | list[float] = parsed.place if parsed.place is not None else parsed.xy
    for entry in entries:
        _apply_place(entry, "goal", target)
    _refresh_spec(spec)
    world.spec = spec
    return _result(
        f"Set the goal for {len(entries)} robot(s).",
        goals={entry.id: {"goal_xy": entry.goal_xy, "goal_place": entry.goal_place} for entry in entries},
    )


def build_and_check(world: World, args: dict[str, Any] | None = None) -> ToolResult:
    if args:
        raise ValueError("build_and_check takes no arguments.")
    simulator = agents.route(world.spec)
    logs: list[str] = []
    build = agents.build(simulator, world.spec, logs.append)
    checks = simulator.validate(world.spec, build)
    world.simulator, world.build, world.checks = simulator, build, checks
    failed = [check for check in checks if not check.passed]
    text = (
        f"Build and checks complete: {len(checks) - len(failed)}/{len(checks)} checks passed."
        if checks else "Build complete; the checker returned no checks."
    )
    return _result(
        text,
        checks=[check.__dict__ for check in checks],
        code=build.code,
        language=build.language,
        scene=build.scene,
        logs=logs,
        passed=not failed,
    )


def run(world: World, args: dict[str, Any] | None = None) -> ToolResult:
    if args:
        raise ValueError("run takes no arguments.")
    if world.simulator is None or world.build is None:
        raise ValueError("Build and check the scene before running it.")
    failures = [check for check in world.checks if not check.passed]
    if failures:
        raise ValueError("Cannot run while checks fail: " + "; ".join(check.name for check in failures))
    for _chunk in world.simulator.run(world.spec, world.build):
        pass
    world.results = world.simulator.summary()
    return _result("Simulation finished.", results=world.results)


def compare(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(CompareArgs, args)
    assert isinstance(parsed, CompareArgs)
    from .batch import run_batch

    results = run_batch(world.spec, parsed.variants, None)
    world.results = {"runs": results}
    rows = [
        {
            "label": result["label"],
            "passed": result["passed"],
            "reason": result["reason"],
            "summary": result["summary"],
            **({"error": result["error"]} if result.get("error") else {}),
        }
        for result in results
    ]
    return _result(f"Compared {len(results)} conditions.", rows=rows)


def stress_test(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(StressTestArgs, args)
    assert isinstance(parsed, StressTestArgs)
    from .batch import run_batch

    variations = agents.plan_stress(world.spec, parsed.question)
    results = run_batch(world.spec, variations, None)
    world.results = {"runs": results}
    return _result(
        f"Stress test completed {len(results)} runs.",
        variations=variations,
        results=results,
    )


def _validate_custom_fragment(fragment: str, build: BuildResult) -> str:
    try:
        wrapper = ET.fromstring(f"<worldbody>{fragment}</worldbody>")
    except ET.ParseError as exc:
        raise ValueError(f"Builder returned invalid MJCF fragment: {exc}") from exc
    allowed = {"body", "geom"}

    def validate_node(node: ET.Element) -> None:
        if node.tag not in allowed:
            raise ValueError(f"Custom MJCF may contain bodies and geoms only; found {node.tag!r}.")
        if node.tag == "body" and node.find("joint") is not None:
            raise ValueError("Custom bodies may not add joints.")
        if node.tag == "body":
            for child in node:
                validate_node(child)
        if node.tag == "geom":
            if list(node):
                raise ValueError("Custom geoms cannot contain child elements.")

    for child in wrapper:
        validate_node(child)
    existing_names = {
        node.get("name")
        for node in ET.fromstring(build.code).iter()
        if node.get("name")
    }
    for node in wrapper.iter():
        name = node.get("name")
        if name in existing_names:
            raise ValueError(f"Custom MJCF name conflicts with the scene: {name}")
    return "".join(ET.tostring(child, encoding="unicode") for child in wrapper)


def _insert_custom_fragment(code: str, fragment: str) -> str:
    root = ET.fromstring(code)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("The generated MJCF has no worldbody.")
    fragment_root = ET.fromstring(f"<worldbody>{fragment}</worldbody>")
    for child in list(fragment_root):
        worldbody.append(child)
    return ET.tostring(root, encoding="unicode")


def build_custom(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(BuildCustomArgs, args)
    assert isinstance(parsed, BuildCustomArgs)
    if world.spec.domain != "robot_terrain":
        raise ValueError("Custom MJCF fragments are available for robot scenes only.")
    simulator = agents.route(world.spec)
    if not hasattr(simulator, "template"):
        raise ValueError("The selected simulator does not support MJCF fragments.")
    template = simulator.build(world.spec)
    base_code = template.code
    messages = [
        {
            "role": "system",
            "content": (
                "You are a MuJoCo scene-detail Builder. Return ONLY an MJCF XML fragment containing "
                "additional <body> and/or <geom> elements. Do not define joints, actuators, assets, "
                "or modify existing robot bodies. Use unique names."
            ),
        },
        {
            "role": "user",
            "content": f"Scene: {world.spec.model_dump_json()}\nRequest: {parsed.request}",
        },
    ]
    errors = []
    retries = max(1, int(config.BUILDER_MAX_RETRIES))
    if config.MOCK_MODE:
        fragment = ""
        retries = 0
    else:
        fragment = ""
        for attempt in range(1, retries + 1):
            reply = llm.chat(messages, model=config.BUILDER_MODEL)
            messages.append({"role": "assistant", "content": reply})
            try:
                fragment = llm.extract_code(reply, "xml").strip()
                fragment = _validate_custom_fragment(fragment, template)
                candidate = simulator.build(world.spec, _insert_custom_fragment(base_code, fragment))
                checks = simulator.validate(world.spec, candidate)
                failed = [check for check in checks if not check.passed]
                if failed:
                    raise ValueError("; ".join(f"{check.name}: {check.detail}" for check in failed))
                world.simulator, world.build, world.checks = simulator, candidate, checks
                return _result(f"Added custom scene detail on attempt {attempt}.", attempts=attempt, scene=candidate.scene)
            except Exception as exc:
                message = f"{type(exc).__name__}: {str(exc)[:300]}"
                errors.append(message)
                messages.append({"role": "user", "content": f"Attempt {attempt} failed: {message}. Return a corrected fragment only."})
    fallback = simulator.build(world.spec)
    world.simulator, world.build, world.checks = simulator, fallback, simulator.validate(world.spec, fallback)
    return _result(
        "The custom detail was not added; using the verified scene template.",
        attempts=retries,
        errors=errors,
        scene=fallback.scene,
    )


def ask_user(world: World, args: dict[str, Any]) -> ToolResult:
    parsed = _validate(AskUserArgs, args)
    assert isinstance(parsed, AskUserArgs)
    world.question = parsed.question
    return _result(parsed.question, question=parsed.question)


def explain(world: World, args: dict[str, Any] | None = None) -> ToolResult:
    parsed = _validate(ExplainArgs, args or {})
    assert isinstance(parsed, ExplainArgs)
    if world.results is None:
        raise ValueError("There are no simulation results to explain yet.")
    metrics = world.results
    if "runs" in metrics:
        text = agents.explain_stress(world.spec, metrics["runs"])
    else:
        text = agents.explain(world.spec, metrics)
    if parsed.focus:
        text = f"{parsed.focus}: {text}"
    return _result(text, explanation=text)


TOOL_FUNCTIONS = {
    "set_world": set_world,
    "spawn_robot": spawn_robot,
    "remove_robot": remove_robot,
    "list_library": list_library,
    "add_obstacles": add_obstacles,
    "clear_obstacles": clear_obstacles,
    "set_goal": set_goal,
    "build_and_check": build_and_check,
    "run": run,
    "compare": compare,
    "build_custom": build_custom,
    "ask_user": ask_user,
    "explain": explain,
}

try:
    from . import batch as _batch
except ImportError:
    pass
else:
    TOOL_FUNCTIONS["stress_test"] = stress_test


def invoke_tool(world: World, name: str, args: dict[str, Any]) -> ToolResult:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        raise ValueError(f"Unknown tool: {name}")
    return function(world, args)


def tool_catalog() -> dict[str, str]:
    return {
        "set_world": "Choose a registered body/surface/terrain preset and physical terrain settings.",
        "spawn_robot": "Add a compatible built-in or library robot and optional start/goal.",
        "remove_robot": "Remove a robot by id, keeping at least one robot in the world.",
        "list_library": "List registered robots, policies, and worlds.",
        "add_obstacles": "Add validated shared obstacles; placement is computed by the simulator.",
        "clear_obstacles": "Remove all obstacles.",
        "set_goal": "Set a named or coordinate goal for one robot or all robots.",
        "build_and_check": "Build the world and return all checks, including failures.",
        "run": "Run the checked world and return metrics and leaderboard.",
        "compare": "Compare the same world under a list of supported condition variations.",
        "build_custom": "Add a constrained, checker-validated MJCF body/geom fragment.",
        "ask_user": "Pause and ask the user a question before continuing.",
        "explain": "Explain the latest run or comparison results.",
        **({"stress_test": "Run the registered stress-test planner and batch runner."} if "stress_test" in TOOL_FUNCTIONS else {}),
    }
