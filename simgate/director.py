"""The Director: one validated plan per request, executed by fixed workflows in code."""

import asyncio
import json
import re
import time
import uuid
from typing import Any, Awaitable, Callable

from . import agents, config, llm, presets
from .library import LibraryRegistry
from .plan import Plan, WorldPlan
from .spec import Robot, SceneSpec
from .tools import ToolResult, World, invoke_tool

StepCallback = Callable[[dict[str, Any]], Awaitable[None]]


class Director:
    def __init__(self, step_callback: StepCallback | None = None) -> None:
        self.step_callback = step_callback
        self._counter = 0
        self._root_id: str | None = None
        self.last_checker_id: str | None = None   # latest Checker hand-off step, parent of any repair
        self.used_rule_planner = False            # the model's plan failed twice; the user is told

    def plan(self, text: str, world_state: World | SceneSpec | dict[str, Any] | None = None) -> Plan:
        """Create a complete validated plan, retry once, then use the deterministic planner."""
        if config.MOCK_MODE:
            return _rule_plan(text, world_state)

        messages = [
            {"role": "system", "content": _plan_system()},
            {"role": "user", "content": f"Current world state:\n{_plan_world_state(world_state)}\n\nRequest:\n{text}"},
        ]
        last_error = ""
        reply = ""
        for attempt in range(2):
            if attempt:
                messages.append({"role": "assistant", "content": reply})
                messages.append({
                    "role": "user",
                    "content": f"The previous response failed validation with this exact error:\n{last_error}\nReturn a corrected Plan JSON only.",
                })
            reply = llm.chat(messages, config.PLANNER_MODEL)
            try:
                plan = Plan.model_validate(llm.extract_json(reply))
                if plan.workflow == "stress_test" and not _has_robot_scene(world_state):
                    raise ValueError("workflow 'stress_test' requires an existing robot scene.")
                if plan.workflow == "edit":
                    _check_edit(plan, text, world_state)
                return plan
            except (ValueError, TypeError) as exc:
                last_error = str(exc)
        self.used_rule_planner = True
        return _rule_plan(text, world_state)

    async def execute_plan(self, text: str, current: SceneSpec | None = None) -> tuple[World, str]:
        from . import workflows

        self._root_id = self._id()
        model = "rules" if config.MOCK_MODE else config.PLANNER_MODEL
        await self._emit_step(
            id=self._root_id,
            parent_id=None,
            agent="Director",
            model=model,
            tool=None,
            args={},
            thought="Planning the requested world in one step.",
            result="",
            status="working",
            ms=0,
        )
        started = time.perf_counter()
        plan = await asyncio.to_thread(self.plan, text, current)
        plan_step = self._id()
        await self._emit_step(
            id=plan_step,
            parent_id=self._root_id,
            agent="Director",
            model=model,
            tool="plan",
            args={"request": text[:300]},
            thought=plan.summary,
            result=(
                f"The model's plan failed validation twice; used the rules planner's {plan.workflow} plan."
                if self.used_rule_planner else f"Validated {plan.workflow} plan."
            ),
            status="done",
            ms=round((time.perf_counter() - started) * 1000),
        )
        world, message = await workflows.execute(plan, current, self)
        if self.used_rule_planner:
            message = f"I used the simple planner for this one. {message}"
        await self._finish_root(message)
        return world, message

    async def _mock_call(self, world: World, name: str, args: dict[str, Any], thought: str) -> ToolResult:
        step_id = self._id()
        started = time.perf_counter()
        await self._emit_step(
            id=step_id, parent_id=self._root_id, agent="Director", model="rules",
            tool=name, args=args, thought=thought, result="", status="working", ms=0,
        )
        result = await asyncio.to_thread(invoke_tool, world, name, args)
        await self._emit_handoffs(name, result, step_id)
        await self._emit_step(
            id=step_id, parent_id=self._root_id, agent="Director", model="rules",
            tool=name, args=args, thought=thought, result=result.text, status="done",
            ms=round((time.perf_counter() - started) * 1000),
        )
        return result

    async def _emit_handoffs(self, tool: str, result: ToolResult, parent_id: str) -> None:
        handoffs = []
        if tool == "build_and_check":
            failed = [
                check for check in result.data.get("checks", [])
                if not check.get("passed", False)
            ]
            handoffs = [
                ("Builder", config.BUILDER_MODEL if config.BUILDER_MODE == "llm" else "template", "Build finished."),
                (
                    "Checker",
                    "rules",
                    "All checks passed." if result.data.get("passed") else "; ".join(
                        f"{check.get('name')}: {check.get('detail')}" for check in failed
                    ) or "Some physics checks need attention.",
                ),
            ]
        elif tool == "build_custom":
            errors = result.data.get("errors", [])
            attempts = result.data.get("attempts", 1)
            for attempt in range(1, attempts + 1):
                detail = errors[attempt - 1] if attempt <= len(errors) else result.text
                status = "warning" if attempt <= len(errors) else "done"
                await self._emit_agent_step(
                    "Builder",
                    config.BUILDER_MODEL,
                    tool,
                    parent_id,
                    f"Custom fragment attempt {attempt}.",
                    detail,
                    status,
                )
            return
        elif tool == "run":
            handoffs = [("Simulator", "MuJoCo", "Simulation completed and metrics were measured.")]
        elif tool == "explain":
            handoffs = [("Explainer", config.PLANNER_MODEL if not config.MOCK_MODE else "rules", result.text)]
        elif tool == "stress_test":
            handoffs = [
                ("Stress planner", config.PLANNER_MODEL if not config.MOCK_MODE else "rules", "Selected condition variations."),
                ("Batch runner", "ProcessPoolExecutor", result.text),
                ("Explainer", config.PLANNER_MODEL if not config.MOCK_MODE else "rules", "Summarized the measured pattern."),
            ]
        for agent, model, message in handoffs:
            status = "warning" if agent == "Checker" and not result.data.get("passed") else "done"
            step_id = await self._emit_agent_step(
                agent, model, tool, parent_id, f"{agent} hand-off.", message, status,
            )
            if agent == "Checker":
                self.last_checker_id = step_id

    async def _emit_agent_step(
        self,
        agent: str,
        model: str,
        tool: str,
        parent_id: str,
        thought: str,
        result: str,
        status: str,
    ) -> str:
        step_id = self._id()
        await self._emit_step(
            id=step_id, parent_id=parent_id, agent=agent, model=model, tool=tool,
            args={}, thought=thought, result="", status="working", ms=0,
        )
        await self._emit_step(
            id=step_id, parent_id=parent_id, agent=agent, model=model, tool=tool,
            args={}, thought=thought, result=result, status=status, ms=0,
        )
        return step_id

    async def _emit_step(self, **payload: Any) -> None:
        if self.step_callback is not None:
            await self.step_callback({"type": "step", **payload})

    async def _finish_root(self, message: str) -> None:
        if self._root_id is not None:
            await self._emit_step(
                id=self._root_id, parent_id=None, agent="Director",
                model=config.PLANNER_MODEL if not config.MOCK_MODE else "rules",
                tool=None, args={}, thought="Director finished this request.",
                result=message, status="done", ms=0,
            )

    def _id(self) -> str:
        self._counter += 1
        return f"step-{self._counter}-{uuid.uuid4().hex[:8]}"


def _plan_system() -> str:
    registry = LibraryRegistry()
    robots = [
        {
            "id": robot.id,
            "name": robot.name,
            "kind": robot.kind,
            "compatible_policies": [policy.id for policy in registry.compatible_policies(robot.id)],
        }
        for robot in registry.robots()
        if robot.source != "builtin"   # built-ins are chosen by kind, not robot_id
    ]
    terrain_presets = agents.presets_text()
    world_ids = [world.id for world in registry.worlds()]
    schema = json.dumps(Plan.model_json_schema(), separators=(",", ":"))
    return f"""You are the Director for a physics simulation sandbox. Make a complete plan in ONE response.
Return only one JSON object matching the Plan schema below. Do not return tools or a sequence of actions.

Workflows:
- build: create a robot_terrain or structure scene and ask before running unless run_now is true.
- race: create multiple robots sharing a course; ask before running unless run_now is true.
- compare: build one scene and compare its variants.
- stress_test: choose controlled variations for the user's robustness question.
- edit: modify only the requested parts of the existing world.
- custom: request a custom MJCF fragment.
- answer: answer a question without simulation.
- Existing structural scenes use domain "structure" with structure and load fields.

Guidance:
- Use only robot, policy, and terrain-preset IDs listed below. Built-in kinds are rover and walker.
- Use robot_id with its compatible policy_id for library robots such as the humanoid.
- Put repeated identical robots in one robots entry with count (1–6); code expands the count.
- summary is a short name for the scene, used as its title: "Three rovers race on Mars hills", not an action
  such as "Create a rover scene on Mars hills".
- Use named places rather than guessing coordinates. A shared goal can be set with goal.
- Never invent physical values. Select known body/surface/terrain options and let code apply defaults.
- run_now is true only when the user explicitly asks to run, race, start, or test now. Otherwise ask before running.
- A question about capabilities or facts with no requested world change uses workflow answer.
- Compare variants must each change a small number of valid conditions. Stress tests state the question.

Short examples:
Race plan: {{"workflow":"race","summary":"Three rovers race to the east side","robots":[{{"kind":"rover","count":3}}],"goal":{{"place":"east"}},"run_now":false}}
Compare plan: {{"workflow":"compare","summary":"Rover on Mars and Earth","robots":[{{"kind":"rover"}}],"variants":[{{"label":"Mars","gravity":3.71}},{{"label":"Earth","gravity":9.81}}],"run_now":false}}
Stress plan: {{"workflow":"stress_test","summary":"Rover robustness test","question":"How robust is it?","run_now":false}}
Edit plan: {{"workflow":"edit","summary":"Rover on a 30° Mars slope","changes":{{"world":{{"terrain_kind":"slope","slope_deg":30}},"title":"Rover on a 30° Mars slope"}}}}
- An edit lists only the fields that change. Include a new title when the change makes the old one wrong.
Answer plan: {{"workflow":"answer","summary":"What SimGate can do","reply":"You can build robot scenes, compare conditions, race robots, or run stress tests."}}

Available library robots and policies:
{json.dumps(robots, separators=(",", ":"))}
Built-in robot kinds: rover, walker

Available real-terrain presets:
{terrain_presets}
Available world preset IDs:
{json.dumps(world_ids, separators=(",", ":"))}

Plan JSON schema:
{schema}"""


def _plan_world_state(world_state: World | SceneSpec | dict[str, Any] | None) -> str:
    if world_state is None:
        return "{}"
    if isinstance(world_state, World):
        return _compact_world_state(world_state)
    if isinstance(world_state, SceneSpec):
        spec = world_state
        compact = {
            "domain": spec.domain,
            "title": spec.title,
            "environment": spec.environment.model_dump(mode="json"),
            "terrain": spec.terrain.model_dump(mode="json") if spec.terrain else None,
            "structure": spec.structure.model_dump(mode="json") if spec.structure else None,
            "load": spec.load.model_dump(mode="json") if spec.load else None,
            "robots": [
                {
                    "id": robot.id,
                    "label": robot.label,
                    "robot": robot.robot.model_dump(mode="json"),
                    "start": robot.start_xy or robot.start_place,
                    "goal": robot.goal_xy or robot.goal_place,
                }
                for robot in spec.robots
            ],
            "obstacles": [obstacle.model_dump(mode="json") for obstacle in spec.obstacles],
        }
        return json.dumps(compact, separators=(",", ":"))
    return json.dumps(world_state, separators=(",", ":"), default=str)[:12000]


def _rule_plan(text: str, world_state: World | SceneSpec | dict[str, Any] | None) -> Plan:
    current = world_state.spec if isinstance(world_state, World) else (
        world_state if isinstance(world_state, SceneSpec) else None
    )
    source = text.lower()
    is_stress = bool(re.search(r"\bstress[- ]test\b|\bhow robust\b|\bunder different conditions\b", source))
    is_compare = bool(re.search(r"\bcompare\b|\bversus\b|\bvs\.?\b", source))
    is_custom = bool(re.search(r"\bcustom\b", source))
    has_world_request = bool(re.search(
        r"\b(build|create|make|put|race|compare|stress|run|simulate|test|spawn|add|remove)\b",
        source,
    ))
    if current is None and not has_world_request and re.search(r"\b(what|how|why|can you|tell me)\b", source):
        return Plan(
            workflow="answer",
            summary="Answering without starting a simulation.",
            reply="I can create robot worlds, set goals and obstacles, compare conditions, and run races or stress tests.",
        )

    planned = agents._mock_plan(text, current)
    if is_stress and (current is None or current.domain != "robot_terrain"):
        return Plan(
            workflow="answer",
            summary="A stress test needs an existing robot scene.",
            reply="Create a robot scene first, then ask me to stress-test it.",
        )
    if is_stress:
        return Plan(
            workflow="stress_test",
            summary="Test the current robot under varied conditions.",
            question=text,
        )
    if planned.domain == "structure":
        if current is not None and current.domain == "structure":
            changes: dict[str, Any] = {}
            if planned.structure != current.structure and planned.structure is not None:
                changes["structure"] = planned.structure.model_dump(mode="json")
            if planned.load != current.load and planned.load is not None:
                changes["load"] = planned.load.model_dump(mode="json")
            if changes:
                changes["title"] = planned.title
            if not changes:
                return Plan(
                    workflow="answer",
                    domain="structure",
                    summary="The structural scene needs no changes.",
                    reply="The current structural scene is ready. Describe a change or ask to run it.",
                )
            return Plan(
                workflow="edit",
                domain="structure",
                summary="Applying the requested structural changes.",
                changes=changes,
            )
        return Plan(
            workflow="build",
            domain="structure",
            summary=planned.title.rstrip(".") + ".",
            structure=planned.structure,
            load=planned.load,
            run_now=_asks_to_run(source),
        )
    if is_custom:
        return Plan(workflow="custom", summary=f"{planned.title} with a custom detail", custom_request=text)
    world = {
        "body": planned.environment.body,
        "surface": planned.environment.surface,
        "terrain_kind": planned.terrain.kind if planned.terrain else None,
        "preset": planned.terrain.preset if planned.terrain and planned.terrain.kind == "real" else None,
        "slope_deg": planned.terrain.slope_deg if planned.terrain and planned.terrain.kind == "slope" else None,
        "roughness": planned.terrain.roughness if planned.terrain else None,
    }
    mission = planned.mission
    goal = None
    if mission is not None:
        if mission.goal_xy is not None:
            goal = {"xy": mission.goal_xy}
        elif mission.goal_place is not None:
            goal = {"place": mission.goal_place}

    if is_compare:
        body_names = [name for name in presets.GRAVITY if re.search(rf"\b{name}\b", source)]
        if len(body_names) < 2:
            body_names = ["earth", "mars"]
        variations = [
            {"label": body.title(), "gravity": presets.GRAVITY[body]}
            for body in body_names[:2]
        ]
        robots = _rule_robot_plans(text, planned.robot)
        return Plan(
            workflow="compare",
            summary=f"{planned.title} on " + " and ".join(variation["label"] for variation in variations),
            world=world,
            robots=robots,
            variants=variations,
        )

    robots = _rule_robot_plans(text, planned.robot)
    race = bool(re.search(r"\brace\b", source)) and sum(robot["count"] for robot in robots) > 1
    requested_run = _asks_to_run(source)
    if current is not None and race:
        return Plan(
            workflow="race",
            summary="Race the requested robots on the current course.",
            world=world,
            robots=robots,
            obstacles=[obstacle.model_dump(mode="json") for obstacle in planned.obstacles],
            goal=goal,
            run_now=requested_run,
        )

    if current is not None and not _requests_scene_change(source):
        return Plan(
            workflow="answer",
            summary="Answering a question about the current scene.",
            reply="The current world is ready for a run; ask me to run it or describe a change.",
        )
    if current is not None:
        changes: dict[str, Any] = {}
        if (
            planned.environment.body != current.environment.body
            or planned.environment.surface != current.environment.surface
            or planned.terrain != current.terrain
        ):
            changes["world"] = world
        if planned.obstacles != current.obstacles:
            changes["obstacles"] = [obstacle.model_dump(mode="json") for obstacle in planned.obstacles]
        if planned.robot != current.robot:
            changes["robot"] = planned.robot.model_dump(mode="json") if planned.robot else None
        if planned.mission != current.mission and planned.mission is not None:
            changes["mission"] = planned.mission.model_dump(mode="json")
        if changes:
            changes["title"] = planned.title
        return Plan(
            workflow="edit" if changes else "answer",
            summary="Applying only the requested changes to the current world." if changes
            else "No scene changes are needed.",
            changes=changes,
            reply=None if changes else "I couldn't identify a scene change. Tell me what you'd like to adjust.",
        )
    return Plan(
        workflow="race" if race else "build",
        summary=planned.title.rstrip(".") + ".",
        world=world,
        robots=robots,
        obstacles=[obstacle.model_dump(mode="json") for obstacle in planned.obstacles],
        goal=goal,
        run_now=requested_run,
    )


def _current_spec(world_state: World | SceneSpec | dict[str, Any] | None) -> SceneSpec | None:
    if isinstance(world_state, World):
        return world_state.spec
    return world_state if isinstance(world_state, SceneSpec) else None


def _check_edit(plan: Plan, text: str, world_state: World | SceneSpec | dict[str, Any] | None) -> None:
    """Apply the edit to a copy of the scene so any problem gets the planner's one retry."""
    from .workflows import apply_edit

    current = _current_spec(world_state)
    if current is None:
        raise ValueError("workflow 'edit' requires an existing scene; use workflow 'build' for a new one.")
    source = text.lower()
    if re.search(r"\bsteeper\b", source) and not re.search(r"\d", source):   # no number given: 10° steeper
        terrain = current.terrain
        base = terrain.slope_deg if terrain is not None and terrain.kind == "slope" else 0.0
        world = plan.changes.world.model_dump(exclude_none=True) if plan.changes.world else {}
        slope = min(base + 10.0, presets.SLOPE_RANGE_DEG[1])
        if world.get("slope_deg") not in (None, slope):
            plan.changes.title = None   # a title the model wrote may name the angle it chose
        world["slope_deg"] = slope
        if world.get("terrain_kind") not in (None, "slope"):
            del world["terrain_kind"]
        plan.changes.world = WorldPlan.model_validate(world)
    before = current.model_copy(deep=True)
    before.assumptions = []
    before.fill_defaults()
    after = apply_edit(current, plan.changes).spec
    if after.model_dump(exclude={"assumptions"}) == before.model_dump(exclude={"assumptions"}):
        raise ValueError(
            f"this edit changes nothing: {json.dumps(plan.changes.given(), default=str)} leaves the scene as it is. "
            "Set the fields that should change to new values."
        )


_CHANGE_VERB = re.compile(
    r"\b(build|create|make|put|race|compare|stress|run|simulate|test|spawn|add|remove|use|set|change|switch|try"
    r"|swap|replace|increase|decrease|raise|lower|give|turn|move)\b"
)
_SCENE_ELEMENT = re.compile(
    r"\b(wheel(?:s|ed)?|legs?|legged|slope|steeper|flatter|gravity|earth|moon|mars|titan|europa|enceladus"
    r"|saturn|jupiter|martian|lunar|rock|regolith|gravel|sand|mud|ice|icy|rocks|boulders?|steps?|stairs|gaps?"
    r"|trench|ramps?|walls?|logs?|obstacles?|rover|walker|dog|spider|hexapod|humanoid|g1|robots?|faster|slower"
    r"|bigger|smaller|larger|hills?|flat|craters?|rocky|goal|summit|top|bottom|peak|north|south|east|west"
    r"|center|centre)\b"
)
_QUESTION = re.compile(r"^\s*(what|why|how|when|where|which|who|is|are|does|do|can you|could you|tell me)\b|\?\s*$")


def _requests_scene_change(source: str) -> bool:
    """A change verb, or a scene element named in anything but a question ("six wheels", "ice")."""
    return bool(_CHANGE_VERB.search(source) or (_SCENE_ELEMENT.search(source) and not _QUESTION.search(source)))


def _asks_to_run(source: str) -> bool:
    """An explicit run request: a run verb anywhere, or "go", "start" or "run it" closing the request."""
    return bool(re.search(r"\b(run|simulate|start|go ahead|test now)\b", source)
                or re.search(r"\b(go|start|run it)\s*[.!]*\s*$", source))


def _has_robot_scene(world_state: World | SceneSpec | dict[str, Any] | None) -> bool:
    if isinstance(world_state, World):
        spec = world_state.spec
    elif isinstance(world_state, SceneSpec):
        spec = world_state
    else:
        return False
    return spec.domain == "robot_terrain" and bool(spec.robots or spec.robot)


def _rule_robot_plans(text: str, fallback: Robot | None) -> list[dict[str, Any]]:
    requested = _requested_robots(text, fallback)
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for robot in requested:
        signature = (
            robot.get("kind"), robot.get("robot_id"), robot.get("policy_id"),
            robot.get("legs"), robot.get("wheels"),
        )
        if signature not in grouped:
            grouped[signature] = {
                key: robot[key]
                for key in ("kind", "robot_id", "policy_id", "label", "legs", "wheels")
                if key in robot
            }
            grouped[signature]["count"] = 0
        grouped[signature]["count"] += 1
    return list(grouped.values())


def _compact_world_state(world: World) -> str:
    spec = world.spec
    state = {
        "domain": spec.domain,
        "environment": {
            "body": spec.environment.body,
            "gravity": spec.environment.gravity,
            "surface": spec.environment.surface,
            "friction": spec.environment.friction,
        },
        "terrain": {
            "kind": spec.terrain.kind if spec.terrain else None,
            "preset": spec.terrain.preset if spec.terrain else None,
            "slope_deg": spec.terrain.slope_deg if spec.terrain else None,
        },
        "structure": spec.structure.model_dump(mode="json") if spec.structure else None,
        "load": spec.load.model_dump(mode="json") if spec.load else None,
        "robots": [
            {
                "id": entry.id,
                "label": entry.label,
                "kind": entry.robot.kind,
                "robot_id": entry.robot.robot_id,
                "policy_id": entry.robot.policy_id,
                "start": entry.start_xy or entry.start_place,
                "goal": entry.goal_xy or entry.goal_place,
            }
            for entry in spec.robots
        ],
        "obstacle_count": len(spec.obstacles),
        "built": world.build is not None,
        "checks": [
            {"name": check.name, "passed": check.passed, "detail": check.detail[:200]}
            for check in world.checks
        ],
        "has_results": world.results is not None,
    }
    return json.dumps(state, separators=(",", ":"), default=str)


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}


def _requested_robots(text: str, fallback: Robot | None) -> list[dict[str, Any]]:
    source = text.lower()
    pattern = re.compile(r"(?:(\d+|one|two|three|four|five|six)\s+)?(g1s?|humanoids?|rovers?|walkers?|hexapods?)")
    matches = list(pattern.finditer(source))
    if re.search(r"\bg1\s+humanoid\b", source):
        matches = [match for match in matches if not match.group(2).startswith("humanoid")]
    parsed: list[dict[str, Any]] = []
    for match in matches:
        amount = match.group(1)
        count = int(amount) if amount and amount.isdigit() else _NUMBER_WORDS.get(amount, 1)
        name = match.group(2)
        if name.startswith(("g1", "humanoid")):
            item = {"robot_id": "g1", "policy_id": "g1_walking", "label": "G1"}
        elif name.startswith("walker") or name.startswith("hexapod"):
            item = {"kind": "walker", "label": "Walker"}
            if re.search(r"\b(?:6|six)[\s-]*leg", source):
                item["legs"] = 6
            if re.search(r"\b(?:4|four)[\s-]*leg", source):
                item["legs"] = 4
        else:
            item = {"kind": "rover", "wheels": 6 if re.search(r"\b(?:6|six)[\s-]*wheel", source) else 4, "label": "Rover"}
        parsed.extend([item.copy() for _ in range(count)])
    if not parsed:
        robot = fallback or Robot()
        if robot.kind == "library":
            return [{"robot_id": robot.robot_id, "policy_id": robot.policy_id, "label": "G1"}]
        item = {"kind": robot.kind, "label": robot.kind.title()}
        if robot.kind == "walker":
            item["legs"] = robot.legs
        else:
            item["wheels"] = robot.wheels
        return [item]
    if len(parsed) > 6:
        raise ValueError("The Director can spawn at most six robots.")
    return parsed
