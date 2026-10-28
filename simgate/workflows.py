"""Fixed workflows that execute validated Director plans."""

import asyncio
from typing import TYPE_CHECKING, Any

from . import agents, config, presets
from .backends.base import blocking_failures
from .plan import Plan
from .spec import Load, Mission, Robot, RobotEntry, SceneSpec, Structure, Terrain
from .tools import World

if TYPE_CHECKING:
    from .director import Director

WorkflowResult = tuple[World, str]


async def execute(plan: Plan, current: SceneSpec | None, director: "Director") -> WorkflowResult:
    if plan.workflow == "answer":
        return World(current.model_copy(deep=True) if current else SceneSpec(domain="robot_terrain")), plan.reply or plan.summary

    if plan.workflow == "edit":
        if current is None:
            raise ValueError("An edit plan needs an existing world.")
        world = World(current.model_copy(deep=True))
        await _apply_changes(world, plan.changes, director)
        return await _build_and_finish(world, plan, director)

    if plan.domain == "structure":
        world = World(SceneSpec(
            domain="structure",
            title=_title(plan),
            structure=plan.structure.model_copy(deep=True) if plan.structure else Structure(),
            load=plan.load.model_copy(deep=True) if plan.load else Load(),
        ))
        ready, message = await _check_and_repair(world, director, "Building and checking the structural scene.")
        if not ready:
            return world, message
        return await _approval_or_run(world, plan, director, "The structural scene is built and checked.")

    if plan.workflow == "custom":
        world = await _prepare_robot_world(plan, current, director)
        result = await director._mock_call(
            world,
            "build_custom",
            {"request": plan.custom_request or ""},
            "Building the requested custom scene detail.",
        )
        if world.build is None or blocking_failures(world.checks):
            return world, result.text
        return await _approval_or_run(world, plan, director, result.text)

    world = await _prepare_robot_world(plan, current, director)
    ready, message = await _check_and_repair(world, director, "Building and checking the planned world.")
    if not ready:
        return world, message

    if plan.workflow == "compare":
        compared = await director._mock_call(
            world,
            "compare",
            {"variants": [variant.model_dump(exclude_none=True) for variant in plan.variants]},
            "Comparing the requested conditions.",
        )
        explanation = await _explain(world, director)
        await _request_run(world, director, "Streaming the baseline run for playback.")
        return world, f"{compared.text} {explanation}".strip()
    if plan.workflow == "stress_test":
        tested = await director._mock_call(
            world,
            "stress_test",
            {"question": plan.question or ""},
            "Running the requested stress test.",
        )
        explanation = await _explain(world, director)
        await _request_run(world, director, "Streaming the baseline run for playback.")
        return world, f"{tested.text} {explanation}".strip()
    return await _approval_or_run(world, plan, director, "The world is built and all checks passed.")


async def _prepare_robot_world(
    plan: Plan,
    current: SceneSpec | None,
    director: "Director",
) -> World:
    world = World(
        current.model_copy(deep=True)
        if current is not None
        else SceneSpec(domain="robot_terrain", title=_title(plan))
    )
    if current is None or plan.world.model_dump(exclude_none=True):
        await director._mock_call(
            world,
            "set_world",
            plan.world.model_dump(exclude_none=True),
            "Selecting the shared world conditions.",
        )
    if current is not None and plan.robots:
        world.spec.robots = []
        world.spec.robot = None
        world.spec.mission = None

    spawned: list[tuple[str, bool]] = []
    for robot_plan in plan.robots:
        for copy_index in range(robot_plan.count):
            label = robot_plan.label or (
                robot_plan.robot_id.replace("_", " ").title() if robot_plan.robot_id else robot_plan.kind.title()
            )
            if robot_plan.count > 1:
                label = f"{label} {copy_index + 1}"
            args: dict[str, Any] = {
                "label": label,
                "kind": robot_plan.kind,
                "robot_id": robot_plan.robot_id,
                "policy_id": robot_plan.policy_id,
                "legs": robot_plan.legs,
                "wheels": robot_plan.wheels,
            }
            if robot_plan.start_place is not None:
                args["start"] = robot_plan.start_place
            if robot_plan.goal_place is not None:
                args["goal"] = robot_plan.goal_place
            result = await director._mock_call(world, "spawn_robot", args, f"Adding {label}.")
            spawned.append((result.data["robot"]["id"], robot_plan.goal_place is not None))

    if plan.obstacles:
        if current is not None:
            world.spec.obstacles = []
        await director._mock_call(
            world,
            "add_obstacles",
            {"obstacles": [obstacle.model_dump(mode="json") for obstacle in plan.obstacles]},
            "Placing the requested obstacles.",
        )
    if plan.goal is not None:
        target = (
            {"xy": plan.goal.xy}
            if plan.goal.xy is not None
            else {"place": plan.goal.place}
        )
        targets = [robot_id for robot_id, has_override in spawned if not has_override]
        if targets and len(targets) == len(spawned):
            await director._mock_call(
                world,
                "set_goal",
                {"robot_id": "all", **target},
                "Assigning the shared goal.",
            )
        else:
            for robot_id in targets:
                await director._mock_call(
                    world,
                    "set_goal",
                    {"robot_id": robot_id, **target},
                    "Assigning the shared goal.",
                )
    return world


def _title(plan: Plan) -> str:
    return plan.summary.rstrip(".")


async def _build_and_finish(world: World, plan: Plan, director: "Director") -> WorkflowResult:
    ready, message = await _check_and_repair(world, director, "Building and checking the updated world.")
    if not ready:
        return world, message
    return await _approval_or_run(world, plan, director, "The updated world is built and checked.")


async def _check_and_repair(
    world: World,
    director: "Director",
    thought: str,
) -> tuple[bool, str]:
    await director._mock_call(world, "build_and_check", {}, thought)
    if not blocking_failures(world.checks):
        return True, ""

    builder_is_llm = not config.MOCK_MODE and config.BUILDER_MODE == "llm"
    attempts = 0
    for attempt in range(1, 4):
        failures = blocking_failures(world.checks)
        if not failures:
            return True, ""
        failure_text = _format_failures(failures)
        checker_id = await _emit_completed_step(
            director,
            agent="Checker",
            model="rules",
            tool="build_and_check",
            parent_id=director._root_id,
            thought="Checker found failed scene checks.",
            result=failure_text,
            status="warning",
        )
        builder_id = await _start_step(
            director,
            agent="Builder",
            model=config.BUILDER_MODEL if builder_is_llm else "template rules",
            tool="repair",
            parent_id=checker_id,
            thought=f"Repair attempt {attempt} using the failed checks.",
            args={"checks": failure_text[:1000]},
        )
        attempts += 1
        if builder_is_llm:
            feedback = failure_text
        else:
            fixed = _apply_rule_fixes(world, failures)
            feedback = None
            if not fixed:
                await _finish_step(
                    director,
                    builder_id,
                    agent="Builder",
                    model="template rules",
                    tool="repair",
                    parent_id=checker_id,
                    thought=f"Repair attempt {attempt} could not find a safe rule-based correction.",
                    result="No safe automatic change applies to these checks.",
                    status="warning",
                )
                await _emit_completed_step(
                    director,
                    agent="Checker",
                    model="rules",
                    tool="build_and_check",
                    parent_id=builder_id,
                    thought="Repair stopped because no safe change was available.",
                    result="The scene was not rerun.",
                    status="warning",
                )
                break

        simulator, build, logs = await asyncio.to_thread(_build_candidate, world, feedback)
        await _finish_step(
            director,
            builder_id,
            agent="Builder",
            model=config.BUILDER_MODEL if builder_is_llm else "template rules",
            tool="repair",
            parent_id=checker_id,
            thought=f"Repair attempt {attempt} completed.",
            result=" ".join(logs) or "Applied the bounded scene repair.",
            status="done",
            args={"checks": failure_text[:1000]},
        )
        checker_id = await _start_step(
            director,
            agent="Checker",
            model="rules",
            tool="build_and_check",
            parent_id=builder_id,
            thought=f"Checking the scene after repair attempt {attempt}.",
            args={},
        )
        checks = await asyncio.to_thread(simulator.validate, world.spec, build)
        world.simulator, world.build, world.checks = simulator, build, checks
        passed = not blocking_failures(checks)
        await _finish_step(
            director,
            checker_id,
            agent="Checker",
            model="rules",
            tool="build_and_check",
            parent_id=builder_id,
            thought=f"Checker finished attempt {attempt}.",
            result="All checks passed." if passed else _format_failures(blocking_failures(checks)),
            status="done" if passed else "warning",
        )
        if passed:
            return True, ""

    failures = blocking_failures(world.checks)
    question = (
        f"The scene still fails checks after {attempts} repair attempt(s): {_format_failures(failures)}. "
        "I stopped without running it. What would you like to change?"
    )
    asked = await director._mock_call(
        world,
        "ask_user",
        {"question": question},
        "Showing the remaining check failures and asking what to change.",
    )
    return False, asked.text


def _apply_rule_fixes(world: World, failures: list[Any]) -> bool:
    fixed = False
    spec = world.spec
    for check in failures:
        if check.name == "Gravity matches reference":
            value = presets.gravity_for(spec.environment.body)
            if value is not None and spec.environment.gravity != value:
                spec.environment.gravity = value
                fixed = True
        elif check.name == "Friction is plausible":
            value = presets.friction_for(spec.environment.surface)
            if value is not None and spec.environment.friction != value:
                spec.environment.friction = value
                fixed = True
        elif check.name == "Slope is plausible" and spec.terrain is not None:
            low, high = presets.SLOPE_RANGE_DEG
            value = min(high, max(low, spec.terrain.slope_deg))
            if spec.terrain.slope_deg != value:
                spec.terrain.slope_deg = value
                fixed = True
    return fixed


def _build_candidate(world: World, feedback: str | None):
    simulator = world.simulator or agents.route(world.spec)
    logs: list[str] = []
    build = agents.build(simulator, world.spec, logs.append, feedback=feedback)
    return simulator, build, logs


def _format_failures(checks: list[Any]) -> str:
    return "; ".join(f"{check.name}: {check.detail}" for check in checks)


async def _start_step(
    director: "Director",
    *,
    agent: str,
    model: str,
    tool: str,
    parent_id: str | None,
    thought: str,
    args: dict[str, Any],
) -> str:
    step_id = director._id()
    await director._emit_step(
        id=step_id,
        parent_id=parent_id,
        agent=agent,
        model=model,
        tool=tool,
        args=args,
        thought=thought,
        result="",
        status="working",
        ms=0,
    )
    return step_id


async def _finish_step(
    director: "Director",
    step_id: str,
    *,
    agent: str,
    model: str,
    tool: str,
    parent_id: str | None,
    thought: str,
    result: str,
    status: str,
    args: dict[str, Any] | None = None,
) -> None:
    await director._emit_step(
        id=step_id,
        parent_id=parent_id,
        agent=agent,
        model=model,
        tool=tool,
        args=args or {},
        thought=thought,
        result=result,
        status=status,
        ms=0,
    )


async def _emit_completed_step(
    director: "Director",
    *,
    agent: str,
    model: str,
    tool: str,
    parent_id: str | None,
    thought: str,
    result: str,
    status: str,
) -> str:
    step_id = await _start_step(
        director,
        agent=agent,
        model=model,
        tool=tool,
        parent_id=parent_id,
        thought=thought,
        args={},
    )
    await _finish_step(
        director,
        step_id,
        agent=agent,
        model=model,
        tool=tool,
        parent_id=parent_id,
        thought=thought,
        result=result,
        status=status,
    )
    return step_id


async def _apply_changes(world: World, changes: dict[str, Any], director: "Director") -> None:
    spec = world.spec
    supported = {
        "world", "environment", "terrain", "robot", "robots", "obstacles", "goal", "mission",
        "structure", "load",
    }
    unknown = set(changes) - supported
    if unknown:
        raise ValueError(f"Unsupported edit fields: {', '.join(sorted(unknown))}")

    if "world" in changes:
        await director._mock_call(world, "set_world", changes["world"], "Changing the requested world settings.")
        spec = world.spec
    if "environment" in changes:
        for key, value in changes["environment"].items():
            if key not in {"body", "surface"}:
                raise ValueError(f"Unsupported environment edit: {key}")
            setattr(spec.environment, key, value)
    if "terrain" in changes:
        values = {**(spec.terrain.model_dump() if spec.terrain else {}), **changes["terrain"]}
        spec.terrain = Terrain.model_validate(values)
    if "robot" in changes:
        robot = Robot.model_validate(changes["robot"])
        spec.robot = robot
        for entry in spec.robots:
            entry.robot = robot.model_copy(deep=True)
    if "robots" in changes:
        spec.robots = [RobotEntry.model_validate(item) for item in changes["robots"]]
    if "obstacles" in changes:
        spec.obstacles = changes["obstacles"]
    if "mission" in changes:
        mission = Mission.model_validate(changes["mission"])
        spec.mission = mission
        if spec.robots:
            entry = spec.robots[0]
            entry.start_xy, entry.start_place = mission.start_xy, mission.start_place
            for robot in spec.robots:
                robot.goal_xy, robot.goal_place = mission.goal_xy, mission.goal_place
    if "structure" in changes:
        spec.structure = Structure.model_validate(changes["structure"])
    if "load" in changes:
        spec.load = Load.model_validate(changes["load"])
    if "goal" in changes:
        goal = changes["goal"]
        args = {"robot_id": "all", **goal}
        await director._mock_call(world, "set_goal", args, "Changing the requested goal.")
        spec = world.spec
    spec.fill_defaults()
    world.spec = spec


async def _approval_or_run(
    world: World,
    plan: Plan,
    director: "Director",
    ready_message: str,
) -> WorkflowResult:
    if not plan.run_now:
        asked = await director._mock_call(
            world,
            "ask_user",
            {"question": "Run it?"},
            "Waiting for approval before running the simulation.",
        )
        return world, asked.text
    await _request_run(world, director, "Running the checked simulation.")
    return world, f"{ready_message} Running it now."


async def _request_run(world: World, director: "Director", thought: str) -> None:
    """The session streams the run (frames, metrics, explanation) exactly like a user-approved run."""
    world.run_requested = True
    await _emit_completed_step(
        director,
        agent="Director",
        model="rules",
        tool="run",
        parent_id=director._root_id,
        thought=thought,
        result="Run requested; the simulator streams it to the browser.",
        status="done",
    )


async def _explain(world: World, director: "Director") -> str:
    result = await director._mock_call(world, "explain", {}, "Explaining the measured results.")
    return result.text
