"""Fixed workflows that execute validated Director plans."""

from typing import TYPE_CHECKING, Any

from . import agents
from .plan import Plan
from .spec import Mission, Robot, RobotEntry, SceneSpec, Terrain
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

    if plan.workflow == "custom":
        world = await _prepare_robot_world(plan, current, director)
        result = await director._mock_call(
            world,
            "build_custom",
            {"request": plan.custom_request or ""},
            "Building the requested custom scene detail.",
        )
        if world.build is None or any(not check.passed for check in world.checks):
            return world, result.text
        return await _approval_or_run(world, plan, director, result.text)

    world = await _prepare_robot_world(plan, current, director)
    checked = await director._mock_call(world, "build_and_check", {}, "Building and checking the planned world.")
    if not checked.data.get("passed", False):
        failures = [
            f"{item.get('name')}: {item.get('detail')}"
            for item in checked.data.get("checks", [])
            if not item.get("passed")
        ]
        return world, "The scene did not pass its checks: " + "; ".join(failures)

    if plan.workflow == "compare":
        compared = await director._mock_call(
            world,
            "compare",
            {"variants": [variant.model_dump(exclude_none=True) for variant in plan.variants]},
            "Comparing the requested conditions.",
        )
        explanation = await _explain(world, director)
        return world, f"{compared.text} {explanation}".strip()
    if plan.workflow == "stress_test":
        tested = await director._mock_call(
            world,
            "stress_test",
            {"question": plan.question or ""},
            "Running the requested stress test.",
        )
        explanation = await _explain(world, director)
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
        else SceneSpec(domain="robot_terrain")
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
        for robot_id, has_override in spawned:
            if not has_override:
                await director._mock_call(
                    world,
                    "set_goal",
                    {"robot_id": robot_id, **target},
                    "Assigning the shared goal.",
                )
    return world


async def _build_and_finish(world: World, plan: Plan, director: "Director") -> WorkflowResult:
    checked = await director._mock_call(world, "build_and_check", {}, "Building and checking the updated world.")
    if not checked.data.get("passed", False):
        failures = [
            f"{item.get('name')}: {item.get('detail')}"
            for item in checked.data.get("checks", [])
            if not item.get("passed")
        ]
        return world, "The scene did not pass its checks: " + "; ".join(failures)
    return await _approval_or_run(world, plan, director, "The updated world is built and checked.")


async def _apply_changes(world: World, changes: dict[str, Any], director: "Director") -> None:
    spec = world.spec
    supported = {"world", "environment", "terrain", "robot", "robots", "obstacles", "goal", "mission"}
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
    result = await director._mock_call(world, "run", {}, "Running the checked simulation.")
    explanation = await _explain(world, director)
    if plan.workflow == "race":
        leaderboard = result.data.get("results", {}).get("leaderboard", [])
        if leaderboard:
            placements = ", ".join(
                f"{index}. {row.get('label', row.get('id'))}" for index, row in enumerate(leaderboard, 1)
            )
            ready_message = f"Race complete. Leaderboard: {placements}."
    return world, f"{ready_message} {explanation}".strip()


async def _explain(world: World, director: "Director") -> str:
    result = await director._mock_call(world, "explain", {}, "Explaining the measured results.")
    return result.text
