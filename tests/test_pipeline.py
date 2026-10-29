"""Smoke tests: run with `python -m pytest -q` or `python tests/test_pipeline.py`.

They run offline in MOCK mode unless you've configured a model.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simgate import agents  # noqa: E402
from simgate.backends.mujoco_backend import MuJoCoSimulator  # noqa: E402
from simgate.backends.opensees_backend import OpenSeesSimulator  # noqa: E402
from simgate.spec import Environment, Mission, Obstacle, Robot, RobotEntry, SceneSpec, Structure, Terrain  # noqa: E402

agents.config.MOCK_MODE = True


def run(spec):
    sim = agents.route(spec)
    build = agents.build(sim, spec, lambda _: None)
    checks = sim.validate(spec, build)
    frames = sum(len(c) for c in sim.run(spec, build))
    return sim, checks, frames


def test_rover_on_mars():
    spec = SceneSpec(domain="robot_terrain", environment=Environment(body="mars", surface="regolith"), terrain=Terrain(kind="hills")).fill_defaults()
    sim, checks, frames = run(spec)
    assert isinstance(sim, MuJoCoSimulator)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert frames == int(spec.duration_s * 30)
    assert sim.summary()["forward_progress_m"] > 1.0


def test_three_rover_race_has_non_overlapping_spawns_and_leaderboard():
    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0),
        duration_s=22,
        robots=[
            RobotEntry(id=f"r{i}", label=f"Rover {i}", robot=Robot(), goal_place="east")
            for i in range(1, 4)
        ],
    ).fill_defaults()
    sim = MuJoCoSimulator()
    build = sim.build(spec)
    checks = sim.validate(spec, build)
    assert all(check.passed for check in checks), [check for check in checks if not check.passed]
    assert [row.id for row in spec.robots] == ["r1", "r2", "r3"]
    assert sum(len(chunk) for chunk in sim.run(spec, build)) == int(spec.duration_s * 30)
    leaderboard = sim.summary()["leaderboard"]
    assert len(leaderboard) == 3 and all(row["goal_reached"] for row in leaderboard)
    assert all(row["brain"] == "wheel controller" for row in leaderboard)
    arrival_times = [row["time_to_goal_s"] for row in leaderboard]
    assert arrival_times == sorted(arrival_times)


def test_two_rovers_collide_in_shared_world():
    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0),
        duration_s=5,
        robots=[
            RobotEntry(id="left", label="Left rover", robot=Robot(), start_xy=[-2, 0], goal_xy=[2, 0]),
            RobotEntry(id="right", label="Right rover", robot=Robot(), start_xy=[2, 0], goal_xy=[-2, 0]),
        ],
    ).fill_defaults()
    sim = MuJoCoSimulator()
    build = sim.build(spec)
    assert sum(len(chunk) for chunk in sim.run(spec, build)) == 150
    assert sim.summary()["robot_contacts"] > 0


def test_legacy_robot_spec_becomes_single_entry():
    spec = SceneSpec(
        domain="robot_terrain",
        robot=Robot(kind="walker"),
        mission=Mission(start_xy=[-2, 1], goal_xy=[3, 1]),
    ).fill_defaults()
    assert len(spec.robots) == 1
    assert spec.robots[0].id == "r1"
    assert spec.robots[0].robot.kind == "walker"
    assert spec.robots[0].mission().goal_xy == [3, 1]


def test_director_plan_schema_validates_ids_and_expanded_robot_limit():
    import pytest
    from pydantic import ValidationError

    from simgate.plan import Plan

    race = Plan.model_validate({
        "workflow": "race",
        "summary": "Race three rovers to the east.",
        "robots": [{"kind": "rover", "count": 3}],
        "goal": {"place": "east"},
    })
    assert race.robots[0].count == 3
    with pytest.raises(ValidationError, match="robot_id 'go3' doesn't exist"):
        Plan.model_validate({
            "workflow": "build",
            "summary": "Build a robot scene.",
            "robots": [{"robot_id": "go3"}],
        })
    with pytest.raises(ValidationError, match="at most six robots"):
        Plan.model_validate({
            "workflow": "build",
            "summary": "Build a robot scene.",
            "robots": [{"kind": "rover", "count": 6}, {"kind": "walker"}],
        })


def test_director_planning_uses_one_call_retries_validation_and_falls_back():
    import asyncio

    from simgate import config, llm
    from simgate.director import Director

    original_mock = config.MOCK_MODE
    original_chat = llm.chat
    calls = []

    def scripted_chat(messages, *args, **kwargs):
        calls.append(messages)
        return (
            '{"workflow":"race","summary":"Race three rovers east.","robots":'
            '[{"kind":"rover","count":3}],"goal":{"place":"east"},"run_now":false}'
        )

    try:
        config.MOCK_MODE = False
        llm.chat = scripted_chat
        race = Director().plan("race three rovers to the east")
        assert len(calls) == 1
        assert race.workflow == "race" and race.robots[0].count == 3

        calls.clear()
        replies = iter([
            '{"workflow":"build","summary":"Use an unknown robot.","robots":[{"robot_id":"go3"}]}',
            '{"workflow":"build","summary":"Build a rover.","robots":[{"kind":"rover"}]}',
        ])
        llm.chat = lambda messages, *args, **kwargs: (
            calls.append(messages) or next(replies)
        )
        corrected = Director().plan("build a rover")
        assert len(calls) == 2
        assert "robot_id 'go3' doesn't exist" in calls[1][-1]["content"]
        assert calls[1][-2] == {
            "role": "assistant",
            "content": '{"workflow":"build","summary":"Use an unknown robot.","robots":[{"robot_id":"go3"}]}',
        }
        assert corrected.robots[0].kind == "rover"

        calls.clear()
        llm.chat = lambda *_args, **_kwargs: (calls.append([]) or "not JSON")
        director = Director()
        fallback = director.plan("race three rovers to the far side")
        assert len(calls) == 2
        assert fallback.workflow == "race" and fallback.robots[0].count == 3
        assert director.used_rule_planner

        world, message = asyncio.run(Director().execute_plan("race three rovers to the far side"))
        assert message.startswith("I used the simple planner for this one.")
        assert world.spec.title == "Three 4-wheel rovers on hills terrain, Earth"
    finally:
        config.MOCK_MODE = original_mock
        llm.chat = original_chat


def test_race_goals_and_run_cues_in_plans():
    from simgate import config
    from simgate.director import Director
    from simgate.plan import Plan

    original_mock = config.MOCK_MODE
    try:
        config.MOCK_MODE = True
        for prompt, place, run_now in [
            ("race a rover and a six-legged walker to the summit on Mars hills, go", "highest", True),
            ("race two rovers to the top", "highest", False),
            ("race three walkers to the bottom, start", "lowest", True),
            ("race two rovers to the far side and run it", "east", True),
        ]:
            plan = Director().plan(prompt)
            assert plan.workflow == "race", prompt
            assert plan.goal is not None and plan.goal.place == place, (prompt, plan.goal)
            assert plan.run_now is run_now, prompt
        assert Director().plan("A rover on Mars hills").run_now is False
        assert Director().plan("A rover on Mars hills, go").run_now is True
    finally:
        config.MOCK_MODE = original_mock

    plan = Plan.model_validate({
        "workflow": "race",
        "summary": "Two rovers race to the far side.",
        "robots": [{"kind": "rover", "count": 2, "goal_place": "far side"}],
        "goal": {"place": "far side"},
    })
    assert plan.goal.place == "east" and plan.robots[0].goal_place == "east"


def test_model_plans_are_corrected_from_the_request_words():
    import asyncio
    import json

    from simgate import config, llm
    from simgate.director import Director

    original_mock, original_chat = config.MOCK_MODE, llm.chat

    def planned(text, **plan):
        llm.chat = lambda *_args, **_kwargs: json.dumps({"workflow": "build", "summary": "A scene", **plan})
        director = Director()
        return director.plan(text), director.guard_notes

    rover = [{"kind": "rover"}]
    try:
        config.MOCK_MODE = False
        # a. world from the words
        plan, notes = planned("A rover on Mars hills", robots=rover, world={"terrain_kind": "hills"})
        assert plan.world.body == "mars" and notes == ["You named Mars, so the world is Mars."]
        plan, _ = planned("a lunar rover", robots=rover, world={"body": "mars"})
        assert plan.world.body == "moon"
        plan, notes = planned("A rover in Gale Crater", robots=rover, world={"terrain_kind": "hills"})
        assert (plan.world.preset, plan.world.terrain_kind, plan.world.body) == ("gale", "real", "mars")
        assert "Gale Crater, Mars" in notes[-1]
        # b. a preset nobody asked for
        plan, notes = planned("A rover on Mars hills", robots=rover,
                              world={"body": "mars", "terrain_kind": "hills", "preset": "real-ares3"})
        assert plan.world.preset is None and plan.world.terrain_kind == "hills"
        assert notes == ["You didn't ask for a real place, so I used generated terrain instead of real-ares3."]
        plan, notes = planned("A rover on real terrain from NASA", robots=rover, world={"preset": "real-jezero"})
        assert plan.world.preset == "real-jezero" and notes == []
        # c. course placement
        course = [{"kind": "rock", "size": "small", "count": 10, "placement": "scattered"},
                  {"kind": "step", "placement": "place", "place": "center"}]
        plan, notes = planned("Rover obstacle course: 10 small rocks and a step", robots=rover, obstacles=course)
        assert [obstacle.placement for obstacle in plan.obstacles] == ["path", "path"]
        assert notes == ["Placed the obstacles along the path in the order given."]
        plan, _ = planned("a rover with rocks scattered everywhere", robots=rover, obstacles=course[:1])
        assert plan.obstacles[0].placement == "scattered"
        plan, _ = planned("a rover with a step in the center", robots=rover, obstacles=course[1:])
        assert (plan.obstacles[0].placement, plan.obstacles[0].place) == ("place", "center")
        # d. run intent
        plan, notes = planned("two rovers to the east, race them", robots=[{"kind": "rover", "count": 2}],
                              goal={"place": "east"}, run_now=False)
        assert plan.run_now and notes == ["You asked to run it, so it runs now."]
        plan, notes = planned("A rover on flat ground", robots=rover)
        assert not plan.run_now and notes == []

        llm.chat = lambda *_args, **_kwargs: '{"workflow":"build","summary":"Rover","robots":[{"kind":"rover"}]}'
        world, _ = asyncio.run(Director().execute_plan("A rover on the Moon"))
        assert world.spec.environment.body == "moon"
        assert "You named Moon, so the world is Moon." in world.spec.assumptions
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat


def test_stress_plan_requires_an_existing_robot_scene():
    from simgate import config, llm
    from simgate.director import Director

    original_mock = config.MOCK_MODE
    original_chat = llm.chat
    structure = SceneSpec(domain="structure").fill_defaults()
    calls = []

    try:
        config.MOCK_MODE = True
        fallback = Director().plan("stress test", structure)
        assert fallback.workflow == "answer"
        assert "Create a robot scene first" in fallback.reply

        config.MOCK_MODE = False
        llm.chat = lambda *_args, **_kwargs: (calls.append(True) or (
            '{"workflow":"stress_test","summary":"Test robustness.","question":"How robust?"}'
        ))
        director = Director()
        fallback = director.plan("stress test", structure)
        assert len(calls) == 2
        assert fallback.workflow == "answer"
        assert director.used_rule_planner
    finally:
        config.MOCK_MODE = original_mock
        llm.chat = original_chat


def test_plan_workflow_expands_race_and_runs_only_when_requested():
    import asyncio

    from simgate import config, llm
    from simgate.director import Director

    original_mock = config.MOCK_MODE
    original_chat = llm.chat
    calls = []

    async def execute(run_now, current=None):
        events = []

        async def emit(event):
            events.append(event)

        def scripted_chat(messages, *_args, **_kwargs):
            calls.append(messages)
            if len(calls) > 1:
                return "Rover 2 reached the goal first."
            return (
                '{"workflow":"race","summary":"Race three rovers east.",'
                f'"robots":[{{"kind":"rover","count":3}}],"goal":{{"place":"east"}},'
                f'"run_now":{str(run_now).lower()}' + "}"
            )

        llm.chat = scripted_chat
        return await Director(emit).execute_plan("race three rovers east", current), events

    try:
        config.MOCK_MODE = False
        (world, message), events = asyncio.run(execute(False))
        assert len(calls) == 1
        assert len(world.spec.robots) == 3
        assert all(robot.goal_place == "east" for robot in world.spec.robots)
        assert message == "Run it?"
        assert any(event.get("tool") == "ask_user" for event in events)

        calls.clear()
        current = SceneSpec(
            domain="robot_terrain",
            terrain=Terrain(kind="flat"),
        ).fill_defaults()
        (world, _), _ = asyncio.run(execute(False, current))
        assert len(calls) == 1
        assert len(world.spec.robots) == 3
        assert world.spec.environment.body == current.environment.body

        calls.clear()
        (world, message), events = asyncio.run(execute(True))
        assert world.run_requested and world.results is None  # the session streams the run
        assert any(event.get("tool") == "run" for event in events)
        assert len(calls) == 1
    finally:
        config.MOCK_MODE = original_mock
        llm.chat = original_chat


def test_run_now_race_streams_frames_before_metrics():
    import asyncio

    from simgate import config, llm
    from simgate.orchestrator import Session

    original_mock = config.MOCK_MODE
    original_chat = llm.chat
    calls = []

    def scripted_chat(messages, *_args, **_kwargs):
        calls.append(messages)
        if len(calls) > 1:
            return "Rover 2 reached the goal first."
        return (
            '{"workflow":"race","summary":"Three rovers race east.",'
            '"robots":[{"kind":"rover","count":3}],"goal":{"place":"east"},"run_now":true}'
        )

    async def scenario():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.handle("race three rovers east, go")
        return session, events

    try:
        config.MOCK_MODE = False
        llm.chat = scripted_chat
        session, events = asyncio.run(scenario())
    finally:
        config.MOCK_MODE = original_mock
        llm.chat = original_chat

    types = [event["type"] for event in events]
    assert "run_start" in types and "frames" in types and "metrics" in types
    assert types.index("run_start") < types.index("frames") < types.index("metrics")
    assert "question" not in types[types.index("run_start"):]
    metrics = next(event["metrics"] for event in events if event["type"] == "metrics")
    assert len(metrics["leaderboard"]) == 3
    assert types[-1] == "explanation" and events[-1]["message"] == "Rover 2 reached the goal first."
    assert len(calls) == 2  # one plan and one post-run explanation
    assert not session.awaiting_approval


def test_plan_workflow_bounds_visible_repairs_and_asks_before_running():
    import asyncio
    from unittest.mock import patch

    from simgate import config, workflows
    from simgate.backends.base import BuildResult, Check
    from simgate.director import Director

    class AlwaysFailSimulator:
        name = "test simulator"
        language = "test"

        def __init__(self):
            self.validation_count = 0

        def build(self, _spec, code=None):
            return BuildResult(code or "template", self.language, {})

        def validate(self, _spec, _build):
            self.validation_count += 1
            return [Check("Deliberately failing check", False, "forced for repair-loop test")]

    simulator = AlwaysFailSimulator()
    events = []

    async def scenario():
        async def emit(event):
            events.append(event)

        with patch.object(config, "MOCK_MODE", True), \
                patch("simgate.agents.route", return_value=simulator), \
                patch.object(workflows, "_apply_rule_fixes", return_value=True):
            return await Director(emit).execute_plan("build a rover")

    world, message = asyncio.run(scenario())
    assert simulator.validation_count == 4  # initial check plus three bounded repairs
    assert "3 repair attempt(s)" in message
    assert world.question == message
    assert len({
        event["id"] for event in events
        if event.get("type") == "step" and event.get("agent") == "Builder"
        and event.get("tool") == "repair"
    }) == 3
    assert not any(event.get("tool") == "run" for event in events)
    assert any(event.get("tool") == "ask_user" for event in events)

    steps = [event for event in events if event.get("type") == "step"]
    checkers = [
        event["id"] for event in steps
        if event["agent"] == "Checker" and event["status"] == "warning" and "Deliberately failing" in event["result"]
    ]
    assert len(checkers) == len(set(checkers)) == 4  # reported once, then once after each of three repairs
    builders = [event for event in steps if event.get("tool") == "repair" and event["status"] != "working"]
    assert [event["parent_id"] for event in builders] == checkers[:3]
    parents = {event["id"]: event.get("parent_id") for event in steps}
    for event in steps:
        if event.get("tool") == "repair" and event.get("agent") == "Builder":
            parent_id = event.get("parent_id")
            assert parents[parent_id] is not None
            assert any(
                child.get("agent") == "Checker"
                and child.get("parent_id") == event["id"]
                and child.get("tool") == "build_and_check"
                for child in steps
            )


def test_advisory_course_warning_does_not_block_or_repair():
    import asyncio

    from simgate.director import Director

    async def scenario(text):
        events = []

        async def emit(event):
            events.append(event)

        world, message = await Director(emit).execute_plan(text)
        return world, message, events

    prompt = "Rover obstacle course: 10 small rocks, a step, stairs and a gap"
    world, message, events = asyncio.run(scenario(prompt))
    course = next(check for check in world.checks if check.name == "Course looks passable for this robot")
    assert not course.passed and course.advisory
    assert message == "Run it?"
    assert not any(event.get("tool") == "repair" for event in events)

    world, message, events = asyncio.run(scenario(prompt + ", run it"))
    assert any(event.get("tool") == "run" and event.get("status") == "done" for event in events)
    assert world.run_requested


def test_answer_first_does_not_leave_an_empty_scene():
    import asyncio

    from simgate.orchestrator import Session

    async def scenario():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.handle("what's the gravity on Mars?")
        assert session.spec is None and session.build is None
        assert not any(event["type"] in ("spec", "scene") for event in events)
        await session.handle("A rover on Mars hills")
        return session, events

    session, events = asyncio.run(scenario())
    assert session.build is not None and len(session.spec.robots) == 1
    assert events[-1]["type"] == "question" and events[-1]["message"] == "Run it?"


def test_new_plan_scenes_keep_the_planned_title():
    import asyncio

    from simgate.director import Director

    for prompt in ("A rover on Mars hills", "a 3-story building in an earthquake"):
        world, _ = asyncio.run(Director().execute_plan(prompt))
        assert world.spec.title == agents.plan(prompt).title


def test_failing_scene_runs_only_with_run_anyway():
    import asyncio
    from unittest.mock import patch

    from simgate.backends.base import BuildResult, Check
    from simgate.orchestrator import Session

    class FailingSimulator:
        name = "test simulator"
        language = "test"
        runs = 0

        def build(self, _spec, code=None):
            return BuildResult(code or "template", self.language, {})

        def validate(self, _spec, _build):
            return [Check("Deliberately failing check", False, "forced")]

        def run(self, _spec, _build):
            FailingSimulator.runs += 1
            yield [{"t": 0.0}]

        def summary(self):
            return {}

    async def scenario():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        with patch("simgate.agents.route", return_value=FailingSimulator()), \
                patch("simgate.agents.explain", return_value="explained"):
            await session.handle("build a rover")
            assert not session.awaiting_approval
            for text in ("run", "run again"):
                await session.handle(text)
                assert events[-1]["type"] == "question"
                assert events[-1]["message"] == (
                    "This scene still fails: Deliberately failing check (forced). "
                    "Describe a change, or say 'run anyway' to run it."
                )
            assert FailingSimulator.runs == 0
            await session.handle("Run anyway")
        return events

    events = asyncio.run(scenario())
    assert FailingSimulator.runs == 1
    assert [event["type"] for event in events][-1] == "explanation"


def test_plan_workflow_preserves_structural_scene_support():
    import asyncio

    from simgate.director import Director

    async def scenario():
        return await Director().execute_plan("a 3-story building in an earthquake")

    world, message = asyncio.run(scenario())
    assert world.spec.domain == "structure"
    assert world.spec.structure.stories == 3
    assert world.build is not None and all(check.passed for check in world.checks)
    assert message == "Run it?"


def test_edit_plan_changes_only_the_requested_scene_fields():
    import asyncio

    from simgate.director import Director

    current = SceneSpec(
        domain="robot_terrain",
        environment=Environment(body="mars", surface="regolith"),
        terrain=Terrain(kind="hills", roughness=0.2),
        robot=Robot(kind="walker", legs=6),
        mission=Mission(start_place="west", goal_place="east"),
        obstacles=[Obstacle(kind="rock", count=2)],
    ).fill_defaults()

    async def scenario():
        return await Director().execute_plan("make the slope steeper", current)

    world, _ = asyncio.run(scenario())
    assert world.spec.terrain.kind == "slope" and world.spec.terrain.slope_deg == 10  # hills count as 0°
    steeper_again = Director().plan("make the slope steeper", world.spec)
    assert steeper_again.changes.world.slope_deg == 20  # +10° on an existing slope, same as the model path
    assert world.spec.environment.body == current.environment.body == "mars"
    assert world.spec.environment.surface == current.environment.surface == "regolith"
    assert world.spec.robot == current.robot
    assert world.spec.mission == current.mission
    assert world.spec.obstacles == current.obstacles


def test_model_edit_plans_are_typed_and_checked_against_the_scene():
    import asyncio

    from simgate import config, llm
    from simgate.director import Director

    current = SceneSpec(
        domain="robot_terrain",
        title="Rover on Mars hills",
        environment=Environment(body="mars", surface="regolith"),
        terrain=Terrain(kind="hills"),
        robot=Robot(kind="rover"),
    ).fill_defaults()
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    calls = []

    def scripted(replies):
        replies = iter(replies)
        llm.chat = lambda messages, *_args, **_kwargs: calls.append(messages) or next(replies)

    try:
        config.MOCK_MODE = False
        scripted(['{"workflow":"edit","summary":"Rover on a 30° slope.","changes":{"terrain":{"slope_deg":30}}}'])
        world, message = asyncio.run(Director().execute_plan("make it a 30 degree slope", current))
        assert len(calls) == 1 and message == "Run it?"
        assert (world.spec.terrain.kind, world.spec.terrain.slope_deg) == ("slope", 30)
        assert any("30° slope" in note for note in world.spec.assumptions)
        assert world.spec.environment.body == "mars" and world.spec.robot == current.robot
        assert world.spec.title == "4-wheel rover on a 30° slope, Mars"  # rebuilt from the edited scene

        calls.clear()
        scripted([
            '{"workflow":"edit","summary":"Mars.","changes":{"world":{"body":"mars"}}}',
            '{"workflow":"edit","summary":"Moon.","changes":{"world":{"body":"moon"},"title":"Rover on Moon hills"}}',
        ])
        plan = Director().plan("put it on the moon", current)
        assert len(calls) == 2
        assert "this edit changes nothing" in calls[1][-1]["content"]
        assert plan.changes.world.body == "moon" and plan.changes.title is None  # the model's title is ignored

        calls.clear()
        scripted(['{"workflow":"edit","summary":"Rover on a 35° slope","changes":'
                  '{"world":{"terrain_kind":"slope","slope_deg":35},"title":"Rover on a 35° slope"}}'])
        plan = Director().plan("make the slope steeper", current)
        assert plan.changes.world.slope_deg == 10  # no number given: 10° steeper than the hills' 0°
        assert plan.changes.title is None  # titles are rebuilt in code, not taken from the model
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat


def test_rules_planner_treats_short_change_requests_as_edits():
    import asyncio

    from simgate.director import Director
    from simgate.orchestrator import Session

    rover = agents.plan("A rover on Mars hills")
    building = agents.plan("A 5-story building in an earthquake")
    for text, current in [
        ("use six wheels", rover), ("six wheels", rover), ("on the moon instead", rover), ("ice", rover),
        ("make the slope steeper and use ice", rover), ("add a wall", rover), ("make it stiffer", building),
    ]:
        plan = Director().plan(text, current)
        assert plan.workflow == "edit", (text, plan.workflow, plan.reply)
    for question in ("what's the gravity on Mars?", "can you explain the slip?", "why did it stop"):
        assert Director().plan(question, rover).workflow == "answer", question

    async def readme_flow():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.handle("A rover on Mars hills")
        await session.handle("use six wheels")
        assert [entry.robot.wheels for entry in session.spec.robots] == [6]
        await session.handle("make the slope steeper and use ice")
        assert session.spec.terrain.kind == "slope" and session.spec.environment.surface == "ice"
        await session.handle("add a wall")
        assert [obstacle.kind for obstacle in session.spec.obstacles] == ["wall"]
        assert session.spec.robots[0].robot.wheels == 6 and session.spec.environment.surface == "ice"
        return events

    events = asyncio.run(readme_flow())
    assert not any(event["type"] == "error" for event in events)


def test_leg_and_wheel_counts_apply_only_to_their_part():
    from simgate.director import Director

    assert (agents.plan("a rover and a six-legged walker").robot.wheels, agents.plan("a six-legged walker").robot.legs) == (4, 6)
    assert agents.plan("a six-wheeled rover").robot.legs == 4
    assert agents.plan("a robot dog, six wheels on the next one").robot.legs == 4
    assert agents.plan("add six rocks").robot.wheels == 4

    plan = Director().plan("race a rover and a six-legged walker to the summit on Mars hills, go")
    assert [(robot.kind, robot.wheels, robot.legs) for robot in plan.robots] == [("rover", 4, None), ("walker", None, 6)]
    assert plan.summary == "4-wheel rover and 6-legged walker on hills terrain, Mars."
    assert Director().plan("race three rovers to the far side").summary == "Three 4-wheel rovers on hills terrain, Earth."


def test_edits_rebuild_the_scene_title():
    import asyncio

    from simgate import config, llm
    from simgate.director import Director
    from simgate.orchestrator import Session

    async def flow():
        session = Session(lambda _event: asyncio.sleep(0))
        titles = []
        for text in ("A rover on Mars hills", "make the slope steeper", "use ice", "call it Cliff run",
                     "use six wheels", "use sand and call it Dune run"):
            await session.handle(text)
            titles.append(session.spec.title)
        return titles, session.spec

    titles, spec = asyncio.run(flow())
    assert titles == [
        "4-wheel rover on hills terrain, Mars",
        "4-wheel rover on a 10° slope, Mars",
        "4-wheel rover on a 10° ice slope, Mars",
        "Cliff run",
        "6-wheel rover on a 10° ice slope, Mars",
        "Dune run",
    ]
    assert spec.environment.surface == "sand" and spec.terrain.slope_deg == 10
    course = agents.plan("Rover obstacle course: 10 small rocks, a step, stairs and a gap")
    assert agents.scene_title(course) == "4-wheel rover on flat terrain, Earth with an obstacle course"

    current = agents.plan("A rover on Mars hills")
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    try:
        config.MOCK_MODE = False
        llm.chat = lambda *_args, **_kwargs: (
            '{"workflow":"edit","summary":"Rover on Mars ice hills",'
            '"changes":{"world":{"surface":"ice"},"title":"Rover on Mars ice hills"}}'
        )
        world, _ = asyncio.run(Director().execute_plan("use ice", current))
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat
    assert world.spec.title == "4-wheel rover on ice hills terrain, Mars"


def test_adding_obstacles_keeps_the_existing_terrain():
    import asyncio

    from simgate.orchestrator import Session

    async def flow():
        session = Session(lambda _event: asyncio.sleep(0))
        for text in ("A rover on Mars hills", "make the slope steeper and use ice", "add a wall"):
            await session.handle(text)
        return session.spec

    spec = asyncio.run(flow())
    assert (spec.terrain.kind, spec.terrain.slope_deg, spec.environment.surface) == ("slope", 10, "ice")
    assert [obstacle.kind for obstacle in spec.obstacles] == ["wall"]
    assert spec.title == "4-wheel rover on a 10° ice slope, Mars with an obstacle course"
    assert agents.plan("Rover obstacle course: 10 small rocks, a step, stairs and a gap").terrain.kind == "flat"


def test_template_repair_uses_registered_physics_presets():
    from simgate import presets, workflows
    from simgate.backends.base import Check
    from simgate.tools import World

    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="slope", slope_deg=80),
    ).fill_defaults()
    spec.environment.gravity = 1.0
    spec.environment.friction = 10.0
    world = World(spec)
    failures = [
        Check("Gravity matches reference", False, "incorrect gravity"),
        Check("Friction is plausible", False, "incorrect friction"),
        Check("Slope is plausible", False, "too steep"),
    ]

    assert workflows._apply_rule_fixes(world, failures)
    assert spec.environment.gravity == presets.gravity_for(spec.environment.body)
    assert spec.environment.friction == presets.friction_for(spec.environment.surface)
    assert spec.terrain.slope_deg == presets.SLOPE_RANGE_DEG[1]


def test_director_tools_validate_and_update_world():
    import pytest
    from pydantic import ValidationError

    from simgate import tools

    world = tools.World(agents.plan("a rover on flat terrain"))
    tools.set_world(world, {"body": "mars", "surface": "regolith"})
    assert world.spec.environment.body == "mars" and world.spec.environment.friction is not None
    with pytest.raises((ValidationError, ValueError)):
        tools.set_world(world, {"body": "jupiter"})
    spawned = tools.spawn_robot(world, {"kind": "walker", "label": "Six-legged scout"})
    assert spawned.data["robot"]["id"] == "r2"
    tools.set_goal(world, {"robot_id": "all", "place": "east"})
    assert all(entry.goal_place == "east" for entry in world.spec.robots)
    added = tools.add_obstacles(world, {"obstacles": [{"kind": "rock", "count": 2}]})
    assert len(added.data["obstacles"]) == 1
    library = tools.list_library(world)
    assert library.data["robots"] and library.data["policies"]
    removed = tools.remove_robot(world, {"id": "r2"})
    assert len(removed.data["robots"]) == 1
    checked = tools.build_and_check(world)
    assert checked.data["checks"]
    assert world.build is not None and world.simulator is not None


def test_set_world_accepts_a_bare_real_terrain_preset_name():
    from simgate import tools

    world = tools.World(agents.plan("a rover on flat terrain"))
    tools.set_world(world, {"preset": "jezero"})
    assert (world.spec.terrain.kind, world.spec.terrain.preset, world.spec.environment.body) == ("real", "jezero", "mars")


def test_no_tool_loop_leftovers():
    from simgate import director, tools

    assert "loop" not in director.__doc__.lower()
    assert not hasattr(tools, "tool_catalog") and not hasattr(tools, "TOOL_ARGUMENT_MODELS")


def test_mock_director_turns_race_into_ordered_tool_calls():
    import asyncio

    from simgate import config
    from simgate.director import Director

    async def scenario(text):
        events = []

        async def emit(event):
            events.append(event)

        config.MOCK_MODE = True
        world, message = await Director(emit).execute_plan(text)
        done_tools = [
            event["tool"] for event in events
            if event.get("type") == "step" and event.get("status") == "done"
            and event.get("agent") == "Director" and event.get("tool")
        ]
        return world, message, done_tools, events

    world, message, tools, events = asyncio.run(scenario("race three rovers to the far side"))
    assert tools == [
        "plan", "set_world", "spawn_robot", "spawn_robot", "spawn_robot",
        "set_goal", "build_and_check", "ask_user",
    ]
    assert len(world.spec.robots) == 3 and all(entry.goal_place == "east" for entry in world.spec.robots)
    assert message == "Run it?" and all(check.passed for check in world.checks)
    assert {"Builder", "Checker"} <= {event["agent"] for event in events if event["type"] == "step"}
    steps = [event for event in events if event["type"] == "step"]
    parents = {}
    for event in steps:
        if event["id"] in parents:
            assert parents[event["id"]] == event.get("parent_id")
        else:
            parents[event["id"]] = event.get("parent_id")
    roots = {step_id for step_id, parent_id in parents.items() if parent_id is None}
    assert len(roots) == 1
    for step_id, parent_id in parents.items():
        visited = {step_id}
        while parent_id is not None:
            assert parent_id in parents and parent_id not in visited
            visited.add(parent_id)
            parent_id = parents[parent_id]
        assert visited & roots

    mixed, _, mixed_tools, _ = asyncio.run(
        scenario("a rover and a 6-legged walker on Mars hills, race to the summit")
    )
    assert [entry.robot.kind for entry in mixed.spec.robots] == ["rover", "walker"]
    assert mixed.spec.robots[1].robot.legs == 6
    assert mixed.spec.environment.body == "mars" and mixed.spec.terrain.kind == "hills"
    assert mixed_tools[-1] == "ask_user"


def test_build_custom_retries_and_emits_attempt_steps():
    import asyncio

    from simgate import config
    from simgate.director import Director

    spec = agents.plan("a rover on flat terrain")
    original_chat = agents.llm.chat
    original_retries = config.BUILDER_MAX_RETRIES
    replies = iter([
        '{"workflow":"custom","summary":"Add a marker above the scene.",'
        '"custom_request":"Add a marker above the scene.","run_now":false}',
        '<geom name="bad_marker" type="cube" size="0.1 0.1 0.1" pos="0 0 20"/>',
        '<geom name="marker" type="sphere" size="0.05" pos="0 0 20"/>',
    ])
    events = []

    async def emit(event):
        events.append(event)

    agents.llm.chat = lambda *_args, **_kwargs: next(replies)
    config.MOCK_MODE = False
    config.BUILDER_MAX_RETRIES = 2
    try:
        world, message = asyncio.run(Director(emit).execute_plan("add a marker", current=spec))
        assert message == "Run it?"
        assert world.build is not None and "marker" in world.build.code
        builder_steps = [
            event for event in events
            if event.get("agent") == "Builder" and event.get("status") != "working"
        ]
        assert [event["status"] for event in builder_steps] == ["warning", "done"]
        assert "Custom fragment attempt 1." == builder_steps[0]["thought"]
        assert "Custom fragment attempt 2." == builder_steps[1]["thought"]
        assert "cube" in builder_steps[0]["result"]
        assert "attempt 2" in builder_steps[1]["result"]
        step_ids = {event["id"] for event in events if event.get("type") == "step"}
        assert all(event.get("parent_id") in step_ids for event in builder_steps)
    finally:
        agents.llm.chat = original_chat
        config.MOCK_MODE = True
        config.BUILDER_MAX_RETRIES = original_retries


def test_plan_prompt_offers_built_in_robots_only_as_kinds():
    import json

    from simgate.director import _plan_system

    prompt = _plan_system()
    listed = prompt.split("Available library robots and policies:\n", 1)[1].splitlines()[0]
    assert [robot["id"] for robot in json.loads(listed)] == ["g1"]
    assert "Built-in robot kinds: rover, walker" in prompt


def test_plan_summaries_are_scene_names():
    import json

    from simgate.director import Director, _plan_system

    prompt = _plan_system()
    assert '"Three rovers race on Mars hills", not an action' in prompt
    examples = [line for line in prompt.splitlines() if line.endswith("}") and ' plan: {"workflow"' in line]
    summaries = [json.loads(line.split(" plan: ", 1)[1])["summary"] for line in examples]
    assert len(summaries) == 11
    assert not any(summary.split()[0] in {"Race", "Compare", "Test", "Make", "Explain", "Create"} for summary in summaries)

    plan = Director().plan("compare a rover on the moon and mars")
    assert plan.summary.startswith("4-wheel rover on") and "Compare" not in plan.summary


def test_plan_prompt_states_world_course_and_preset_rules():
    import json

    from simgate.director import _plan_system
    from simgate.plan import Plan

    prompt = _plan_system()
    assert "A planet or moon named in the request goes in world.body" in prompt
    assert 'Obstacles for a course use placement "path", listed in the order the user gave them.' in prompt
    assert "Only use a preset when the user asks for a real place" in prompt
    examples = {
        line.split(" plan: ", 1)[0]: Plan.model_validate(json.loads(line.split(" plan: ", 1)[1]))
        for line in prompt.splitlines() if ' plan: {"workflow"' in line
    }
    assert examples["Named world"].world.body == "moon" and examples["Named world"].world.preset is None
    assert [obstacle.placement for obstacle in examples["Course"].obstacles] == ["path"] * 3
    assert examples["Real place"].world.preset == "real-gale"
    assert len(examples) == 11  # every example in the prompt validates as a Plan

    assert "30" not in prompt.split("Short examples:", 1)[1].split("Available library robots", 1)[0]
    steeper = examples["Steeper edit"].changes
    assert steeper.world.terrain_kind == "slope" and steeper.world.slope_deg is None
    robot_edit = examples["Robot edit"].changes
    assert robot_edit.robot.model_dump(exclude_unset=True) == {"wheels": 6} and robot_edit.robots is None

    from simgate.workflows import apply_edit

    rover = agents.plan("A rover on Mars hills")
    rover.robots[0].robot.wheel_radius = 0.2
    edited = apply_edit(rover, robot_edit).spec
    assert edited.robots[0].robot.wheels == 6 and edited.robots[0].robot.wheel_radius == 0.2


def test_plan_prompt_uses_compact_current_world_state():
    from simgate.backends.base import BuildResult
    from simgate.director import _plan_world_state
    from simgate.tools import World

    world = World(SceneSpec(domain="robot_terrain").fill_defaults())
    world.build = BuildResult(
        "CODE_BULK" * 10000,
        "mjcf-xml",
        {
            "geoms": [{} for _ in range(30000)],
            "mesh_assets": {"mesh": {"vertices": "VERTEX_BULK" * 10000}},
        },
    )
    compact = _plan_world_state(world)
    assert len(compact) < 2000
    assert "CODE_BULK" not in compact and "VERTEX_BULK" not in compact
    assert "mesh_assets" not in compact and '"built":true' in compact


def test_icy_slope_physics():
    # Friction 0.1 is below tan(25°) ≈ 0.47, so the rover cannot hold position and should not climb.
    spec = SceneSpec(domain="robot_terrain", environment=Environment(surface="ice"), terrain=Terrain(kind="slope", slope_deg=25, roughness=0.1)).fill_defaults()
    sim, _, _ = run(spec)
    assert sim.summary()["forward_progress_m"] < 1.0


def test_walkers_walk_without_falling():
    for legs in (4, 6):
        spec = SceneSpec(domain="robot_terrain", environment=Environment(body="mars", surface="regolith"),
                         terrain=Terrain(kind="hills"), robot=Robot(kind="walker", legs=legs)).fill_defaults()
        sim, checks, _ = run(spec)
        assert all(c.passed for c in checks), [c for c in checks if not c.passed]
        m = sim.summary()
        assert not m["flipped"] and m["forward_progress_m"] > 1.0, m


def test_building_textbook_and_resonance():
    calm = SceneSpec(domain="structure", structure=Structure(stories=5, section="heavy")).fill_defaults()
    soft = SceneSpec(domain="structure", structure=Structure(stories=5, section="light")).fill_defaults()
    sim_a, checks, _ = run(calm)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    sim_b, _, _ = run(soft)
    assert isinstance(sim_a, OpenSeesSimulator)
    assert sim_b.summary()["max_story_drift_pct"] > sim_a.summary()["max_story_drift_pct"]


def test_real_terrain_pipeline(tmp_dir=None):
    """Fake HiRISE-style GeoTIFF -> converter -> preset -> simulation, with orientation preserved."""
    import json
    import subprocess
    import tempfile

    import numpy as np

    from simgate import terrain_data
    from simgate.backends.mujoco_backend import GRID, make_heights

    try:
        import rasterio  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("pip install rasterio to run it") from None
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        tif, out = Path(tmp) / "fake.tif", Path(tmp) / "terrains"
        subprocess.run([sys.executable, str(root / "scripts/make_test_dtm.py"), str(tif)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(root / "scripts/hirise_to_simgate.py"), "--input", str(tif), "--name", "t",
                        "--label", "Test terrain", "--credit", "Synthetic test data", "--out-dir", str(out)],
                       check=True, capture_output=True)
        meta = json.loads((out / "t.json").read_text())
        assert meta["body"] == "mars" and meta["typical_slope_deg"] <= 22 and 35 < meta["size_m"] < 45, meta

        old_dir = terrain_data.TERRAIN_DIR
        terrain_data.TERRAIN_DIR = out
        terrain_data.load_preset.cache_clear()
        try:
            spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="real", preset="t", roughness=0),
                             robot=Robot(kind="rover", wheels=6)).fill_defaults()
            assert spec.environment.body == "mars" and spec.environment.gravity == 3.71
            # Away from the start pad, the simulated terrain must equal the source data (catches flipped rows/cols).
            heights, src = make_heights(spec)[0], terrain_data.resample(terrain_data.load_preset("t")[0], GRID)
            far = np.zeros_like(heights, bool)
            far[:, GRID * 2 // 3:] = True
            diff = (heights - heights[far].min()) - (src - src[far].min())
            assert float(np.abs(diff[far]).max()) < 1e-6
            _, checks, _ = run(spec)
            assert all(c.passed for c in checks), [c for c in checks if not c.passed]
        finally:
            terrain_data.TERRAIN_DIR = old_dir
            terrain_data.load_preset.cache_clear()


def test_fall_diagnostics_on_steep_slope():
    # Walker facing up a 25° slope: it slides back and tips over; the evidence must say so.
    spec = SceneSpec(domain="robot_terrain", environment=Environment(surface="gravel"),
                     terrain=Terrain(kind="slope", slope_deg=25, roughness=0.1), robot=Robot(kind="walker")).fill_defaults()
    sim, _, _ = run(spec)
    m = sim.summary()
    assert m["flipped"] and m["fall_direction"] in ("forward", "backward", "left", "right"), m
    assert m["fall_started_at_s"] <= m["flipped_at_s"]
    assert 20 < m["local_slope_deg"] < 32 and m["slope_along_heading_deg"] > 15   # facing uphill on a ~25° slope
    assert 0 <= m["slip_ratio"] <= 1 and m["slip_ratio_last_1s_before_fall"] is not None
    assert 8 <= len(m["tilt_trace"]) <= 11


def test_slope_geometry_and_no_fall_fields():
    # Rover driving up a 10° slope that rises toward +x: uphill ahead, flat sideways, no fall, wheels rolling.
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="slope", slope_deg=10, roughness=0),
                     robot=Robot(kind="rover")).fill_defaults()
    sim, _, _ = run(spec)
    m = sim.summary()
    assert not m["flipped"] and m["fall_direction"] is None and m["fall_started_at_s"] is None
    assert abs(m["slope_along_heading_deg"] - 10) < 1.5 and abs(m["cross_slope_deg"]) < 1.5, m
    assert abs(m["max_slope_on_path_deg"] - 10) < 2      # no artificial steep edges around the spawn point
    assert m["slip_ratio"] < 0.1


def test_run_once_per_approval():
    import asyncio

    from simgate.orchestrator import Session

    async def scenario():
        events = []

        async def emit(ev):
            events.append(ev)

        s = Session(emit)
        await s.handle("rover on the moon")
        await s.handle("run")
        await s.handle("run")             # a double click: must not run again
        starts_after_double = sum(e["type"] == "run_start" for e in events)
        await s.handle("run again")       # explicit: runs again
        return starts_after_double, sum(e["type"] == "run_start" for e in events), events

    after_double, total, events = asyncio.run(scenario())
    assert after_double == 1 and total == 2
    assert any(e["type"] == "question" and "already ran" in e["message"] for e in events)


def mission_run(robot, terrain, mission, env=None):
    spec = SceneSpec(domain="robot_terrain", environment=env or Environment(), terrain=terrain,
                     robot=robot, mission=mission).fill_defaults()
    sim, checks, frames = run(spec)
    return spec, sim.summary(), checks, frames


def test_rover_steers_to_goal_off_its_heading():
    # Facing east, goal 5 m away at 45° to its left: it must turn, arrive, and stop the run early.
    spec, m, checks, frames = mission_run(Robot(kind="rover"), Terrain(kind="flat", roughness=0),
                                          Mission(start_xy=[-5, 0], start_heading_deg=0, goal_xy=[-1.5, 3.5]))
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    assert m["goal_reached"] and m["outcome"] == "reached" and m["final_distance_m"] <= 1.0, m
    assert m["path_efficiency"] > 0.8 and frames < spec.duration_s * 30


def test_walker_reaches_goal_on_hills():
    _, m, _, _ = mission_run(Robot(kind="walker", legs=6), Terrain(kind="hills"),
                             Mission(start_xy=[-7, 0], goal_xy=[6, 5]), Environment(body="mars", surface="regolith"))
    assert m["goal_reached"] and not m["flipped"], m


def test_goal_up_an_icy_slope_is_missed():
    # Friction 0.1 on 25°: the rover can't climb, and the summary must say it missed and why.
    _, m, checks, _ = mission_run(Robot(kind="rover"), Terrain(kind="slope", slope_deg=25, roughness=0.1),
                                  Mission(goal_place="highest"), Environment(surface="ice"))
    assert not m["goal_reached"] and m["outcome"] in ("stuck", "out_of_time", "left_terrain", "fell"), m
    assert m["closest_distance_m"] > 2
    assert any(c.name == "Straight path looks passable" for c in checks)


def test_named_places_and_mock_missions():
    spec = agents.plan("rover on the moon, drive from the west to the crater floor")
    assert spec.mission.start_place == "west" and spec.mission.goal_place == "lowest" and spec.duration_s == 30
    sim = agents.route(spec)
    build = agents.build(sim, spec, lambda _: None)
    ms = build.scene["mission"]
    assert ms["start"][0] < -5 and ms["goal"] is not None
    assert abs(ms["heading_deg"] - math.degrees(math.atan2(ms["goal"][1] - ms["start"][1], ms["goal"][0] - ms["start"][0]))) < 0.2
    assert agents.plan("rover on the moon").mission is None


def test_click_to_set_goal():
    import asyncio

    from simgate.orchestrator import Session

    async def scenario():
        events = []

        async def emit(ev):
            events.append(ev)

        s = Session(emit)
        await s.handle("rover on mars")
        await s.set_point("goal", 4.0, -3.0)
        return s, events

    s, events = asyncio.run(scenario())
    assert s.spec.mission.goal_xy == [4.0, -3.0] and s.awaiting_approval
    assert [e for e in events if e["type"] == "scene"][-1]["scene"]["mission"]["goal"] == (4.0, -3.0)


def course_run(robot, obstacles, terrain=None):
    spec = SceneSpec(domain="robot_terrain", terrain=terrain or Terrain(kind="flat", roughness=0),
                     robot=robot, obstacles=obstacles).fill_defaults()
    sim, checks, _ = run(spec)
    return spec, sim, checks, sim.summary()


def test_step_height_decides_pass_or_stop():
    # Measured limits: the 4-wheel rover climbs ~0.09 m; the 6-wheel rover's rear wheels push it up ~0.2 m.
    _, _, checks, low = course_run(Robot(kind="rover"), [Obstacle(kind="step", height_m=0.07)])
    assert low["goal_reached"] and low["obstacles_cleared"] == 1, low
    assert next(c for c in checks if c.name == "Course looks passable for this robot").passed
    _, _, checks, high = course_run(Robot(kind="rover"), [Obstacle(kind="step", height_m=0.15)])
    assert not high["goal_reached"] and high["stopped_by"]["kind"] == "step" and high["stopped_by"]["touched_at_s"], high
    assert not next(c for c in checks if c.name == "Course looks passable for this robot").passed
    _, _, _, six = course_run(Robot(kind="rover", wheels=6), [Obstacle(kind="step", height_m=0.15)])
    assert six["goal_reached"], six


def test_walker_stopped_by_gap():
    _, _, _, m = course_run(Robot(kind="walker", legs=6), [Obstacle(kind="gap", size="small")])
    assert not m["goal_reached"] and m["stopped_by"]["kind"] == "gap", m
    assert 0.15 < m["stopped_by"]["width_m"] < 0.35 and m["robot_rough_limits"]["gap_m"] == 0.0


def test_course_layout():
    obstacles = [Obstacle(kind="rock", count=10, size="small"), Obstacle(kind="step", size="small"),
                 Obstacle(kind="stairs", size="small"), Obstacle(kind="ramp"), Obstacle(kind="gap", size="small"),
                 Obstacle(kind="log", count=2, size="small")]
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=Robot(kind="rover", wheels=6),
                     obstacles=obstacles).fill_defaults()
    assert spec.terrain.size_m > 20 and spec.mission.goal_place == "east"          # grew to fit; finish line added
    sim = agents.route(spec)
    build = agents.build(sim, spec, lambda _: None)
    course, ms = build.scene["course"], build.scene["mission"]
    assert [p["kind"] for p in course] == ["rock", "step", "stairs", "ramp", "gap", "log"]
    along = [p["along"] for p in course]
    assert along == sorted(along)                                                   # in the order asked for
    gap = math.dist(ms["start"], ms["goal"])
    for p in course:
        assert p["along"] - p["length"] / 2 > 1.5 and p["along"] + p["length"] / 2 < gap - 1.5, p   # start/goal clear
    names = {g["name"] for g in build.scene["geoms"]}
    assert sum(n.startswith("obs0_rock_") for n in names) == 10 and all(any(n.startswith(p["name"]) for n in names) for p in course)
    assert all(c.passed for c in sim.validate(spec, build) if c.name != "Course looks passable for this robot")


def test_mock_course_parsing():
    spec = agents.plan("Rover obstacle course: 10 small rocks, a big step, stairs and a narrow trench")
    assert [(o.kind, o.size, o.count) for o in spec.obstacles] == [("rock", "small", 10), ("step", "large", 1),
                                                                      ("stairs", "medium", 1), ("gap", "small", 1)]
    assert spec.terrain.kind == "flat" and spec.environment.surface == "rock"
    edited = agents.edit(spec, "add a wall")
    assert [o.kind for o in edited.obstacles][-1] == "wall" and len(edited.obstacles) == 5
    assert agents.edit(edited, "remove the obstacles").obstacles == []


def test_planner_routes_by_domain():
    assert agents.plan("a 3-story building in an earthquake").domain == "structure"
    assert agents.plan("rover on the moon").domain == "robot_terrain"


def test_batch_stress_conditions_and_ghosts():
    import base64
    import json
    import zlib

    from simgate.batch import MAX_BATCH_GHOST_BYTES, run_batch

    spec = SceneSpec(
        domain="robot_terrain",
        environment=Environment(surface="rock"),
        terrain=Terrain(kind="slope", slope_deg=25, roughness=0.1),
        duration_s=4.0,
    ).fill_defaults()
    variations = [
        {"label": "Moon gravity", "gravity": 1.62},
        {"label": "Sand", "surface": "sand"},
        {"label": "Ice", "surface": "ice"},
        {"label": "Side push", "push_n": 30},
        {"label": "New seed", "seed": 12},
        {"label": "Steeper", "slope_deg": 30},
        {"label": "Invalid knob", "not_a_condition": True},
    ]
    results = run_batch(spec, variations, timeout=90)
    by_label = {result["label"]: result for result in results}
    assert len(results) == len(variations) + 1 and results[0]["label"] == "Baseline"
    assert by_label["Baseline"]["passed"]
    assert not by_label["Ice"]["passed"]
    assert by_label["Invalid knob"]["error"]
    flat = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0),
        duration_s=4.0,
    ).fill_defaults()
    push_results = run_batch(flat, [{"label": "Side push", "push_n": 30}], timeout=90)
    baseline, pushed = push_results[0]["summary"], push_results[1]["summary"]
    assert (abs(pushed["lateral_drift_m"]) > abs(baseline["lateral_drift_m"])
            or pushed["max_tilt_deg"] > baseline["max_tilt_deg"]), (baseline, pushed)
    assert by_label["Baseline"]["ghost"] and by_label["Moon gravity"]["ghost"]
    assert by_label["Sand"]["ghost"] and by_label["Side push"]["ghost"]
    for label in ("New seed", "Steeper"):
        assert by_label[label]["ghost"] is None
        assert by_label[label]["ghost_status"] == "not in ghost view"

    ghost = by_label["Baseline"]["ghost"]
    frames = json.loads(zlib.decompress(base64.b64decode(ghost["data"])))
    assert len(frames) == ghost["frame_count"]
    assert 55 <= len(frames) <= 61
    assert all(len(pose) == 7 and all(round(value, 3) == value for value in pose)
               for _, poses in frames for pose in poses)
    payload_size = len(json.dumps(results, separators=(",", ":")).encode("utf-8"))
    assert payload_size < 5 * 1024 * 1024 and MAX_BATCH_GHOST_BYTES < 5 * 1024 * 1024


def failed_run(spec):
    sim, _, _ = run(spec)
    return sim.summary()


def test_diagnose_confirms_the_step_and_rules_out_low_gravity():
    from simgate.diagnose import MAX_RERUNS, diagnose

    # On Mars the 4-wheel rover climbs 0.15 m, so a 0.2 m step is the failing case there.
    spec = SceneSpec(domain="robot_terrain", environment=Environment(body="mars", surface="regolith"),
                     terrain=Terrain(kind="flat", roughness=0), robot=Robot(kind="rover"),
                     obstacles=[Obstacle(kind="step", height_m=0.2)]).fill_defaults()
    steps = []
    report = diagnose(spec, failed_run(spec), on_step=steps.append)
    assert report["reproduced"] is True and report["runs"] <= MAX_RERUNS
    verdicts = {h["cause"]: h["verdict"] for h in report["hypotheses"]}
    assert verdicts["obstacle"] == "confirmed" and verdicts["low gravity"] == "ruled out", verdicts
    assert any(h.get("control") for h in report["hypotheses"])
    assert "Confirmed: obstacle" in report["conclusion"] and "low gravity" in report["conclusion"]
    assert [(step["agent"], step["tool"]) for step in steps if step["status"] != "working"] == [
        ("Batch runner", "reproduce"), ("Explainer", "hypotheses"), ("Batch runner", "experiments"),
        ("Explainer", "decision"), ("Explainer", "conclusion")]
    assert report["decisions"] == [{"agent": "Explainer", "decision": "conclude", "reason": "obstacle passed on its own."}]


def test_diagnose_confirms_slipping_on_an_icy_slope():
    from simgate.diagnose import diagnose

    spec = SceneSpec(domain="robot_terrain", environment=Environment(surface="ice"),
                     terrain=Terrain(kind="slope", slope_deg=25, roughness=0.1)).fill_defaults()
    report = diagnose(spec, failed_run(spec))
    slipping = next(h for h in report["hypotheses"] if h["cause"] == "slipping")
    assert slipping["intervention"] == {"surface": "rock"} and slipping["verdict"] == "confirmed"
    assert report["conclusion"].startswith("Confirmed: slipping")


def test_diagnose_does_not_diagnose_a_passing_run():
    from simgate.diagnose import diagnose

    spec = SceneSpec(domain="robot_terrain", environment=Environment(body="mars", surface="regolith"),
                     terrain=Terrain(kind="flat", roughness=0), robot=Robot(kind="rover"),
                     obstacles=[Obstacle(kind="step", height_m=0.15)]).fill_defaults()
    report = diagnose(spec, failed_run(spec))
    assert report["reproduced"] is None and report["runs"] == 0 and report["hypotheses"] == []
    assert report["conclusion"].startswith("This run passed")


def test_experiment_knobs_and_per_robot_metrics():
    from simgate.batch import _validate_variation, obstacle_entry_index
    from simgate.diagnose import robot_metrics

    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat"), robot=Robot(kind="rover"),
                     obstacles=[Obstacle(kind="rock", count=10), Obstacle(kind="step", count=2),
                                Obstacle(kind="gap")]).fill_defaults()
    assert [obstacle_entry_index(spec, name) for name in ("obs0_rock", "obs1_step", "obs2_step", "obs3_gap")] == [0, 1, 1, 2]
    _, removed, _ = _validate_variation(spec, {"remove_obstacle": "obs2_step"})
    assert [(o.kind, o.count) for o in removed.obstacles] == [("rock", 10), ("step", 1), ("gap", 1)]
    _, slower, _ = _validate_variation(spec, {"speed_scale": 0.5})
    assert slower.robot.drive_speed == slower.robots[0].robot.drive_speed == 3.0
    _, bigger, _ = _validate_variation(spec, {"robot": {"wheel_radius": 0.2, "wheels": 6}})
    assert (bigger.robots[0].robot.wheel_radius, bigger.robots[0].robot.wheels) == (0.2, 6)
    for bad in ({"robot": {"kind": "walker"}}, {"robot": {"wings": 2}}, {"face_uphill": True}, {"no_push": True}):
        try:
            _validate_variation(spec, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")

    multi = {"robots": {"r1": {"goal_reached": True, "final_distance_m": 0.5},
                        "r2": {"goal_reached": False, "final_distance_m": 4.0, "flipped": True,
                               "obstacles": {"stopped_by": {"name": "obs0_step", "kind": "step"}}}}}
    second = robot_metrics(multi, "r2")
    assert not second["passed"] and second["stopped_by"]["name"] == "obs0_step" and second["flipped"]
    assert robot_metrics(multi, "r1")["passed"]


def step_course(height, kind="step"):
    return SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=Robot(kind="rover"),
                     obstacles=[Obstacle(kind=kind, height_m=height)]).fill_defaults()


def test_redesign_finds_the_smallest_passing_design_and_keeps_the_world():
    from simgate.batch import _validate_variation
    from simgate.diagnose import pin_task
    from simgate.redesign import BOUNDS, redesign

    spec = step_course(0.15)
    summary = failed_run(spec)
    before = spec.model_dump()
    report = redesign(spec, summary)
    assert spec.model_dump() == before   # never applied automatically
    winner = report["winner"]
    assert winner is not None and set(winner["changes"]) <= {"wheel_radius", "wheels"}, winner
    assert winner["robustness"]["total"] == 3 and winner["robustness"]["passed"] >= 2
    assert report["tried_count"] <= 12 and "passes" in report["message"] and "use this design" in report["message"]
    world = pin_task(spec, summary)
    _, designed, _ = _validate_variation(world, {"robot": winner["changes"]})
    for part in ("environment", "terrain", "obstacles", "duration_s"):
        assert getattr(designed, part) == getattr(world, part), part
    assert designed.robots[0].goal_xy == world.robots[0].goal_xy and designed.robots[0].goal_xy is not None
    passing = [c for r in report["rounds"] for c in r["candidates"] if c["passed"]]
    assert winner["distance"] == min(c["distance"] for c in passing)
    for candidate in (c for r in report["rounds"] for c in r["candidates"]):
        for name, value in candidate["changes"].items():
            low, high = BOUNDS["rover"][name]
            assert low <= value <= high, (name, value)

    kept = redesign(spec, summary, "keep 4 wheels")
    assert set(kept["winner"]["changes"]) == {"wheel_radius"}
    assert all("wheels" not in c["changes"] for r in kept["rounds"] for c in r["candidates"])


def test_redesign_reports_the_limit_when_nothing_passes():
    from simgate.redesign import redesign

    spec = step_course(1.0, "wall")
    report = redesign(spec, failed_run(spec))
    assert report["winner"] is None and report["tried_count"] <= 12 and len(report["rounds"]) <= 3
    assert report["message"].startswith("No design passed") and report["limit"] == "wheel radius already at the 0.35 m maximum"
    assert report["best_attempt"]["changes"]


def test_design_constraints_narrow_the_design_space():
    from simgate.redesign import DesignSpace

    rover = Robot(kind="rover")
    space = DesignSpace(rover, "make it pass, keep it under 15 kg")
    assert space.bounds["mass_kg"] == (3.0, 15.0)
    assert space.clean({"mass_kg": 30, "wheel_radius": 0.9, "legs": 6})[0] == {"mass_kg": 15.0, "wheel_radius": 0.35}
    only = DesignSpace(rover, "only change the wheels")
    assert set(only.knobs) == {"wheel_radius", "wheels"}
    assert DesignSpace(rover, "keep 4 wheels and don't change the speed").fixed == {"wheels": 4, "drive_speed": 6.0}
    walker = DesignSpace(Robot(kind="walker"), "keep six legs")
    assert walker.fixed == {"legs": 6} and "mass_kg" not in walker.bounds
    g1 = DesignSpace(Robot(kind="library", robot_id="g1", policy_id="g1_walking"), None)
    assert g1.knobs == ["command_speed_m_s"]


def test_mock_explainer_tests_more_only_when_the_best_is_partly():
    from simgate.diagnose import rule_decision

    spec = SceneSpec(domain="robot_terrain", environment=Environment(surface="ice"),
                     terrain=Terrain(kind="slope", slope_deg=25)).fill_defaults()
    slip = {"cause": "slipping", "intervention": {"surface": "rock"}, "verdict": "partly", "improvement": 0.5}
    steep = {"cause": "steep ground", "intervention": {"slope_deg": 15}, "verdict": "partly", "improvement": 0.4}
    speed = {"cause": "too fast", "intervention": {"speed_scale": 0.6}, "verdict": "ruled out", "improvement": 0.0}
    both = rule_decision(spec, [slip, steep, speed], remaining=2)
    assert both["decision"] == "test more" and both["next"] == [
        {"cause": "slipping + steep ground", "intervention": {"surface": "rock", "slope_deg": 15}}]
    one = rule_decision(spec, [slip, speed], remaining=1)
    assert one["decision"] == "test more" and one["next"][0]["cause"] == "slipping (stronger)"
    assert rule_decision(spec, [slip, steep], remaining=0)["decision"] == "conclude"
    assert rule_decision(spec, [dict(slip, verdict="confirmed"), steep], remaining=2)["decision"] == "conclude"
    assert rule_decision(spec, [dict(slip, verdict="ruled out"), speed], remaining=2)["decision"] == "conclude"


def scripted_models(replies):
    """A fake chat model that answers by role: the reply list for the first matching system-prompt key."""
    calls = []

    def chat(messages, model=None, temperature=0.2):
        system = messages[0]["content"]
        role = next(key for key in replies if key in system)
        calls.append((role, model, messages))
        return replies[role].pop(0)

    return chat, calls


def test_model_explainer_decisions_are_validated_bounded_and_visible():
    import json

    from simgate import config, llm
    from simgate.diagnose import MAX_RERUNS, diagnose

    spec = SceneSpec(domain="robot_terrain", environment=Environment(body="mars", surface="regolith"),
                     terrain=Terrain(kind="flat", roughness=0), robot=Robot(kind="rover"),
                     obstacles=[Obstacle(kind="step", height_m=0.2)]).fill_defaults()
    summary = failed_run(spec)
    chat, calls = scripted_models({
        "candidate\ncauses": [json.dumps({"hypotheses": [
            {"cause": "low gravity", "why": "gravity is 3.71 m/s²", "intervention": {"gravity": 9.81}},
            {"cause": "wheels too small", "why": "stopped at a 0.2 m step", "intervention": {"wheel_radius": 0.3}},
            {"cause": "step height", "why": "stuck 7.9 m short at the 0.2 m step", "intervention": {"obstacle_scale": 0.9}},
            {"cause": "slipping", "why": "slip was 1%", "intervention": {"surface": "rock"}, "control": True}]})],
        "deciding what to do": [
            json.dumps({"decision": "test more", "reason": "The step only partly explains it; trying a lower step.",
                        "next": [{"cause": "step height (stronger)", "intervention": {"obstacle_scale": 0.25}},
                                 {"cause": "bad", "intervention": {"mass_kg": 5}}]}),
            json.dumps({"decision": "conclude", "reason": "The lower step passed, so the evidence is enough."})],
        "opening sentence is already written": ["The step height explains it."],
    })
    steps = []
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    try:
        config.MOCK_MODE, llm.chat = False, chat
        report = diagnose(spec, summary, on_step=steps.append)
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat

    causes = [h["cause"] for h in report["hypotheses"]]
    assert "wheels too small" not in causes and causes[-1] == "step height (stronger)"
    step_height = next(h for h in report["hypotheses"] if h["cause"] == "step height")
    assert step_height["intervention"] == {"obstacle_scale": 0.5}   # the allowed mapping, not the model's 0.9
    assert any(d["reason"] == "step height: obstacle_scale 0.9 isn't an allowed experiment; testing 0.5 instead."
               for d in report["decisions"])
    assert sum(1 for h in report["hypotheses"] if h.get("follow_up")) == 1
    assert report["runs"] <= MAX_RERUNS and report["hypotheses"][-1]["verdict"] == "confirmed"
    assert [role for role, _, _ in calls] == ["candidate\ncauses", "deciding what to do", "deciding what to do",
                                          "opening sentence is already written"]
    reasons = [step["result"] for step in steps if step["status"] != "working"]
    for decision in report["decisions"]:
        assert any(decision["reason"] in reason for reason in reasons), decision
    assert any("wheel_radius is not an allowed experiment" in d["reason"] for d in report["decisions"])
    assert any("mass_kg" not in d["reason"] and d["decision"] == "dropped" and d["reason"].startswith("bad:")
               for d in report["decisions"])
    assert "Test more: The step only partly explains it" in " ".join(reasons)
    assert "Adjusted: step height: obstacle_scale 0.9 isn't an allowed experiment; testing 0.5 instead." in reasons
    assert "Dropped: wheels too small: wheel_radius is not an allowed experiment." in reasons
    assert report["conclusion"] == "Confirmed: step height and step height (stronger). The step height explains it."


def test_model_designer_candidates_are_clamped_and_stop_ends_early():
    import json

    from simgate import config, llm
    from simgate.diagnose import robot_metrics
    from simgate.redesign import redesign

    spec = step_course(0.15)
    summary = failed_run(spec)
    diagnosis = {"reproduced": True, "hypotheses": [], "reproduced_metrics": robot_metrics(summary)}
    chat, calls = scripted_models({"You are the Designer": [
        json.dumps({"strategy": "bigger change", "reason": "Nothing is close yet.",
                    "candidates": [{"changes": {"drive_speed": 0.5}, "why": "slower"},
                                   {"changes": {"wings": 2}, "why": "not a knob"}]}),
        json.dumps({"strategy": "stop", "reason": "Speed alone can't climb a step.", "candidates": []}),
    ]})
    steps = []
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    try:
        config.MOCK_MODE, llm.chat = False, chat
        report = redesign(spec, summary, diagnosis=diagnosis, on_step=steps.append)
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat

    assert [model for _, model, _ in calls] == [config.BUILDER_MODEL] * 2
    assert len(report["rounds"]) == 1 and report["tried_count"] == 1
    assert report["rounds"][0]["candidates"][0]["changes"] == {"drive_speed": 2.0}   # clamped to the 2 rad/s minimum
    assert report["decisions"][-1] == {"agent": "Designer", "decision": "stop", "reason": "Speed alone can't climb a step."}
    reasons = " ".join(step["result"] for step in steps)
    assert "wheel speed 0.5 clamped to 2 rad/s" in reasons and "wings isn't a design knob" in reasons
    assert "Stop: Speed alone can't climb a step." in reasons
    for decision in report["decisions"]:
        assert decision["reason"] in reasons, decision
    assert report["winner"] is None and report["message"].startswith("No design passed")


def test_model_loops_have_code_backstops():
    import json

    from simgate import config, llm
    from simgate.diagnose import model_decision, robot_metrics
    from simgate.redesign import redesign

    spec = step_course(0.2)
    confirmed = {"cause": "obstacle", "intervention": {"remove_obstacle": "obs0_step"}, "verdict": "confirmed",
                 "key_numbers": {"passed": [False, True]}}
    partly = {"cause": "friction", "intervention": {"friction": 1.35}, "verdict": "partly",
              "key_numbers": {"passed": [False, False]}}
    decisions = []
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    try:
        config.MOCK_MODE = False
        llm.chat = lambda *_args, **_kwargs: json.dumps({"decision": "test more", "reason": "Check friction.", "next": [
            {"cause": "obstacle + friction", "intervention": {"remove_obstacle": "obs0_step", "friction": 1.35}},
            {"cause": "friction (stronger)", "intervention": {"friction": 2.0}}]})
        decided = model_decision(spec, [confirmed, partly], 2, decisions, [])
        assert [item["cause"] for item in decided["next"]] == ["friction (stronger)"]
        assert decisions[-1]["reason"] == "obstacle + friction: it adds to obstacle, which already passes on its own."

        step = step_course(0.15)
        summary = failed_run(step)
        diagnosis = {"reproduced": True, "hypotheses": [], "reproduced_metrics": robot_metrics(summary)}
        chat, calls = scripted_models({"You are the Designer": [
            json.dumps({"strategy": "bigger change", "reason": "Try slower wheels.",
                        "candidates": [{"changes": {"drive_speed": 5.0}}]}),
            json.dumps({"strategy": "refine", "reason": "Again.", "candidates": [{"changes": {"drive_speed": 5.0}}]}),
            json.dumps({"strategy": "stop", "reason": "Done.", "candidates": []}),
        ]})
        llm.chat = chat
        steps = []
        report = redesign(step, summary, diagnosis=diagnosis, on_step=steps.append)
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat
    assert len(report["rounds"]) >= 2 and report["rounds"][1]["candidates"]
    fallback = next(d for d in report["decisions"] if d["decision"] == "fallback")
    assert "all tried already or not allowed" in fallback["reason"]
    assert any(fallback["reason"] in step["result"] for step in steps)
    system = calls[0][2][0]["content"]
    assert "6 wheels let the rear wheels push the front ones up a step" in system
    assert '"robot_limits": {"climb_m": 0.09' in calls[0][2][1]["content"]


def test_director_plans_diagnose_and_redesign_in_one_call():
    import json

    from simgate import config, llm
    from simgate.director import Director
    from simgate.workflows import SessionState

    spec = step_course(0.25)
    failed = SessionState(summary={"goal_reached": False, "final_distance_m": 8.0, "outcome": "stuck"})
    passed = SessionState(summary={"goal_reached": True, "final_distance_m": 0.5, "outcome": "reached"})
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    calls = []

    def planned(text, reply, state=failed):
        llm.chat = lambda messages, *_args, **_kwargs: calls.append(messages) or json.dumps(reply)
        director = Director()
        return director.plan(text, spec, state), director.guard_notes

    try:
        config.MOCK_MODE = False
        plan, _ = planned("why did it fail?", {"workflow": "diagnose", "summary": "Why the rover got stuck"})
        assert plan.workflow == "diagnose" and len(calls) == 1
        calls.clear()
        plan, _ = planned("make it pass, keep it under 15 kg",
                          {"workflow": "redesign", "summary": "A rover that passes", "constraints": "light"})
        assert plan.workflow == "redesign" and plan.constraints == "make it pass, keep it under 15 kg" and len(calls) == 1
        # the guard: the model chose something else, but the text clearly asks why after a failed run
        calls.clear()
        plan, notes = planned("why did it fail?", {"workflow": "answer", "summary": "Answer", "reply": "It got stuck."})
        assert plan.workflow == "diagnose" and len(calls) == 1
        assert notes[-1] == "You asked why it failed after a failed run, so I'm using the diagnose workflow."
        plan, notes = planned("why did it fail?", {"workflow": "answer", "summary": "Answer", "reply": "It passed."},
                              state=passed)
        assert plan.workflow == "answer" and notes == []   # no failed run: no guard
        # no robot scene: diagnose fails validation and gets the retry
        calls.clear()
        llm.chat = lambda messages, *_args, **_kwargs: calls.append(messages) or json.dumps(
            {"workflow": "diagnose", "summary": "Why"})
        Director().plan("why did it fail?", None, failed)
        assert len(calls) == 2 and "requires an existing robot scene" in calls[1][-1]["content"]
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat

    config.MOCK_MODE = True
    assert Director().plan("why did it fail?", spec, failed).workflow == "diagnose"
    redesign_plan = Director().plan("make it pass, keep it under 15 kg", spec, failed)
    assert redesign_plan.workflow == "redesign" and redesign_plan.constraints == "make it pass, keep it under 15 kg"
    assert Director().plan("what would work?", spec, failed).workflow == "redesign"
    assert Director().plan("why did it fail?", spec, SessionState()).workflow == "answer"   # nothing ran yet


def test_use_this_design_applies_only_the_winner_and_rebuilds_the_title():
    import asyncio

    from simgate import config, llm
    from simgate.director import Director
    from simgate.workflows import SessionState

    spec = step_course(0.25)
    spec.robots[0].robot.wheel_radius = spec.robot.wheel_radius = 0.18
    state = SessionState(redesign={"type": "redesign", "robot_id": "r1",
                                   "winner": {"changes": {"wheels": 6}, "description": "wheels 4 → 6"}})
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    try:
        config.MOCK_MODE = False
        llm.chat = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no model call expected"))
        world, message = asyncio.run(Director().execute_plan("use this design", spec, state))
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat
    robot = world.spec.robots[0].robot
    assert robot.wheels == 6 and robot.wheel_radius == 0.18
    assert robot.model_dump(exclude={"wheels"}) == spec.robots[0].robot.model_dump(exclude={"wheels"})
    for part in ("environment", "terrain", "obstacles", "duration_s"):
        assert getattr(world.spec, part) == getattr(spec, part), part
    assert world.spec.title == "6-wheel rover on flat terrain, Earth with an obstacle course"
    assert message == "Run it?"
    nothing = Director().plan("use this design", spec, SessionState())
    assert nothing.workflow == "answer" and "no passing design" in nothing.reply


def test_an_answer_keeps_the_built_scene():
    import asyncio

    from simgate.orchestrator import Session

    async def flow():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.handle("A rover on Mars hills")
        build = session.build
        await session.handle("what's the gravity on Mars?")
        assert session.build is build
        await session.handle("run")
        return events

    events = asyncio.run(flow())
    assert any(event["type"] == "run_start" for event in events)


def test_show_runs_replays_the_winner_without_applying_it():
    import asyncio

    from simgate.orchestrator import Session

    async def flow():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.handle("Rover obstacle course with a big step")
        await session.handle("run")
        spec, build = session.spec.model_copy(deep=True), session.build
        session.last_redesign = {"type": "redesign", "robot_id": "r1",
                                 "winner": {"changes": {"wheels": 6}, "description": "wheels 4 → 6"}}
        start = len(events)
        await session.show_design()
        replay = events[start:]
        assert session.spec == spec and session.build is build   # shown, not applied
        start = len(events)
        await session.handle("run again")
        return replay, events[start:]

    replay, rerun = asyncio.run(flow())
    types = [event["type"] for event in replay]
    assert types.index("scene") < types.index("run_start") < types.index("frames") < types.index("metrics")
    scene = next(event["scene"] for event in replay if event["type"] == "scene")
    assert next(e["metrics"] for e in replay if e["type"] == "metrics")["goal_reached"]
    assert "wheels 4 → 6" in replay[-1]["message"] and "use this design" in replay[-1]["message"]
    rerun_types = [event["type"] for event in rerun]
    assert rerun_types.index("scene") < rerun_types.index("run_start")   # the original scene comes back first
    assert next(e["scene"] for e in rerun if e["type"] == "scene") != scene


def test_browser_handles_the_analysis_events():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    page = (root / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    for needed in ('case "diagnosis"', 'case "redesign"', 'id="diagnosisCard"', 'id="redesignCard"',
                   'chips(q, ev.chips, null)', 'send($("useDesign").textContent.toLowerCase())', '"Use fragile design"', 'type: "show_design"',
                   '"✓ confirmed"', '"~ partly"', '"✕ ruled out"', 'Designer: [55, 190]', '"Batch runner": [175, 190]',
                   'if (ev.chips)'):
        assert needed in page, needed
    assert 'msg.get("type") == "show_design"' in (root / "server.py").read_text(encoding="utf-8")


def test_full_session_why_then_make_it_pass():
    import asyncio

    from simgate.orchestrator import Session

    async def flow():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        turns = {}
        for text in ("Rover obstacle course with a big step", "run", "why did it fail?", "make it pass",
                     "use this design", "run"):
            start = len(events)
            await session.handle(text)
            turns.setdefault(text, []).append(events[start:])
        return session, turns

    session, turns = asyncio.run(flow())
    first_run, final_run = turns["run"]
    assert not next(e["metrics"] for e in first_run if e["type"] == "metrics")["goal_reached"]
    assert first_run[-1]["type"] == "explanation"

    why = turns["why did it fail?"][0]
    diagnosis = next(e["report"] for e in why if e["type"] == "diagnosis")
    assert diagnosis["reproduced"] and any(h["verdict"] == "confirmed" and h["cause"] == "obstacle"
                                           for h in diagnosis["hypotheses"])
    assert why[-1]["type"] == "question" and why[-1]["chips"] == ["Make it pass"]
    assert not any(e["type"] in ("scene", "spec") for e in why)   # the scene itself didn't change

    fix = turns["make it pass"][0]
    redesign = next(e["report"] for e in fix if e["type"] == "redesign")
    assert redesign["winner"] and redesign["winner"]["robustness"]["total"] == 3
    assert fix[-1]["chips"] == ["Use this design"]
    for turn, agents_seen in ((why, {"Explainer", "Batch runner"}), (fix, {"Designer", "Batch runner", "Checker"})):
        steps = [e for e in turn if e["type"] == "step"]
        ids = {e["id"] for e in steps}
        assert all(e["parent_id"] in ids for e in steps if e["parent_id"] is not None)
        assert agents_seen <= {e["agent"] for e in steps}

    applied = session.spec.robots[0].robot
    for name, value in redesign["winner"]["changes"].items():
        assert getattr(applied, name) == value
    assert session.spec.title.startswith(("6-wheel rover", "4-wheel rover")) and "obstacle course" in session.spec.title
    types = [e["type"] for e in final_run]
    assert types.index("run_start") < types.index("frames") < types.index("metrics")
    assert next(e["metrics"] for e in final_run if e["type"] == "metrics")["goal_reached"]


def session_turns(*texts):
    """Run messages through orchestrator.Session (the browser's path); return the session and each turn's events."""
    import asyncio

    from simgate.orchestrator import Session

    async def flow():
        events, turns = [], []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        for text in texts:
            start = len(events)
            await session.handle(text)
            turns.append(events[start:])
        return session, turns

    return asyncio.run(flow())


def test_race_on_an_existing_course_builds_and_checks():
    session, (_, race) = session_turns("Rover obstacle course with a big step", "race three rovers to the far side")
    assert not [e for e in race if e["type"] == "error"], [e["message"] for e in race if e["type"] == "error"]
    assert len(session.spec.robots) == 3 and session.spec.obstacles
    checks = next(e["checks"] for e in race if e["type"] == "checks")
    assert any(check["name"] == "Robot starts and goals clear obstacles" for check in checks)


MIXED_RACE = "race a rover and a six-legged walker to the summit on Mars hills, go"


def test_stress_test_on_a_race_reports_each_robot():
    session, (_, stress) = session_turns(MIXED_RACE, "stress test")
    results = [e["result"] for e in stress if e["type"] == "batch_result"]
    assert results and not [r["error"] for r in results if r.get("error")], [r.get("error") for r in results]
    ids = [entry.id for entry in session.spec.robots]
    assert all(list(r["robots"]) == ids for r in results)
    report = next(e for e in stress if e["type"] == "report")
    assert [column["id"] for column in report["robots"]] == ids
    assert all(set(row["robots"]) == set(ids) and all("passed" in cell for cell in row["robots"].values())
               for row in report["rows"])
    baseline = next(r for r in results if r["label"] == "Baseline")
    payload = next(r for r in results if r["changes"].get("payload_kg"))
    for robot_id in ids:   # the payload reaches every robot
        before = baseline["summary"]["robots"][robot_id]["robot"]["mass_kg"]
        assert payload["summary"]["robots"][robot_id]["robot"]["mass_kg"] == before + payload["changes"]["payload_kg"]


def test_race_diagnosis_and_redesign_target_one_robot():
    from simgate.redesign import BOUNDS

    session, (race, rover, walker, fix) = session_turns(
        MIXED_RACE, "why did the rover fail?", "why did it fail?", "make the walker pass")
    robots = next(e["metrics"] for e in race if e["type"] == "metrics")["robots"]
    assert robots["r1"]["goal_reached"] and not robots["r2"]["goal_reached"]
    assert [robots[robot_id]["robot"]["kind"] for robot_id in ("r1", "r2")] == ["rover", "walker"]

    named = next(e["report"] for e in rover if e["type"] == "diagnosis")   # the rover won: nothing to explain
    assert named["robot_id"] == "r1" and named["runs"] == 0 and "didn't fail" in named["conclusion"]

    diagnosis = next(e["report"] for e in walker if e["type"] == "diagnosis")
    assert diagnosis["robot_id"] == "r2" and diagnosis["reproduced"]
    assert not [h for h in diagnosis["hypotheses"] if h["verdict"] == "not tested"]
    first = [e for e in walker if e["type"] == "step" and e["status"] != "working" and e["agent"] != "Director"][0]
    assert first["tool"] == "target" and "Walker (r2)" in first["result"] and "Rover stays unchanged" in first["result"]

    redesign = next(e["report"] for e in fix if e["type"] == "redesign")
    assert redesign["robot_id"] == "r2" and redesign["kind"] == "walker"
    tried = [c["changes"] for r in redesign["rounds"] for c in r["candidates"]]
    assert tried and all(set(changes) <= set(BOUNDS["walker"]) for changes in tried)
    checks = [e for e in fix if e["type"] == "step" and e["agent"] == "Checker" and e["status"] != "working"]
    assert checks and all(e["status"] == "done" and "World unchanged" in e["result"] for e in checks)


def test_failure_chips_name_the_robot_that_failed():
    _, (race, why, rover) = session_turns(MIXED_RACE, "Why did the Walker fail?", "why did the rover fail?")
    explained = [e for e in race if e["type"] == "explanation"][-1]
    assert explained["chips"] == ["Why did the Walker fail?", "Make the Walker pass"]
    assert next(e["report"] for e in why if e["type"] == "diagnosis")["robot_id"] == "r2"
    assert why[-1]["chips"] == ["Make the Walker pass"]
    assert rover[-1]["chips"] == ["Why did the Walker fail?", "Make the Walker pass"]   # the rover won

    _, (_, won) = session_turns("race three rovers to the far side", "run")
    assert all(robot["goal_reached"] for robot in next(e["metrics"] for e in won if e["type"] == "metrics")["robots"].values())
    assert "chips" not in [e for e in won if e["type"] == "explanation"][-1]


def test_crashed_runs_are_a_system_problem_not_physics():
    from simgate import batch, diagnose

    real = batch.run_batch

    def crashing(which):
        def run(spec, variations, on_result=None, timeout=batch.RUN_TIMEOUT_SECONDS, baseline=True):
            results = real(spec, variations, None, timeout, baseline)
            for index, result in enumerate(results):
                if which(index, result):
                    results[index] = batch._result_error(index, result["label"], result["changes"], "RuntimeError: boom")
                if on_result:
                    on_result(results[index])
            return results
        return run

    stress_explained = []
    original_explain = agents._mock_explain_stress
    try:   # half the stress runs crash: the rest are explained, the crashed ones are only reported
        agents._mock_explain_stress = lambda results: stress_explained.append(results) or original_explain(results)
        batch.run_batch = crashing(lambda index, result: index % 2 == 1)
        _, (_, stress) = session_turns("Rover obstacle course with a big step", "stress test")
        warning = next(e for e in stress if e["type"] == "step" and e["tool"] == "crashed")
        total = next(e for e in stress if e["type"] == "batch_done")["total"]
        assert warning["status"] == "warning" and warning["result"].startswith(f"{total // 2} of {total} runs crashed: RuntimeError: boom")
        assert stress_explained and not any(r.get("error") for r in stress_explained[-1])
        message = stress[-1]["message"]
        assert stress[-1]["type"] == "explanation" and message.startswith(f"{total // 2} of {total} runs crashed")
        assert f"of {total - total // 2} runs passed" in message   # the pattern counts only runs that ran

        stress_explained.clear()   # every run crashes: no Explainer at all
        batch.run_batch = crashing(lambda index, result: True)
        _, (_, stress) = session_turns("Rover obstacle course with a big step", "stress test")
        assert not stress_explained and not [e for e in stress if e["type"] == "agent" and e["agent"] == "Explainer"]
        assert "runs crashed" in stress[-1]["message"] and "nothing to analyse" in stress[-1]["message"]
    finally:
        batch.run_batch, agents._mock_explain_stress = real, original_explain

    try:   # the reproduction runs, every experiment crashes: no verdicts, decisions or conclusion from them
        diagnose.run_batch = crashing(lambda index, result: result["label"] != "Baseline")
        _, (_, why) = session_turns("Rover obstacle course with a big step, go", "why did it fail?")
    finally:
        diagnose.run_batch = real
    report = next(e["report"] for e in why if e["type"] == "diagnosis")
    assert report["reproduced"] and report["outcome"] == "crashed"
    assert report["conclusion"].startswith(f"{len(report['hypotheses'])} of {len(report['hypotheses'])} runs crashed")
    tools = [e["tool"] for e in why if e["type"] == "step" and e["status"] != "working"]
    assert "crashed" in tools and "decision" not in tools and "conclusion" not in tools


def test_mixed_race_time_limit_uses_the_slowest_robot():
    from simgate import batch, diagnose

    real, rerun_durations = batch.run_batch, []

    def recording(spec, variations, *args, **kwargs):
        rerun_durations.append(spec.duration_s)
        return real(spec, variations, *args, **kwargs)

    try:
        diagnose.run_batch = recording
        session, (race, _) = session_turns(MIXED_RACE, "why did it fail?")
    finally:
        diagnose.run_batch = real
    assert session.spec.duration_s == 45.0
    assert next(e["duration_s"] for e in race if e["type"] == "run_start") == 45.0
    assert rerun_durations and set(rerun_durations) == {45.0}   # reruns pinned to the original task

    rovers, _ = session_turns("race three rovers to the far side")
    assert rovers.spec.duration_s == 30.0


def test_explainers_get_each_robots_kind_and_mark_untested_gaits():
    import json

    from simgate import config, llm

    plan = {"workflow": "race", "summary": "Rover and walker race to the summit", "world": {"body": "mars", "terrain_kind": "hills"},
            "robots": [{"kind": "rover"}, {"kind": "walker", "legs": 6}], "goal": {"place": "highest"}, "run_now": False}
    chat, calls = scripted_models({
        "Plan JSON schema:": [json.dumps(plan)],
        "You are the Explainer in a physics sandbox": ["The walker ran out of time. Try a gait of 1.5 Hz."],
        "You are the Stress planner": ["{}", "{}"],   # invalid twice: the compatible defaults are used
        "You are the Stress-test Explainer": ["The walker missed most runs. Try a 20° stride at 2.0 Hz."],
    })
    original_mock, original_chat = config.MOCK_MODE, llm.chat
    try:
        config.MOCK_MODE, llm.chat = False, chat
        _, (_, run, stress) = session_turns("race a rover and a walker to the summit", "run", "stress test")
    finally:
        config.MOCK_MODE, llm.chat = original_mock, original_chat

    explained = [e for e in run if e["type"] == "explanation"][-1]["message"]
    assert explained == "The walker ran out of time. Try a gait of 1.5 Hz. (untested settings)"
    assert stress[-1]["message"].endswith("Try a 20° stride at 2.0 Hz. (untested settings)")

    system, user = next(messages for role, _, messages in calls if role == "You are the Explainer in a physics sandbox")
    assert "Never compare a property one robot doesn't" in system["content"]
    robots = json.loads(user["content"].split("Metrics: ", 1)[1])["robots"]
    rover, walker = (next(m for m in robots.values() if m["kind"] == kind) for kind in ("rover", "walker"))
    assert "drive_speed" in rover["robot"] and not {"gait_freq_hz", "stride_deg", "legs"} & set(rover["robot"])
    assert "gait_freq_hz" in walker["robot"] and not {"drive_speed", "wheels", "wheel_radius"} & set(walker["robot"])

    system, user = next(messages for role, _, messages in calls if role == "You are the Stress-test Explainer")
    assert "never compare a property" in system["content"]
    for result in json.loads(user["content"].split("Stress-test results: ", 1)[1]):
        for m in result["summary"]["robots"].values():
            assert m["kind"] in ("rover", "walker")
            assert not ({"drive_speed", "wheels"} if m["kind"] == "walker" else {"gait_freq_hz", "legs"}) & set(m["robot"])

    from simgate.agents import mark_untested_gait
    from simgate.spec import SceneSpec
    walker_scene = SceneSpec(domain="robot_terrain", robots=[RobotEntry(id="r1", label="W", robot=Robot(kind="walker"))]).fill_defaults()
    for text in ("Keep the 2.0 Hz gait and 15° stride.", "A 1.5 Hz gait is untested but may help.", "It tilted 13° on a 12° slope."):
        assert mark_untested_gait(text, walker_scene) == text


def test_mock_stress_explainer_reads_each_racer():
    _, (race, stress) = session_turns(MIXED_RACE, "stress test")
    robots = next(e["metrics"] for e in race if e["type"] == "metrics")["robots"]
    assert robots["r1"]["goal_reached"] and not robots["r2"]["goal_reached"]
    rows = next(e for e in stress if e["type"] == "report")["rows"]
    rover_passes = sum(row["robots"]["r1"]["passed"] for row in rows)
    walker_passes = sum(row["robots"]["r2"]["passed"] for row in rows)
    message = stress[-1]["message"]
    assert message.startswith(f"Rover passed {rover_passes} of {len(rows)} runs; Walker passed {walker_passes} of {len(rows)} runs.")
    assert "Baseline: Rover reached; Walker out_of_time." in message
    assert "0 m forward progress" not in message


def model_explainer_only(replies):
    """Plan and edit with the rules, but explain runs with a scripted model; returns (restore, calls)."""
    from simgate import config, llm

    chat, calls = scripted_models(replies)
    original_explain, original_chat = agents.explain, llm.chat

    def explain(spec, metrics):
        config.MOCK_MODE = False
        try:
            return original_explain(spec, metrics)
        finally:
            config.MOCK_MODE = True

    agents.explain, llm.chat = explain, chat

    def restore():
        agents.explain, llm.chat = original_explain, original_chat

    return restore, calls


def test_explainer_uses_code_facts_and_retries_contradictions():
    restore, calls = model_explainer_only({"You are the Explainer in a physics sandbox": [
        "On flat ground the rover won and the walker ran out of time.",
        "On the hills the rover won and the walker ran out of time."]})
    try:   # "flat" for hills: one retry with the exact problem, and the corrected reply is used
        _, (race,) = session_turns(MIXED_RACE)
    finally:
        restore()
    assert [e for e in race if e["type"] == "explanation"][-1]["message"] == "On the hills the rover won and the walker ran out of time."
    messages = calls[-1][2]
    assert "Describe the terrain and tilt only with the words given in FACTS" in messages[0]["content"]
    assert "- terrain: hills" in messages[1]["content"] and "tilt: steady" in messages[1]["content"]
    assert 'calls the terrain flat, but FACTS say the terrain is "hills"' in messages[-1]["content"]

    restore, calls = model_explainer_only({"You are the Explainer in a physics sandbox": [
        "The rover crossed the step and stayed stable at 56.9°.", "It stayed steady, tilting 56.9 degrees."]})
    try:   # a 56.9° tilt called stable twice: the rules text is used instead
        _, (_, _, run) = session_turns("Rover obstacle course with a big step", "use six wheels", "run")
    finally:
        restore()
    from simgate.facts import tilt_label

    assert [tilt_label(t) for t in (8, 20, 40, 70)] == ["steady", "leaning", "nearly tipped", "tipped over"]
    tilt = next(e["metrics"] for e in run if e["type"] == "metrics")["max_tilt_deg"]
    assert tilt > 35 and len(calls) == 2
    assert f'calls a {tilt:g}° tilt stable or steady; FACTS label it "{tilt_label(tilt)}"' in calls[-1][2][-1]["content"]
    message = [e for e in run if e["type"] == "explanation"][-1]["message"]
    assert message.startswith("The rover reached the goal") and "stable" not in message and "steady" not in message


def test_diagnose_conclusion_opening_is_code_and_contradictions_are_rejected():
    from simgate import config, llm
    from simgate.diagnose import conclusion

    def tested(cause, verdict, numbers):
        return {"cause": cause, "intervention": {"gravity": 9.81}, "verdict": verdict, "key_numbers": numbers}

    obstacle = tested("obstacle", "confirmed", {"passed": [False, True], "distance_left_m": [8.1, 0.8]})
    friction = tested("friction", "partly", {"passed": [False, False], "distance_left_m": [8.1, 5.0]})
    gravity = tested("low gravity", "ruled out", {"passed": [False, False], "distance_left_m": [8.1, 8.0]})
    key = "opening sentence is already written"

    def run(hypotheses, replies):
        chat, calls = scripted_models({key: replies})
        original_mock, original_chat = config.MOCK_MODE, llm.chat
        try:
            config.MOCK_MODE, llm.chat = False, chat
            return conclusion(hypotheses, []), calls
        finally:
            config.MOCK_MODE, llm.chat = original_mock, original_chat

    # Confirmed, then "none explains it": one retry with the exact problem; the corrected text follows the opening.
    text, calls = run([obstacle, gravity], ["None of the tested causes explains it on its own.",
                                            "Confirmed: obstacle. Without the step it reached the goal; gravity made no difference."])
    assert text == "Confirmed: obstacle. Without the step it reached the goal; gravity made no difference."
    assert '"Confirmed: obstacle."' in calls[0][2][0]["content"]
    assert 'It says "None of the tested causes", but obstacle is confirmed' in calls[-1][2][-1]["content"]

    # Nothing confirmed, then a cause claimed as confirmed (twice): the rules text follows the opening.
    text, calls = run([friction, gravity], ["Friction was confirmed as the cause of the failure.",
                                            "The cause is friction, which is confirmed."])
    assert len(calls) == 2 and 'no tested cause is confirmed' in calls[-1][2][-1]["content"]
    assert text.startswith("No single tested cause explains it. Friction partly explains it")
    assert "confirmed" not in text.lower()


def test_less_slip_without_getting_closer_is_ruled_out():
    from simgate.diagnose import verdict

    _, (_, why) = session_turns("Rover obstacle course with a big step, go", "why did it fail?")
    report = next(e["report"] for e in why if e["type"] == "diagnosis")
    slipping = next(h for h in report["hypotheses"] if h["cause"] == "slipping")
    before, after = slipping["key_numbers"]["distance_left_m"]
    assert after > 0.7 * before and slipping["key_numbers"]["slip"][1] <= 0.5 * slipping["key_numbers"]["slip"][0]
    assert slipping["verdict"] == "ruled out"   # slip fell, but it got no closer
    assert slipping["numbers_text"].endswith(
        f"slip fell from {slipping['key_numbers']['slip'][0]:.0%} to {slipping['key_numbers']['slip'][1]:.0%}, but it got no closer")
    assert "Ruled out: slipping" in report["conclusion"] and "partly" not in report["conclusion"]

    base = {"passed": False, "final_distance_m": 10.0, "flipped": True, "slip_ratio": 0.4}
    assert verdict(base, {"passed": False, "final_distance_m": 7.0, "flipped": True}) == "partly"     # 30% closer
    assert verdict(base, {"passed": False, "final_distance_m": 9.9, "flipped": False}) == "partly"    # fall avoided
    assert verdict(base, {"passed": False, "final_distance_m": 9.0, "flipped": True, "slip_ratio": 0.1}) == "ruled out"


def test_robustness_decides_the_winner():
    from simgate import redesign

    real, checked = redesign.robustness, []

    def scripted(scores):
        def robustness(world, changes, robot_id):
            checked.append(changes)
            passed = scores[min(len(checked), len(scores)) - 1]
            return {"passed": passed, "total": 3, "failed": ["seed 1", "seed 2", "friction -10%"][passed:]}
        return robustness

    try:   # the smallest passing design holds up 1/3: the next-best passing design is tried and wins at 3/3
        redesign.robustness = scripted([1, 3])
        _, (_, _, fix) = session_turns("Rover obstacle course with a big step, go", "why did it fail?", "make it pass")
    finally:
        redesign.robustness = real
    report = next(e["report"] for e in fix if e["type"] == "redesign")
    assert len(checked) == 2 and report["winner"]["changes"] == checked[1] and report["winner"]["robustness"]["passed"] == 3
    assert any(d["decision"] == "next-best" and "passed only 1/3" in d["reason"] for d in report["decisions"])
    assert fix[-1]["chips"] == ["Use this design"]

    checked.clear()
    try:   # nothing holds up: no winner, the best one is shown as fragile and can still be applied
        redesign.robustness = scripted([1])
        session, (_, _, fix, use) = session_turns("Rover obstacle course with a big step, go", "why did it fail?",
                                                  "make it pass", "use fragile design")
    finally:
        redesign.robustness = real
    report = next(e["report"] for e in fix if e["type"] == "redesign")
    assert report["winner"] is None and report["fragile"]["fragile"] and report["fragile"]["changes"] in checked
    assert "Found a design that passed once but isn't reliable (1/3)" in report["message"]
    assert "Smallest change that passes" not in report["message"]
    assert fix[-1]["chips"] == ["Use fragile design"]
    for name, value in report["fragile"]["changes"].items():
        assert getattr(session.spec.robots[0].robot, name) == value
    assert not [e for e in use if e["type"] == "error"]


def test_race_stress_report_counts_each_robot():
    from pathlib import Path

    _, (_, stress) = session_turns(MIXED_RACE, "stress test")
    report = next(e for e in stress if e["type"] == "report")
    rows = report["rows"]
    rover = sum(row["robots"]["r1"]["passed"] for row in rows)
    walker = sum(row["robots"]["r2"]["passed"] for row in rows)
    assert rover > 0 and walker < len(rows)   # some runs had a robot pass and another fail
    assert report["overall"] == f"Rover: {rover}/{len(rows)} · Walker: {walker}/{len(rows)}"
    assert all(row["passed"] is None for row in rows)   # no row marked failed because one robot failed
    page = (Path(__file__).resolve().parent.parent / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    assert "report.overall ||" in page and '"race"' in page


def test_mock_tips_fit_the_scene():
    def tip(text):
        session, (turn,) = session_turns(text)
        return session.spec, [e for e in turn if e["type"] == "explanation"][-1]["message"].rsplit("Try:", 1)[1]

    spec, course = tip("Rover obstacle course with a big step, go")
    assert spec.obstacles and "bigger wheels" in course and "six wheels" in course and "slope" not in course
    spec, flat = tip("a rover on flat ground, go")
    assert spec.terrain.kind == "flat" and not spec.obstacles and "slope" not in flat
    spec, walker = tip("a four-legged walker on hills, go")
    assert spec.robot.kind == "walker" and spec.robot.legs == 4 and "six legs" in walker

    from simgate.agents import _mock_tip

    slope = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="slope", slope_deg=30)).fill_defaults()
    assert "less steep" in _mock_tip(slope, {"flipped": True})
    assert "steeper" in _mock_tip(slope, {"flipped": False}) and "less" not in _mock_tip(slope, {"flipped": False})


def test_stress_trigger_requires_robot_scene():
    import asyncio
    from unittest.mock import patch

    from simgate import config
    from simgate.orchestrator import Session

    def fake_batch(spec, variations, on_result):
        result = {"id": "run-0", "label": "Baseline", "changes": {}, "summary": {},
                  "passed": True, "reason": "made progress", "ghost": None, "ghost_status": "unavailable"}
        on_result(result)
        return [result]

    async def scenario():
        events = []

        async def emit(event):
            events.append(event)

        with patch.object(config, "MOCK_MODE", True), \
                patch.object(agents, "plan_stress", return_value=[]) as planner, \
                patch("simgate.batch.run_batch", side_effect=fake_batch):
            empty = Session(emit)
            await empty.handle("stress test")
            assert not any(event["type"] == "batch_start" for event in events)
            assert planner.call_count == 0

            events.clear()
            structure = Session(emit)
            structure.spec = SceneSpec(domain="structure").fill_defaults()
            await structure.handle("how robust?")
            assert not any(event["type"] == "batch_start" for event in events)
            assert planner.call_count == 0

            events.clear()
            robot = Session(emit)
            robot.spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat")).fill_defaults()
            await robot.handle("Will it work on Mars?")
            assert planner.call_count == 1
            assert any(event["type"] == "batch_start" for event in events)
            assert any(event["type"] == "report" for event in events)

    assert agents.is_stress_test("stress test")
    assert agents.is_stress_test("how robust?")
    assert agents.is_stress_test("test it under different conditions")
    assert agents.is_stress_test("Will it work on Mars?")
    asyncio.run(scenario())


def test_stress_planner_falls_back_when_conditions_do_not_apply():
    import json
    from unittest.mock import patch

    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat"),
    ).fill_defaults()
    invalid_plan = {
        "variations": [
            {"label": "Angle", "slope_deg": 30},
            {"label": "Moon", "gravity": 1.62},
            {"label": "Mars", "gravity": 3.71},
            {"label": "Earth", "gravity": 9.81},
            {"label": "Sand", "surface": "sand"},
            {"label": "Ice", "surface": "ice"},
            {"label": "Payload", "payload_kg": 5},
            {"label": "Push", "push_n": 30},
        ],
    }
    with patch.object(agents.config, "MOCK_MODE", False), \
            patch.object(agents.llm, "chat", return_value=json.dumps(invalid_plan)) as chat:
        variations = agents.plan_stress(spec, "stress test")
    assert chat.call_count == 2
    assert all(variation["label"].startswith("Fallback: ") for variation in variations)
    assert all("slope_deg" not in variation for variation in variations)


_G1_BASE_BUILD = None


def g1_spec(*, duration_s=8.0, mission=None, obstacles=None):
    return SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0),
        robot=Robot(kind="library", robot_id="g1", policy_id="g1_walking"),
        mission=mission,
        obstacles=obstacles or [],
        duration_s=duration_s,
    ).fill_defaults()


def g1_base_build():
    global _G1_BASE_BUILD
    if _G1_BASE_BUILD is None:
        _G1_BASE_BUILD = MuJoCoSimulator().build(g1_spec(duration_s=2.0))
    return _G1_BASE_BUILD


def test_g1_stands_for_two_seconds_at_zero_command():
    import mujoco

    from simgate.library import LibraryRegistry

    spec = g1_spec(duration_s=2.0)
    sim = MuJoCoSimulator()
    build = g1_base_build()
    model, data = sim._make(build)
    mujoco.mj_forward(model, data)
    runner = sim._policy_runner(spec, model, data)
    command = runner.card.command
    runner.set_command({name: 0.0 for name in command.names})
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
    settle_steps = round(0.5 / model.opt.timestep)
    for step in range(round(2.0 / model.opt.timestep)):
        if step < settle_steps:
            runner.hold_default()
        else:
            runner.step(step * model.opt.timestep - 0.5)
        mujoco.mj_step(model, data)

    height = float(data.xpos[root][2])
    up_z = float(data.xmat[root].reshape(3, 3)[2, 2])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, up_z))))
    standing_height = LibraryRegistry().robot("g1").standing_height_m
    assert abs(height - standing_height) <= 0.1 * standing_height, height
    assert tilt < 45, tilt


def test_g1_walks_two_metres_in_ten_seconds():
    spec = g1_spec(duration_s=10.0)
    sim = MuJoCoSimulator()
    build = g1_base_build()
    list(sim.run(spec, build))
    summary = sim.summary()
    assert not summary["flipped"] and summary["forward_progress_m"] >= 2.0, summary


def test_g1_reaches_diagonal_five_metre_goal():
    mission = Mission(
        start_xy=[-3.0, 0.0],
        start_heading_deg=0.0,
        goal_xy=[-3.0 + 5.0 / math.sqrt(2), 5.0 / math.sqrt(2)],
    )
    spec = g1_spec(duration_s=10.0, mission=mission)
    sim = MuJoCoSimulator()
    build = sim.build(spec)
    start, goal = build.scene["mission"]["start"], build.scene["mission"]["goal"]
    assert abs(math.dist(start, goal) - 5.0) <= 0.02
    list(sim.run(spec, build))
    summary = sim.summary()
    assert summary["goal_reached"] and summary["outcome"] == "reached", summary


def test_g1_named_mapping_policy_validation_and_mesh_budget():
    import json
    import mujoco

    from simgate.library import LibraryRegistry, PolicyCard
    from simgate.policy_runtime import PolicyRunner

    spec = g1_spec(
        duration_s=8.0,
        mission=Mission(start_xy=[-6.0, 0.0], goal_xy=[6.0, 0.0]),
        obstacles=[Obstacle(kind="rock", size="small", count=2)],
    )
    sim = MuJoCoSimulator()
    build = sim.build(spec)
    model, data = sim._make(build)
    runner = sim._policy_runner(spec, model, data)
    card = LibraryRegistry().policy("g1_walking")
    addresses = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in card.joint_order
    ]
    assert runner.qpos_addresses.tolist() == addresses
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_zone") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_flag") >= 0
    assert build.scene["course"]
    head_mesh_id = next(
        int(geom["mesh_id"])
        for geom in build.scene["geoms"]
        if geom["type"] == "mesh" and build.scene["mesh_assets"][str(geom["mesh_id"])]["name"] == "head_link"
    )
    head_geom_id = next(
        geom["id"] for geom in build.scene["geoms"]
        if geom["type"] == "mesh" and geom["mesh_id"] == head_mesh_id
    )
    pelvis_geom_id = next(
        geom["id"] for geom in build.scene["geoms"]
        if geom["type"] == "mesh" and build.scene["mesh_assets"][str(geom["mesh_id"])]["name"] == "pelvis"
    )
    assert 0.4 < build.scene["poses"][head_geom_id][2] - build.scene["poses"][pelvis_geom_id][2] < 0.6
    assert "pos" not in build.scene["mesh_assets"][str(head_mesh_id)]
    assert "quat" not in build.scene["mesh_assets"][str(head_mesh_id)]
    checks = {check.name: check for check in sim.validate(spec, build)}
    for name in ("Library policy loads and validates", "Library robot stands for 1 s at zero command"):
        assert checks[name].passed, checks[name]
    event = {"type": "scene", "scene": build.scene, "code": build.code, "language": build.language}
    assert len(json.dumps(event, separators=(",", ":")).encode("utf-8")) < 5 * 1024 * 1024

    payload = card.model_dump()
    payload["obs_layout"][0]["size"] = 2
    bad_card = PolicyCard.model_validate(payload)

    class WrongObservationSize:
        input_size = card.observation_size
        output_size = len(card.joint_order)

        def run(self, observation):
            raise AssertionError("dimension validation should happen before inference")

        def reset(self):
            pass

    try:
        PolicyRunner(bad_card, model, data, inference_model=WrongObservationSize())
    except ValueError as exc:
        assert "expects 47 observations" in str(exc)
    else:
        raise AssertionError("a policy/card observation-size mismatch must be rejected")


def test_policy_with_out_of_range_targets_is_rejected_on_load():
    import numpy as np

    from simgate.library import LibraryRegistry
    from simgate.policy_runtime import PolicyRunner

    card = LibraryRegistry().policy("g1_walking")
    model, data = MuJoCoSimulator()._make(g1_base_build())

    class FarOutOfRange:
        input_size = card.observation_size
        output_size = len(card.joint_order)

        def run(self, observation):
            return np.full((1, len(card.joint_order)), 40.0)   # 10 rad past every default angle

        def reset(self):
            pass

    try:
        PolicyRunner(card, model, data, inference_model=FarOutOfRange())
    except ValueError as exc:
        assert "outside the joint range" in str(exc), exc
    else:
        raise AssertionError("a policy whose targets leave the joint ranges must be rejected on load")


def test_g1_planner_and_library_selection_use_registered_cards():
    import asyncio

    from simgate.orchestrator import Session

    spec = agents.plan("A G1 humanoid")
    assert (spec.robot.kind, spec.robot.robot_id, spec.robot.policy_id) == (
        "library", "g1", "g1_walking"
    )

    async def select():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.select_library("g1", "g1_walking", "mars-regolith")
        selected = agents.plan("a robot on flat terrain")
        session._apply_library_selection(selected)
        assert selected.robot.robot_id == "g1" and selected.environment.body == "mars"
        assert events[0]["type"] == "library_selected"

    asyncio.run(select())


def test_library_lists_only_each_robots_own_compatible_policies():
    from pathlib import Path

    from simgate.library import LibraryRegistry

    registry = LibraryRegistry()
    compatible = {robot.id: [policy.id for policy in registry.compatible_policies(robot.id)] for robot in registry.robots()}
    assert compatible["rover"] == ["rover_wheel_controller"], compatible
    assert compatible["walker"] == ["walker_gait"], compatible
    assert compatible["g1"] == ["g1_walking"], compatible
    page = (Path(__file__).resolve().parent.parent / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    assert "policy.joint_order.length > 0 && policy.joint_order.length === robot.joint_names.length" in page


def _run_one(name, fn) -> str:
    """Run one test for the plain-script runner and return its report line."""
    try:
        fn()
    except unittest.SkipTest as reason:
        return f"skip {name} ({reason})"
    return f"ok  {name}"


def test_script_runner_reports_skips():
    def skipped():
        raise unittest.SkipTest("needs a library")

    assert _run_one("test_x", skipped) == "skip test_x (needs a library)"
    assert _run_one("test_y", lambda: None) == "ok  test_y"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(_run_one(name, fn))
