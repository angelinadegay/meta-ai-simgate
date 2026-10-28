"""Smoke tests: run with `python -m pytest -q` or `python tests/test_pipeline.py`.

They run offline in MOCK mode unless you've configured a model.
"""
import math
import sys
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


def test_mock_director_turns_race_into_ordered_tool_calls():
    import asyncio

    from simgate import config
    from simgate.director import Director

    async def scenario(text):
        events = []

        async def emit(event):
            events.append(event)

        config.MOCK_MODE = True
        world, message = await Director(emit).execute(text)
        done_tools = [
            event["tool"] for event in events
            if event.get("type") == "step" and event.get("status") == "done"
            and event.get("agent") == "Director" and event.get("tool")
        ]
        return world, message, done_tools, events

    world, message, tools, events = asyncio.run(scenario("race three rovers to the far side"))
    assert tools == ["set_world", "spawn_robot", "spawn_robot", "spawn_robot", "set_goal", "build_and_check", "ask_user"]
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


def test_director_recovers_from_invalid_replies_and_failed_checks():
    import asyncio

    from simgate import config
    from simgate.director import Director
    from simgate.tools import ToolResult

    original_chat = agents.llm.chat
    original_invoke = __import__("simgate.director", fromlist=["invoke_tool"]).invoke_tool

    async def execute_script(replies, invoke=None):
        iterator = iter(replies)
        agents.llm.chat = lambda *_args, **_kwargs: next(iterator)
        if invoke is not None:
            import simgate.director as director_module
            director_module.invoke_tool = invoke
        config.MOCK_MODE = False
        return await Director().execute("prepare a robot")

    try:
        recovered, message = asyncio.run(execute_script([
            "not JSON",
            '{"thought":"Recovered format","final":"Recovered."}',
        ]))
        assert message == "Recovered."

        recovered, message = asyncio.run(execute_script([
            '{"thought":"Try unknown tool","tool":"erase_world","args":{}}',
            '{"thought":"Recovered tool choice","final":"Stopped safely."}',
        ]))
        assert message == "Stopped safely."

        calls = []

        def fake_invoke(_world, name, args):
            calls.append(name)
            if name == "build_and_check" and calls.count(name) == 1:
                return ToolResult("A gravity check failed.", {"passed": False})
            if name == "build_and_check":
                return ToolResult("All checks pass.", {"passed": True})
            if name == "ask_user":
                return ToolResult("Run it?", {"question": "Run it?"})
            return ToolResult("Updated.", {})

        recovered, message = asyncio.run(execute_script([
            '{"thought":"Check the draft","tool":"build_and_check","args":{}}',
            '{"thought":"Repair the world","tool":"set_world","args":{"body":"mars"}}',
            '{"thought":"Check the repair","tool":"build_and_check","args":{}}',
            '{"thought":"Ask before running","tool":"ask_user","args":{"question":"Run it?"}}',
        ], fake_invoke))
        assert message == "Run it?"
        assert calls == ["build_and_check", "set_world", "build_and_check", "ask_user"]
    finally:
        agents.llm.chat = original_chat
        import simgate.director as director_module
        director_module.invoke_tool = original_invoke
        config.MOCK_MODE = True


def test_build_custom_retries_and_emits_attempt_steps():
    import asyncio

    from simgate import config
    from simgate.director import Director

    spec = agents.plan("a rover on flat terrain")
    original_chat = agents.llm.chat
    original_retries = config.BUILDER_MAX_RETRIES
    replies = iter([
        '{"thought":"Add a custom marker","tool":"build_custom","args":{"request":"Add a marker above the scene."}}',
        '<geom name="bad_marker" type="cube" size="0.1 0.1 0.1" pos="0 0 20"/>',
        '<geom name="marker" type="sphere" size="0.05" pos="0 0 20"/>',
        '{"thought":"The marker is ready","final":"Custom scene detail added."}',
    ])
    events = []

    async def emit(event):
        events.append(event)

    agents.llm.chat = lambda *_args, **_kwargs: next(replies)
    config.MOCK_MODE = False
    config.BUILDER_MAX_RETRIES = 2
    try:
        world, message = asyncio.run(Director(emit).execute("add a marker", current=spec))
        assert message == "Custom scene detail added."
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
        print("skip test_real_terrain_pipeline (pip install rasterio to run it)")
        return
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
