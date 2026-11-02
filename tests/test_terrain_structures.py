import math

import numpy as np
import pytest

from simgate.spec import (Mission, Obstacle, Robot, RobotEntry, SceneSpec, Terrain,
                          TerrainStructure)


def test_scene_spec_accepts_terrain_structure_requests_and_empty_default():
    empty = SceneSpec(domain="robot_terrain").fill_defaults()
    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0),
        robot=Robot(kind="rover"),
        structures=[
            TerrainStructure(kind="habitat", size="medium", placement="near_start"),
            TerrainStructure(kind="bridge", size="small", material="steel", placement="over_gap"),
            TerrainStructure(kind="lander", size="large", placement="lowest"),
        ],
    ).fill_defaults()

    assert empty.structures == []
    assert [item.placement for item in spec.structures] == ["near_start", "over_gap", "lowest"]
    assert spec.structures[0].material == "unspecified"
    assert spec.structures[1].material == "steel"


def flat_scene(*, structures, size=40.0):
    return SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0, size_m=size),
        robots=[RobotEntry(id="r1", label="Rover", robot=Robot(kind="rover"),
                           start_xy=[-12, 0], goal_xy=[12, 0])],
        structures=structures,
    ).fill_defaults()


def test_habitat_placement_is_clear_of_robot_starts_and_straight_path():
    from simgate.backends.mujoco_backend import footprint_radius, make_heights
    from simgate.structures import _point_rect_distance, place_structures

    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0, size_m=40),
        robots=[
            RobotEntry(id="r1", label="Rover A", robot=Robot(kind="rover"),
                       start_xy=[-12, 0], goal_xy=[12, 0]),
            RobotEntry(id="r2", label="Rover B", robot=Robot(kind="rover"),
                       start_xy=[-12, 8], goal_xy=[12, 0]),
        ],
        structures=[TerrainStructure(kind="habitat", placement="near_start")],
    ).fill_defaults()
    heights = make_heights(spec)[0]
    placed, notes = place_structures(spec, heights)

    assert len(placed) == 1, notes
    record = placed[0]
    center = tuple(record["footprint"]["center_xy"])
    yaw = math.radians(record["orientation"]["yaw_deg"])
    for entry in spec.robots:
        radius = footprint_radius(entry.robot)
        assert _point_rect_distance(tuple(entry.start_xy), center, yaw,
                                    record["footprint"]["length_m"] / 2,
                                    record["footprint"]["width_m"] / 2) >= radius + 0.25
    assert not any("path" in note.lower() for note in notes)


def test_bridge_spans_measured_path_gap_with_bearings():
    from simgate.backends.mujoco_backend import make_heights
    from simgate.backends.obstacles import ground, measure_gap, plan_course
    from simgate.structures import place_structures

    spec = flat_scene(structures=[TerrainStructure(kind="bridge", placement="over_gap")])
    spec.obstacles = [Obstacle(kind="gap", size="medium", placement="path")]
    heights = make_heights(spec)[0]
    mission = spec.robots[0].mission()
    start, goal = mission.start_xy, mission.goal_xy
    slots, _ = plan_course(spec, heights, start, goal, spec.terrain.seed)
    slot = next(item for item in slots if item["ob"].kind == "gap")
    scratch = heights.copy()
    from simgate.backends.obstacles import cut_gaps
    cut_gaps([slot], scratch, spec.terrain.size_m)
    opening = measure_gap(scratch, spec.terrain.size_m, *slot["centre"], slot["d"])

    placed, notes = place_structures(spec, heights)

    assert len(placed) == 1, notes
    bridge = placed[0]
    assert bridge["gap"]["center"] == [round(value, 2) for value in slot["centre"]]
    assert bridge["dimensions"]["span_m"] == round(opening + 1.0, 2)
    assert bridge["gap"]["bearings_each_m"] == 0.5


def test_habitat_note_reports_measured_ground_slope():
    from simgate.structures import place_structures

    spec = flat_scene(structures=[TerrainStructure(kind="habitat", placement="near_goal")])
    half = spec.terrain.size_m / 2
    coords = np.linspace(-half, half, 160)
    x_grid, _ = np.meshgrid(coords, coords)
    heights = np.tan(math.radians(2.1)) * (x_grid + half)

    placed, notes = place_structures(spec, heights)

    assert len(placed) == 1, notes
    assert placed[0]["ground_slope_deg"] == 2.1
    assert any("Habitat on 2.1° ground" in note and "from the goal" in note for note in notes)


def test_no_structures_returns_empty_without_mutating_spec_or_heights():
    from simgate.backends.mujoco_backend import make_heights
    from simgate.structures import place_structures

    spec = flat_scene(structures=[])
    before_spec = spec.model_dump(mode="json")
    heights = make_heights(spec)[0]
    before_heights = heights.copy()

    assert place_structures(spec, heights) == ([], [])
    assert spec.model_dump(mode="json") == before_spec
    assert np.array_equal(heights, before_heights)


def test_structures_are_named_static_collision_geoms():
    from simgate.backends.mujoco_backend import MuJoCoSimulator

    spec = flat_scene(structures=[TerrainStructure(kind="lander", placement="near_goal")])
    without_structures = spec.model_copy(deep=True)
    without_structures.structures = []
    with_metadata = MuJoCoSimulator().build(spec)
    without_metadata = MuJoCoSimulator().build(without_structures)

    assert with_metadata.code != without_metadata.code
    assert with_metadata.scene["geoms"] != without_metadata.scene["geoms"]
    assert len(with_metadata.scene["structures"]) == 1
    assert with_metadata.scene["structure_notes"]
    names = {geom["name"] for geom in with_metadata.scene["geoms"]}
    assert {"struct0_lander_body", *(f"struct0_lander_leg{i}" for i in range(4)),
            *(f"struct0_lander_pad{i}" for i in range(4))} <= names
    assert "contype=\"1\" conaffinity=\"1\"" in with_metadata.code


def _run_scene(spec):
    from simgate.backends.mujoco_backend import MuJoCoSimulator

    sim = MuJoCoSimulator()
    build = sim.build(spec)
    for _ in sim.run(spec, build):
        pass
    return sim, build


def test_bridge_lets_rover_cross_medium_trench_and_tracker_counts_gap_cleared():
    obstacle = Obstacle(kind="gap", size="medium", placement="path")
    unbridged = flat_scene(structures=[])
    unbridged.obstacles = [obstacle]
    unbridged_sim, _ = _run_scene(unbridged)
    unbridged_summary = unbridged_sim.summary()
    assert not unbridged_summary["goal_reached"]
    assert unbridged_summary["stopped_by"]["kind"] == "gap"

    bridged = flat_scene(structures=[TerrainStructure(kind="bridge", placement="over_gap")])
    bridged.obstacles = [obstacle]
    sim, build = _run_scene(bridged)
    summary = sim.summary()

    assert summary["goal_reached"], summary
    assert summary["obstacles_cleared"] == 1, summary
    assert any(geom["name"] == "struct0_bridge_deck" for geom in build.scene["geoms"])


def test_rover_reaches_goal_by_driving_up_habitat_ramp():
    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0, size_m=40),
        robot=Robot(kind="rover"),
        mission=Mission(start_xy=[-12.0, 0.0], goal_xy=[0.0, 0.0], goal_radius_m=1.0),
        structures=[TerrainStructure(kind="habitat", placement="near_goal")],
    ).fill_defaults()

    sim, build = _run_scene(spec)

    assert sim.summary()["goal_reached"], sim.summary()
    names = {geom["name"] for geom in build.scene["geoms"]}
    assert "struct0_habitat_body" in names and "struct0_habitat_ramp" in names


def test_structure_colliders_do_not_overlap_robot_at_initial_spawn():
    import mujoco

    from simgate.backends.mujoco_backend import MuJoCoSimulator

    spec = flat_scene(structures=[TerrainStructure(kind="lander", placement="near_start")])
    build = MuJoCoSimulator().build(spec)
    model = mujoco.MjModel.from_xml_string(build.code)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    structure_geoms = {
        geom for geom in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith("struct0_")
    }
    robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
    robot_geoms = {
        geom for geom in range(model.ngeom)
        if int(model.body_rootid[int(model.geom_bodyid[geom])]) == robot_root
    }
    overlaps = [
        (int(data.contact[index].geom1), int(data.contact[index].geom2))
        for index in range(data.ncon)
        if ({int(data.contact[index].geom1), int(data.contact[index].geom2)} & structure_geoms)
        and ({int(data.contact[index].geom1), int(data.contact[index].geom2)} & robot_geoms)
    ]

    assert not overlaps, overlaps


@pytest.mark.parametrize("mock", [True, False])
def test_mars_base_planning_builds_structures_and_resolves_airlock_goal(monkeypatch, mock):
    import asyncio
    import json

    from simgate import config, llm
    from simgate.backends.mujoco_backend import MuJoCoSimulator, make_heights
    from simgate.backends.obstacles import plan_course
    from simgate.director import Director
    from simgate.structures import structure_goal_xy
    from simgate.workflows import _prepare_robot_world

    monkeypatch.setattr(config, "MOCK_MODE", mock)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: json.dumps({
        "workflow": "build", "summary": "Mars base rover mission",
        "world": {"body": "mars", "terrain_kind": "flat", "roughness": 0},
        "robots": [{"kind": "rover"}],
        "structures": [{"kind": "habitat", "placement": "near_goal"}],
        "goal": {"place": "east"},
    }))
    director = Director()
    plan = director.plan("Build a Mars base on flat terrain with a habitat, lander, and bridge. "
                         "Drive a rover from the lander to the habitat airlock.")

    assert not director.used_rule_planner
    assert {item.kind: item.placement for item in plan.structures} == {
        "habitat": "near_goal", "lander": "near_start", "bridge": "over_gap"}
    assert plan.goal.structure == "habitat_airlock"
    assert plan.robots[0].start_place == "start_area"
    assert sum(item.kind == "gap" and item.placement == "path" for item in plan.obstacles) == 1

    world = asyncio.run(_prepare_robot_world(plan, None, director))
    spec = world.spec.fill_defaults()
    assert spec.robots[0].goal_structure == "habitat_airlock"
    sim = MuJoCoSimulator()
    build = sim.build(spec)
    assert {item["kind"] for item in build.scene["structures"]} == {"habitat", "lander", "bridge"}
    assert list(sim.mission["goal"]) == structure_goal_xy(build.scene["structures"], "habitat_airlock")
    assert spec.mission.goal_xy == list(sim.mission["goal"])
    lander = next(item for item in build.scene["structures"] if item["kind"] == "lander")
    assert math.dist(sim.mission["start"], lander["footprint"]["center_xy"]) <= 6.1

    heights = make_heights(spec)[0]
    slots, _ = plan_course(spec, heights, sim.mission["start"], sim.mission["goal"], spec.terrain.seed)
    gap = next(item for item in slots if item["ob"].kind == "gap")
    bridge = next(item for item in build.scene["structures"] if item["kind"] == "bridge")
    assert bridge["gap"]["center"] == [round(value, 2) for value in gap["centre"]]


def test_bridge_guard_discloses_real_terrain_trench_and_keeps_existing_gap(monkeypatch):
    from simgate import director, plan as plan_module

    presets = {"test-mars": {"label": "Test Mars", "body": "mars"}}
    monkeypatch.setattr(director, "list_presets", lambda: presets)
    monkeypatch.setattr(plan_module, "list_presets", lambda: presets)
    plan = plan_module.Plan(
        workflow="build", summary="Bridge on real terrain", robots=[{"kind": "rover"}],
        world={"terrain_kind": "real", "preset": "test-mars"},
    )
    notes = director._apply_text_guards(plan, "Build a rover and bridge on Test Mars real terrain.")
    assert "Added trench, not in the terrain data." in notes
    assert len(plan.obstacles) == 1 and plan.obstacles[0].kind == "gap"
    plan.obstacles[0].size = "small"
    director._apply_text_guards(plan, "Build a rover and bridge on Test Mars real terrain.")
    assert len(plan.obstacles) == 1 and plan.obstacles[0].size == "small"
    assert len(plan.structures) == 1


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("target,kind", [("habitat_airlock", "habitat"), ("lander", "lander")])
def test_structure_goal_resolves_for_single_robot_and_race(count, target, kind):
    from simgate.backends.mujoco_backend import MuJoCoSimulator
    from simgate.structures import structure_goal_xy

    spec = SceneSpec(
        domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0, size_m=40),
        robots=[RobotEntry(id=f"r{index}", label=f"Rover {index}", robot=Robot(kind="rover"),
                           start_xy=[-12, index * 3], goal_structure=target) for index in range(count)],
        structures=[TerrainStructure(kind=kind, placement="near_goal")],
    ).fill_defaults()
    sim = MuJoCoSimulator()
    build = sim.build(spec)
    point = structure_goal_xy(build.scene["structures"], target)
    assert all(entry.goal_xy == point for entry in spec.robots)
    assert list(sim.mission["goal"]) == point
    assert all(list(mission["goal"]) == point for mission in sim.missions.values())
    rebuilt = MuJoCoSimulator().build(spec)
    assert rebuilt.scene["structures"] == build.scene["structures"]
