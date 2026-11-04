import json

import mujoco
import numpy as np
import pytest

from simgate.backends.base import blocking_failures
from simgate.backends.mujoco_backend import MuJoCoSimulator, make_heights, resolve_mission
from simgate.indoor import get_layout
from simgate.spec import Mission, Robot, SceneSpec, Terrain


def apartment(mission=None):
    return SceneSpec(domain="robot_terrain", terrain=Terrain(kind="indoor", preset="apartment"),
                     robot=Robot(kind="rover"), mission=mission).fill_defaults()


def test_apartment_layout_builds_at_true_sizes_and_earth_gravity():
    spec = apartment()
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    model = mujoco.MjModel.from_xml_string(build.code)
    assert build.scene["indoor"]["size_m"] == [14, 10]
    assert model.opt.gravity.tolist() == [0, 0, -9.81]
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "indoor_wall_west")
    assert model.geom_size[wall] == pytest.approx([0.06, 5, 1.3])
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert model.geom_contype[ground] == model.geom_conaffinity[ground] == 0
    assert not blocking_failures(simulator.validate(spec, build))
    json.dumps(get_layout())


def test_rover_drives_kitchen_to_living_room():
    spec = apartment(Mission(start_place="kitchen", goal_place="living_room", goal_radius_m=0.5))
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    for _ in simulator.run(spec, build, record=False):
        pass
    assert simulator.summary()["goal_reached"], simulator.summary()


def test_apartment_wall_blocks_rover():
    spec = apartment(Mission(start_xy=[-4.5, -3.8], goal_xy=[2.5, -3.8], goal_radius_m=0.5))
    spec.duration_s, spec.time_limit_given = 12, True
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    for _ in simulator.run(spec, build, record=False):
        pass
    assert not simulator.summary()["goal_reached"]
    assert simulator.summary()["forward_progress_m"] < 4


def test_indoor_floor_friction_and_real_thicknesses():
    model = mujoco.MjModel.from_xml_string(MuJoCoSimulator().build(apartment()).code)
    expected = {"indoor_floor_kitchen_tile": (0.6, 0.002), "indoor_floor_living_wood": (0.5, 0.002),
                "indoor_floor_rug": (0.8, 0.008), "indoor_floor_wet": (0.15, 0.002)}
    for name, (friction, thickness) in expected.items():
        tag = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_friction[tag][0] == pytest.approx(friction)
        assert 2 * model.geom_size[tag][2] == pytest.approx(thickness)
    layout = get_layout()
    assert sorted(door["width_m"] for door in layout["doorways"]) == [0.65, 0.84, 0.85, 0.88, 0.9, 1.2]
    assert all(0.015 <= door["threshold_m"] <= 0.02 for door in layout["doorways"])
    for step in (1, 2, 3):
        tag = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"indoor_bedroom_step_{step}")
        assert model.geom_pos[tag][2] + model.geom_size[tag][2] == pytest.approx(step * 0.17)


def test_bedroom_ramp_is_wide_six_degrees_and_meets_platform():
    layout = get_layout()
    ramp = next(geom for geom in layout["geoms"] if geom["name"] == "indoor_ramp")
    assert ramp["pitch_deg"] == 6
    assert ramp["size_m"][1] >= 1.0
    model = mujoco.MjModel.from_xml_string(MuJoCoSimulator().build(apartment()).code)
    ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "indoor_ramp")
    rotation = model.geom_quat[ramp_id]
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, rotation)
    matrix = matrix.reshape(3, 3)
    half_length, _, half_thickness = model.geom_size[ramp_id]
    low, high = [model.geom_pos[ramp_id] + matrix @ [x, 0, half_thickness]
                 for x in (half_length, -half_length)]
    assert low[2] == pytest.approx(0, abs=1e-6)
    assert high[2] == pytest.approx(0.512, abs=1e-6)
    assert high[0] < -1.06
    for wall in layout["geoms"]:
        if wall["name"].startswith("indoor_wall_bedroom_east"):
            assert abs(wall["position"][1] - 3.2) >= wall["size_m"][1] / 2 + 0.7


@pytest.mark.parametrize("place", ["kitchen", "living_room", "hallway", "bedroom"])
def test_indoor_named_places_resolve(place):
    spec = apartment(Mission(start_place=place, goal_place="living_room"))
    resolved = resolve_mission(spec, make_heights(spec)[0])
    assert resolved["start"] == tuple(get_layout()["places"][place])


@pytest.mark.parametrize("mock", [True, False])
@pytest.mark.parametrize("prompt,goal", [("A rover in an apartment from the kitchen to the living room", "living_room"),
                                       ("A rover inside a house from the kitchen to the bedroom", "bedroom")])
def test_indoor_rules_and_model_guards_preserve_room_missions(monkeypatch, mock, prompt, goal):
    import asyncio
    from simgate import config, llm
    from simgate.director import Director
    from simgate.workflows import _prepare_robot_world

    monkeypatch.setattr(config, "MOCK_MODE", mock)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: json.dumps({"workflow": "build", "summary": "Rover indoors",
        "world": {"terrain_kind": "flat"}, "robots": [{"kind": "rover"}]}))
    director = Director()
    plan = director.plan(prompt)
    assert plan.world.terrain_kind == "indoor" and plan.world.preset == "apartment"
    assert plan.robots[0].start_place == "kitchen" and plan.goal.place == goal
    assert not director.used_rule_planner
    world = asyncio.run(_prepare_robot_world(plan, None, director))
    build = MuJoCoSimulator().build(world.spec)
    assert build.scene["indoor"]["id"] == "apartment"
    assert build.scene["mission"]["goal"] == tuple(get_layout()["places"][goal])


@pytest.mark.parametrize("place,zone,expected", [("kitchen", "indoor_floor_kitchen_tile", 0.6),
                                                ("living_room", "indoor_floor_rug", 0.8)])
def test_floor_zone_friction_is_used_by_actual_wheel_contacts(place, zone, expected):
    spec = apartment(Mission(start_place=place))
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    model, data = simulator._make(build)
    for _ in range(500):
        mujoco.mj_step(model, data)
    tag = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, zone)
    contacts = [data.contact[index] for index in range(data.ncon)
                if tag in (data.contact[index].geom1, data.contact[index].geom2)]
    assert contacts
    assert all(contact.friction[0] == pytest.approx(expected) for contact in contacts)


def test_apartment_library_selection_preserves_layout_and_default_mission():
    import asyncio
    from simgate.library import LibraryRegistry
    from simgate.orchestrator import Session

    async def exercise():
        async def emit(event):
            pass

        registry = LibraryRegistry()
        policy = registry.compatible_controllers("rover")[0]
        session = Session(emit)
        await session.select_library("rover", policy.id, "apartment", start_scene=True)
        assert session.spec.terrain.kind == "indoor"
        assert session.build.scene["indoor"]["id"] == "apartment"
        assert session.build.scene["mission"]["start"] == (-4.5, -2.5)
        assert session.build.scene["mission"]["goal"] == (2.4, -2.5)

    asyncio.run(exercise())


def test_indoor_map_points_respect_rectangular_floor_bounds():
    spec = apartment()
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    with pytest.raises(ValueError, match="outside the physics terrain"):
        simulator.update_mission_point(spec, build, "start", 0, 6)
    updated, changed, _ = simulator.update_mission_point(spec, build, "goal", 2, -3)
    assert updated.mission.goal_xy == [2, -3] and changed.code == build.code


def test_indoor_dimensions_are_not_expanded_by_outdoor_course_sizing():
    from simgate.spec import Obstacle

    spec = apartment()
    spec.obstacles = [Obstacle(kind="stairs", count=10)]
    spec.fill_defaults()
    assert spec.terrain.size_m == 14


def test_apartment_building_earthquake_request_is_not_an_indoor_robot_world(monkeypatch):
    from simgate import agents, config

    monkeypatch.setattr(config, "MOCK_MODE", True)
    spec = agents.plan("a 5-story apartment block in an earthquake")
    assert spec.domain == "structure" and spec.structure.stories == 5