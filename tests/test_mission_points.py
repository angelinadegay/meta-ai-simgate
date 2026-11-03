import asyncio
import copy

import mujoco
import numpy as np
import pytest

from simgate.backends import mujoco_backend
from simgate.backends.mujoco_backend import MuJoCoSimulator
from simgate.orchestrator import Session
from simgate.spec import Mission, Obstacle, Robot, RobotEntry, SceneSpec, Terrain, TerrainStructure


def existing_world(count=1):
    return SceneSpec(
        domain="robot_terrain", terrain=Terrain(kind="hills", roughness=0.15, size_m=40),
        robots=[RobotEntry(id=f"r{index}", label=f"Rover {index}", robot=Robot(),
                           start_xy=[-12, index * 3], goal_xy=[12, 0]) for index in range(count)],
        obstacles=[Obstacle(kind="rock", placement="path")],
        structures=[TerrainStructure(kind="lander", placement="near_goal")],
    ).fill_defaults()


@pytest.mark.parametrize("which,point", [("start", (-9, 3)), ("goal", (10, 5))])
@pytest.mark.parametrize("count", [1, 2])
def test_map_points_preserve_world_and_emit_only_incremental_update(monkeypatch, which, point, count):
    async def exercise():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        spec = existing_world(count)
        simulator = MuJoCoSimulator()
        build = simulator.build(spec)
        session.spec, session.sim, session.build = spec, simulator, build
        session.checks = simulator.validate(spec, build)
        before_scene = copy.deepcopy(build.scene)
        before_terrain = spec.terrain.model_dump()
        before_obstacles = [item.model_dump() for item in spec.obstacles]
        before_model = simulator._make(build)[0]

        def forbidden(*args, **kwargs):
            raise AssertionError("Map clicks must not regenerate terrain or re-place objects.")

        monkeypatch.setattr(simulator, "template", forbidden)
        monkeypatch.setattr(mujoco_backend, "make_heights", forbidden)
        await session.set_point(which, *point)
        assert not [event for event in events if event["type"] == "error"], events
        assert not [event for event in events if event["type"] == "scene"]
        patch = next(event for event in events if event["type"] == "mission_updated")
        assert session.build.code == build.code
        assert session.spec.terrain.model_dump() == before_terrain
        assert [item.model_dump() for item in session.spec.obstacles] == before_obstacles
        for field in ("geoms", "structures", "course", "terrain_size_m"):
            assert session.build.scene[field] == before_scene[field]
        model, data = simulator._make(session.build)
        assert np.array_equal(model.hfield_data, before_model.hfield_data)
        if which == "goal":
            assert not patch["moved_poses"]
            assert session.build.scene["poses"] == before_scene["poses"]
            assert all(entry.goal_xy == list(point) for entry in session.spec.robots)
        else:
            assert patch["moved_poses"]
            assert session.spec.robots[0].start_xy == list(point)
            root_name = "r0_rover" if count == 2 else "rover"
            root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
            assert data.xpos[root][:2] == pytest.approx(point)
            if count == 2:
                assert session.build.scene["robots"]["r1"]["mission"] == before_scene["robots"]["r1"]["mission"]
        assert any(event["type"] == "checks" for event in events)

    asyncio.run(exercise())


def test_map_point_outside_physics_area_does_not_change_world():
    simulator = MuJoCoSimulator()
    spec = existing_world()
    build = simulator.build(spec)
    before = copy.deepcopy(build.scene)
    with pytest.raises(ValueError, match="outside the physics terrain"):
        simulator.update_mission_point(spec, build, "start", 1000, 0)
    assert build.scene == before


def test_start_override_is_used_when_simulation_runs():
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0, size_m=40),
                     robot=Robot(), mission=Mission(start_xy=[-12, 0], goal_xy=[12, 0]), duration_s=2).fill_defaults()
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    spec, build, _ = simulator.update_mission_point(spec, build, "start", -8, 4)
    frames = list(simulator.run(spec, build))
    assert frames[0][0]["poses"][build.scene["follow_geom"]][:2] == pytest.approx([-8, 4], abs=0.1)


def test_start_relocates_selected_race_robot_only():
    simulator = MuJoCoSimulator()
    spec = existing_world(2)
    build = simulator.build(spec)
    original_first = copy.deepcopy(build.scene["robots"]["r0"]["mission"])
    updated, changed, patch = simulator.update_mission_point(spec, build, "start", -7, 5, robot_id="r1")
    assert changed.scene["robots"]["r0"]["mission"] == original_first
    assert updated.robots[1].start_xy == [-7, 5]
    model, data = simulator._make(changed)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r1_rover")
    assert data.xpos[root][:2] == pytest.approx([-7, 5])
    assert all(int(model.body_rootid[int(model.geom_bodyid[int(geom)])]) == root for geom in patch["moved_poses"])


def test_clicked_goal_refreshes_automatic_time_without_rebuilding(monkeypatch):
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0),
                     robot=Robot(kind="walker", legs=4)).fill_defaults()
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    assert spec.duration_s == 8

    def forbidden(*args, **kwargs):
        raise AssertionError("Timing refresh must not regenerate terrain.")

    monkeypatch.setattr(mujoco_backend, "make_heights", forbidden)
    updated, changed, patch = simulator.update_mission_point(spec, build, "goal", 0, 0)
    assert updated.duration_s >= 20
    assert patch["duration_s"] == updated.duration_s and patch["timing_note"]
    assert changed.code == build.code and changed.scene["geoms"] == build.scene["geoms"]


def test_clicked_goal_preserves_explicit_time_limit():
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=Robot(),
                     duration_s=8, time_limit_given=True).fill_defaults()
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    updated, _, patch = simulator.update_mission_point(spec, build, "goal", 0, 0)
    assert updated.duration_s == 8 and patch["timing_note"] is None