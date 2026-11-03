import math

import numpy as np

from simgate.backends.mujoco_backend import MuJoCoSimulator, Navigator, resolve_mission
from simgate.batch import _validate_variation
from simgate.routing import compare_routes, plan_route, segment_cells
from simgate.spec import Mission, Robot, RobotEntry, SceneSpec, Terrain
from simgate.traversability import features


def _scene(kind="flat", start=(-3, 0), goal=(3, 0)):
    return SceneSpec(
        domain="robot_terrain", terrain=Terrain(kind=kind, size_m=10, roughness=0, seed=7),
        robot=Robot(), mission=Mission(start_xy=list(start), goal_xy=list(goal), goal_radius_m=0.25),
        duration_s=40, time_limit_given=True,
    ).fill_defaults()


def _grid(scene):
    grid = features(scene)
    grid["classes"] = np.where(grid["blocked"], "grey", "green")
    return grid


def _assert_safe(route):
    for a, b in zip(route["path_xy"], route["path_xy"][1:]):
        cells = segment_cells(route["navigation_grid"], a, b)
        assert cells is not None
        assert all(route["navigation_grid"]["allowed"][row][col] for row, col in cells)


def test_astar_and_smoothing_avoid_red_cells_and_wall_corners():
    scene = _scene()
    grid = _grid(scene)
    grid["classes"][4:16, 9] = "red"
    grid["classes"][4:14, 11] = "grey"
    grid["blocked"][4:14, 11] = True
    result = plan_route(scene, grid)
    assert result["found"]
    assert len(result["path_xy"]) > 2
    assert all(grid["classes"][cell] not in ("red", "grey") for cell in result["cells"])
    _assert_safe(result)


def test_route_goes_around_wet_patch_and_prefers_green_to_amber():
    scene = _scene()
    grid = _grid(scene)
    grid["classes"][7:13, 8:12] = "red"
    grid["friction"][7:13, 8:12] = 0.15
    result = plan_route(scene, grid)
    assert result["found"]
    _assert_safe(result)
    grid["classes"][7:13, 8:12] = "amber"
    result = plan_route(scene, grid)
    assert result["found"]
    assert all(grid["classes"][cell] == "green" for cell in result["cells"])
    _assert_safe(result)


def test_no_route_names_step_limit_and_shows_best_partial_route():
    scene = _scene()
    grid = _grid(scene)
    grid["classes"][:, 10] = "red"
    grid["max_step_m"][:, 10] = 0.17
    grid["limits"] = {"step": {"reliable_ok": 0.05, "first_fail": 0.08}}
    result = plan_route(scene, grid)
    assert not result["found"]
    assert "0.17 m step" in result["reason"]
    assert "climbs about 0.05 m" in result["reason"]
    assert len(result["partial_path_xy"]) > 1
    assert max(point[0] for point in result["partial_path_xy"]) < 0.25
    _assert_safe(result)


def test_bedroom_route_uses_ramp_while_direct_steps_remain_red():
    scene = _scene("indoor", start=(-4.5, -2.5), goal=(-4.5, 3.2))
    grid = _grid(scene)
    grid["classes"][(grid["max_step_m"] > 0.05) & ~grid["blocked"]] = "red"
    grid["limits"] = {"step": {"reliable_ok": 0.05, "first_fail": 0.08}}
    result = plan_route(scene, grid)
    assert result["found"], result["reason"]
    assert max(point[0] for point in result["path_xy"]) > 2
    row = int(np.argmin(abs(grid["y_m"] - 1.25)))
    col = int(np.argmin(abs(grid["x_m"] + 4.5)))
    assert grid["classes"][row, col] == "red"
    _assert_safe(result)


def test_rover_physically_reaches_bedroom_via_ramp():
    scene = _scene("indoor", start=(-4.5, -2.5), goal=(-4.5, 3.2))
    grid = _grid(scene)
    grid["classes"][(grid["max_step_m"] >= 0.06) & ~grid["blocked"]] = "red"
    result = compare_routes(scene, grid)
    assert result["route"]["found"], result["text"]
    safe = result["runs"][1]["summary"]
    assert safe["goal_reached"], result["text"]
    assert not result["runs"][0]["summary"]["goal_reached"]


def test_routed_turns_count_as_progress_but_motionless_steering_still_sticks():
    from simgate.backends.mujoco_backend import Outcome

    turning, stopped = Outcome(), Outcome()
    for t in (0, 1, 2, 3, 4):
        turning.update(t, np.zeros(2), None, False, np.array([math.cos(t * .2), math.sin(t * .2)]))
        stopped.update(t, np.zeros(2), None, False, np.array([1., 0.]))
    assert turning.decided is None
    assert stopped.decided["reason"] == "stuck"


def test_routed_rover_brakes_before_turns_without_changing_legacy_commands():
    mission = {"start": [0, 0], "goal": [2, 0], "waypoints": [[1, 0]]}
    legacy = Navigator(mission, "rover")
    routed = Navigator({**mission, "navigation_grid": {"allowed": []}}, "rover")
    assert legacy.command(np.array([0., 0.]), np.array([0., 1.]), 0)[1] == 1
    assert routed.command(np.array([0., 0.]), np.array([0., 1.]), 0)[1] == 0
    assert routed.command(np.array([.5, 0.]), np.array([1., 0.]), 0)[1] < 1


def test_waypoints_survive_batch_defaults_and_mission_resolution():
    scene = _scene("indoor")
    _, varied, _ = _validate_variation(scene, {
        "waypoints": [[-2, 1], [2, 1]], "robot_id": "r1",
    })
    assert varied.robots[0].waypoints == [(-2, 1), (2, 1)]
    assert varied.mission.waypoints == varied.robots[0].waypoints
    mission = resolve_mission(varied, np.zeros((128, 128)))
    assert mission["waypoints"] == [(-2, 1), (2, 1)]
    assert mission["waypoint_radius_m"] == 0.4
    assert resolve_mission(_scene(), np.zeros((128, 128)))["waypoint_radius_m"] == 1


def test_waypoint_navigation_reaches_goal_behind_a_wall():
    scene = _scene("indoor", start=(-3, -3.6), goal=(1, -3.6))
    grid = _grid(scene)
    route = plan_route(scene, grid)
    assert route["found"], route["reason"]
    _assert_safe(route)
    navigator = Navigator({
        "goal": route["goal_xy"], "radius": 0.25,
        "waypoints": route["path_xy"][1:-1],
        "waypoint_radius_m": 0.4, "navigation_grid": route["navigation_grid"],
    }, "rover")
    position = np.array(route["start_xy"], dtype=float)
    yaw = 0.0
    for tick in range(12000):
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        turn, go = navigator.command(position, heading, tick * 0.02)
        yaw += turn * 0.04
        following = position + 0.008 * go * np.array([math.cos(yaw), math.sin(yaw)])
        covered = segment_cells(route["navigation_grid"], position, following)
        assert covered is not None
        assert all(route["navigation_grid"]["allowed"][row][col] for row, col in covered)
        position = following
        if navigator.reached_at is not None:
            break
    assert navigator.reached_at is not None


def test_paired_runs_keep_seed_speed_and_report_measured_paths():
    scene = _scene()
    grid = _grid(scene)

    def fake_batch(working, variations, **kwargs):
        assert len(variations) == 2
        assert kwargs == {"baseline": False, "record": False}
        assert variations[0]["seed"] == variations[1]["seed"] == 7
        assert working.robot == scene.robot
        assert variations[0]["waypoints"] == []
        assert variations[1]["navigation_grid"]
        return [
            {"id": "run-1", "summary": {
                "outcome": "stuck", "stopped_by": {"kind": "threshold"}, "path_length_m": 10,
            }},
            {"id": "run-2", "summary": {
                "goal_reached": True, "time_to_goal_s": 48, "path_length_m": 13.1,
            }},
        ]

    result = compare_routes(scene, grid, run_batch_fn=fake_batch)
    assert result["text"] == (
        "Straight: stuck at the threshold. Safe route: reached the goal in 48 s, 31% longer path."
    )


def test_no_route_does_not_launch_a_comparison():
    scene = _scene()
    grid = _grid(scene)
    grid["classes"][:, 10] = "red"

    def unexpected_batch(*args, **kwargs):
        raise AssertionError("Must not drive an unsafe partial route as if it reached the goal.")

    result = compare_routes(scene, grid, run_batch_fn=unexpected_batch)
    assert result["runs"] == []
    assert "No safe route" in result["text"]


def test_mujoco_waypoints_reach_goal_behind_kitchen_wall():
    scene = _scene("indoor", start=(-3, -3.6), goal=(1, -3.6))
    result = compare_routes(scene, _grid(scene))
    assert len(result["runs"]) == 2
    assert all(not run.get("error") for run in result["runs"]), result["runs"]
    straight, safe = result["runs"]
    assert straight["summary"]["outcome"] == "stuck", result["text"]
    assert safe["summary"]["goal_reached"], (result["route"]["path_xy"], safe["summary"])


def test_off_world_goal_returns_a_partial_route_and_explanation():
    scene = _scene()
    result = plan_route(scene, _grid(scene), goal_xy=[20, 0])
    assert not result["found"]
    assert "off-world" in result["reason"]
    assert len(result["partial_path_xy"]) > 1
    _assert_safe(result)


def test_multi_robot_template_preserves_selected_waypoints():
    scene = _scene()
    scene.robots.append(RobotEntry(
        id="r2", label="Other rover", robot=Robot(),
        start_xy=[-3, 2], goal_xy=[3, 2],
    ))
    _, varied, _ = _validate_variation(scene, {
        "robot_id": "r2", "waypoints": [[0, 3]],
    })
    simulator = MuJoCoSimulator()
    simulator.template(varied)
    assert simulator.missions["r1"]["waypoints"] == []
    assert simulator.missions["r2"]["waypoints"] == [(0, 3)]


def test_waypoint_switch_does_not_cut_a_red_corner():
    scene = _scene()
    grid = _grid(scene)
    grid["classes"][9:11, 9:11] = "red"
    route = plan_route(scene, grid)
    navigator = Navigator({
        "goal": [3, 0], "radius": 0.25, "waypoint_radius_m": 1,
        "waypoints": [[-0.75, 1.25], [0.75, 1.25]],
        "navigation_grid": route["navigation_grid"],
    }, "rover")
    navigator.command(np.array([-0.75, 0.3]), np.array([0, 1]), 1)
    assert len(navigator.waypoints) == 2
