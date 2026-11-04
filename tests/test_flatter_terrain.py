import math
import re

from simgate import agents
from simgate.backends.mujoco_backend import (
    MuJoCoSimulator,
    apply_gentler_route,
    make_heights,
    move_start_to_flatter,
    resolve_mission,
)
from simgate.spec import Mission, Robot, SceneSpec, Terrain


def gale_g1() -> SceneSpec:
    return SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="real", preset="gale"),
        robot=Robot(kind="library", robot_id="g1", controller_id="g1_stand"),
        mission=Mission(start_place="highest", goal_place="lowest"),
    ).fill_defaults()


def test_g1_on_gale_gets_flatter_start_and_its_stand_check_reports_the_slope():
    spec = gale_g1()
    requested_start = resolve_mission(spec, make_heights(spec)[0])["start"]
    simulator = MuJoCoSimulator()
    build = agents.build(simulator, spec, lambda _: None)
    checks = simulator.validate(spec, build)

    assert math.dist(requested_start, simulator.mission["start"]) <= 5.0
    assert any("Moved the start" in note and "flatter ground" in note for note in spec.assumptions)
    # the standing-only G1 holds a couple of degrees; Gale's flattest nearby ground is still ~14°, and the check says so
    stand = next(check for check in checks if check.name == "Library robot stands for 1 s at zero command")
    assert not stand.passed and "Starting footprint ground slope" in stand.detail


def test_explicit_flatter_start_uses_fifteen_metre_radius():
    spec = gale_g1()
    requested_start = resolve_mission(spec, make_heights(spec)[0])["start"]

    note = move_start_to_flatter(spec, search_radius_m=15.0)

    distance = math.dist(requested_start, spec.robots[0].start_xy)
    assert note is not None
    assert 5.0 < distance <= 15.0
    assert note in spec.assumptions


def test_gentler_route_preserves_distance_and_direction():
    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="hills", size_m=40, seed=0),
        robot=Robot(kind="walker"),
        mission=Mission(start_place="west", goal_place="east"),
    ).fill_defaults()
    original = resolve_mission(spec, make_heights(spec)[0])
    original_start, original_goal = original["start"], original["goal"]

    note = apply_gentler_route(spec)

    entry = spec.robots[0]
    new_start, new_goal = entry.start_xy, entry.goal_xy
    original_vector = (original_goal[0] - original_start[0], original_goal[1] - original_start[1])
    new_vector = (new_goal[0] - new_start[0], new_goal[1] - new_start[1])
    slopes = re.search(r"average slope ([\d.]+)° instead of ([\d.]+)°", note or "")
    assert note is not None and note in spec.assumptions
    assert math.isclose(math.dist(new_start, new_goal), math.dist(original_start, original_goal), abs_tol=0.1)
    assert math.isclose(new_vector[0], original_vector[0], abs_tol=0.01)
    assert math.isclose(new_vector[1], original_vector[1], abs_tol=0.01)
    assert slopes and float(slopes.group(1)) < float(slopes.group(2))
