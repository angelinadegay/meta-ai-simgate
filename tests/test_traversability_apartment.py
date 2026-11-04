"""The apartment demo: kitchen to bedroom, the ramp and the stairs, and honest spot checks."""
import numpy as np

from simgate.spec import Robot, RobotEntry, SceneSpec, Terrain
from simgate.traversability import features, map_report, verify


def _apartment(robot=None):
    return SceneSpec(domain="robot_terrain", terrain=Terrain(kind="indoor", preset="apartment"),
                     robots=[RobotEntry(id="r1", label="Rover", robot=robot or Robot(kind="rover"),
                                        start_place="kitchen", goal_place="bedroom")]).fill_defaults()


def test_rover_reaches_the_bedroom_via_the_ramp_and_straight_gets_stuck_at_the_stairs():
    report = map_report(_apartment(), "r1")
    straight, safe = report["comparison"]["runs"]
    assert straight["label"] == "Straight" and straight["summary"]["outcome"] == "stuck"
    assert safe["label"] == "Safe route" and safe["summary"]["goal_reached"], report["comparison"]["text"]
    assert report["headline"].startswith("The rover can reach the bedroom via the ramp (")
    assert "The stairs are too high: 0.17 m, it handles about" in report["headline"]
    # the ramp is judged on evidence: the indoor scan measured slope past the ramp's 6°
    assert {scan["knob"] for scan in report["scan"]["scans"]} >= {"slope_deg", "step", "friction"}
    assert report["limits"]["slope_deg"]["reliable_ok"] >= 6
    assert "tile 0.6, wood 0.5, rug 0.8, wet patch 0.15" in report["limits_scope"] and "rock" not in report["limits_scope"]
    assert all("selected area" not in spot["label"] for spot in report["spot_checks"])


def test_indoor_scene_takes_the_apartments_friction_not_an_outdoor_preset():
    assert _apartment().environment.friction == 0.5


def _step_map():
    scene = _apartment()
    grid = features(scene, scene.robots[0])
    classes = np.where(grid["blocked"], "grey", np.where(grid["max_step_m"] >= 0.1, "red", "green"))
    return scene, {**grid, "classes": classes}


def _runs(path_for):
    """A fake batch: every spot passes, driving the path path_for(variation) gives."""
    def run_batch(scene, variations, **kwargs):
        return [{"id": f"spot-run-{index}", "label": variation["label"],
                 "summary": {"outcome": "reached", "goal_reached": True, "flipped": False,
                             "path_xy": path_for(variation)}}
                for index, variation in enumerate(variations, 1)]
    return run_batch


def test_a_spot_that_drives_around_a_step_cannot_correct_it():
    scene, grid = _step_map()
    around = verify(scene, scene.robots[0], grid,   # it "passes", but its path stays on the hallway floor
                    run_batch_fn=_runs(lambda variation: [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]))
    stepped = [spot for spot in around["spot_checks"] if spot["predicted"] == "red" and spot["step_at_spot_m"] >= 0.1]
    assert stepped, "the map has red step spots to check"
    for spot in stepped:
        assert not spot["judged"] and not spot["corrected"] and spot["outcome"].startswith("went around")
        assert spot["max_step_crossed_m"] < spot["step_at_spot_m"]
    assert not around["corrections"] and "didn't count" in around["agreement_text"]

    over = verify(scene, scene.robots[0], grid,   # the same spots, driven over the step: now they can correct it
                  run_batch_fn=_runs(lambda variation: [variation["start_xy"], variation["goal_xy"]]))
    crossed = [spot for spot in over["spot_checks"] if spot["predicted"] == "red" and spot["step_at_spot_m"] >= 0.1]
    assert any(spot["corrected"] and spot["max_step_crossed_m"] >= spot["step_at_spot_m"] for spot in crossed)
