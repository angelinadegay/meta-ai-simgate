import numpy as np
import pytest

from simgate.finder import limits_world
from simgate.library.cards import RobotCard
from simgate.spec import Robot, SceneSpec, Terrain
from simgate.traversability import classify


def _scene(terrain):
    return SceneSpec(domain="robot_terrain", terrain=terrain, robot=Robot()).fill_defaults()


def _report(scene, limits):
    return {"robot_id": scene.robots[0].id, "world": limits_world(scene, scene.robots[0].id), "limits": limits}


def _good_limits():
    return [
        {"knob": "slope_deg", "reliable_ok": 20, "first_fail": 30},
        {"knob": "roughness_m", "reliable_ok": 0.1, "first_fail": 0.2},
        {"knob": "step", "reliable_ok": 0.2, "first_fail": 0.3},
        {"knob": "friction", "reliable_ok": 0.4, "first_fail": 0.3},
    ]


def test_classification_is_green_within_every_reliable_limit():
    scene = _scene(Terrain(kind="flat", size_m=8))

    result = classify(scene, finder_report=_report(scene, _good_limits()))

    assert result["classes"][8, 8] == "green"
    assert result["scan"] is None
    assert "gravity" in result["limits_scope"] and "surface" in result["limits_scope"]
    assert result["honesty"] == (
        "Map from measured limits (step ±1 cm, slope ±1.9°, …); real places combine features, "
        "so the map is checked with spot runs below."
    )


def test_slope_thresholds_make_fail_red_and_unreliable_band_amber():
    scene = _scene(Terrain(kind="slope", slope_deg=12, roughness=0, size_m=8))
    other_limits = [limit for limit in _good_limits() if limit["knob"] != "slope_deg"]
    row = int(np.argmin(np.abs(classify(scene, finder_report=_report(scene, _good_limits()))["y_m"])))
    col = int(np.argmin(np.abs(classify(scene, finder_report=_report(scene, _good_limits()))["x_m"])))

    amber = classify(scene, finder_report=_report(
        scene, [{"knob": "slope_deg", "reliable_ok": 10, "first_fail": 15}, *other_limits]
    ))
    red = classify(scene, finder_report=_report(
        scene, [{"knob": "slope_deg", "reliable_ok": 10, "first_fail": 12}, *other_limits]
    ))

    assert amber["classes"][row, col] == "amber"
    assert red["classes"][row, col] == "red"


def test_missing_limit_is_amber_not_green(monkeypatch):
    from simgate import finder

    monkeypatch.setattr(finder, "scan_map_limits", lambda *args, **kwargs: {
        "runs": 0, "limits": [], "baselines": [],
    })
    scene = _scene(Terrain(kind="flat", size_m=8))
    limits = [limit for limit in _good_limits() if limit["knob"] != "roughness_m"]

    result = classify(scene, finder_report=_report(scene, limits))

    assert result["classes"][8, 8] == "amber"
    assert "roughness" in result["scan_note"]


def test_narrow_doorway_is_red_for_wide_robot():
    scene = _scene(Terrain(kind="indoor"))
    wide_robot = RobotCard(id="wide", name="Wide rover", license="test", credit="test", kind="wheeled",
                           source="builtin", root_body="base", actuator_type="torque",
                           standing_height_m=0.3, mass_kg=10, footprint_length_m=0.4,
                           footprint_width_m=0.58)
    report = _report(scene, [
        {"knob": "step", "reliable_ok": 1.0, "first_fail": 2.0},
        {"knob": "friction", "reliable_ok": 0.1, "first_fail": 0.05},
    ])

    result = classify(scene, wide_robot, report)
    row = int(np.argmin(np.abs(result["y_m"] - 0.625)))
    col = int(np.argmin(np.abs(result["x_m"] - 5.0)))

    assert np.isclose(result["clearance_width_m"][row, col], 0.65)
    assert np.isclose(result["width_limit_m"], 0.68)
    assert result["classes"][row, col] == "red"


def test_wet_patch_is_red_below_measured_friction_fail():
    scene = _scene(Terrain(kind="indoor"))
    report = _report(scene, [
        {"knob": "step", "reliable_ok": 1.0, "first_fail": 2.0},
        {"knob": "friction", "reliable_ok": 0.25, "first_fail": 0.2},
    ])

    result = classify(scene, finder_report=report)
    row = int(np.argmin(np.abs(result["y_m"] + 3.875)))
    col = int(np.argmin(np.abs(result["x_m"] + 5.625)))

    assert result["friction"][row, col] == 0.15
    assert result["classes"][row, col] == "red"


def test_missing_world_limits_get_a_scan_within_sixteen_runs(monkeypatch):
    from simgate import finder

    def run_without_simulation(self, jobs):
        rows = []
        for setting, changes in jobs[:max(0, self.left)]:
            row = {"id": f"r{len(self.evidence) + 1}", **setting, "change": changes,
                   "broke": False, "failure": None}
            self.evidence.append(row)
            rows.append(row)
        return rows

    monkeypatch.setattr(finder.Runner, "run", run_without_simulation)
    scene = _scene(Terrain(kind="flat", size_m=8))

    result = classify(scene)

    assert result["scan"]["runs"] <= 16
    assert result["scan"]["status"] == "measured"
    assert "ran a quick scan" in result["scan_note"]
    assert np.all(result["classes"][~result["blocked"]] == "green")
    assert result["baselines"]


def test_baseline_passes_make_flat_green_but_unseen_slope_stays_amber(monkeypatch):
    from simgate import finder

    scene = _scene(Terrain(kind="flat", size_m=8))
    report = _report(scene, [])
    report["baselines"] = [{"run_id": "baseline-1", "passed": True,
                            "values": {"slope_deg": 0, "roughness": 0, "step": 0, "friction": 0.5}}]
    monkeypatch.setattr(finder, "scan_map_limits", lambda *args, **kwargs: {
        "runs": 0, "limits": [], "baselines": [],
    })
    flat = classify(scene, finder_report=report)
    assert flat["classes"][8, 8] == "green"
    slope = scene.model_copy(deep=True)
    slope.terrain.kind, slope.terrain.slope_deg = "slope", 6
    higher = classify(slope, finder_report=report)
    assert higher["classes"][8, 8] == "amber"
    assert higher["limits"]["slope_deg"]["reliable_ok"] == 0
    assert higher["honesty"] == flat["honesty"]
    report["baselines"][0]["passed"] = False
    assert classify(scene, finder_report=report)["classes"][8, 8] == "amber"
    report["baselines"][0]["passed"] = True
    report["world"]["gravity_m_s2"] = 1.62
    assert classify(scene, finder_report=report)["classes"][8, 8] == "amber"


def test_baseline_friction_is_a_lower_bound_not_an_upper_bound(monkeypatch):
    from simgate import finder

    scene = _scene(Terrain(kind="indoor"))
    report = _report(scene, [
        {"knob": "step", "reliable_ok": 0.05, "first_fail": 0.1},
        {"knob": "friction", "reliable_ok": None, "first_fail": None, "never_broke": True},
    ])
    report["baselines"] = [{"passed": True, "values": {
        "slope_deg": 0, "roughness": 0, "step": 0, "friction": 0.5,
    }}]
    result = classify(scene, finder_report=report)
    row, col = (np.argmin(abs(result[axis] - coordinate)) for axis, coordinate in [("y_m", -3), ("x_m", -3)])
    assert result["classes"][row, col] == "green"
    row, col = (np.argmin(abs(result[axis] - coordinate)) for axis, coordinate in [("y_m", -4.1), ("x_m", -5.7)])
    assert result["classes"][row, col] == "amber"


def test_finder_retains_only_successful_physically_flat_baselines():
    from simgate.finder import _map_baseline

    scene = _scene(Terrain(kind="flat"))
    passed = _map_baseline(scene, {"broke": False}, "r1")
    assert passed["values"]["slope_deg"] == passed["values"]["step"] == passed["values"]["roughness"] == 0
    assert passed["values"]["friction"] == scene.environment.friction
    assert _map_baseline(scene, {"broke": True}, "r2") is None
    assert _map_baseline(scene, {"broke": None}, "r3") is None
    scene.terrain.kind = "slope"
    scene.terrain.slope_deg = 6
    assert _map_baseline(scene, {"broke": False}, "r4") is None


def test_baseline_classified_apartment_uses_ramp_not_the_red_stairs():
    from simgate.routing import plan_route, segment_cells
    from simgate.spec import Mission

    scene = _scene(Terrain(kind="indoor"))
    scene.robots[0].start_place, scene.robots[0].goal_place = "kitchen", "bedroom"
    scene.mission = Mission(start_place="kitchen", goal_place="bedroom")
    report = _report(scene, [
        {"knob": "slope_deg", "last_ok": 9.0, "reliable_ok": None, "first_fail": None, "never_broke": True},
        {"knob": "step", "reliable_ok": None, "first_fail": 0.06, "never_broke": False},
        {"knob": "friction", "reliable_ok": None, "first_fail": None, "never_broke": True},
    ])
    report["baselines"] = [{"passed": True, "values": {
        "slope_deg": 0, "roughness": 0, "step": 0.0225, "friction": scene.environment.friction,
    }}]
    grid = classify(scene, finder_report=report)
    col = int(np.argmin(abs(grid["x_m"] + 4.5)))
    assert all(grid["classes"][row, col] == "red"
               for row, y in enumerate(grid["y_m"]) if 0.6 < y < 1.9)
    route = plan_route(scene, grid)
    assert route["found"], route["reason"]
    assert max(point[0] for point in route["path_xy"]) > 3
    for start, goal in zip(route["path_xy"], route["path_xy"][1:]):
        assert all(grid["classes"][cell] not in ("red", "grey")
                   for cell in segment_cells(route["navigation_grid"], start, goal))


@pytest.mark.parametrize("kind", ["indoor", "slope"])
def test_quick_scan_isolates_features_from_mission_and_validates_every_variation(monkeypatch, kind):
    from simgate import finder
    from simgate.batch import _validate_variation

    scene = _scene(Terrain(kind=kind, size_m=14, slope_deg=12))
    scene.robots[0].start_xy, scene.robots[0].goal_xy = [-4, -2.5], [-4.5, 3.2]
    original = scene.model_dump(mode="json")
    trials = []

    def run_without_simulation(self, jobs):
        trials.append(self.scene)
        assert self.scene.terrain.kind in ("flat", "slope", "rocky")
        assert self.scene.environment == scene.environment
        assert not self.scene.structures and self.scene.disturbance is None
        assert self.scene.mission.start_xy == [-1.5, 0]
        assert self.scene.mission.goal_xy == [1.5, 0]
        assert self.scene.robots[0].mission() == self.scene.mission
        rows = []
        for setting, changes in jobs[:max(0, self.left)]:
            _validate_variation(self.scene, {"label": "probe", **(changes or {"seed": self.scene.terrain.seed})})
            row = {"id": f"r{len(self.evidence) + 1}", **setting, "change": changes,
                   "broke": False, "failure": None}
            self.evidence.append(row)
            rows.append(row)
        return rows

    monkeypatch.setattr(finder.Runner, "run", run_without_simulation)
    result = finder.scan_map_limits(scene)
    assert 0 < result["runs"] <= 16
    assert trials
    assert scene.model_dump(mode="json") == original
    assert result["world"] == limits_world(scene, scene.robots[0].id)
    if kind == "indoor":
        assert {scan["knob"] for scan in result["scans"]} == {"slope_deg", "step", "friction"}   # slope: the ramp
