import numpy as np

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


def test_missing_limit_is_amber_not_green():
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
    assert np.all(result["classes"][~result["blocked"]] == "amber")
