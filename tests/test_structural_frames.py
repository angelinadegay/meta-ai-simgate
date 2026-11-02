import math

import pytest

from simgate.structural.frames import check_habitat, check_lander


def assert_gravity_scaling(earth, mars):
    ratio = 3.71 / 9.81
    for original, scaled in zip(earth["frame"]["members"], mars["frame"]["members"]):
        assert scaled["local_forces"] == pytest.approx([value * ratio for value in original["local_forces"]], abs=1e-6)
        assert scaled["stress_pa"] == pytest.approx(original["stress_pa"] * ratio)
    for tag, displacement in earth["frame"]["displacements_m"].items():
        assert mars["frame"]["displacements_m"][tag] == pytest.approx([value * ratio for value in displacement], abs=1e-10)
    assert mars["utilisation_pct"] == pytest.approx(earth["utilisation_pct"] * ratio)


def test_habitat_forces_scale_with_local_gravity():
    assert_gravity_scaling(check_habitat(gravity_m_s2=9.81), check_habitat(gravity_m_s2=3.71))


def test_habitat_thicker_shielding_increases_utilisation_and_mass():
    thin = check_habitat(gravity_m_s2=3.71)
    thick = check_habitat(gravity_m_s2=3.71, shielding_thickness_m=1.0)
    assert thick["utilisation_pct"] > thin["utilisation_pct"]
    assert thick["shielding_mass_per_rib_kg"] == 2 * thin["shielding_mass_per_rib_kg"]
    assert thin["equipment_mass_kg"] == 2000
    assert thin["equipment_mass_per_rib_kg"] * thin["rib_count"] == 2000
    assert thin["shielding_mass_per_rib_kg"] * thin["rib_count"] == 0.5 * 1600 * 4 * 6
    assert set(thin["checks"]) == {"stress", "deflection"}
    assert "visual model isn't analysed" in thin["honesty"]


def test_lander_forces_scale_with_gravity_in_both_cases():
    earth, mars = check_lander(gravity_m_s2=9.81), check_lander(gravity_m_s2=3.71)
    for name in ("static", "landing_what_if"):
        assert_gravity_scaling(earth["cases"][name], mars["cases"][name])


def test_lander_landing_case_is_explicitly_a_three_g_local_what_if():
    report = check_lander(gravity_m_s2=3.71)
    static, landing = report["cases"]["static"], report["cases"]["landing_what_if"]
    assert not static["what_if"] and landing["what_if"]
    assert "landing what-if" in landing["label"] and "static equivalent" in landing["label"]
    assert landing["frame"]["gravity_m_s2"] == pytest.approx(3 * 3.71)
    assert landing["utilisation_pct"] == pytest.approx(3 * static["utilisation_pct"])
    assert "landing what-if" in report["summary"]
    assert report["governing"]["case"] == "landing_what_if"
    assert set(landing["checks"]) == {"stress", "buckling"}
    assert "visual model isn't analysed" in report["honesty"]


@pytest.mark.parametrize("checker", [check_habitat, check_lander])
def test_frame_loads_balance_support_reactions(checker):
    report = checker()
    frame = report["frame"] if "frame" in report else report["cases"]["static"]["frame"]
    assert sum(reaction[1] for reaction in frame["reactions"].values()) == pytest.approx(
        -sum(load[1] for load in frame["applied_loads_n"].values()))
    assert sum(reaction[0] for reaction in frame["reactions"].values()) == pytest.approx(0, abs=1e-6)


def test_frame_solver_matches_fixed_ended_beam_hand_calculation():
    from simgate.structural.frames import _solve_frame

    frame = _solve_frame({1: (0, 0), 2: (4, 0), 3: (2, 0)}, [(1, 3, "roof"), (3, 2, "roof")],
                         (1, 2), True, {3: 1000}, 9.81, "SHS150x150x8")
    area = 0.15**2 - 0.134**2
    inertia = (0.15**4 - 0.134**4) / 12
    centre_load = (1000 + area * 4 * 2700 / 2) * 9.81
    expected_deflection = centre_load * 4**3 / (192 * 69e9 * inertia)
    expected_stress = centre_load * 4 / 8 / (inertia / 0.075)
    assert frame["displacements_m"][3][1] == pytest.approx(-expected_deflection, rel=0.01)
    assert all(member["stress_pa"] == pytest.approx(expected_stress, rel=0.01) for member in frame["members"])


def test_lander_buckling_capacity_matches_euler_hand_calculation():
    report = check_lander()
    case = report["cases"]["static"]
    check = case["checks"]["buckling"]
    leg = next(member for member in case["frame"]["members"] if member["tag"] == check["member"])
    length = math.hypot(4.5 / 4, 0.6 * 4)
    inertia = (0.1**4 - 0.088**4) / 12
    expected_capacity = math.pi**2 * 69e9 * inertia / length**2
    assert leg["group"] == "leg" and leg["axial_force_n"] < 0
    assert check["capacity_n"] == pytest.approx(expected_capacity)
    assert check["compression_n"] == pytest.approx(-leg["axial_force_n"])


def test_overloaded_frames_fail_without_hiding_which_case_failed():
    habitat = check_habitat(gravity_m_s2=3.71, shielding_thickness_m=10)
    lander = check_lander(body_mass_kg=1_000_000)
    assert not habitat["passed"] and habitat["utilisation_pct"] > 100
    assert not lander["passed"] and not lander["cases"]["static"]["passed"]
    assert not lander["cases"]["landing_what_if"]["passed"]


@pytest.mark.parametrize("checker,kwargs", [
    (check_habitat, {"diameter_m": 0}), (check_habitat, {"module_length_m": -1}),
    (check_habitat, {"shielding_thickness_m": -1}), (check_habitat, {"equipment_mass_kg": math.nan}),
    (check_habitat, {"section": "missing"}), (check_lander, {"footprint_m": 0}),
    (check_lander, {"height_m": math.inf}), (check_lander, {"body_mass_kg": -1}),
    (check_lander, {"gravity_m_s2": math.nan}), (check_lander, {"section": "missing"}),
])
def test_invalid_frame_inputs_are_rejected(checker, kwargs):
    with pytest.raises(ValueError):
        checker(**kwargs)


def test_failed_frame_analysis_clears_opensees_state(monkeypatch):
    from simgate.structural import frames

    monkeypatch.setattr(frames.ops, "analyze", lambda steps: -1)
    with pytest.raises(RuntimeError, match="frame static analysis failed"):
        check_habitat()
    assert frames.ops.getNodeTags() == []