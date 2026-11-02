import math

import pytest

from simgate.structural.truss import (E_STEEL_PA, STEEL_AREAS_M2, STEEL_DENSITY_KG_M3,
                                      build_warren_truss, solve_truss)


def test_three_panel_truss_matches_method_of_joints():
    """6 m span, 2 m depth, 10 kN downward at (3, 2); reactions are P/2.

    At the left joint, sin(theta)=2/sqrt(5), so the first web is
    -P*sqrt(5)/4 and the bottom chord is P/4. Successive joint equilibrium
    gives the arrays below; symmetry supplies the right half. Unit-load
    virtual work gives central deflection P*(19+15*sqrt(5))/(8*E*A).
    """
    load = 10_000.0
    result = solve_truss(point_loads={6: (0.0, -load)}, gravity_m_s2=0)
    diagonal = load * math.sqrt(5) / 4
    expected = [load / 4, 3 * load / 4, load / 4, -load / 2, -load / 2,
                -diagonal, diagonal, -diagonal, -diagonal, diagonal, -diagonal]
    assert len(result.truss.nodes) == 7 and len(result.truss.members) == 11
    area = STEEL_AREAS_M2["SHS60x60x4"]
    for member, force in zip(result.truss.members, expected):
        assert result.member_forces_n[member.tag] == pytest.approx(force, rel=0.01)
        assert result.member_stresses_pa[member.tag] == pytest.approx(force / area, rel=0.01)
    for reaction in result.reactions_n.values():
        assert reaction == pytest.approx((0, load / 2), abs=1e-6)
    deflection = load * (19 + 15 * math.sqrt(5)) / (8 * E_STEEL_PA * area)
    assert result.displacements_m[6][1] == pytest.approx(-deflection, rel=0.01)
    assert result.max_vertical_deflection_m == pytest.approx(deflection, rel=0.01)


def test_member_forces_stresses_and_deflections_scale_with_point_loads():
    baseline = solve_truss(point_loads={2: (750, -3000), 6: (0, -10_000)}, gravity_m_s2=0)
    doubled = solve_truss(point_loads={2: (1500, -6000), 6: (0, -20_000)}, gravity_m_s2=0)
    for tag in baseline.member_forces_n:
        assert doubled.member_forces_n[tag] == pytest.approx(2 * baseline.member_forces_n[tag])
        assert doubled.member_stresses_pa[tag] == pytest.approx(2 * baseline.member_stresses_pa[tag])
    for tag in baseline.displacements_m:
        assert doubled.displacements_m[tag] == pytest.approx(tuple(2 * value for value in baseline.displacements_m[tag]))
    assert doubled.max_deflection_m == pytest.approx(2 * baseline.max_deflection_m)


def test_self_weight_scales_with_gravity_and_balances_reactions():
    earth = solve_truss(gravity_m_s2=9.81)
    mars = solve_truss(gravity_m_s2=3.71)
    zero = solve_truss(gravity_m_s2=0)
    ratio = 3.71 / 9.81
    expected_mass = STEEL_DENSITY_KG_M3 * STEEL_AREAS_M2["SHS60x60x4"] * (10 + 6 * math.sqrt(5))
    assert earth.total_mass_kg == pytest.approx(expected_mass)
    assert mars.total_mass_kg == earth.total_mass_kg
    assert -sum(force[1] for force in earth.applied_loads_n.values()) == pytest.approx(expected_mass * 9.81)
    assert sum(force[1] for force in earth.reactions_n.values()) == pytest.approx(expected_mass * 9.81)
    for tag in earth.member_forces_n:
        assert mars.member_forces_n[tag] == pytest.approx(ratio * earth.member_forces_n[tag])
        assert zero.member_forces_n[tag] == pytest.approx(0, abs=1e-9)
    assert mars.max_deflection_m == pytest.approx(ratio * earth.max_deflection_m)


def test_member_sections_control_area_mass_and_stiffness():
    light = solve_truss(point_loads={6: (0, -10_000)}, gravity_m_s2=0, chord_section="SHS40x40x3")
    heavy = solve_truss(point_loads={6: (0, -10_000)}, gravity_m_s2=0, chord_section="SHS80x80x4")
    ratio = STEEL_AREAS_M2["SHS80x80x4"] / STEEL_AREAS_M2["SHS40x40x3"]
    assert heavy.member_forces_n == pytest.approx(light.member_forces_n)
    assert heavy.total_mass_kg == pytest.approx(ratio * light.total_mass_kg)
    assert heavy.max_deflection_m == pytest.approx(light.max_deflection_m / ratio)
    mixed = build_warren_truss(chord_section="SHS80x80x4", web_section="SHS40x40x3")
    assert all(member.section == ("SHS40x40x3" if member.group == "web" else "SHS80x80x4")
               for member in mixed.members)


@pytest.mark.parametrize("kwargs", [
    {"span_m": 0}, {"depth_m": -1}, {"span_m": math.nan}, {"depth_m": math.inf},
    {"panels": 0}, {"panels": 2.5}, {"panels": True}, {"chord_section": "missing"},
    {"gravity_m_s2": -1}, {"gravity_m_s2": math.nan}, {"gravity_m_s2": math.inf},
    {"point_loads": {99: (0, -1)}}, {"point_loads": {6: (0, math.nan)}},
    {"point_loads": {6: (1,)}},
])
def test_invalid_truss_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_truss(**kwargs)


@pytest.mark.parametrize("panels", [1, 2, 4, 6])
def test_joint_equilibrium_for_other_panel_counts(panels):
    result = solve_truss(span_m=12, depth_m=3, panels=panels, point_loads={panels + 2: (500, -4000)},
                         chord_section="SHS80x80x4", web_section="SHS40x40x3")
    balance = {tag: list(force) for tag, force in result.applied_loads_n.items()}
    for tag, reaction in result.reactions_n.items():
        for axis in (0, 1):
            balance[tag][axis] += reaction[axis]
    for member in result.truss.members:
        start, end = result.truss.nodes[member.start], result.truss.nodes[member.end]
        force = result.member_forces_n[member.tag]
        for axis in (0, 1):
            component = force * (end[axis] - start[axis]) / member.length_m
            balance[member.start][axis] += component
            balance[member.end][axis] -= component
    assert all(abs(value) < 1e-6 for pair in balance.values() for value in pair)
    assert result.max_deflection_m > 0


def test_failed_analysis_clears_opensees_model(monkeypatch):
    from simgate.structural import truss

    monkeypatch.setattr(truss.ops, "analyze", lambda steps: -1)
    with pytest.raises(RuntimeError, match="static analysis failed"):
        solve_truss()
    assert truss.ops.getNodeTags() == []


def test_bridge_rover_passes_and_reports_every_panel_position():
    from simgate.structural.bridge import check_bridge

    report = check_bridge(span_m=6, robot_mass_kg=10)
    assert report["passed"] and report["utilisation_pct"] < 100
    assert report["depth_m"] == 0.75 and report["panels"] == 4
    assert [case["position_m"] for case in report["load_cases"]] == [0, 1.5, 3, 4.5, 6]
    assert report["governing"]["position_m"] == 3
    assert "rover at mid-span" in report["summary"]
    assert report["honesty"] == "2D steel truss, 15 members, pinned joints; the visual model isn't analysed"
    assert set(report["checks"]) == {"yield", "buckling", "sag"}


def test_bridge_heavy_load_fails_and_one_catalogue_size_up_passes():
    from simgate.structural.bridge import check_bridge

    small = check_bridge(span_m=6, robot_mass_kg=5000, section="SHS40x40x3")
    larger = check_bridge(span_m=6, robot_mass_kg=5000, section="SHS60x60x4")
    assert not small["passed"] and small["utilisation_pct"] > 100
    assert larger["passed"] and larger["utilisation_pct"] < 100


def test_bridge_long_span_fails():
    from simgate.structural.bridge import check_bridge

    report = check_bridge(span_m=60, robot_mass_kg=10)
    assert not report["passed"] and report["utilisation_pct"] > 100
    assert not report["checks"]["buckling"]["passed"]


def test_bridge_utilisation_scales_with_world_gravity():
    from simgate.structural.bridge import check_bridge

    earth = check_bridge(span_m=6, robot_mass_kg=10, gravity_m_s2=9.81)
    mars = check_bridge(span_m=6, robot_mass_kg=10, gravity_m_s2=3.71)
    assert mars["utilisation_pct"] / earth["utilisation_pct"] == pytest.approx(3.71 / 9.81)
    for mode in earth["checks"]:
        assert mars["checks"][mode]["utilisation_pct"] == pytest.approx(
            earth["checks"][mode]["utilisation_pct"] * 3.71 / 9.81)


def test_bridge_envelope_checks_match_independent_limits():
    from simgate.structural.bridge import check_bridge

    report = check_bridge(span_m=6, robot_mass_kg=5000)
    inertia = (0.04**4 - 0.034**4) / 12
    for mode, check in report["checks"].items():
        solution = solve_truss(span_m=6, depth_m=0.75, panels=4, chord_section="SHS40x40x3",
                               point_loads={check["load_node"]: (0, -5000 * 9.81)})
        if mode == "sag":
            sag = max(-pair[1] for pair in solution.displacements_m.values())
            expected = 100 * sag / (6 / 360)
            assert check["member"] is None
        else:
            member = next(item for item in solution.truss.members if item.tag == check["member"])
            force = solution.member_forces_n[member.tag]
            if mode == "yield":
                expected = 100 * abs(force / member.area_m2) / (235e6 / 1.5)
            else:
                assert force < 0
                expected = 100 * -force / (math.pi**2 * 200e9 * inertia / member.length_m**2)
        assert check["utilisation_pct"] == pytest.approx(expected)
    assert report["utilisation_pct"] == max(item["utilisation_pct"] for item in report["checks"].values())


@pytest.mark.parametrize("kwargs", [
    {"span_m": 0}, {"span_m": math.inf}, {"robot_mass_kg": -1}, {"robot_mass_kg": math.nan},
    {"gravity_m_s2": -1}, {"gravity_m_s2": math.inf}, {"section": "missing"},
])
def test_invalid_bridge_inputs_are_rejected(kwargs):
    from simgate.structural.bridge import check_bridge

    with pytest.raises(ValueError):
        check_bridge(**{"span_m": 6, "robot_mass_kg": 10, **kwargs})