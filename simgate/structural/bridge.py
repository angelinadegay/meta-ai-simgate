"""Static screening of a separate steel truss idealisation, not visual bridge geometry."""
import math
from typing import Any

from .truss import E_STEEL_PA, STEEL_SECOND_MOMENTS_M4, STEEL_YIELD_PA, solve_truss


def check_bridge(span_m: float, robot_mass_kg: float, gravity_m_s2: float = 9.81,
                 section: str = "SHS40x40x3", robot_label: str = "rover") -> dict[str, Any]:
    """Envelope self-weight plus one robot at every lower-chord panel point.

    A single truss carries the full robot weight, with no load-sharing credit.
    S235 nominal square hollow sections have sharp-corner area/inertia. Euler
    buckling uses K=1 and member-length restraint in both planes at each joint.
    No deck mass, impact, fatigue, connections or lateral system are modelled.
    This screening calculation is not a bridge design certification.
    """
    if not math.isfinite(span_m) or span_m <= 0:
        raise ValueError("span_m must be finite and positive.")
    if not math.isfinite(robot_mass_kg) or robot_mass_kg < 0:
        raise ValueError("robot_mass_kg must be finite and nonnegative.")
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 < 0:
        raise ValueError("gravity_m_s2 must be finite and nonnegative.")
    if section not in STEEL_SECOND_MOMENTS_M4:
        raise ValueError(f"Unknown steel section {section!r}.")
    panels = max(2, math.ceil(span_m / 1.5))
    depth = span_m / 8
    allowable_stress = STEEL_YIELD_PA / 1.5
    allowable_sag = span_m / 360
    envelopes: dict[str, dict[str, Any]] = {}
    cases = []
    result = None
    for node in range(1, panels + 2):
        result = solve_truss(span_m=span_m, depth_m=depth, panels=panels,
                             point_loads={node: (0.0, -robot_mass_kg * gravity_m_s2)},
                             gravity_m_s2=gravity_m_s2, chord_section=section)
        position = result.truss.nodes[node][0]
        base = {"load_node": node, "position_m": position}
        per_case = {}
        for mode in ("yield", "buckling"):
            candidates = []
            for member in result.truss.members:
                force = result.member_forces_n[member.tag]
                demand = abs(force) if mode == "yield" else max(0.0, -force)
                capacity = (allowable_stress * member.area_m2 if mode == "yield" else
                            math.pi**2 * E_STEEL_PA * STEEL_SECOND_MOMENTS_M4[section] / member.length_m**2)
                candidates.append({**base, "mode": mode, "member": member.tag,
                                   "member_nodes": [member.start, member.end], "force_n": force,
                                   "demand_n": demand, "capacity_n": capacity,
                                   "stress_pa": result.member_stresses_pa[member.tag],
                                   "utilisation_pct": 100 * demand / capacity})
            per_case[mode] = max(candidates, key=lambda item: item["utilisation_pct"])
        sag_node = min(result.displacements_m, key=lambda tag: result.displacements_m[tag][1])
        sag = max(0.0, -result.displacements_m[sag_node][1])
        per_case["sag"] = {**base, "mode": "sag", "member": None, "deflected_node": sag_node,
                           "sag_m": sag, "limit_m": allowable_sag,
                           "utilisation_pct": 100 * sag / allowable_sag}
        for mode, candidate in per_case.items():
            if mode not in envelopes or candidate["utilisation_pct"] > envelopes[mode]["utilisation_pct"]:
                envelopes[mode] = candidate
        cases.append({**base, "utilisation_pct": max(item["utilisation_pct"] for item in per_case.values())})

    governing = max(envelopes.values(), key=lambda item: item["utilisation_pct"])
    position = governing["position_m"]
    location = "mid-span" if math.isclose(position, span_m / 2, abs_tol=1e-8) else f"{position:.2f} m from the left support"
    member_text = f", member {governing['member']}" if governing["member"] is not None else ""
    utilisation = governing["utilisation_pct"]
    return {
        "passed": utilisation <= 100,
        "utilisation_pct": utilisation,
        "governing": governing,
        "checks": {mode: {**item, "passed": item["utilisation_pct"] <= 100} for mode, item in envelopes.items()},
        "summary": f"{utilisation:.0f}% with the {robot_label} at {location} ({governing['mode']}{member_text}).",
        "honesty": (f"2D steel truss, {len(result.truss.members)} members, pinned joints; "
                    "the visual model isn't analysed"),
        "assumptions": ("S235 nominal SHS; yield / 1.5; Euler K=1 with panel-point restraint in both planes; "
                        "sag limit span/360. One truss carries full robot weight. Truss self-weight only; "
                        "no deck, impact, connections or lateral-system analysis. Screening, not certification."),
        "span_m": span_m, "depth_m": depth, "panels": panels, "section": section,
        "robot_mass_kg": robot_mass_kg, "robot_label": robot_label, "gravity_m_s2": gravity_m_s2,
        "truss_mass_kg": result.total_mass_kg, "load_cases": cases,
    }