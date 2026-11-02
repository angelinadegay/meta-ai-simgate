"""Aluminium portal-rib and opposite-lander-leg screening using OpenSees.

SI units, linear elasticity, rigid member joints; these are separate idealised
frames, not the visual models. OpenSees state is process-global: use sequential
analyses or separate processes, as with the truss engine.
"""
import math
from typing import Any

import openseespy.opensees as ops

E_ALUMINIUM_PA = 69e9
ALUMINIUM_DENSITY_KG_M3 = 2700.0
ALUMINIUM_YIELD_PA = 240e6
ALUMINIUM_SECTIONS = {
    "SHS100x100x6": (0.100, 0.006),
    "SHS150x150x8": (0.150, 0.008),
    "SHS200x200x10": (0.200, 0.010),
}


def _validate_values(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")


def _solve_frame(nodes: dict[int, tuple[float, float]], members: list[tuple[int, int, str]],
                 supports: tuple[int, int], fixed_bases: bool, masses_kg: dict[int, float],
                 gravity_m_s2: float, section: str) -> dict[str, Any]:
    if section not in ALUMINIUM_SECTIONS:
        raise ValueError(f"Unknown aluminium section {section!r}.")
    width, thickness = ALUMINIUM_SECTIONS[section]
    area = width**2 - (width - 2 * thickness)**2
    inertia = (width**4 - (width - 2 * thickness)**4) / 12
    modulus = inertia / (width / 2)
    loads = {tag: [0.0, -masses_kg.get(tag, 0.0) * gravity_m_s2] for tag in nodes}
    member_records = []
    total_mass = 0.0
    for tag, (start, end, group) in enumerate(members, 1):
        length = math.dist(nodes[start], nodes[end])
        mass = area * length * ALUMINIUM_DENSITY_KG_M3
        total_mass += mass
        for node in (start, end):
            loads[node][1] -= mass * gravity_m_s2 / 2
        member_records.append({"tag": tag, "nodes": [start, end], "group": group, "length_m": length})
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, point in nodes.items():
            ops.node(tag, *point)
        for tag in supports:
            ops.fix(tag, 1, 1, int(fixed_bases))
        ops.geomTransf("Linear", 1)
        for member in member_records:
            ops.element("elasticBeamColumn", member["tag"], *member["nodes"], area, E_ALUMINIUM_PA, inertia, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for tag, load in loads.items():
            ops.load(tag, *load, 0.0)
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise RuntimeError("OpenSees aluminium frame static analysis failed.")
        displacements = {tag: [float(ops.nodeDisp(tag, axis)) for axis in (1, 2)] for tag in nodes}
        for member in member_records:
            forces = [float(value) for value in ops.eleResponse(member["tag"], "localForce")]
            if not all(math.isfinite(value) for value in forces):
                raise RuntimeError("OpenSees returned non-finite frame forces.")
            member.update({"local_forces": forces, "axial_force_n": forces[3],
                           "max_moment_nm": max(abs(forces[2]), abs(forces[5])),
                           "stress_pa": max(abs(forces[0]), abs(forces[3])) / area
                                        + max(abs(forces[2]), abs(forces[5])) / modulus})
        if not all(math.isfinite(value) for pair in displacements.values() for value in pair):
            raise RuntimeError("OpenSees returned non-finite frame deflections.")
        ops.reactions()
        reactions = {tag: [float(ops.nodeReaction(tag, axis)) for axis in (1, 2, 3)] for tag in supports}
        return {"nodes": nodes, "members": member_records, "displacements_m": displacements,
                "applied_loads_n": loads, "reactions": reactions, "section": section,
                "area_m2": area, "inertia_m4": inertia, "section_modulus_m3": modulus,
                "frame_mass_kg": total_mass, "gravity_m_s2": gravity_m_s2}
    finally:
        ops.wipe()


def _checks(frame: dict[str, Any], deflection_limit_m: float | None = None,
            legs_only: bool = False) -> dict[str, dict[str, Any]]:
    members = [member for member in frame["members"] if not legs_only or member["group"] == "leg"]
    allowable = ALUMINIUM_YIELD_PA / 1.5
    governing = max(members, key=lambda member: member["stress_pa"])
    checks = {"stress": {"member": governing["tag"], "stress_pa": governing["stress_pa"],
                          "allowable_pa": allowable, "utilisation_pct": 100 * governing["stress_pa"] / allowable}}
    if legs_only:
        candidates = []
        for member in members:
            capacity = math.pi**2 * E_ALUMINIUM_PA * frame["inertia_m4"] / member["length_m"]**2
            compression = max(0.0, -member["axial_force_n"])
            candidates.append({"member": member["tag"], "compression_n": compression,
                               "capacity_n": capacity, "effective_length_m": member["length_m"],
                               "utilisation_pct": 100 * compression / capacity})
        checks["buckling"] = max(candidates, key=lambda item: item["utilisation_pct"])
    if deflection_limit_m is not None:
        node = max(frame["displacements_m"], key=lambda tag: math.hypot(*frame["displacements_m"][tag]))
        displacement = math.hypot(*frame["displacements_m"][node])
        checks["deflection"] = {"member": None, "node": node, "deflection_m": displacement,
                                "limit_m": deflection_limit_m, "utilisation_pct": 100 * displacement / deflection_limit_m}
    return {mode: {**item, "mode": mode, "passed": item["utilisation_pct"] <= 100} for mode, item in checks.items()}


def _report(frame: dict[str, Any], checks: dict[str, dict[str, Any]], label: str,
            honesty: str, assumptions: str) -> dict[str, Any]:
    governing = max(checks.values(), key=lambda item: item["utilisation_pct"])
    utilisation = governing["utilisation_pct"]
    location = f"member {governing['member']}" if governing["member"] is not None else f"joint {governing['node']}"
    return {"passed": utilisation <= 100, "utilisation_pct": utilisation, "governing": governing,
            "checks": checks, "summary": f"{utilisation:.0f}% under {label} ({governing['mode']}, {location}).",
            "honesty": honesty, "assumptions": assumptions, "frame": frame}


def check_habitat(module_length_m: float = 6.0, diameter_m: float = 4.0, height_m: float = 3.0,
                  gravity_m_s2: float = 9.81, shielding_thickness_m: float = 0.5,
                  equipment_mass_kg: float = 2000.0, section: str = "SHS150x150x8") -> dict[str, Any]:
    """One fixed-base portal rib, with tributary roof shielding and equipment.

    Ribs divide module length evenly at <=1.5 m spacing. Shielding covers the
    projected roof plan at 1600 kg/m3. Roof loads and member self-weight are
    lumped at nodes; equipment is conservatively carried by the roof rib.
    """
    _validate_values(module_length_m=module_length_m, diameter_m=diameter_m, height_m=height_m,
                     gravity_m_s2=gravity_m_s2, shielding_thickness_m=shielding_thickness_m,
                     equipment_mass_kg=equipment_mass_kg)
    if min(module_length_m, diameter_m, height_m) <= 0:
        raise ValueError("Habitat dimensions must be positive.")
    ribs = max(1, math.ceil(module_length_m / 1.5))
    tributary = module_length_m / ribs
    equipment = equipment_mass_kg / ribs
    shielding = shielding_thickness_m * 1600 * diameter_m * tributary
    segments = 8
    nodes = {1: (0.0, 0.0), 2: (diameter_m, 0.0)}
    nodes.update({index + 3: (diameter_m * index / segments, height_m) for index in range(segments + 1)})
    members = [(1, 3, "column"), (2, segments + 3, "column")]
    members += [(index + 3, index + 4, "roof") for index in range(segments)]
    masses = {index + 3: (equipment + shielding) / segments * (0.5 if index in (0, segments) else 1)
              for index in range(segments + 1)}
    frame = _solve_frame(nodes, members, (1, 2), True, masses, gravity_m_s2, section)
    report = _report(
        frame, _checks(frame, deflection_limit_m=diameter_m / 360), "habitat gravity loads",
        f"2D aluminium portal rib, {len(members)} frame elements, fixed bases; the visual model isn't analysed",
        "6061-T6 nominal SHS, E=69 GPa, assumed yield 240 MPa; yield / 1.5; "
        "axial plus bending stress; displacement limit diameter/360. "
        "One portal rib approximates an arch; equipment and projected-roof shielding shared by tributary length. "
        "Nodal gravity loads; no pressure shell, skin mass, joints, buckling, lateral stability or foundation analysis. "
        "Screening, not certification.",
    )
    report.update({"rib_count": ribs, "tributary_length_m": tributary, "equipment_mass_kg": equipment_mass_kg,
                   "equipment_mass_per_rib_kg": equipment, "shielding_mass_per_rib_kg": shielding,
                   "shielding_thickness_m": shielding_thickness_m, "shielding_density_kg_m3": 1600.0})
    return report


def check_lander(footprint_m: float = 4.5, height_m: float = 4.0, gravity_m_s2: float = 9.81,
                 body_mass_kg: float = 2000.0, section: str = "SHS100x100x6") -> dict[str, Any]:
    """Two opposite legs carry half the body mass, plus this frame's self-weight.

    Feet are pinned; leg/body connections are rigid. The landing what-if is
    exactly 3 times local gravity applied to all masses, not 3 Earth g or a
    transient touchdown prediction. Neither case includes the other as an
    additional load. Body mass is an explicit assumption, not visual mass.
    """
    _validate_values(footprint_m=footprint_m, height_m=height_m, gravity_m_s2=gravity_m_s2,
                     body_mass_kg=body_mass_kg)
    if min(footprint_m, height_m) <= 0:
        raise ValueError("Lander dimensions must be positive.")
    leg_height = 0.6 * height_m
    nodes = {1: (-footprint_m / 2, 0.0), 2: (footprint_m / 2, 0.0),
             3: (-footprint_m / 4, leg_height), 4: (footprint_m / 4, leg_height)}
    members = [(1, 3, "leg"), (2, 4, "leg"), (3, 4, "body")]
    masses = {3: body_mass_kg / 4, 4: body_mass_kg / 4}
    honesty = "2D aluminium frame, 2 opposite legs + body, pinned feet; the visual model isn't analysed"
    assumptions = (f"Assumed body mass {body_mass_kg:g} kg. 6061-T6 nominal SHS, E=69 GPa, assumed yield 240 MPa; "
                   "yield / 1.5; combined axial/bending leg stress; Euler K=1 with "
                   "leg-length restraint in both planes. One opposite pair carries half the assumed body mass "
                   "plus its own frame weight. No body strength, propellant, impact dynamics, joints, terrain "
                   "compliance or lateral-system analysis. Screening, not certification.")
    cases = {}
    for name, multiplier, label in (("static", 1.0, "local-gravity weight"),
                                    ("landing_what_if", 3.0, "landing what-if: 3 x local gravity (static equivalent)")):
        frame = _solve_frame(nodes, members, (1, 2), False, masses, gravity_m_s2 * multiplier, section)
        case = _report(frame, _checks(frame, legs_only=True), label, honesty, assumptions)
        case.update({"label": label, "what_if": name == "landing_what_if", "gravity_multiplier": multiplier})
        cases[name] = case
    governing_case = max(cases, key=lambda name: cases[name]["utilisation_pct"])
    governing = cases[governing_case]
    return {"passed": all(case["passed"] for case in cases.values()),
            "utilisation_pct": governing["utilisation_pct"],
            "governing": {**governing["governing"], "case": governing_case},
            "summary": governing["summary"], "honesty": honesty, "assumptions": assumptions,
            "body_mass_kg": body_mass_kg, "pair_body_mass_kg": body_mass_kg / 2,
            "gravity_m_s2": gravity_m_s2, "cases": cases}