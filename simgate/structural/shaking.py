"""Synthetic 2D structural base shaking, separate from baseline mission physics."""
import copy
import math

import openseespy.opensees as ops

from . import frames
from .checks import structure_section
from .truss import E_STEEL_PA, STEEL_DENSITY_KG_M3, STEEL_SECOND_MOMENTS_M4, STEEL_YIELD_PA, build_warren_truss

DT_S = 0.01
DURATION_S = 4.0
FREQUENCY_HZ = 1.5
PGA_LOCAL_G = 0.2


def _shake(record: dict, gravity: float) -> dict:
    baseline = copy.deepcopy(record["structural_check"])
    is_truss = record["kind"] == "bridge"
    if not is_truss and record["kind"] == "lander":
        baseline = baseline["cases"]["static"]
    checks = copy.deepcopy(baseline["checks"])
    if gravity == 0:
        return {"passed": True, "utilisation_pct": 0.0, "checks": checks, "period_s": None}
    if is_truss:
        dimensions = record["dimensions"]
        truss = build_warren_truss(dimensions["span_m"], baseline["depth_m"], baseline["panels"], structure_section(record))
        nodes = truss.nodes
        masses = {tag: 0.0 for tag in nodes}
        for member in truss.members:
            mass = STEEL_DENSITY_KG_M3 * member.area_m2 * member.length_m
            masses[member.start] += mass / 2
            masses[member.end] += mass / 2
        masses[baseline["governing"]["load_node"]] += baseline["robot_mass_kg"]
        members = [{"tag": member.tag, "nodes": [member.start, member.end], "length_m": member.length_m,
                    "area_m2": member.area_m2} for member in truss.members]
        supports = [1, baseline["panels"] + 1]
        if record.get("mid_support"):
            supports.append(baseline["panels"] // 2 + 1)
        degrees = 2
    else:
        frame = copy.deepcopy(baseline["frame"])
        nodes, members = frame["nodes"], frame["members"]
        masses = {tag: -load[1] / gravity for tag, load in frame["applied_loads_n"].items()}
        supports, degrees = [1, 2], 3
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", degrees)
        for tag, point in nodes.items():
            ops.node(tag, *point)
            ops.mass(tag, masses[tag], masses[tag], *([0.0] if degrees == 3 else []))
        for index, tag in enumerate(supports):
            fixed = [int(index == 0), 1] if is_truss else [1, 1, int(record["kind"] == "habitat")]
            ops.fix(tag, *fixed)
        if is_truss:
            ops.uniaxialMaterial("Elastic", 1, E_STEEL_PA)
            for member in members:
                ops.element("truss", member["tag"], *member["nodes"], member["area_m2"], 1)
        else:
            ops.geomTransf("Linear", 1)
            for member in members:
                ops.element("elasticBeamColumn", member["tag"], *member["nodes"], frame["area_m2"],
                            frames.E_ALUMINIUM_PA, frame["inertia_m4"], 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for tag, mass in masses.items():
            ops.load(tag, 0.0, -mass * gravity, *([0.0] if degrees == 3 else []))
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise RuntimeError("Synthetic shaking gravity equilibrium failed.")
        ops.loadConst("-time", 0.0)
        eigen = float(ops.eigen("-fullGenLapack", 1)[0])
        if not math.isfinite(eigen) or eigen <= 0:
            raise RuntimeError("The structure has no valid shaking mode.")
        frequency = math.sqrt(eigen)
        ops.rayleigh(2 * 0.05 * frequency, 0.0, 0.0, 0.0)
        acceleration = []
        for step in range(int(DURATION_S / DT_S) + 1):
            time = step * DT_S
            envelope = min(1.0, time, max(0.0, DURATION_S - time))
            acceleration.append(PGA_LOCAL_G * gravity * math.sin(2 * math.pi * FREQUENCY_HZ * time) * envelope)
        ops.timeSeries("Path", 2, "-dt", DT_S, "-values", *acceleration)
        ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
        ops.wipeAnalysis()
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.algorithm("Linear")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")
        for step in range(int(DURATION_S / DT_S)):
            if ops.analyze(1, DT_S) != 0:
                raise RuntimeError("Synthetic shaking transient analysis failed.")
            if is_truss:
                instantaneous = {}
                for mode in ("yield", "buckling"):
                    candidates = []
                    for member in members:
                        force = float(ops.eleResponse(member["tag"], "axialForce")[0])
                        capacity = (STEEL_YIELD_PA / 1.5 * member["area_m2"] if mode == "yield" else
                                    math.pi**2 * E_STEEL_PA * STEEL_SECOND_MOMENTS_M4[structure_section(record)] / member["length_m"]**2)
                        demand = abs(force) if mode == "yield" else max(0.0, -force)
                        candidates.append({"mode": mode, "member": member["tag"], "utilisation_pct": 100 * demand / capacity})
                    instantaneous[mode] = max(candidates, key=lambda item: item["utilisation_pct"])
                sag_node = min(nodes, key=lambda tag: float(ops.nodeDisp(tag, 2)))
                sag = max(0.0, -float(ops.nodeDisp(sag_node, 2)))
                instantaneous["sag"] = {"mode": "sag", "member": None, "deflected_node": sag_node,
                                         "utilisation_pct": 100 * sag / (dimensions["span_m"] / 360)}
            else:
                frame["displacements_m"] = {tag: [float(ops.nodeDisp(tag, axis)) for axis in (1, 2)] for tag in nodes}
                for member in members:
                    forces = [float(value) for value in ops.eleResponse(member["tag"], "localForce")]
                    member["axial_force_n"] = forces[3]
                    member["stress_pa"] = (max(abs(forces[0]), abs(forces[3])) / frame["area_m2"]
                                            + max(abs(forces[2]), abs(forces[5])) / frame["section_modulus_m3"])
                instantaneous = frames._checks(frame, record["dimensions"]["module_diameter_m"] / 360
                                                if record["kind"] == "habitat" else None,
                                                legs_only=record["kind"] == "lander")
            for mode, candidate in instantaneous.items():
                value = candidate["utilisation_pct"]
                if not math.isfinite(value):
                    raise RuntimeError("Synthetic shaking returned non-finite results.")
                if value > checks[mode]["utilisation_pct"]:
                    checks[mode] = {**candidate, "time_s": round((step + 1) * DT_S, 2), "passed": value <= 100}
        governing = max(checks.values(), key=lambda item: item["utilisation_pct"])
        return {"passed": governing["utilisation_pct"] <= 100, "utilisation_pct": governing["utilisation_pct"],
                "governing": governing, "checks": checks, "period_s": 2 * math.pi / frequency}
    finally:
        ops.wipe()


def synthetic_marsquake(records: list[dict], gravity_m_s2: float) -> dict:
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 < 0:
        raise ValueError("Synthetic shaking requires finite nonnegative gravity.")
    results = []
    for record in records:
        result = _shake(record, gravity_m_s2)
        results.append({**result, "kind": record["kind"], "request_index": record["request_index"]})
    return {"type": "structure_shaking", "synthetic": True,
            "label": "Synthetic marsquake what-if", "pga_m_s2": PGA_LOCAL_G * gravity_m_s2,
            "frequency_hz": FREQUENCY_HZ, "duration_s": DURATION_S, "structures": results,
            "honesty": ("Synthetic 1.5 Hz sinusoidal base shaking, 0.2 x local gravity, 4 s; not measured Mars seismic data. "
                        "Linear-elastic 2D models, 5% first-mode mass damping, relative structural displacement. "
                        "Bridge robot held at its governing static load point. No 3D response, yielding, impact, "
                        "foundation or radiation analysis. Baseline mission physics and checks are unchanged.")}