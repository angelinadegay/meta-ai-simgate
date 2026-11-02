"""Small-displacement, pin-jointed 2D Warren trusses in SI units.

Loads use +x right and +y up; positive axial forces and stresses mean tension.
The left end is pinned and the right end is a horizontal roller. Steel stays
linear elastic: yielding, buckling, bending and connection effects are excluded.
OpenSees owns process-global state; run analyses sequentially or in separate
processes, never alongside another OpenSees model in the same process.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math

import openseespy.opensees as ops

E_STEEL_PA = 200e9
STEEL_DENSITY_KG_M3 = 7850.0
STEEL_AREAS_M2 = {
    "SHS40x40x3": 0.000444,
    "SHS60x60x4": 0.000896,
    "SHS80x80x4": 0.001216,
}
STEEL_YIELD_PA = 235e6
STEEL_SECOND_MOMENTS_M4 = {
    "SHS40x40x3": (0.04**4 - 0.034**4) / 12,
    "SHS60x60x4": (0.06**4 - 0.052**4) / 12,
    "SHS80x80x4": (0.08**4 - 0.072**4) / 12,
}


@dataclass(frozen=True)
class Member:
    tag: int
    start: int
    end: int
    group: str
    section: str
    area_m2: float
    length_m: float


@dataclass(frozen=True)
class Truss:
    nodes: dict[int, tuple[float, float]]
    members: list[Member]
    supports: tuple[int, int]


@dataclass(frozen=True)
class TrussResult:
    truss: Truss
    applied_loads_n: dict[int, tuple[float, float]]
    displacements_m: dict[int, tuple[float, float]]
    reactions_n: dict[int, tuple[float, float]]
    member_forces_n: dict[int, float]
    member_stresses_pa: dict[int, float]
    max_deflection_m: float
    max_vertical_deflection_m: float
    total_mass_kg: float


def build_warren_truss(span_m: float = 6.0, depth_m: float = 2.0, panels: int = 3,
                       chord_section: str = "SHS60x60x4", web_section: str | None = None) -> Truss:
    """Build geometry without touching OpenSees.

    Bottom panel points are numbered 1..panels+1, left to right. Upper points
    are panels+2..2*panels+1 at half-panel offsets. Odd panel counts therefore
    have a central upper load point. Each bay has two diagonal web members.
    Catalogue SHS areas use nominal square dimensions, ignoring corner radii.
    """
    if not math.isfinite(span_m) or span_m <= 0 or not math.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("span_m and depth_m must be finite and positive.")
    if isinstance(panels, bool) or not isinstance(panels, int) or panels < 1:
        raise ValueError("panels must be a positive integer.")
    web_section = chord_section if web_section is None else web_section
    for section in (chord_section, web_section):
        if section not in STEEL_AREAS_M2:
            raise ValueError(f"Unknown steel section {section!r}; choose from {', '.join(STEEL_AREAS_M2)}.")
    bay = span_m / panels
    nodes = {index + 1: (index * bay, 0.0) for index in range(panels + 1)}
    nodes.update({panels + 2 + index: ((index + 0.5) * bay, depth_m) for index in range(panels)})
    members = []

    def add_member(start: int, end: int, group: str, section: str) -> None:
        members.append(Member(len(members) + 1, start, end, group, section,
                              STEEL_AREAS_M2[section], math.dist(nodes[start], nodes[end])))

    for index in range(panels):
        add_member(index + 1, index + 2, "bottom_chord", chord_section)
    for index in range(panels - 1):
        add_member(panels + 2 + index, panels + 3 + index, "top_chord", chord_section)
    for index in range(panels):
        upper = panels + 2 + index
        add_member(index + 1, upper, "web", web_section)
        add_member(upper, index + 2, "web", web_section)
    return Truss(nodes, members, (1, panels + 1))


def solve_truss(span_m: float = 6.0, depth_m: float = 2.0, panels: int = 3,
                point_loads: Mapping[int, tuple[float, float]] | None = None,
                gravity_m_s2: float = 9.81, chord_section: str = "SHS60x60x4",
                web_section: str | None = None) -> TrussResult:
    """Solve statics for nodal (Fx, Fy) loads in newtons plus member self-weight.

    Gravity is a nonnegative downward acceleration; use zero to omit self-weight.
    Each member contributes half its weight to each endpoint. Results include
    all nodal displacements, support reactions and unrounded axial quantities.
    Maximum deflection is the largest nodal displacement magnitude, in metres.
    The model is wiped on entry and exit, including when analysis fails.
    """
    truss = build_warren_truss(span_m, depth_m, panels, chord_section, web_section)
    if not math.isfinite(gravity_m_s2) or gravity_m_s2 < 0:
        raise ValueError("gravity_m_s2 must be finite and nonnegative.")
    loads = {tag: [0.0, 0.0] for tag in truss.nodes}
    for tag, force in ({} if point_loads is None else point_loads).items():
        if isinstance(tag, bool) or not isinstance(tag, int) or tag not in truss.nodes:
            raise ValueError(f"Unknown panel point {tag!r}.")
        if len(force) != 2 or not all(math.isfinite(value) for value in force):
            raise ValueError("Point loads must be finite (Fx, Fy) pairs in newtons.")
        loads[tag] = [float(force[0]), float(force[1])]
    mass = 0.0
    for member in truss.members:
        member_mass = STEEL_DENSITY_KG_M3 * member.area_m2 * member.length_m
        mass += member_mass
        for tag in (member.start, member.end):
            loads[tag][1] -= member_mass * gravity_m_s2 / 2

    ops.wipe()
    try:
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        for tag, coordinates in truss.nodes.items():
            ops.node(tag, *coordinates)
        ops.fix(truss.supports[0], 1, 1)
        ops.fix(truss.supports[1], 0, 1)
        ops.uniaxialMaterial("Elastic", 1, E_STEEL_PA)
        for member in truss.members:
            ops.element("truss", member.tag, member.start, member.end, member.area_m2, 1)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for tag, force in loads.items():
            ops.load(tag, *force)
        ops.constraints("Plain")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise RuntimeError("OpenSees truss static analysis failed.")
        displacements = {tag: (float(ops.nodeDisp(tag, 1)), float(ops.nodeDisp(tag, 2)))
                         for tag in truss.nodes}
        forces = {member.tag: float(ops.eleResponse(member.tag, "axialForce")[0])
                  for member in truss.members}
        if not all(math.isfinite(value) for pair in displacements.values() for value in pair) \
                or not all(math.isfinite(value) for value in forces.values()):
            raise RuntimeError("OpenSees returned non-finite truss results.")
        ops.reactions()
        reactions = {tag: (float(ops.nodeReaction(tag, 1)), float(ops.nodeReaction(tag, 2)))
                     for tag in truss.supports}
        return TrussResult(
            truss=truss,
            applied_loads_n={tag: (force[0], force[1]) for tag, force in loads.items()},
            displacements_m=displacements,
            reactions_n=reactions,
            member_forces_n=forces,
            member_stresses_pa={member.tag: forces[member.tag] / member.area_m2 for member in truss.members},
            max_deflection_m=max(math.hypot(*pair) for pair in displacements.values()),
            max_vertical_deflection_m=max(abs(pair[1]) for pair in displacements.values()),
            total_mass_kg=mass,
        )
    finally:
        ops.wipe()