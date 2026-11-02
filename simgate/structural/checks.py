"""Evaluate a placed structure without choosing or moving its site."""
from typing import Any

from .bridge import check_bridge
from . import frames
from .frames import ALUMINIUM_SECTIONS
from .truss import STEEL_AREAS_M2


def structure_section(record: dict[str, Any]) -> str:
    if record.get("section"):
        return record["section"]
    if record["kind"] == "bridge":
        return {"small": "SHS40x40x3", "medium": "SHS60x60x4", "large": "SHS80x80x4"}[record["size"]]
    return "SHS150x150x8" if record["kind"] == "habitat" else "SHS100x100x6"


def stronger_section(record: dict[str, Any]) -> str | None:
    catalogue = list(STEEL_AREAS_M2 if record["kind"] == "bridge" else ALUMINIUM_SECTIONS)
    section = structure_section(record)
    index = catalogue.index(section)
    return catalogue[index + 1] if index + 1 < len(catalogue) else None


def evaluate_structure(record: dict[str, Any], gravity_m_s2: float, robot_mass_kg: float = 0,
                       robot_label: str = "rover") -> dict[str, Any]:
    dimensions = record["dimensions"]
    section = structure_section(record)
    if record["kind"] == "bridge":
        return check_bridge(span_m=dimensions["span_m"], robot_mass_kg=robot_mass_kg,
                            gravity_m_s2=gravity_m_s2, section=section, robot_label=robot_label,
                            mid_support=record.get("mid_support", False))
    if record["kind"] == "habitat":
        return frames.check_habitat(module_length_m=dimensions["module_length_m"],
                             diameter_m=dimensions["module_diameter_m"], height_m=dimensions["module_height_m"],
                             gravity_m_s2=gravity_m_s2, section=section,
                             shielding_thickness_m=record.get("shielding_thickness_m", 0.5))
    if record["kind"] == "lander":
        return frames.check_lander(footprint_m=dimensions["footprint_m"], height_m=dimensions["height_m"],
                            gravity_m_s2=gravity_m_s2, section=section)
    raise ValueError(f"No structural check is available for {record['kind']!r}.")