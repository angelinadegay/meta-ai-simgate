"""Metre-scale, JSON-compatible indoor layouts built from static boxes."""
import copy
import math
import re

import numpy as np

ROOM_PLACES = {"kitchen", "living_room", "hallway", "bedroom"}


def requested(text):
    building = re.search(r"\b(?:earthquake|tower|frame|apartment block)\b|\b\d+[- ]stor(?:y|ey|ies)", text, re.I)
    if building and not re.search(r"\b(?:robot|rover|humanoid|inside|indoors?)\b", text, re.I):
        return False
    return bool(re.search(r"\bapartment\b|\bindoor[s]?\b|\b(?:in|inside) (?:a |the )?house\b|\bkitchen\b|\bliving room\b|\bbedroom\b", text, re.I))


def requested_places(text):
    rooms = r"kitchen|living[ _]room|hallway|bedroom"
    start = re.search(rf"\bfrom (?:the )?({rooms})\b", text, re.I)
    goal = re.search(rf"\bto (?:the )?({rooms})\b", text, re.I)
    return (start.group(1).lower().replace(" ", "_") if start else None,
            goal.group(1).lower().replace(" ", "_") if goal else None)


def requested_stops(text):
    """Ordered destinations explicitly requested with 'to ..., then to ...'."""
    return [match.group(1).lower().replace(" ", "_") for match in re.finditer(
        r"\bto (?:the )?(kitchen|living[ _]room|hallway|bedroom)\b", text, re.I)]


def _apartment():
    layout = {"id": "apartment", "label": "Apartment", "size_m": [14.0, 10.0],
              "wall_height_m": 2.6, "wall_thickness_m": 0.12,
              "places": {"kitchen": [-4.5, -2.5], "living_room": [2.4, -2.5],
                         "hallway": [0, 0], "bedroom": [-4.5, 3.2]}, "geoms": [],
              "doorways": [{"id": "kitchen_living", "width_m": 0.9, "threshold_m": 0.018},
                           {"id": "kitchen_hall", "width_m": 0.88, "threshold_m": 0.015},
                           {"id": "living_hall", "width_m": 0.84, "threshold_m": 0.018},
                           {"id": "bedroom_hall", "width_m": 0.85, "threshold_m": 0.020},
                           {"id": "ramp_access", "width_m": 1.2, "threshold_m": 0.015},
                           {"id": "narrow", "width_m": 0.65, "threshold_m": 0.015}]}

    def box(name, size, position, surface, friction=0.5, pitch=0, priority=1):
        layout["geoms"].append({"name": f"indoor_{name}", "size_m": list(size), "position": list(position),
                                 "surface": surface, "friction": friction, "pitch_deg": pitch, "priority": priority})

    def wall(name, start, end):
        size = [abs(end[0] - start[0]) or 0.12, abs(end[1] - start[1]) or 0.12, 2.6]
        box(name, size, [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 1.3], "wall")

    box("floor_base", [14, 10, 0.1], [0, 0, -0.05], "base", priority=0)
    box("floor_kitchen_tile", [5.88, 4.28, 0.002], [-4, -2.8, 0.001], "tile", 0.6)
    box("floor_living_wood", [7.88, 4.28, 0.002], [3, -2.8, 0.001], "wood", 0.5)
    box("floor_hall_wood", [13.88, 1.2, 0.002], [0, 0, 0.001], "wood", 0.5)
    box("floor_north_wood", [7.88, 4.28, 0.002], [3, 2.8, 0.001], "wood", 0.5)
    box("floor_bedroom_approach", [5.88, 1.2, 0.002], [-4, 1.26, 0.001], "wood", 0.5)
    box("floor_rug", [3, 2, 0.008], [2.5, -2.5, 0.006], "rug", 0.8, priority=2)
    box("floor_wet", [1.1, 0.8, 0.002], [-5.7, -4.1, 0.003], "wet", 0.15, priority=2)
    box("bedroom_platform", [5.88, 3.08, 0.51], [-4, 3.4, 0.255], "wood", 0.5)
    box("floor_bedroom_wood", [5.88, 3.08, 0.002], [-4, 3.4, 0.511], "wood", 0.5)
    for step in range(3):
        height = (step + 1) * 0.17
        box(f"bedroom_step_{step + 1}", [0.85, 0.4, height], [-4.5, 0.86 + step * 0.4, height / 2], "wood", 0.5)
    for name, start, end in [("wall_south", [-7, -5], [7, -5]), ("wall_north", [-7, 5], [7, 5]),
                              ("wall_west", [-7, -5], [-7, 5]), ("wall_east", [7, -5], [7, 5]),
                              ("wall_kitchen_lower", [-1, -4.94], [-1, -2.95]),
                              ("wall_kitchen_upper", [-1, -2.05], [-1, -0.6]),
                              ("wall_bedroom_east_lower", [-1, 0.6], [-1, 2.4]),
                              ("wall_bedroom_east_upper", [-1, 4.0], [-1, 4.94])]:
        wall(name, start, end)
    for side, y, openings in [("south", -0.6, [(-4.94, -4.06), (2.08, 2.92)]),
                               ("north", 0.6, [(-4.925, -4.075), (2.0, 3.2), (4.675, 5.325)])]:
        cursor = -7
        for index, (left, right) in enumerate(openings):
            wall(f"wall_hall_{side}_{index}", [cursor, y], [left, y])
            threshold = 0.020 if side == "north" and index == 0 else 0.018 if side == "south" and index == 1 else 0.015
            box(f"threshold_{side}_{index}", [right - left, 0.12, threshold],
                [(left + right) / 2, y, threshold / 2], "threshold", 0.6)
            cursor = right
        wall(f"wall_hall_{side}_end", [cursor, y], [7, y])
    box("threshold_kitchen_living", [0.12, 0.9, 0.018], [-1, -2.5, 0.009], "threshold", 0.6)
    rise, angle, thickness = 0.512, math.radians(6), 0.08
    run = rise / math.tan(angle)
    high_x = -1.25
    # The upper face joins the bedroom platform through the east-wall opening.
    box("ramp", [math.hypot(run, rise), 1.4, thickness],
        [high_x + run / 2 - thickness / 2 * math.sin(angle), 3.2,
         rise / 2 - thickness / 2 * math.cos(angle)], "ramp", 0.6, pitch=6)
    box("kitchen_counter", [2.2, 0.65, 0.9], [-5.6, -4.55, 0.45], "counter", 0.6)
    return layout


LAYOUTS = {"apartment": _apartment()}


def get_layout(preset="apartment"):
    if preset not in LAYOUTS:
        raise ValueError(f"Unknown indoor layout {preset!r}; choose apartment.")
    return copy.deepcopy(LAYOUTS[preset])


def height_grid(preset, resolution):
    layout = get_layout(preset)
    size = max(layout["size_m"])
    axis = np.linspace(-size / 2, size / 2, resolution)
    east, north = np.meshgrid(axis, axis)
    heights = np.zeros_like(east)
    for geom in layout["geoms"]:
        if geom["surface"] in {"wall", "counter", "base"}:
            continue
        x, y, z = geom["position"]
        length, width, thickness = geom["size_m"]
        inside = (np.abs(east - x) <= length / 2) & (np.abs(north - y) <= width / 2)
        top = z + thickness / 2 - (east - x) * math.tan(math.radians(geom["pitch_deg"]))
        heights[inside] = np.maximum(heights[inside], top[inside])
    return heights


def geometry_xml(preset):
    colors = {"wall": "0.82 0.84 0.81", "tile": "0.72 0.77 0.75", "wood": "0.49 0.37 0.25",
              "rug": "0.23 0.43 0.44", "wet": "0.48 0.66 0.71", "counter": "0.48 0.53 0.52"}
    parts = []
    for geom in get_layout(preset)["geoms"]:
        size = " ".join(f"{value / 2:.6f}" for value in geom["size_m"])
        position = " ".join(f"{value:.6f}" for value in geom["position"])
        color = colors.get(geom["surface"], colors["wood"])
        pitch = math.radians(geom["pitch_deg"]) / 2
        parts.append(f'<geom name="{geom["name"]}" type="box" size="{size}" pos="{position}" '
                 f'quat="{math.cos(pitch):.8f} 0 {math.sin(pitch):.8f} 0" friction="{geom["friction"]} 0.005 0.0001" '
                     f'priority="{geom["priority"]}" contype="1" conaffinity="1" rgba="{color} 1"/>')
    return "\n".join(parts)