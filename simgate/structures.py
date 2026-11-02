"""Deterministic, metadata-only placements for structures on robot terrain scenes."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .spec import SceneSpec, TerrainStructure

SIZE_SCALE = {"small": 0.75, "medium": 1.0, "large": 1.25}
MAX_PAD_SLOPE_DEG = 8.0
SEARCH_RADIUS_M = 6.0
PATH_CLEARANCE_M = 0.25
START_CLEARANCE_M = 0.25
STRUCTURE_CLEARANCE_M = 0.5
TERRAIN_SETBACK_M = 1.0
BEARING_M = 0.5
HABITAT_RAMP_LIMIT_DEG = 12.0
HABITAT_RAMP_RISE_M = 1.0
HABITAT_RAMP_WIDTH_M = 1.5


def _robot_mission(spec: SceneSpec, entry, heights: np.ndarray) -> dict:
    from .backends.mujoco_backend import resolve_mission

    robot_spec = spec.model_copy(deep=True)
    robot_spec.robot = entry.robot.model_copy(deep=True)
    robot_spec.mission = entry.mission()
    if robot_spec.mission.start_structure is not None:
        robot_spec.mission.start_xy, robot_spec.mission.start_place = None, "start_area"
    if robot_spec.mission.goal_structure is not None:
        robot_spec.mission.goal_xy, robot_spec.mission.goal_place = None, None
    return resolve_mission(robot_spec, heights)


def _local(point, center, yaw: float) -> tuple[float, float]:
    dx, dy = point[0] - center[0], point[1] - center[1]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return dx * cosine + dy * sine, -dx * sine + dy * cosine


def _point_rect_distance(point, center, yaw: float, half_length: float, half_width: float) -> float:
    along, across = _local(point, center, yaw)
    dx = max(abs(along) - half_length, 0.0)
    dy = max(abs(across) - half_width, 0.0)
    return math.hypot(dx, dy)


def _segment_intersects_rect(start, end, center, yaw: float, half_length: float, half_width: float) -> bool:
    start_local, end_local = _local(start, center, yaw), _local(end, center, yaw)
    dx, dy = end_local[0] - start_local[0], end_local[1] - start_local[1]
    lower, upper = 0.0, 1.0
    for p, q in ((-dx, start_local[0] + half_length), (dx, half_length - start_local[0]),
                 (-dy, start_local[1] + half_width), (dy, half_width - start_local[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        ratio = q / p
        if p < 0:
            if ratio > upper:
                return False
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return False
            upper = min(upper, ratio)
    return lower <= upper


def _slope_grid(heights: np.ndarray, size_m: float) -> np.ndarray:
    cell = size_m / (heights.shape[0] - 1)
    dy, dx = np.gradient(heights, cell, cell)
    return np.degrees(np.arctan(np.hypot(dx, dy)))


def _footprint_slopes(slope: np.ndarray, size_m: float, center, yaw: float,
                      length: float, width: float) -> np.ndarray | None:
    n = slope.shape[0]
    half, cell = size_m / 2, size_m / (n - 1)
    along_count = max(5, int(math.ceil(length / 0.4)) + 1)
    across_count = max(5, int(math.ceil(width / 0.4)) + 1)
    values = []
    for along in np.linspace(-length / 2, length / 2, along_count):
        for across in np.linspace(-width / 2, width / 2, across_count):
            x = center[0] + math.cos(yaw) * along - math.sin(yaw) * across
            y = center[1] + math.sin(yaw) * along + math.cos(yaw) * across
            if abs(x) > half - cell or abs(y) > half - cell:
                return None
            col = int(round((x + half) / cell))
            row = int(round((y + half) / cell))
            values.append(float(slope[row, col]))
    return np.asarray(values, dtype=float)


def _bridge_bearing_slopes(slope: np.ndarray, size_m: float, center, yaw: float,
                            opening: float, width: float) -> np.ndarray | None:
    values = []
    offset = opening / 2 + BEARING_M / 2
    for sign in (-1, 1):
        bearing = (center[0] + sign * math.cos(yaw) * offset,
                   center[1] + sign * math.sin(yaw) * offset)
        samples = _footprint_slopes(slope, size_m, bearing, yaw, BEARING_M, width)
        if samples is None:
            return None
        values.extend(samples.tolist())
    return np.asarray(values, dtype=float)


def _dimensions(request: TerrainStructure, widest_robot: float, body: str = "earth") -> dict[str, Any]:
    scale = SIZE_SCALE[request.size]
    if request.kind == "habitat":
        length, diameter = 6.0 * scale, 4.0 * scale
        rise = HABITAT_RAMP_RISE_M * scale
        run = rise / math.tan(math.radians(HABITAT_RAMP_LIMIT_DEG))
        ramp_width = HABITAT_RAMP_WIDTH_M * scale
        return {"record": {"module_length_m": round(length, 2), "module_diameter_m": round(diameter, 2),
                           "module_height_m": round(diameter, 2),
                           "airlock_rise_m": round(rise, 2),
                           "ramp": {"run_m": round(run, 2), "width_m": round(ramp_width, 2),
                                    "slope_deg": HABITAT_RAMP_LIMIT_DEG}},
                "length": length + run, "width": max(diameter, ramp_width), "offset": run / 2}
    if request.kind == "lander":
        insight = body == "mars" and (Path(__file__).resolve().parents[1] / "assets/nasa/insight_lander.glb").is_file()
        if insight:
            footprint = 6.0 * scale
            return {"record": {"height_m": round(2.25 * scale, 2), "footprint_m": round(footprint, 2),
                               "deck_span_m": round(1.5 * scale, 2), "deck_height_m": round(1.0 * scale, 2),
                               "body_height_m": round(0.35 * scale, 2), "leg_footprint_m": round(2.5 * scale, 2),
                               "legs": 3, "visual_model": "nasa_insight"},
                    "length": footprint, "width": footprint, "offset": 0.0}
        footprint = 4.5 * scale
        return {"record": {"height_m": round(4.0 * scale, 2), "footprint_m": round(footprint, 2), "legs": 4},
                "length": footprint, "width": footprint, "offset": 0.0}
    minimum_deck = 1.2 * widest_robot + 0.24
    deck_width = max(minimum_deck, minimum_deck * scale)
    return {"record": {"deck_width_m": round(deck_width, 2), "bearing_each_m": BEARING_M},
            "length": 0.0, "width": deck_width, "offset": 0.0}


def _candidate_centers(target, radius: float, step: float):
    span = int(math.floor(radius / step))
    candidates = []
    for row in range(-span, span + 1):
        for col in range(-span, span + 1):
            dx, dy = col * step, row * step
            distance = math.hypot(dx, dy)
            if distance <= radius + 1e-9:
                candidates.append((distance, (target[0] + dx, target[1] + dy)))
    candidates.sort(key=lambda item: (item[0], item[1][1], item[1][0]))
    return candidates


def footprint_inside_terrain(center, yaw: float, length: float, width: float, size_m: float) -> bool:
    extent_x = abs(math.cos(yaw)) * length / 2 + abs(math.sin(yaw)) * width / 2
    extent_y = abs(math.sin(yaw)) * length / 2 + abs(math.cos(yaw)) * width / 2
    limit = size_m / 2 - TERRAIN_SETBACK_M
    return abs(center[0]) + extent_x <= limit and abs(center[1]) + extent_y <= limit


def prepare_structure_terrain(spec: SceneSpec) -> None:
    if not spec.structures or spec.terrain is None:
        return
    if spec.terrain.kind == "real":
        if spec.terrain.structure_crop_size_m is None and len(spec.structures) > 1:
            from .terrain_data import load_structure_patch

            requested = max(80.0, spec.terrain.size_m * 2)
            _, meta = load_structure_patch(spec.terrain.preset, requested)
            previous = spec.terrain.size_m
            spec.terrain.structure_crop_size_m = requested
            spec.terrain.size_m = float(meta["size_m"])
            spec.assumptions.append(f"Selected a larger measured {meta['label']} patch: {previous:g} m to "
                                    f"{spec.terrain.size_m:g} m for full structure footprints and 1 m setbacks; "
                                    "source elevations were not stretched or flattened.")
        return
    dimensions = [_dimensions(item, max(entry.robot.chassis_width for entry in spec.robots))
                  for item in spec.structures]
    minimum = max(40.0 if len(spec.structures) > 1 else 20.0,
                  max(item["length"] + item["width"] + 2 * TERRAIN_SETBACK_M + 8 for item in dimensions))
    required = math.ceil(minimum / 10) * 10
    if spec.terrain.size_m < required:
        previous = spec.terrain.size_m
        spec.terrain.size_m = float(required)
        note = (f"Expanded generated terrain from {previous:g} m to {required:g} m so full structure "
                "footprints and their 1 m terrain setbacks fit inside the physics area.")
        if note not in spec.assumptions:
            spec.assumptions.append(note)


def _route(spec: SceneSpec, heights: np.ndarray):
    from .backends.mujoco_backend import footprint_radius

    if not spec.robots:
        return None
    missions = [_robot_mission(spec, entry, heights) for entry in spec.robots]
    primary = missions[0]
    start = tuple(primary["start"])
    goal = tuple(primary["goal"]) if primary["goal"] is not None else None
    heading = math.radians(primary["heading_deg"])
    route_goal = goal
    if route_goal is None and spec.obstacles:
        route_goal = (start[0] + 0.7 * spec.terrain.size_m * math.cos(heading),
                      start[1] + 0.7 * spec.terrain.size_m * math.sin(heading))
    yaw = (math.atan2(route_goal[1] - start[1], route_goal[0] - start[0])
           if route_goal is not None and math.dist(start, route_goal) > 1e-8 else heading)
    starts = [(tuple(mission["start"]), footprint_radius(entry.robot))
              for entry, mission in zip(spec.robots, missions)]
    return start, goal, route_goal, yaw, starts


def _path_gaps(spec: SceneSpec, heights: np.ndarray, start, route_goal, choose_site: bool = False):
    from .backends import obstacles as course

    if route_goal is None or not spec.obstacles:
        return []
    slots, _ = course.plan_course(spec, heights, start, route_goal, spec.terrain.seed)
    gaps = [slot for slot in slots if slot["ob"].kind == "gap" and slot["ob"].placement == "path"]
    if not gaps:
        return []
    if choose_site and spec.terrain.kind == "real" and len(gaps) == 1:
        slot = gaps[0]
        route_length = math.dist(start, route_goal)
        slope = _slope_grid(heights, spec.terrain.size_m)
        yaw = math.atan2(slot["d"][1], slot["d"][0])
        width = 1.2 * max(entry.robot.chassis_width + (2 * entry.robot.wheel_width + 0.02
                    if entry.robot.kind == "rover" else 0) for entry in spec.robots) + 0.24
        candidates = []
        for fraction in np.linspace(0.2, 0.8, 25):
            centre = np.asarray(start) + slot["d"] * route_length * fraction
            if min(math.dist(centre, start), math.dist(centre, route_goal)) < 3:
                continue
            values = _bridge_bearing_slopes(slope, spec.terrain.size_m, centre, yaw, slot["value"], width)
            if values is not None and footprint_inside_terrain(centre, yaw, slot["value"] + 2 * BEARING_M,
                                                               width, spec.terrain.size_m):
                candidates.append((float(values.max()), abs(fraction - 0.5), float(fraction), centre))
        if candidates:
            _, _, fraction, centre = min(candidates, key=lambda item: item[:2])
            slot["ob"].path_fraction = fraction
            slot["centre"], slot["along"] = centre, route_length * fraction
    lowered = heights.copy()
    course.cut_gaps(gaps, lowered, spec.terrain.size_m)
    return [{"slot": slot,
             "opening": course.measure_gap(lowered, spec.terrain.size_m, *slot["centre"], slot["d"])}
            for slot in gaps]


def _target(request: TerrainStructure, spec: SceneSpec, heights: np.ndarray, start, goal, gaps, used_gaps):
    from .backends.mujoco_backend import place_xy

    if request.placement == "near_start":
        return start, "start", None
    if request.placement == "near_goal":
        return (goal, "goal", None) if goal is not None else (None, "goal", "there is no goal in this scene")
    if request.placement == "over_gap":
        if request.kind != "bridge":
            return None, "gap", "only a bridge can use over_gap placement"
        gap = next((item for item in gaps if tuple(item["slot"]["centre"]) not in used_gaps), None)
        if gap is None:
            return None, "gap", "there is no path gap obstacle to span"
        return tuple(gap["slot"]["centre"]), "gap", gap
    if request.kind == "bridge":
        return None, request.placement, "a bridge must use over_gap placement"
    return place_xy(heights, spec.terrain.size_m, request.placement), request.placement, None


def place_structures(spec: SceneSpec, heights: np.ndarray) -> tuple[list[dict], list[str]]:
    """Place descriptive terrain-structure metadata without modifying the spec or heightfield."""
    if not spec.structures:
        return [], []
    if spec.domain != "robot_terrain" or spec.terrain is None:
        return [], ["Terrain structures need a robot terrain scene; none were placed."]
    if not spec.robots:
        return [], ["No robot spawn was available to place terrain structures safely."]

    from .backends import obstacles as course
    from .backends.mujoco_backend import EDGE_MARGIN, height_at

    route = _route(spec, heights)
    if route is None:
        return [], ["No robot route was available to place terrain structures safely."]
    start, goal, route_goal, route_yaw, starts = route
    gaps = _path_gaps(spec, heights, start, route_goal)
    slope = _slope_grid(heights, spec.terrain.size_m)
    widest_robot = max(entry.robot.chassis_width + (2 * entry.robot.wheel_width + 0.02
                        if entry.robot.kind == "rover" else 0) for entry in spec.robots)
    cell = spec.terrain.size_m / (heights.shape[0] - 1)
    candidate_step = max(0.5, cell * 4)
    size_m = spec.terrain.size_m
    placed: list[dict] = []
    notes: list[str] = []
    used_gaps: set[tuple[float, float]] = set()
    occupied: list[tuple[tuple[float, float], float]] = []

    requests = sorted(enumerate(spec.structures), key=lambda item: item[1].kind == "bridge")
    resolved_bridge_route = False
    for request_index, request in requests:
        if request.kind == "bridge" and not resolved_bridge_route:
            resolved_bridge_route = True
            primary = spec.robots[0]
            if primary.goal_structure is not None:
                route_spec = spec.model_copy(deep=True)
                route_spec.robots[0].goal_xy = structure_goal_xy(placed, primary.goal_structure)
                route_spec.robots[0].goal_place = None
                route_spec.robots[0].goal_structure = None
                if primary.start_structure == "lander":
                    route_spec.robots[0].start_xy = structure_goal_xy(placed, "lander")
                    route_spec.robots[0].start_place = None
                    route_spec.robots[0].start_structure = None
                start, goal, route_goal, route_yaw, starts = _route(route_spec, heights)
                gaps = _path_gaps(route_spec, heights, start, route_goal, choose_site=True)
                for original, selected in zip(spec.obstacles, route_spec.obstacles):
                    original.path_fraction = selected.path_fraction
        target, target_name, extra = _target(request, spec, heights, start, goal, gaps, used_gaps)
        if target is None:
            notes.append(f"{request.kind.title()} not placed: {extra}.")
            continue
        dimensions = _dimensions(request, widest_robot, spec.environment.body)
        length, width, offset = dimensions["length"], dimensions["width"], dimensions["offset"]
        bridge_gap = extra if request.placement == "over_gap" else None
        if bridge_gap:
            slot = bridge_gap["slot"]
            yaw = math.atan2(slot["d"][1], slot["d"][0])
            center = tuple(slot["centre"])
            opening = float(bridge_gap["opening"])
            dimensions["record"].update({"gap_opening_m": round(opening, 2),
                                         "span_m": round(opening + 2 * BEARING_M, 2)})
            length = opening + 2 * BEARING_M
        else:
            yaw = route_yaw + math.pi if request.kind == "habitat" and request.placement == "near_goal" else route_yaw

        best = None
        best_site_slope = None
        search_radius = 0.0 if bridge_gap else (max(SEARCH_RADIUS_M, size_m * 0.6)
                               if spec.terrain.kind == "real" else SEARCH_RADIUS_M)
        for distance, footprint_center in _candidate_centers(target, search_radius, candidate_step):
            if not footprint_inside_terrain(footprint_center, yaw, length, width, size_m):
                continue
            module_center = (footprint_center[0] - math.cos(yaw) * offset,
                             footprint_center[1] - math.sin(yaw) * offset)
            if bridge_gap:
                opening = bridge_gap["opening"]
                pad_offset = opening / 2 + BEARING_M / 2
                bearing_values = []
                for sign in (-1, 1):
                    bearing_center = (center[0] + sign * math.cos(yaw) * pad_offset,
                                      center[1] + sign * math.sin(yaw) * pad_offset)
                    samples = _footprint_slopes(slope, size_m, bearing_center, yaw, BEARING_M, width)
                    if samples is None:
                        bearing_values = []
                        break
                    bearing_values.extend(samples.tolist())
                values = np.asarray(bearing_values) if bearing_values else None
            else:
                values = _footprint_slopes(slope, size_m, footprint_center, yaw, length, width)
            if values is None:
                continue
            slope_score = (float(values.max()), float(values.mean()))
            if best_site_slope is None or slope_score < best_site_slope:
                best_site_slope = slope_score
            half_length, half_width = length / 2, width / 2
            if any(_point_rect_distance(spawn, footprint_center, yaw, half_length, half_width)
                   < radius + START_CLEARANCE_M for spawn, radius in starts):
                continue
            named_goal = goal is not None and math.dist(target, goal) < 0.25
            if request.placement not in ("near_goal", "over_gap") and not named_goal and goal is not None and \
                    _segment_intersects_rect(start, goal, footprint_center, yaw,
                                             half_length + PATH_CLEARANCE_M, half_width + PATH_CLEARANCE_M):
                continue
            footprint_radius = math.hypot(half_length, half_width)
            if any(math.dist(footprint_center, other_center) <
                   footprint_radius + other_radius + STRUCTURE_CLEARANCE_M
                   for other_center, other_radius in occupied):
                continue
            if slope_score[0] > MAX_PAD_SLOPE_DEG:
                continue
            ranked = (slope_score[0], slope_score[1], distance, module_center[1], module_center[0])
            if best is None or ranked < best[0]:
                best = (ranked, module_center, footprint_center, slope_score, distance)

        if best is None:
            slope_note = f"; flattest available footprint was {best_site_slope[0]:.1f}°" if best_site_slope else ""
            notes.append(f"{request.kind.title()} not placed near the {target_name}: no clear pad within "
                         f"{SEARCH_RADIUS_M:g} m at or below {MAX_PAD_SLOPE_DEG:g}°{slope_note}.")
            continue

        _, module_center, footprint_center, slope_score, distance = best
        if request.kind == "habitat":
            body_radius = math.hypot(dimensions["record"]["module_length_m"] / 2,
                                     dimensions["record"]["module_diameter_m"] / 2)
        elif request.kind == "lander":
            body_radius = dimensions["record"]["footprint_m"] / math.sqrt(2)
        else:
            body_radius = 0.45
        z = height_at(heights, size_m, *module_center, radius_m=body_radius)
        if bridge_gap:
            pad_offset = bridge_gap["opening"] / 2 + BEARING_M / 2
            bearing_points = [
                (center[0] + sign * math.cos(yaw) * pad_offset,
                 center[1] + sign * math.sin(yaw) * pad_offset)
                for sign in (-1, 1)
            ]
            bearing_heights = [course.ground(heights, size_m, *point) for point in bearing_points]
            z = sum(bearing_heights) / 2
            span = bridge_gap["opening"] + 2 * BEARING_M
            pitch = math.degrees(math.atan2(bearing_heights[0] - bearing_heights[1], span))
        else:
            bearing_points, bearing_heights, pitch = [], [], 0.0
        record = {
            "request_index": request_index,
            "kind": request.kind, "size": request.size, "material": request.material,
            "section": request.section, "mid_support": request.mid_support,
            "shielding_thickness_m": request.shielding_thickness_m,
            "placement": request.placement,
            "position": {"x": round(module_center[0], 2), "y": round(module_center[1], 2), "z": round(z, 2)},
            "orientation": {"yaw_deg": round(math.degrees(yaw), 1), "pitch_deg": round(pitch, 2)},
            "footprint": {"center_xy": [round(value, 2) for value in footprint_center],
                          "length_m": round(length, 2), "width_m": round(width, 2)},
            "ground_slope_deg": round(slope_score[0], 1),
            "distance_to_target_m": round(distance, 1),
            "dimensions": dimensions["record"],
        }
        access_points = {}
        if request.kind == "habitat":
            run = float(dimensions["record"]["ramp"]["run_m"])
            rise = float(dimensions["record"]["airlock_rise_m"])
            airlock_xy = (module_center[0] + math.cos(yaw) *
                          (dimensions["record"]["module_length_m"] / 2 + 0.2),
                          module_center[1] + math.sin(yaw) *
                          (dimensions["record"]["module_length_m"] / 2 + 0.2))
            access_points["habitat_airlock"] = {"x": round(airlock_xy[0], 2), "y": round(airlock_xy[1], 2),
                                                 "z": round(z + rise, 2)}
        elif request.kind == "lander":
            approach_offset = dimensions["record"]["footprint_m"] / 2 + 0.75
            approach_xy = (module_center[0] - math.cos(yaw) * approach_offset,
                           module_center[1] - math.sin(yaw) * approach_offset)
            safe_edge = size_m / 2 * EDGE_MARGIN
            approaches = [(module_center[0] + math.cos(yaw + angle) * approach_offset,
                           module_center[1] + math.sin(yaw + angle) * approach_offset)
                          for angle in (math.pi, 0, math.pi / 2, -math.pi / 2)]
            approach_xy = next((point for point in approaches
                                if max(abs(point[0]), abs(point[1])) <= safe_edge), None)
            if approach_xy is None:
                notes.append("Lander not placed: there is no clear adjacent access point inside the spawn bounds.")
                continue
            access_points["lander"] = {"x": round(approach_xy[0], 2), "y": round(approach_xy[1], 2),
                                        "z": round(height_at(heights, size_m, *approach_xy), 2)}
        if access_points:
            record["access_points"] = access_points
        if bridge_gap:
            slot = bridge_gap["slot"]
            record["gap"] = {"obstacle": slot["ob"].kind,
                              "center": [round(value, 2) for value in slot["centre"]],
                              "opening_m": round(bridge_gap["opening"], 2),
                              "bearings_each_m": BEARING_M,
                              "bearings": [{"x": round(point[0], 2), "y": round(point[1], 2),
                                            "z": round(elevation, 3)}
                                           for point, elevation in zip(bearing_points, bearing_heights)]}
            used_gaps.add(tuple(slot["centre"]))
            notes.append(f"Bridge spans a {bridge_gap['opening']:.2f} m path gap "
                         f"(span {record['dimensions']['span_m']:.2f} m, 0.5 m bearings).")
        else:
            occupied.append((footprint_center, math.hypot(length / 2, width / 2)))
            notes.append(f"{request.kind.title()} on {slope_score[0]:.1f}° ground, {distance:.0f} m from the {target_name}.")
        placed.append(record)

    return placed, notes


def structure_goal_xy(placements: list[dict], target: str) -> list[float]:
    """Resolve a named structure goal to a navigable access point in placement metadata."""
    for record in placements:
        point = (record.get("access_points") or {}).get(target)
        if point is not None:
            return [float(point["x"]), float(point["y"])]
    raise ValueError(f"No placed structure provides the requested goal {target!r}.")


def structure_geoms(placements: list[dict]) -> str:
    """MJCF worldbody geoms for static structure collisions, using the metadata from place_structures()."""
    xml = []

    def quat(yaw_deg: float, pitch_deg: float = 0.0) -> str:
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        qz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        qy = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = qy
        result = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                  w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
        return " ".join(f"{value:.6f}" for value in result)

    def box(name: str, size, pos, yaw: float, pitch: float = 0.0) -> None:
        xml.append(f'    <geom name="{name}" type="box" size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" '
                   f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{quat(yaw, pitch)}" '
                   'contype="1" conaffinity="1" rgba="0.48 0.49 0.50 1"/>')

    for index, record in enumerate(placements):
        kind = record["kind"]
        x, y, base_z = (float(record["position"][key]) for key in ("x", "y", "z"))
        yaw = float(record["orientation"].get("yaw_deg", 0.0))
        yaw_rad = math.radians(yaw)
        cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
        dimensions = record["dimensions"]
        prefix = f"struct{index}_{kind}"

        if kind == "habitat":
            length = float(dimensions["module_length_m"])
            diameter = float(dimensions["module_diameter_m"])
            height = float(dimensions["module_height_m"])
            rise = float(dimensions["airlock_rise_m"])
            ramp = dimensions["ramp"]
            radius = diameter / 2
            module_pos = (x, y, base_z + radius)
            xml.append(f'    <geom name="{prefix}_body" type="cylinder" '
                       f'size="{radius:.4f} {length / 2:.4f}" '
                       f'pos="{module_pos[0]:.4f} {module_pos[1]:.4f} {module_pos[2]:.4f}" '
                       f'quat="{quat(yaw, 90.0)}" contype="1" conaffinity="1" '
                       'rgba="0.58 0.59 0.60 1"/>')
            run, ramp_width = float(ramp["run_m"]), float(ramp["width_m"])
            ramp_center = (x + cosine * (length / 2 + run / 2),
                           y + sine * (length / 2 + run / 2), base_z + rise / 2)
            pitch = math.degrees(math.atan2(rise, run))
            box(f"{prefix}_ramp", (run / 2, ramp_width / 2, 0.06), ramp_center, yaw, pitch)

        elif kind == "lander":
            footprint = float(dimensions["footprint_m"])
            leg_count = int(dimensions["legs"])
            body_length = body_width = float(dimensions.get("deck_span_m", footprint * 0.48))
            body_height = float(dimensions.get("body_height_m", 1.5))
            body_bottom = (base_z + float(dimensions["deck_height_m"]) - body_height / 2
                           if "deck_height_m" in dimensions else base_z + float(dimensions["height_m"]) - body_height)
            box(f"{prefix}_body", (body_length / 2, body_width / 2, body_height / 2),
                (x, y, body_bottom + body_height / 2), yaw)
            inner = body_length * 0.45
            outer = float(dimensions.get("leg_footprint_m", footprint)) / 2 - 0.2
            pad_half, pad_height = 0.28, 0.12
            for leg in range(leg_count):
                angle = yaw_rad + math.radians(45 + 360 * leg / leg_count)
                inner_xy = (x + inner * math.cos(angle), y + inner * math.sin(angle))
                pad_xy = (x + outer * math.cos(angle), y + outer * math.sin(angle))
                pad_ground = base_z
                leg_top = (inner_xy[0], inner_xy[1], body_bottom)
                leg_bottom = (pad_xy[0], pad_xy[1], pad_ground + pad_height)
                xml.append(f'    <geom name="{prefix}_leg{leg}" type="capsule" '
                           f'fromto="{leg_top[0]:.4f} {leg_top[1]:.4f} {leg_top[2]:.4f} '
                           f'{leg_bottom[0]:.4f} {leg_bottom[1]:.4f} {leg_bottom[2]:.4f}" '
                           'size="0.12" contype="1" conaffinity="1" rgba="0.48 0.49 0.50 1"/>')
                box(f"{prefix}_pad{leg}", (pad_half, pad_half, pad_height / 2),
                    (pad_xy[0], pad_xy[1], base_z + pad_height / 2), yaw)
            if dimensions.get("visual_model") == "nasa_insight":
                radius = (footprint - body_length) / 4
                for side in (-1, 1):
                    across = side * (body_length / 2 + radius)
                    panel_center = (x + cosine * across, y + sine * across,
                                    base_z + dimensions["deck_height_m"])
                    xml.append(f'    <geom name="{prefix}_solar_{side}" type="cylinder" '
                               f'size="{radius:.4f} 0.025" pos="{panel_center[0]:.4f} {panel_center[1]:.4f} {panel_center[2]:.4f}" '
                               'contype="1" conaffinity="1" rgba="0.16 0.22 0.28 1"/>')

        elif kind == "bridge":
            gap = record["gap"]
            span = float(dimensions["span_m"])
            deck_width = float(dimensions["deck_width_m"])
            thickness = 0.2
            pitch = float(record["orientation"].get("pitch_deg", 0.0))
            deck_center = (x, y, base_z - thickness / 2 + 0.03)
            box(f"{prefix}_deck", (span / 2, deck_width / 2, thickness / 2), deck_center, yaw, pitch)
            rail_width, rail_height = 0.12, 0.25
            if record.get("mid_support"):
                box(f"{prefix}_mid_support", (0.15, 0.15, 0.22),
                    (x, y, base_z - 0.38), yaw)
            for side, sign in (("left", -1), ("right", 1)):
                across = sign * (deck_width / 2 - rail_width / 2)
                rail_center = (x - sine * across,
                               y + cosine * across,
                               base_z + 0.03 + rail_height / 2)
                box(f"{prefix}_rail_{side}", (span / 2, rail_width / 2, rail_height / 2),
                    rail_center, yaw, pitch)

    return "\n".join(xml)
