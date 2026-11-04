"""Footprint-safe routes over corrected traversability maps and paired driving evidence."""
from __future__ import annotations

import heapq
import math
from typing import Any, Callable

import numpy as np

from .spec import RobotEntry, SceneSpec
from .traversability import _indoor_geometry, _robot_dimensions


def segment_cells(grid: dict, start, end) -> list[tuple[int, int]] | None:
    """Supercover of a segment, including both cells at an edge or corner."""
    for cx, cy, yaw, half_length, half_width in grid.get("blockers", []):
        radius = grid.get("radius_m", 0.0)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        origin = ((start[0] - cx) * cosine + (start[1] - cy) * sine,
                  -(start[0] - cx) * sine + (start[1] - cy) * cosine)
        velocity = ((end[0] - start[0]) * cosine + (end[1] - start[1]) * sine,
                    -(end[0] - start[0]) * sine + (end[1] - start[1]) * cosine)
        near, far = 0.0, 1.0
        for coordinate, speed, extent in zip(origin, velocity, (half_length + radius, half_width + radius)):
            if abs(speed) < 1e-12:
                if abs(coordinate) > extent:
                    near = math.inf
                    break
            else:
                left, right = (-extent - coordinate) / speed, (extent - coordinate) / speed
                near, far = max(near, min(left, right)), min(far, max(left, right))
        if near <= far:
            return None
    cell = float(grid["cell_size_m"])
    xs, ys = grid["x_m"], grid["y_m"]
    x0, y0 = float(xs[0]) - cell / 2, float(ys[0]) - cell / 2
    a = ((float(start[0]) - x0) / cell, (float(start[1]) - y0) / cell)
    b = ((float(end[0]) - x0) / cell, (float(end[1]) - y0) / cell)
    if any(not (0 <= p[0] < len(xs) and 0 <= p[1] < len(ys)) for p in (a, b)):
        return None
    cuts = {0.0, 1.0}
    for axis in (0, 1):
        delta = b[axis] - a[axis]
        if abs(delta) > 1e-12:
            for boundary in range(math.ceil(min(a[axis], b[axis])), math.floor(max(a[axis], b[axis])) + 1):
                t = (boundary - a[axis]) / delta
                if 0 < t < 1:
                    cuts.add(t)
    cuts = sorted(cuts)
    samples = cuts + [(left + right) / 2 for left, right in zip(cuts, cuts[1:])]
    cells = set()
    for t in samples:
        x, y = (a[axis] + (b[axis] - a[axis]) * t for axis in (0, 1))
        cols = {math.floor(x)}
        rows = {math.floor(y)}
        if abs(x - round(x)) < 1e-9:
            cols.add(round(x) - 1)
        if abs(y - round(y)) < 1e-9:
            rows.add(round(y) - 1)
        cells.update((row, col) for row in rows for col in cols
                     if 0 <= row < len(ys) and 0 <= col < len(xs))
    return sorted(cells)


def _search(allowed: np.ndarray, costs: np.ndarray, start, goal, edge_clear=None):
    queue = [(math.dist(start, goal), 0.0, start)]
    best_cost = {start: 0.0}
    parents = {}
    closest = start
    while queue:
        _, distance, current = heapq.heappop(queue)
        if distance != best_cost[current]:
            continue
        if (math.dist(current, goal), distance) < (math.dist(closest, goal), best_cost[closest]):
            closest = current
        if current == goal:
            closest = goal
            break
        row, col = current
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]) or not allowed[nr, nc]:
                continue
            if dr and dc and (not allowed[row, nc] or not allowed[nr, col]):
                continue
            if edge_clear is not None and not edge_clear(current, (nr, nc)):
                continue
            candidate = distance + math.hypot(dr, dc) * (costs[row, col] + costs[nr, nc]) / 2
            if candidate < best_cost.get((nr, nc), math.inf):
                best_cost[nr, nc] = candidate
                parents[nr, nc] = current
                heapq.heappush(queue, (candidate + math.dist((nr, nc), goal), candidate, (nr, nc)))
    path = [closest]
    while path[-1] in parents:
        path.append(parents[path[-1]])
    return path[::-1], closest == goal


def _clearance_mask(scene: SceneSpec, grid: dict, robot) -> np.ndarray:
    allowed = np.isin(grid["classes"], ("green", "amber")) & ~np.asarray(grid["blocked"], dtype=bool)
    length, width = _robot_dimensions(robot)
    radius = width / 2 + 0.01
    xs, ys = np.meshgrid(grid["x_m"], grid["y_m"])
    if scene.terrain.kind == "indoor":
        from .indoor import get_layout

        layout = get_layout(scene.terrain.preset)
        sx, sy = layout["size_m"]
        allowed &= (np.abs(xs) + radius < sx / 2) & (np.abs(ys) + radius < sy / 2)
        blockers, _ = _indoor_geometry(scene.terrain.preset)
        # Features use a fixed footprint; reserve its rotational sweep beside failed terrain.
        turn_margin = math.hypot(length, width) / 2 - min(length, width) / 2
        red = np.asarray(grid["classes"]) == "red"
        cell = float(grid["cell_size_m"])
        reach = math.ceil(turn_margin / cell + 0.5)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                if math.hypot(max(abs(dr) - 0.5, 0), max(abs(dc) - 0.5, 0)) * cell > turn_margin:
                    continue
                r0, r1 = max(0, -dr), min(allowed.shape[0], allowed.shape[0] - dr)
                c0, c1 = max(0, -dc), min(allowed.shape[1], allowed.shape[1] - dc)
                allowed[r0:r1, c0:c1] &= ~red[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        for cx, cy, yaw, hl, hw in blockers:
            along = (xs - cx) * math.cos(yaw) + (ys - cy) * math.sin(yaw)
            across = -(xs - cx) * math.sin(yaw) + (ys - cy) * math.cos(yaw)
            allowed &= ~((np.abs(along) <= hl + radius) & (np.abs(across) <= hw + radius))
    else:
        # Outdoor grey cells already include footprint overlap; expand physical grey
        # cell extents as well so smoothing cannot graze their corners.
        original = np.asarray(grid["blocked"], dtype=bool)
        cell = float(grid["cell_size_m"])
        reach = math.ceil(radius / cell + 0.5)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                if math.hypot(max(abs(dr) - 0.5, 0), max(abs(dc) - 0.5, 0)) * cell > radius:
                    continue
                r0, r1 = max(0, -dr), min(allowed.shape[0], allowed.shape[0] - dr)
                c0, c1 = max(0, -dc), min(allowed.shape[1], allowed.shape[1] - dc)
                allowed[r0:r1, c0:c1] &= ~original[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return allowed


WALL_SWEEP_COST = 2.0   # per cell, beside a wall: below an amber cell's 5, so a doorway is still used


def _near_walls(grid: dict, blockers: list[tuple], robot) -> np.ndarray:
    """Cells where turning the robot in place would sweep its corners into a wall (within half its footprint
    diagonal). Costlier, not forbidden: routes keep to the middle of doorways and corridors, where a long robot
    can make its turn, and still use a doorway when there is no other way."""
    length, width = _robot_dimensions(robot)
    sweep = math.hypot(length, width) / 2
    xs, ys = np.meshgrid(grid["x_m"], grid["y_m"])
    near = np.zeros(xs.shape, dtype=bool)
    for cx, cy, yaw, hl, hw in blockers:
        along = (xs - cx) * math.cos(yaw) + (ys - cy) * math.sin(yaw)
        across = -(xs - cx) * math.sin(yaw) + (ys - cy) * math.cos(yaw)
        near |= np.hypot(np.maximum(np.abs(along) - hl, 0), np.maximum(np.abs(across) - hw, 0)) < sweep
    return near


def _blocking_reason(scene: SceneSpec, grid: dict, start, goal, allowed) -> str:
    relaxed = np.where(np.asarray(grid["classes"]) == "red", 20.0, 1.0)
    relaxed[~allowed & (np.asarray(grid["classes"]) != "red")] = 100.0
    path, _ = _search(np.ones(allowed.shape, dtype=bool), relaxed, start, goal)
    blockers = [cell for cell in path if not allowed[cell]]
    for row, col in blockers:
        step = float(grid.get("max_step_m", np.zeros(allowed.shape))[row, col])
        limit = grid.get("limits", {}).get("step", {}).get("reliable_ok")
        first_fail = grid.get("limits", {}).get("step", {}).get("first_fail")
        if (limit is not None and step > limit) or (first_fail is not None and step >= first_fail):
            capability = (f"this robot climbs about {limit:.2f} m" if limit is not None else
                          f"this robot first failed at a {first_fail:.2f} m step")
            if scene.terrain.kind == "indoor":
                from .indoor import get_layout

                steps = [geom for geom in get_layout(scene.terrain.preset)["geoms"]
                         if geom["name"].startswith("indoor_bedroom_step_")]
                gx, gy = grid["x_m"][goal[1]], grid["y_m"][goal[0]]
                if steps and gx < -1 and gy > 1.8:
                    rise = min(geom["size_m"][2] for geom in steps)
                    return (f"No safe route: the direct bedroom approach has {len(steps)} steps of "
                            f"{rise:.2f} m; {capability}.")
            return f"No safe route: a {step:.2f} m step blocks the way; {capability}."
        friction = float(grid.get("friction", np.ones(allowed.shape))[row, col])
        fail = grid.get("limits", {}).get("friction", {}).get("first_fail")
        if fail is not None and friction <= fail:
            return (f"No safe route: the wet/slippery patch (friction {friction:.2f}) blocks the way; "
                    f"this robot first failed at friction {fail:.2f}.")
        for feature, key, unit in (("slope", "slope_deg", "degrees"), ("roughness", "roughness_m", "m")):
            value = float(grid.get(key, np.zeros(allowed.shape))[row, col])
            fail = grid.get("limits", {}).get("slope_deg" if feature == "slope" else feature, {}).get("first_fail")
            if fail is not None and value >= fail:
                return f"No safe route: {feature} of {value:.2f} {unit} exceeds the measured failure limit {fail:.2f}."
        if np.asarray(grid["classes"])[row, col] == "red":
            width = grid.get("clearance_width_m")
            if width is not None and width[row, col] < grid.get("width_limit_m", 0):
                return f"No safe route: the opening is only {width[row, col]:.2f} m wide for this robot."
            return "No safe route: a red area blocked by measured limits or a failed spot run separates start and goal."
    return "No safe route: walls, structures, or the world edge leave no opening clear of the robot's width."


def plan_route(scene: SceneSpec, map_result: dict[str, Any], *, robot: RobotEntry | None = None,
               start_xy=None, goal_xy=None) -> dict[str, Any]:
    """A* on corrected colors; return a smoothed route or the closest reachable partial route."""
    from .backends.mujoco_backend import MuJoCoSimulator

    working = scene.model_copy(deep=True).fill_defaults()
    entry = next((entry for entry in working.robots if robot is None or entry.id == robot.id), None)
    if entry is None:
        raise ValueError("Safe routing requires a robot in this scene.")
    simulator = MuJoCoSimulator()
    simulator.template(working)
    mission = simulator.missions.get(entry.id, simulator.mission)
    start_xy = tuple(start_xy) if start_xy is not None else tuple(mission["start"])
    goal_xy = tuple(goal_xy) if goal_xy is not None else mission["goal"]
    if goal_xy is None:
        raise ValueError("Safe routing requires a goal.")
    goal_xy = tuple(goal_xy)
    cell = float(map_result["cell_size_m"])
    x0, y0 = map_result["x_m"][0] - cell / 2, map_result["y_m"][0] - cell / 2

    def index(point, *, clamp=False):
        row, col = math.floor((point[1] - y0) / cell), math.floor((point[0] - x0) / cell)
        if not (0 <= row < len(map_result["y_m"]) and 0 <= col < len(map_result["x_m"])):
            if not clamp:
                raise ValueError("The route start must be inside the mapped world.")
            return (min(max(row, 0), len(map_result["y_m"]) - 1),
                    min(max(col, 0), len(map_result["x_m"]) - 1))
        return row, col

    start, goal = index(start_xy), index(goal_xy, clamp=True)
    goal_outside = not (x0 <= goal_xy[0] < x0 + len(map_result["x_m"]) * cell
                        and y0 <= goal_xy[1] < y0 + len(map_result["y_m"]) * cell)
    allowed = _clearance_mask(working, map_result, entry.robot)
    costs = np.where(np.asarray(map_result["classes"]) == "amber", 5.0, 1.0)
    slope = np.asarray(map_result.get("slope_deg", np.zeros(allowed.shape)))
    clearance = np.asarray(map_result.get("clearance_width_m", np.full(allowed.shape, 100.0)))
    green = np.asarray(map_result["classes"]) == "green"
    costs += green * (0.1 * np.clip(slope / 45, 0, 1)
                      + 0.1 * np.clip(_robot_dimensions(entry.robot)[1] / np.maximum(clearance, 0.1), 0, 1))
    blockers = _indoor_geometry(working.terrain.preset)[0] if working.terrain.kind == "indoor" else []
    costs += WALL_SWEEP_COST * _near_walls(map_result, blockers, entry.robot)
    nav_grid = {"cell_size_m": cell, "x_m": list(map_result["x_m"]), "y_m": list(map_result["y_m"]),
                "allowed": allowed.tolist(), "blockers": blockers,
                "radius_m": _robot_dimensions(entry.robot)[1] / 2 + 0.01}

    def safe_segment(a, b):
        cover = segment_cells(nav_grid, a, b)
        return cover is not None and all(allowed[r, c] for r, c in cover)

    def center(node):
        return float(map_result["x_m"][node[1]]), float(map_result["y_m"][node[0]])

    start_clear = safe_segment(start_xy, center(start))
    goal_clear = not goal_outside and safe_segment(center(goal), goal_xy)
    if not start_clear:
        allowed[start] = False
    if not goal_clear:
        allowed[goal] = False
    nav_grid["allowed"] = allowed.tolist()
    cells, reached = _search(
        allowed, costs, start, goal,
        edge_clear=(lambda a, b: safe_segment(center(a), center(b))) if blockers else None,
    ) if allowed[start] else ([start], False)
    reached = reached and goal_clear
    points = [center(node) for node in cells]
    if allowed[start]:
        points.insert(0, start_xy)
    if reached:
        points.append(goal_xy)

    def segment_cost(a, b):
        count = max(2, math.ceil(math.dist(a, b) / (cell / 4)))
        return math.dist(a, b) * sum(
            costs[index((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))]
            for t in (np.arange(count) + 0.5) / count
        ) / count

    # Keep smoothing from trading a short amber crossing for a longer risky shortcut.
    prefix_cost = [0.0]
    for a, b in zip(points, points[1:]):
        prefix_cost.append(prefix_cost[-1] + segment_cost(a, b))
    smoothed = points[:1]
    current = 0
    while current < len(points) - 1:
        chosen = current + 1
        for candidate in range(len(points) - 1, current + 1, -1):
            cover = segment_cells(nav_grid, points[current], points[candidate])
            original = [index(point) for point in points[current:candidate + 1]]
            if cover and safe_segment(points[current], points[candidate]) \
                    and max(costs[r, c] for r, c in cover) <= max(costs[r, c] for r, c in original) + 1e-9 \
                    and segment_cost(points[current], points[candidate]) <= (
                        prefix_cost[candidate] - prefix_cost[current]
                    ) + 1e-9:
                chosen = candidate
                break
        smoothed.append(points[chosen])
        current = chosen
    if reached and not all(safe_segment(a, b) for a, b in zip(smoothed, smoothed[1:])):
        reached = False
        smoothed = [start_xy]
    reason = "Safe route found." if reached else _blocking_reason(working, map_result, start, goal, allowed)
    if goal_outside:
        reason = "No safe route: the goal is off-world, beyond the mapped terrain edge."
    if map_result.get("unreliable"):
        reason += " The source map is flagged unreliable; this is a tentative route, not a safety guarantee."
    return {"found": reached, "path_xy": smoothed if allowed[start] else [],
            "partial_path_xy": [] if reached else smoothed if allowed[start] else [],
            "cells": cells, "start_xy": start_xy, "goal_xy": goal_xy, "reason": reason,
            "length_m": sum(math.dist(a, b) for a, b in zip(smoothed, smoothed[1:])),
            "navigation_grid": nav_grid, "robot_id": entry.id}


def compare_routes(scene: SceneSpec, map_result: dict[str, Any], *, robot: RobotEntry | None = None,
                   run_batch_fn: Callable[..., list[dict]] | None = None) -> dict[str, Any]:
    """Drive straight and planned routes in one parallel batch, keeping world seed and speed identical."""
    from . import batch

    working = scene.model_copy(deep=True).fill_defaults()
    entry = next((entry for entry in working.robots if robot is None or entry.id == robot.id), None)
    if entry is None:
        raise ValueError("Safe routing requires a robot in this scene.")
    if not entry.mission().has_goal():
        return {"route": None, "runs": [],
                "text": "Map only: no goal selected. Set a goal to plan and compare routes."}
    route = plan_route(scene, map_result, robot=robot)
    if not route["found"]:
        return {"route": route, "runs": [], "text": route["reason"]}
    entry = next(entry for entry in working.robots if entry.id == route["robot_id"])
    speed = (entry.robot.drive_speed * entry.robot.wheel_radius if entry.robot.kind == "rover"
             else entry.robot.command_speed_m_s if entry.robot.kind == "library" else 0.25)
    working.duration_s = max(working.duration_s, 10 + 3 * route["length_m"] / max(speed, 0.1))
    working.time_limit_given = True
    common = {"robot_id": entry.id, "start_xy": list(route["start_xy"]),
              "goal_xy": list(route["goal_xy"]), "seed": working.terrain.seed}
    runs = (run_batch_fn or batch.run_batch)(working, [
        {**common, "label": "Straight", "waypoints": [], "navigation_grid": None},
        {**common, "label": "Safe route", "waypoints": route["path_xy"][1:-1],
         "navigation_grid": route["navigation_grid"]},
    ], baseline=False, record=False)
    if len(runs) != 2:
        raise RuntimeError("Route comparison requires both straight and safe run results.")

    def describe(run):
        if run.get("error"):
            return f"run error: {run['error']}"
        summary = run.get("summary") or {}
        metrics = (summary.get("robots") or {}).get(entry.id, summary)
        if metrics.get("goal_reached"):
            elapsed = metrics.get("time_to_goal_s")
            return f"reached the goal in {elapsed:g} s" if elapsed is not None else "reached the goal"
        outcome = metrics.get("outcome")
        if outcome == "fell" or metrics.get("flipped"):
            elapsed = metrics.get("flipped_at_s")
            slope = metrics.get("local_slope_deg")
            text = f"fell at {elapsed:g} s" if elapsed is not None else "fell"
            return text + (f" on a {slope:g} degree patch" if slope is not None else "")
        if outcome == "stuck":
            obstacle = metrics.get("stopped_by") or (metrics.get("obstacles") or {}).get("stopped_by") or {}
            return "stuck" + (f" at the {obstacle['kind']}" if obstacle.get("kind") else "")
        return str(outcome or "no goal outcome reported")

    metrics = [(run.get("summary") or {}).get("robots", {}).get(entry.id, run.get("summary") or {}) for run in runs]
    straight_length, safe_length = (metric.get("path_length_m") for metric in metrics)
    longer = None
    if not any(run.get("error") for run in runs) and straight_length and safe_length is not None:
        longer = 100 * (safe_length / straight_length - 1)
    text = f"Straight: {describe(runs[0])}. Safe route: {describe(runs[1])}"
    if longer is not None:
        text += f", {abs(longer):.0f}% {'longer' if longer >= 0 else 'shorter'} path"
    if map_result.get("unreliable"):
        text += ". Source map flagged unreliable; route evidence does not certify the whole map"
    return {"route": route, "runs": runs, "path_change_percent": longer, "text": text + "."}
