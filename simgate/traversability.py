"""Per-cell terrain and geometry measurements for robot-terrain scenes."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .backends.mujoco_backend import GRID, MuJoCoSimulator
from .indoor import get_layout, height_grid
from .library.cards import RobotCard
from .spec import Robot, RobotEntry, SceneSpec


def _robot_dimensions(robot: Robot | RobotCard) -> tuple[float, float]:
    if isinstance(robot, RobotCard):
        length, width = robot.footprint_length_m, robot.footprint_width_m
        if length is not None and width is not None:
            return float(length), float(width)
        if robot.kind == "wheeled" and robot.track_m is not None and robot.wheels:
            wheel_radius = max(wheel.radius_m for wheel in robot.wheels)
            return max(2 * robot.track_m, 2 * wheel_radius), robot.track_m + wheel_radius
        raise ValueError(f"Robot card {robot.id!r} does not define its footprint dimensions.")

    if robot.kind == "rover":
        return robot.chassis_length + 0.1, robot.chassis_width + robot.wheel_width + 0.02
    if robot.kind == "walker":
        return robot.chassis_length + 0.12, robot.chassis_width + 0.15 + 0.07
    if robot.kind == "library":
        from .library import LibraryRegistry

        return _robot_dimensions(LibraryRegistry().robot(robot.robot_id))
    raise ValueError(f"Unsupported robot type {robot.kind!r}.")


def _bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray,
              x0: float, y0: float, dx: float, dy: float) -> np.ndarray:
    cols = np.clip((x - x0) / dx, 0, field.shape[1] - 1.001)
    rows = np.clip((y - y0) / dy, 0, field.shape[0] - 1.001)
    c0, r0 = cols.astype(int), rows.astype(int)
    fc, fr = cols - c0, rows - r0
    top = field[r0, c0] * (1 - fc) + field[r0, c0 + 1] * fc
    bottom = field[r0 + 1, c0] * (1 - fc) + field[r0 + 1, c0 + 1] * fc
    return top * (1 - fr) + bottom * fr


def _rect(center, yaw: float, length: float, width: float) -> tuple[float, float, float, float, float]:
    return float(center[0]), float(center[1]), yaw, length / 2, width / 2


def _indoor_geometry(preset: str) -> tuple[list[dict], list[dict]]:
    layout = get_layout(preset)
    blockers, surfaces = [], []
    for geom in layout["geoms"]:
        surface = geom["surface"]
        if surface in ("wall", "counter"):
            blockers.append(_rect(geom["position"], math.radians(geom["pitch_deg"]),
                                  geom["size_m"][0], geom["size_m"][1]))
        elif surface not in ("base",):
            surfaces.append(geom)
    return blockers, surfaces


def _geometry_from_course(course: list[Any]) -> tuple[list[tuple], list[tuple]]:
    blockers, steps = [], []
    for item in course:
        kind = item.kind
        center = item.xy
        yaw = math.radians(item.yaw_deg)
        length, width = item.length, 3.0
        if kind == "wall":
            length = 0.24
        elif kind in ("rock", "boulder"):
            width = max(0.1, min(2.3 * item.height_m, item.length))
        elif kind == "gap":
            continue
        blockers.append(_rect(center, yaw, length, width))
        if kind in ("step", "stairs", "ramp"):
            steps.append((center[0], center[1], yaw, length / 2, width / 2, item.height_m))
    return blockers, steps


def _geometry_from_structures(placements: list[dict]) -> list[tuple]:
    blockers = []
    for item in placements:
        if item["kind"] == "bridge":
            continue
        footprint = item["footprint"]
        blockers.append(_rect(footprint["center_xy"], math.radians(item["orientation"].get("yaw_deg", 0.0)),
                              footprint["length_m"], footprint["width_m"]))
    return blockers


def _point_in_rect(x: float, y: float, rect: tuple) -> bool:
    cx, cy, yaw, half_length, half_width = rect
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = x - cx, y - cy
    along, across = dx * cosine + dy * sine, -dx * sine + dy * cosine
    return abs(along) <= half_length and abs(across) <= half_width


def _rectangles_overlap(first: tuple, second: tuple) -> bool:
    # Separating-axis test for the robot rectangle and one scene rectangle.
    for rect in (first, second):
        cx, cy, yaw, half_length, half_width = rect
        axes = ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw)))
        for ax, ay in axes:
            first_center = first[0] * ax + first[1] * ay
            second_center = second[0] * ax + second[1] * ay
            first_radius = (first[3] * abs(math.cos(first[2]) * ax + math.sin(first[2]) * ay)
                            + first[4] * abs(-math.sin(first[2]) * ax + math.cos(first[2]) * ay))
            second_radius = (second[3] * abs(math.cos(second[2]) * ax + math.sin(second[2]) * ay)
                             + second[4] * abs(-math.sin(second[2]) * ax + math.cos(second[2]) * ay))
            if abs(first_center - second_center) > first_radius + second_radius:
                return False
    return True


def _ray_distance(x: float, y: float, direction: tuple[float, float], rect: tuple) -> float | None:
    cx, cy, yaw, half_length, half_width = rect
    cosine, sine = math.cos(yaw), math.sin(yaw)
    dx, dy = x - cx, y - cy
    origin = (dx * cosine + dy * sine, -dx * sine + dy * cosine)
    velocity = (direction[0] * cosine + direction[1] * sine,
                -direction[0] * sine + direction[1] * cosine)
    near, far = -math.inf, math.inf
    for coordinate, speed, half_extent in zip(origin, velocity, (half_length, half_width)):
        if abs(speed) < 1e-12:
            if abs(coordinate) > half_extent:
                return None
            continue
        first, second = (-half_extent - coordinate) / speed, (half_extent - coordinate) / speed
        near, far = max(near, min(first, second)), min(far, max(first, second))
    return max(near, 0.0) if far >= max(near, 0.0) else None


def _clearance(x: float, y: float, blockers: list[tuple], size_x: float, size_y: float) -> float:
    if any(_point_in_rect(x, y, blocker) for blocker in blockers):
        return 0.0
    distances = []
    for direction, boundary in (((-1.0, 0.0), x + size_x / 2), ((1.0, 0.0), size_x / 2 - x),
                                ((0.0, -1.0), y + size_y / 2), ((0.0, 1.0), size_y / 2 - y)):
        hit = boundary
        for blocker in blockers:
            distance = _ray_distance(x, y, direction, blocker)
            if distance is not None and distance > 1e-8:
                hit = min(hit, distance)
        distances.append(hit)
    return min(distances[0] + distances[1], distances[2] + distances[3])


def _terrain_step_max(heights: np.ndarray, row: int, col: int, radius_rows: int, radius_cols: int,
                      threshold: float = 0.02) -> float:
    row0, row1 = max(0, row - radius_rows), min(heights.shape[0] - 1, row + radius_rows)
    col0, col1 = max(0, col - radius_cols), min(heights.shape[1] - 1, col + radius_cols)
    window = heights[row0:row1 + 1, col0:col1 + 1]
    step_candidates = []
    if window.shape[0] > 1:
        differences = np.abs(np.diff(window, axis=0))
        step_candidates.append(float(differences.max() - np.median(differences)))
    if window.shape[1] > 1:
        differences = np.abs(np.diff(window, axis=1))
        step_candidates.append(float(differences.max() - np.median(differences)))
    return max((difference for difference in step_candidates if difference >= threshold), default=0.0)


def features(scene: SceneSpec, robot: Robot | RobotCard | RobotEntry | None = None) -> dict[str, Any]:
    """Measure terrain and geometry over a 0.5 m outdoor or 0.25 m indoor grid.

    Returns 2-D NumPy arrays for ``slope_deg``, ``roughness_m``, ``max_step_m``,
    ``clearance_width_m``, ``friction``, and ``blocked``, with ``x_m`` and ``y_m``
    containing the corresponding cell-center coordinates.
    """
    if not isinstance(scene, SceneSpec) or scene.domain != "robot_terrain":
        raise ValueError("Traversability features require a robot-terrain SceneSpec.")
    if scene.terrain is None:
        raise ValueError("Traversability features require a terrain.")
    working = scene.model_copy(deep=True)
    working.fill_defaults()
    selected = robot.robot if isinstance(robot, RobotEntry) else robot
    selected = selected or working.robot
    if selected is None:
        raise ValueError("Traversability features require a robot.")
    if isinstance(selected, Robot):
        working.robot = selected.model_copy(deep=True)
        if working.robots:
            working.robots[0].robot = selected.model_copy(deep=True)
    length_m, width_m = _robot_dimensions(selected)
    terrain = working.terrain
    blockers: list[tuple] = []
    step_obstacles: list[tuple] = []
    indoor_surfaces: list[dict] = []

    if terrain.kind == "indoor":
        layout = get_layout(terrain.preset)
        size_x, size_y = map(float, layout["size_m"])
        cell_size = 0.25
        heights = height_grid(terrain.preset, GRID)
        blockers, indoor_surfaces = _indoor_geometry(terrain.preset)
        x0 = y0 = -max(size_x, size_y) / 2
        cell_x = cell_y = max(size_x, size_y) / (GRID - 1)
    else:
        size_x = size_y = float(terrain.size_m)
        cell_size = 0.5
        simulator = MuJoCoSimulator()
        _, heights, _ = simulator.template(working)
        course_blockers, step_obstacles = _geometry_from_course(simulator.course)
        blockers.extend(course_blockers)
        blockers.extend(_geometry_from_structures(simulator.structure_placements))
        x0 = y0 = -size_x / 2
        cell_x = cell_y = size_x / (heights.shape[1] - 1)

    nx, ny = math.ceil(size_x / cell_size), math.ceil(size_y / cell_size)
    x_centers = -size_x / 2 + (np.arange(nx) + 0.5) * cell_size
    y_centers = -size_y / 2 + (np.arange(ny) + 0.5) * cell_size
    gradient_y, gradient_x = np.gradient(heights, cell_y, cell_x)
    robot_rects = [[_rect((x, y), 0.0, length_m, width_m) for x in x_centers] for y in y_centers]
    arrays = {
        "slope_deg": np.zeros((ny, nx), dtype=float),
        "roughness_m": np.zeros((ny, nx), dtype=float),
        "max_step_m": np.zeros((ny, nx), dtype=float),
        "clearance_width_m": np.zeros((ny, nx), dtype=float),
        "friction": np.full((ny, nx), float(working.environment.friction), dtype=float),
        "blocked": np.zeros((ny, nx), dtype=bool),
    }

    along = np.linspace(-length_m / 2, length_m / 2, max(5, math.ceil(length_m / 0.12) + 1))
    across = np.linspace(-width_m / 2, width_m / 2, max(5, math.ceil(width_m / 0.12) + 1))
    offsets_x, offsets_y = np.meshgrid(along, across)
    offsets_x, offsets_y = offsets_x.ravel(), offsets_y.ravel()
    centered_x, centered_y = offsets_x - offsets_x.mean(), offsets_y - offsets_y.mean()
    for row, y in enumerate(y_centers):
        for col, x in enumerate(x_centers):
            xs, ys = x + offsets_x, y + offsets_y
            z = _bilinear(heights, xs, ys, x0, y0, cell_x, cell_y)
            dx = _bilinear(gradient_x, xs, ys, x0, y0, cell_x, cell_y)
            dy = _bilinear(gradient_y, xs, ys, x0, y0, cell_x, cell_y)
            arrays["slope_deg"][row, col] = np.degrees(np.arctan(np.hypot(dx, dy))).max()

            slope_x = float(centered_x @ (z - z.mean()) / max(centered_x @ centered_x, 1e-12))
            slope_y = float(centered_y @ (z - z.mean()) / max(centered_y @ centered_y, 1e-12))
            residual = z - z.mean() - slope_x * centered_x - slope_y * centered_y
            arrays["roughness_m"][row, col] = float(np.sqrt(np.mean(residual * residual)))

            source_col = int(round((x - x0) / cell_x))
            source_row = int(round((y - y0) / cell_y))
            radius_cols = max(1, math.ceil(length_m / (2 * cell_x)))
            radius_rows = max(1, math.ceil(width_m / (2 * cell_y)))
            arrays["max_step_m"][row, col] = _terrain_step_max(heights, source_row, source_col,
                                                                radius_rows, radius_cols)

            footprint = robot_rects[row][col]
            outside = (abs(x) + length_m / 2 > size_x / 2 or abs(y) + width_m / 2 > size_y / 2)
            if terrain.kind == "indoor":
                collision = outside or any(_point_in_rect(x, y, obstacle) for obstacle in blockers)
            else:
                collision = outside or any(_rectangles_overlap(footprint, obstacle) for obstacle in blockers)
            arrays["blocked"][row, col] = collision
            arrays["clearance_width_m"][row, col] = _clearance(x, y, blockers, size_x, size_y)
            for obstacle in step_obstacles:
                if _rectangles_overlap(footprint, obstacle[:5]):
                    arrays["max_step_m"][row, col] = max(arrays["max_step_m"][row, col], obstacle[5])

            surfaces = [geom for geom in indoor_surfaces
                        if abs(x - geom["position"][0]) <= geom["size_m"][0] / 2
                        and abs(y - geom["position"][1]) <= geom["size_m"][1] / 2]
            if surfaces:
                surface = max(surfaces, key=lambda geom: geom["priority"])
                arrays["friction"][row, col] = float(surface["friction"])

    return {
        "cell_size_m": cell_size,
        "x_m": x_centers,
        "y_m": y_centers,
        **arrays,
    }


def _report_limits(report: dict | None) -> dict[str, dict]:
    if not report:
        return {}
    limits = report.get("limits", [])
    if isinstance(limits, dict):
        return {key: value for key, value in limits.items() if isinstance(value, dict)}
    return {limit["knob"]: limit for limit in limits if isinstance(limit, dict) and limit.get("knob")}


def _roughness_at(scene: SceneSpec, robot: Robot | RobotCard | RobotEntry, parameter: float) -> float | None:
    terrain_scene = scene.model_copy(deep=True)
    if terrain_scene.terrain.kind == "flat":
        terrain_scene.terrain.kind = "rocky"
    terrain_scene.terrain.roughness = max(0.0, float(parameter))
    grid = features(terrain_scene, robot)
    values = grid["roughness_m"][~grid["blocked"]]
    return float(np.median(values)) if values.size else None


def _physical_value(
    limit: dict,
    side: str,
    value: float | None,
    scene: SceneSpec,
    robot: Robot | RobotCard | RobotEntry,
) -> float | None:
    if value is None:
        return None
    direct = limit.get(f"{side}_m")
    if isinstance(direct, (int, float)):
        return float(direct)
    recorded = limit.get("physical_values", {}).get(side)
    if isinstance(recorded, (int, float)):
        return float(recorded)
    knob = limit.get("knob")
    if knob in ("step", "obstacle_scale"):
        if knob == "step" or limit.get("unit") == "m":
            return float(value)
        scale = limit.get("physical_scale")
        if scale is None:
            from .backends.obstacles import capability

            model = robot.robot if isinstance(robot, RobotEntry) else robot
            if isinstance(model, RobotCard):
                model = scene.robot
            if model is None:
                return None
            scale = capability(scene.model_copy(update={"robot": model}))["climb_m"]
        return float(value) * float(scale)
    if knob == "roughness":
        return _roughness_at(scene, robot, float(value))
    return float(value)


def _thresholds(
    limit: dict | None,
    scene: SceneSpec,
    robot: Robot | RobotCard | RobotEntry,
) -> tuple[float | None, float | None]:
    if not limit:
        return None, None
    reliable = limit.get("reliable_ok_m", limit.get("reliable_ok"))
    if reliable is None and not limit.get("never_broke", False):
        from .finder import reliable_ok_point

        point = reliable_ok_point(limit)
        reliable = point["value"] if point is not None else None
    failed = limit.get("first_fail_m", limit.get("first_fail"))
    if limit.get("never_broke"):
        failed = None
    return (_physical_value(limit, "last_ok", reliable, scene, robot),
            _physical_value(limit, "first_fail", failed, scene, robot))


def classify(
    scene: SceneSpec,
    robot: Robot | RobotCard | RobotEntry | None = None,
    finder_report: dict | None = None,
    on_step: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Classify each measured cell green, amber, red, or grey from this world's Finder limits.

    The result includes ``classes`` (a 2-D array of color names), ``limits``, scope and
    honesty text, and a scan note. If limits are absent or for another world, the missing
    limits are scanned under the Finder's 16-run cap before classification.
    """
    from . import finder

    if not isinstance(scene, SceneSpec) or scene.domain != "robot_terrain" or not scene.robots:
        raise ValueError("Classification requires a filled robot-terrain SceneSpec.")
    working = scene.model_copy(deep=True).fill_defaults()
    if isinstance(robot, RobotEntry):
        entry = next((candidate for candidate in working.robots if candidate.id == robot.id), None)
    else:
        entry = working.robots[0]
    if entry is None:
        raise ValueError("The selected robot is not in this scene.")
    selected = robot or entry.robot
    if isinstance(selected, Robot) and entry.robot != selected:
        working.robot = selected.model_copy(deep=True)
        entry.robot = selected.model_copy(deep=True)
    robot_id = entry.id
    world = finder.limits_world(working, robot_id)

    matching = bool(finder_report and finder_report.get("world") == world
                    and finder_report.get("robot_id", robot_id) == robot_id)
    limits_by_knob = _report_limits(finder_report if matching else None)
    needed = ["slope_deg", "roughness", "step", "friction"]
    if working.terrain.kind == "indoor":
        needed = ["step", "friction"]
    present = set(limits_by_knob)
    aliases = {"step": "obstacle_scale", "roughness": "roughness_m"}
    missing = [key for key in needed if key not in present and aliases.get(key) not in present]
    scan_report = None
    if missing:
        scan_report = finder.scan_map_limits(working, robot_id, missing=missing, on_step=on_step)
        limits_by_knob.update(_report_limits(scan_report))

    measured = features(working, selected)
    classes = np.full(measured["blocked"].shape, "green", dtype="<U8")
    threshold_names = {
        "slope_deg": ("slope_deg", "slope_deg", True),
        "roughness": ("roughness", "roughness_m", True),
        "step": ("obstacle_scale", "max_step_m", True),
        "friction": ("friction", "friction", False),
    }
    threshold_details = {}
    for label, (knob, feature, higher_is_harsher) in threshold_names.items():
        limit = limits_by_knob.get(knob) or limits_by_knob.get(label)
        if label == "roughness":
            limit = limits_by_knob.get("roughness") or limits_by_knob.get("roughness_m")
        reliable, first_fail = _thresholds(limit, working, selected)
        threshold_details[label] = {"reliable_ok": reliable, "first_fail": first_fail,
                                    "measured": reliable is not None or first_fail is not None}
        values = measured[feature]
        if reliable is None and first_fail is None:
            classes[~measured["blocked"] & (classes == "green")] = "amber"
            continue
        if first_fail is not None:
            fail_mask = values >= first_fail if higher_is_harsher else values <= first_fail
            classes[fail_mask & ~measured["blocked"]] = "red"
        if reliable is None:
            classes[~measured["blocked"] & (classes == "green")] = "amber"
        else:
            safe_mask = values <= reliable if higher_is_harsher else values >= reliable
            classes[~safe_mask & ~measured["blocked"] & (classes == "green")] = "amber"

    _, width_m = _robot_dimensions(selected)
    width_limit = width_m + 0.1
    narrow = measured["clearance_width_m"] < width_limit
    classes[narrow & ~measured["blocked"]] = "red"
    classes[measured["blocked"]] = "grey"

    gravity = float(working.environment.gravity)
    surface = str(working.environment.surface)
    scope = (f"Limits apply to {working.environment.body} gravity ({gravity:g} m/s²) and the "
             f"{surface} surface (base friction {float(working.environment.friction):.2f}).")
    if scan_report:
        reason = "No matching Finder result for this world" if not matching else "Finder limits were missing"
        scan_note = (f"{reason} for {', '.join(missing)}; ran a quick scan "
                     f"({scan_report['runs']} of 16 allowed runs).")
    else:
        scan_note = "Used matching Finder limits; no scan was needed."
    return {
        **measured,
        "classes": classes,
        "width_limit_m": width_limit,
        "limits": threshold_details,
        "world": world,
        "limits_scope": scope,
        "scan": scan_report,
        "scan_note": scan_note,
        "honesty": "Map from measured limits (step ±1 cm, slope ±1.9°, …); real places combine features, so the map is checked with spot runs below.",
    }
