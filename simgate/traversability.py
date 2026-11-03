"""Per-cell terrain and geometry measurements for robot-terrain scenes."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .backends.mujoco_backend import GRID, MuJoCoSimulator
from .indoor import get_layout, height_grid
from .library.cards import RobotCard
from .spec import Robot, RobotEntry, SceneSpec


def _robot_dimensions(robot: Robot | RobotCard | RobotEntry) -> tuple[float, float]:
    if isinstance(robot, RobotEntry):
        robot = robot.robot
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
    baselines = [baseline for report in (finder_report if matching else None, scan_report)
                 if report for baseline in report.get("baselines", [])
                 if baseline.get("passed") is True]

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
        passed_values = [baseline["values"][label] for baseline in baselines
                         if label in baseline.get("values", {})]
        if reliable is not None:
            passed_values.append(reliable)
        reliable = (max(passed_values) if higher_is_harsher else min(passed_values)) if passed_values else None
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
            safe_mask = values <= reliable + 1e-8 if higher_is_harsher else values >= reliable - 1e-8
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
        "baselines": baselines,
        "scan_note": scan_note,
        "honesty": "Map from measured limits (step ±1 cm, slope ±1.9°, …); real places combine features, so the map is checked with spot runs below.",
    }


def _spot_route(
    result: dict[str, Any],
    row: int,
    col: int,
    distance_m: float,
    preferred_axis: str | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float, str] | None:
    x, y = float(result["x_m"][col]), float(result["y_m"][row])
    cell_size = float(result["cell_size_m"])
    axes = [preferred_axis] if preferred_axis else []
    axes.extend(axis for axis in ("x", "y") if axis not in axes)
    blocked = result["blocked"]
    for axis in axes:
        dx, dy = (1.0, 0.0) if axis == "x" else (0.0, 1.0)
        points = np.linspace(-distance_m / 2, distance_m / 2, max(3, math.ceil(distance_m / (cell_size / 2)) + 1))
        valid = True
        for offset in points:
            px, py = x + dx * offset, y + dy * offset
            route_col = int(np.argmin(np.abs(result["x_m"] - px)))
            route_row = int(np.argmin(np.abs(result["y_m"] - py)))
            if abs(float(result["x_m"][route_col]) - px) > cell_size * 0.76 \
                    or abs(float(result["y_m"][route_row]) - py) > cell_size * 0.76 \
                    or blocked[route_row, route_col]:
                valid = False
                break
        if valid:
            heading = 0.0 if axis == "x" else 90.0
            start = (round(x - dx * distance_m / 2, 3), round(y - dy * distance_m / 2, 3))
            goal = (round(x + dx * distance_m / 2, 3), round(y + dy * distance_m / 2, 3))
            return start, goal, heading, axis
    return None


def _indoor_feature_target(scene: SceneSpec) -> tuple[float, float, str, str] | None:
    """Return a representative doorway crossing, or a step if the layout has no doorway."""
    if scene.terrain is None or scene.terrain.kind != "indoor":
        return None
    layout = get_layout(scene.terrain.preset)
    doorways = []
    steps = []
    for geom in layout["geoms"]:
        name = geom["name"]
        x, y = geom["position"][:2]
        if name.startswith("indoor_threshold_"):
            size_x, size_y = geom["size_m"][:2]
            opening_width = max(size_x, size_y)
            axis = "y" if size_x >= size_y else "x"
            doorways.append((opening_width, x, y, axis))
        elif name.startswith("indoor_bedroom_step_"):
            steps.append((x, y))
    if doorways:
        _, x, y, axis = min(doorways)
        return x, y, axis, "doorway"
    if steps:
        x, y = steps[len(steps) // 2]
        return x, y, "y", "step"
    return None


def _spot_name(scene: SceneSpec, result: dict[str, Any], row: int, col: int,
               special_target: tuple[float, float, str, str] | None) -> str:
    x, y = float(result["x_m"][col]), float(result["y_m"][row])
    if special_target and math.dist((x, y), special_target[:2]) <= float(result["cell_size_m"]):
        return f"the {special_target[3]}"
    if scene.terrain is not None and scene.terrain.kind == "indoor":
        layout = get_layout(scene.terrain.preset)
        surfaces = [
            geom for geom in layout["geoms"]
            if geom["surface"] in ("rug", "wet")
            and abs(x - geom["position"][0]) <= geom["size_m"][0] / 2
            and abs(y - geom["position"][1]) <= geom["size_m"][1] / 2
        ]
        if surfaces:
            return "the wet patch" if surfaces[0]["surface"] == "wet" else "the rug"
        if result["max_step_m"][row, col] >= 0.05:
            return "the step"
    return "the selected area"


def _spot_cells(
    result: dict[str, Any],
    row: int,
    col: int,
    axis: str,
    distance_m: float,
    length_m: float,
    width_m: float,
) -> list[tuple[int, int]]:
    x, y = float(result["x_m"][col]), float(result["y_m"][row])
    dx, dy = (1.0, 0.0) if axis == "x" else (0.0, 1.0)
    half_length = length_m / 2 + float(result["cell_size_m"]) / 2
    half_width = width_m / 2 + float(result["cell_size_m"]) / 2
    rows, cols = np.indices(result["blocked"].shape)
    xs, ys = result["x_m"][cols], result["y_m"][rows]
    along = (xs - x) * dx + (ys - y) * dy
    across = np.abs((xs - x) * dy - (ys - y) * dx)
    return [
        (int(r), int(c))
        for r, c in zip(*np.where(
            (np.abs(along) <= distance_m / 2 + half_length)
            & (across <= half_width)
            & ~result["blocked"]
        ))
    ]


def _classify_spot_outcome(run: dict[str, Any], robot_id: str) -> tuple[bool | None, str]:
    if run.get("error"):
        return None, str(run["error"])
    summary = run.get("summary") or {}
    if not isinstance(summary, dict):
        return None, "Run returned no robot metrics."
    robots = summary.get("robots")
    metrics = robots.get(robot_id, summary) if isinstance(robots, dict) else summary
    if not isinstance(metrics, dict):
        return None, "Run returned no robot metrics."
    outcome = str(metrics.get("outcome") or "").lower()
    stopped = (metrics.get("stopped_early") or {}).get("reason")
    if metrics.get("flipped") or outcome in ("fell", "fall"):
        return False, "fell"
    if metrics.get("stuck") or outcome == "stuck" or stopped == "stuck":
        return False, "got stuck"
    if not any(key in metrics for key in ("outcome", "flipped", "stuck", "goal_reached",
                                          "forward_progress_m")):
        return None, "Run returned no fall/stuck outcome evidence."
    return True, "crossed without falling or getting stuck"


def verify(
    scene: SceneSpec,
    robot: Robot | RobotCard | RobotEntry | None,
    map_result: dict[str, Any],
    *,
    run_batch_fn: Callable[..., list[dict]] | None = None,
) -> dict[str, Any]:
    """Run parallel short spot checks and correct map cells when judged predictions disagree."""
    from . import batch

    working = scene.model_copy(deep=True).fill_defaults()
    if not working.robots:
        raise ValueError("Spot verification requires a filled robot-terrain SceneSpec.")
    if isinstance(robot, RobotEntry):
        entry = next((candidate for candidate in working.robots if candidate.id == robot.id), None)
    else:
        entry = working.robots[0]
    if entry is None:
        raise ValueError("The selected robot is not in this scene.")
    selected = robot.robot if isinstance(robot, RobotEntry) else robot
    selected = selected or entry.robot
    if isinstance(selected, RobotCard) and (
        entry.robot.kind != "library" or entry.robot.robot_id != selected.id
    ):
        raise ValueError("Spot verification must simulate the selected robot's matching library-card policy.")
    if isinstance(selected, Robot) and selected != entry.robot:
        entry.robot = selected.model_copy(deep=True)
    # Spot checks belong to one robot; other racers must not affect its result.
    working.robots = [entry]
    working.robot = entry.robot.model_copy(deep=True)
    robot_id = entry.id

    classes = np.array(map_result["classes"], copy=True)
    measured = np.array(map_result.get("measured", np.zeros(classes.shape, dtype=bool)), dtype=bool, copy=True)
    run_ids = np.array(map_result.get("run_ids", np.full(classes.shape, "", dtype="<U32")), copy=True)
    grid = {**map_result, "classes": classes}
    distance_m = 1.5 if working.terrain.kind == "indoor" else 3.0
    length_m, width_m = _robot_dimensions(selected)
    candidates: dict[str, list[tuple[int, int, tuple, str]]] = {
        color: [] for color in ("green", "amber", "red")
    }
    special_target = _indoor_feature_target(working)
    preferred = None
    special_cell = None
    if special_target:
        target_x, target_y, preferred, _ = special_target
        special_col = int(np.argmin(np.abs(grid["x_m"] - target_x)))
        special_row = int(np.argmin(np.abs(grid["y_m"] - target_y)))
        special_cell = (special_row, special_col)

    for color in candidates:
        for row, col in zip(*np.where(classes == color)):
            preferred_axis = preferred if special_cell == (int(row), int(col)) else None
            route = _spot_route(grid, int(row), int(col), distance_m, preferred_axis)
            if route is not None:
                start, goal, heading, axis = route
                candidates[color].append((int(row), int(col), (start, goal, heading, axis), ""))

    selected_spots = []
    if special_cell:
        row, col = special_cell
        color = str(classes[row, col])
        special = next((candidate for candidate in candidates.get(color, [])
                        if candidate[0] == row and candidate[1] == col), None)
        if special:
            selected_spots.append(special)
    for color, available in candidates.items():
        picks = []
        if selected_spots and selected_spots[0] in available:
            picks.append(selected_spots[0])
        while len(picks) < min(2, len(available)):
            remaining = [candidate for candidate in available if candidate not in picks]
            if not remaining:
                break
            if not picks and not selected_spots:
                chosen = remaining[0]
            else:
                previous = selected_spots + picks
                chosen = max(
                    remaining,
                    key=lambda candidate: min(
                        (grid["x_m"][candidate[1]] - grid["x_m"][other[1]]) ** 2
                        + (grid["y_m"][candidate[0]] - grid["y_m"][other[0]]) ** 2
                        for other in previous
                    ),
                )
            picks.append(chosen)
        selected_spots.extend(candidate for candidate in picks if candidate not in selected_spots)

    variations = []
    for index, (row, col, route, _) in enumerate(selected_spots, start=1):
        start, goal, heading, _ = route
        name = _spot_name(working, grid, row, col, special_target)
        variations.append({
            "label": f"Spot {index}: {name}",
            "robot_id": robot_id,
            "start_xy": list(start),
            "goal_xy": list(goal),
            "start_heading_deg": heading,
            "goal_radius_m": max(0.2, min(0.4, float(grid["cell_size_m"]) / 2)),
        })

    verification_scene = working.model_copy(deep=True)
    verification_scene.duration_s = 14.0 if working.terrain.kind == "indoor" else 24.0
    runner = run_batch_fn or batch.run_batch
    results = runner(verification_scene, variations, baseline=False, record=False) if variations else []
    if len(results) != len(variations):
        raise RuntimeError(f"Spot batch returned {len(results)} results for {len(variations)} runs.")

    spots = []
    agreement = 0
    disagreements = []
    for index, (candidate, run) in enumerate(zip(selected_spots, results), start=1):
        row, col, route, _ = candidate
        predicted = str(map_result["classes"][row, col])
        passed, outcome = _classify_spot_outcome(run, robot_id)
        judged = predicted in ("green", "red") and passed is not None
        agrees = judged and ((predicted == "green") == passed)
        if agrees:
            agreement += 1
        run_id = str(run.get("id", f"spot-{index}"))
        start, goal, heading, axis = route
        cells = _spot_cells(grid, row, col, axis, distance_m, length_m, width_m)
        label = _spot_name(working, grid, row, col, special_target)
        corrected = False
        if judged and not agrees:
            actual_class = "green" if passed else "red"
            for cell_row, cell_col in cells:
                classes[cell_row, cell_col] = actual_class
                measured[cell_row, cell_col] = True
                run_ids[cell_row, cell_col] = run_id
            corrected = True
            prediction = "safe" if predicted == "green" else "unsafe"
            actual = outcome
            disagreements.append(f"{label} was predicted {prediction} but the robot {actual}; map corrected")
        spots.append({
            "id": run_id,
            "label": run.get("label", f"Spot {index}"),
            "site": label,
            "row": row,
            "col": col,
            "predicted": predicted,
            "passed": passed,
            "outcome": outcome,
            "judged": judged,
            "agreed": agrees,
            "corrected": corrected,
            "cells": cells,
            "start_xy": start,
            "goal_xy": goal,
            "start_heading_deg": heading,
        })

    amber_count = sum(spot["predicted"] == "amber" for spot in spots)
    errors = [f"{spot['id']}: {spot['outcome']}" for spot in spots if spot["passed"] is None]
    report = f"{agreement} of {len(spots)} spot checks agreed with the map."
    if amber_count:
        report += f" {amber_count} amber spot{'s were' if amber_count != 1 else ' was'} reported, not judged."
    if disagreements:
        count = len(disagreements)
        report += f" {count} spot{'s' if count != 1 else ''} disagreed: " + "; ".join(disagreements) + "."
    if errors:
        report += " Spot run errors: " + "; ".join(errors) + "."
    unreliable = agreement < 4
    if unreliable:
        report += " Map flagged unreliable: fewer than 4 spot checks agreed."
    return {
        **map_result,
        "classes": classes,
        "measured": measured,
        "run_ids": run_ids,
        "spot_checks": spots,
        "agreement": agreement,
        "agreement_total": len(spots),
        "agreement_text": report,
        "corrections": disagreements,
        "unreliable": unreliable,
    }


def map_report(scene: SceneSpec, robot_id: str | None = None, finder_report: dict | None = None,
               on_step: Callable[[dict], None] | None = None, *, via: list[str] | None = None) -> dict:
    """Measure, classify, spot-correct and drive; return a JSON-ready viewer report with replay recipes."""
    from . import batch, cost
    from .routing import compare_routes

    working = scene.model_copy(deep=True).fill_defaults()
    entry = next((entry for entry in working.robots if robot_id is None or entry.id == robot_id), None)
    if entry is None:
        raise ValueError("The requested robot is not in this world.")
    recipes = {}
    run_numbers = {}

    def step(status, result=""):
        if on_step:
            on_step({"agent": "Batch runner", "tool": "traversability", "status": status,
                     "thought": "Measuring, spot-checking and routing on this world's terrain.", "result": result})

    def runner(prefix):
        def execute(base, variations, **kwargs):
            rows = batch.run_batch(base, variations, **kwargs)
            for row, variation in zip(rows, variations):
                run_numbers[prefix] = run_numbers.get(prefix, 0) + 1
                row["id"] = f"{prefix}-run-{run_numbers[prefix]}"
                recipes[row["id"]] = {"scene_spec": base.model_dump(mode="json"), "variation": variation}
            return rows
        return execute

    step("working")
    with cost.track() as spent:
        result = classify(working, entry, finder_report, on_step)
        result = verify(working, entry, result, run_batch_fn=runner("spot"))
        if via:
            legs = []
            destinations = [*via, entry.goal_place or entry.goal_xy]
            if destinations[-1] is None or entry.goal_structure is not None:
                raise ValueError("A multi-leg map needs a named or coordinate final goal.")
            leg_scene = working.model_copy(deep=True)
            for index, destination in enumerate(destinations, 1):
                target = next(robot for robot in leg_scene.robots if robot.id == entry.id)
                target.goal_place = destination if isinstance(destination, str) else None
                target.goal_xy = destination if isinstance(destination, list) else None
                target.goal_structure = None
                leg_scene.fill_defaults()
                leg = compare_routes(leg_scene, result, robot=target, run_batch_fn=runner("route"))
                leg["label"] = f"Leg {index}: to {destination.replace('_', ' ') if isinstance(destination, str) else destination}"
                for run in leg["runs"]:
                    run["label"] = f"{leg['label']} — {run['label']}"
                legs.append(leg)
                if leg["route"] is None:
                    raise RuntimeError("A requested route leg did not resolve its destination.")
                target.start_xy, target.start_place, target.start_structure = list(leg["route"]["goal_xy"]), None, None
                target.start_heading_deg = None
            reached_legs = []
            for leg in legs:
                if leg["route"]:
                    reached_legs.append(leg["route"])
                if not leg["route"] or not leg["route"]["found"]:
                    break
            route = dict(reached_legs[-1]) if reached_legs else None
            if route:
                complete = len(reached_legs) == len(legs) and all(leg["found"] for leg in reached_legs)
                path = [point for leg in reached_legs for point in leg["path_xy"]]
                route.update(start_xy=reached_legs[0]["start_xy"], goal_xy=legs[-1]["route"]["goal_xy"],
                             path_xy=path, partial_path_xy=[] if complete else path, found=complete,
                             cells=[cell for leg in reached_legs for cell in leg["cells"]],
                             length_m=sum(leg["length_m"] for leg in reached_legs))
            comparison = {"route": route, "legs": legs, "runs": [run for leg in legs for run in leg["runs"]],
                          "text": " ".join(f"{leg['label']}. {leg['text']}" for leg in legs)
                          + " Legs are independent comparisons from each named stop, not one continuous run."}
        else:
            comparison = compare_routes(working, result, robot=entry, run_batch_fn=runner("route"))
        navigable = ~result["blocked"]
        total = max(1, int(navigable.sum()))
        percentages = {color: round(100 * int((result["classes"][navigable] == color).sum()) / total, 1)
                       for color in ("green", "amber", "red")}
        report = {
            **result, "type": "traversability", "robot_id": entry.id, "robot_label": entry.label,
            "percentages": percentages, "percentage_basis": "non-blocked cells",
            "comparison": comparison, "replays": recipes, "cost": spent.snapshot(),
            "summary": (f"{entry.label}: {percentages['green']}% green, {percentages['amber']}% amber, "
                        f"{percentages['red']}% red (non-blocked cells). "
                        f"{result['agreement_text']} {comparison['text']}"),
        }
    step("done", report["summary"])

    def json_value(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [json_value(item) for item in value]
        return value

    return json_value(report)
