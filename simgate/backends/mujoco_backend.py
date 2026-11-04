"""MuJoCo backend: robots on generated terrain."""
import hashlib
import base64
import json
import math
import xml.etree.ElementTree as ET
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import mujoco
import numpy as np

from .. import presets
from ..library import LibraryRegistry
from ..policy_runtime import PolicyRunner
from ..simple_control import SimpleController
from ..spec import Mission, SceneSpec
from ..terrain_data import load_ortho_image, load_preset, resample, scene_heights
from . import obstacles as course
from .base import BuildResult, Check, Simulator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENE_EVENT_BUDGET_BYTES = 5 * 1024 * 1024
GRID = 160         # heightfield resolution (GRID x GRID): 0.13 m cells on the default 20 m terrain
FPS = 30           # recorded frames per simulated second


# ── Terrain ────────────────────────────────────────────────────────────────

TERRAIN_RGB = {  # display colors by world, then by surface
    "mars": (0.66, 0.36, 0.24), "moon": (0.58, 0.57, 0.55), "titan": (0.60, 0.45, 0.26),
    "europa": (0.80, 0.86, 0.92), "enceladus": (0.86, 0.90, 0.95),
    "ice": (0.80, 0.87, 0.93), "sand": (0.78, 0.67, 0.46), "mud": (0.36, 0.28, 0.20),
}


def terrain_rgb(spec: SceneSpec) -> tuple[float, float, float]:
    env = spec.environment
    return TERRAIN_RGB.get(env.surface) if env.surface in ("ice", "sand", "mud") else TERRAIN_RGB.get(env.body, (0.50, 0.46, 0.40))


def smooth_noise(rng: np.random.Generator, coarse: int) -> np.ndarray:
    """Random coarse grid, bilinearly upsampled to GRID x GRID (no blocky steps)."""
    c = rng.normal(0, 1, (coarse, coarse))
    src = np.linspace(0, coarse - 1, GRID)
    rows = np.array([np.interp(src, np.arange(coarse), c[i]) for i in range(coarse)])
    return np.array([np.interp(src, np.arange(coarse), rows[:, j]) for j in range(GRID)]).T

def default_start(size_m: float) -> tuple[float, float]:
    return -size_m / 2 * 0.7, 0.0


def make_heights(spec: SceneSpec, start_xy: tuple[float, float] | None = None) -> tuple[np.ndarray, float]:
    """Return heights H[r, c] in metres (row r -> y increasing, col c -> x increasing)."""
    t = spec.terrain
    rng = np.random.default_rng(t.seed)
    half = t.size_m / 2
    xs = np.linspace(-half, half, GRID)
    X, Y = np.meshgrid(xs, xs)          # X varies along columns, Y along rows
    H = np.zeros_like(X)
    if t.kind == "indoor":
        from ..indoor import height_grid

        H = height_grid(t.preset or "apartment", GRID)
        return H, max(float(H.max()), 0.01)

    if t.kind == "slope":
        H += np.tan(math.radians(t.slope_deg)) * (X + half)
    if t.kind == "hills":
        for _ in range(6):
            cx, cy = rng.uniform(-half, half, 2)
            r = rng.uniform(2, 5)
            H += rng.uniform(0.3, 1.2) * (0.5 + t.roughness) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r**2))
    if t.kind == "craters":
        for _ in range(5):
            cx, cy = rng.uniform(-half * 0.8, half * 0.8, 2)
            r = rng.uniform(1.5, 3.5)
            d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / r
            H += (0.6 + t.roughness) * (0.35 * np.exp(-((d - 1.0) ** 2) / 0.05) - 0.5 * np.exp(-(d**2) / 0.4))
    if t.kind == "real":
        # Real elevation data (e.g. NASA HiRISE), already in metres with row 0 at -y.
        H += resample(scene_heights(t), GRID) * t.height_scale
    if t.kind in ("rocky", "hills", "craters", "slope", "real") and t.roughness > 0:
        # Small-scale bumps (coarse random grid, bilinearly upsampled); "rocky" gets the most.
        # On real terrain these stand in for sub-metre roughness the orbital data can't resolve.
        # They're faded out around the spawn point so the robot lands cleanly. The terrain's own
        # shape is never flattened there: that used to create steep artificial edges around the pad.
        scale = 0.3 if t.kind == "rocky" else 0.06
        start_x, start_y = start_xy or default_start(t.size_m)
        pad = np.exp(-(((X - start_x) ** 2) + (Y - start_y) ** 2) / 1.5)
        H += scale * t.roughness * smooth_noise(rng, coarse=GRID // 6) * (1 - pad)

    H -= H.min()
    zmax = max(float(H.max()), 0.01)
    return H, zmax


def find_flattest_start(
    H: np.ndarray,
    size_m: float,
    requested: tuple[float, float],
    search_radius_m: float,
    footprint_radius_m: float,
    avoid: list[tuple[tuple[float, float], float]] | None = None,
) -> tuple[tuple[float, float], float, float]:
    """Find the safest flat footprint on this heightfield within a bounded distance.

    The maximum slope under the footprint is minimized first, followed by mean footprint slope and distance from
    the requested start. Returns the chosen point and the before/after maximum footprint slopes in degrees.
    """
    cell = size_m / (GRID - 1)
    dy, dx = np.gradient(H, cell, cell)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    half = size_m / 2
    safe_edge = half * EDGE_MARGIN
    requested_col = int(round((requested[0] + half) / cell))
    requested_row = int(round((requested[1] + half) / cell))
    footprint_cells = max(0, int(math.ceil(footprint_radius_m / cell)))
    offsets = [(row, col) for row in range(-footprint_cells, footprint_cells + 1)
               for col in range(-footprint_cells, footprint_cells + 1)
               if (row * cell) ** 2 + (col * cell) ** 2 <= footprint_radius_m ** 2 + 1e-9]
    if not offsets:
        offsets = [(0, 0)]

    def score(row: int, col: int) -> tuple[float, float] | None:
        rows = [row + dr for dr, _ in offsets]
        cols = [col + dc for _, dc in offsets]
        if min(rows) < 0 or max(rows) >= GRID or min(cols) < 0 or max(cols) >= GRID:
            return None
        x, y = -half + col * cell, -half + row * cell
        if abs(x) + footprint_radius_m > safe_edge or abs(y) + footprint_radius_m > safe_edge:
            return None
        if any(math.hypot(x - other[0], y - other[1]) < footprint_radius_m + other_radius + 0.1
               for other, other_radius in (avoid or [])):
            return None
        values = slope[rows, cols]
        return float(values.max()), float(values.mean())

    before = score(requested_row, requested_col)
    if before is None:
        before = (float("inf"), float("inf"))
    span = int(math.ceil(search_radius_m / cell))
    row_start, row_end = max(0, requested_row - span), min(GRID, requested_row + span + 1)
    col_start, col_end = max(0, requested_col - span), min(GRID, requested_col + span + 1)
    best: tuple[tuple[float, float, float], tuple[float, float], tuple[int, int]] | None = None
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            x, y = -half + col * cell, -half + row * cell
            distance = math.hypot(x - requested[0], y - requested[1])
            if distance > search_radius_m + 1e-9:
                continue
            candidate_score = score(row, col)
            if candidate_score is None:
                continue
            ranked = (candidate_score[0], candidate_score[1], distance)
            if best is None or ranked < best[0]:
                best = (ranked, candidate_score, (row, col))
    if best is None:
        return requested, before[0], before[0]
    _, after, (row, col) = best
    chosen = (-half + col * cell, -half + row * cell)
    distance = math.hypot(chosen[0] - requested[0], chosen[1] - requested[1])
    if distance < cell / 2 or after[0] >= before[0] - 0.1:
        return requested, before[0], before[0]
    return (round(chosen[0], 2), round(chosen[1], 2)), round(before[0], 1), round(after[0], 1)


def auto_flatter_start(spec: SceneSpec, robot, requested: tuple[float, float], H: np.ndarray,
                       search_radius_m: float = 5.0,
                       avoid: list[tuple[tuple[float, float], float]] | None = None
                       ) -> tuple[tuple[float, float], str | None]:
    """Move a legged or library robot's implicit/place start to flatter terrain when a better spot is nearby."""
    terrain = spec.terrain
    rough = bool(terrain and (terrain.kind == "real" or
                              (terrain.kind in {"hills", "rocky", "craters", "slope"} and terrain.roughness > 0)))
    if spec.domain != "robot_terrain" or robot.kind not in {"walker", "library"} or not rough:
        return requested, None
    chosen, before, after = find_flattest_start(H, terrain.size_m, requested, search_radius_m,
                                                 footprint_radius(robot), avoid)
    if chosen == requested:
        return requested, None
    distance = math.dist(requested, chosen)
    note = (f"Moved the start {distance:.1f} m to flatter ground: {after:g}° instead of {before:g}° "
            "across the robot's footprint.")
    if note not in spec.assumptions:
        spec.assumptions.append(note)
    return chosen, note


def _refresh_default_time_limit(spec: SceneSpec) -> None:
    if spec.time_limit_given:
        return
    from ..timing import time_limit

    if (resolved := time_limit(spec)) is None:
        return
    spec.duration_s, note = resolved
    spec.assumptions = [assumption for assumption in spec.assumptions
                        if not (assumption.startswith("Gave ") and "to reach the goal:" in assumption)]
    if note not in spec.assumptions:
        spec.assumptions.append(note)


def move_start_to_flatter(spec: SceneSpec, search_radius_m: float = 15.0) -> str | None:
    """Move a single legged/library robot's start within ``search_radius_m`` and record why."""
    if spec.domain != "robot_terrain" or len(spec.robots) != 1:
        return None
    entry = spec.robots[0]
    terrain = spec.terrain
    rough = bool(terrain and (terrain.kind == "real" or
                              (terrain.kind in {"hills", "rocky", "craters", "slope"} and terrain.roughness > 0)))
    if entry.robot.kind not in {"walker", "library"} or not rough:
        return None
    original_heights, _ = make_heights(spec)
    requested = resolve_mission(spec, original_heights)["start"]
    heights, _ = make_heights(spec, requested)
    chosen, note = auto_flatter_start(spec, entry.robot, requested, heights, search_radius_m)
    if note is None:
        return None
    entry.start_xy, entry.start_place = list(chosen), None
    spec.mission = spec.mission or entry.mission()
    spec.mission.start_xy, spec.mission.start_place = list(chosen), None
    spec.fill_defaults()
    if note not in spec.assumptions:
        spec.assumptions.append(note)
    return note


def find_gentler_route(spec: SceneSpec, search_radius_m: float = 15.0) -> tuple[tuple[float, float], tuple[float, float], str] | None:
    """Translate a single-robot route to flatter ground without changing its direction or length."""
    if spec.domain != "robot_terrain" or len(spec.robots) != 1:
        return None
    entry = spec.robots[0]
    terrain = spec.terrain
    if terrain is None:
        return None
    base_heights, _ = make_heights(spec)
    mission = resolve_mission(spec, base_heights)
    start = np.asarray(mission["start"], dtype=float)
    if mission["goal"] is None:
        heading = math.radians(mission["heading_deg"])
        goal = start + np.array([math.cos(heading), math.sin(heading)]) * 0.7 * terrain.size_m
    else:
        goal = np.asarray(mission["goal"], dtype=float)
    heights, _ = make_heights(spec, tuple(start))
    cell = terrain.size_m / (GRID - 1)
    dy, dx = np.gradient(heights, cell, cell)
    slopes = np.degrees(np.arctan(np.hypot(dx, dy)))
    footprint = footprint_radius(entry.robot)
    half, safe_edge = terrain.size_m / 2, terrain.size_m / 2 * EDGE_MARGIN

    def at(point: np.ndarray) -> tuple[float, float] | None:
        x, y = float(point[0]), float(point[1])
        if abs(x) + footprint > safe_edge or abs(y) + footprint > safe_edge:
            return None
        col = int(round((x + half) / cell)); row = int(round((y + half) / cell))
        radius_cells = max(0, int(math.ceil(footprint / cell)))
        r0, r1 = row - radius_cells, row + radius_cells + 1
        c0, c1 = col - radius_cells, col + radius_cells + 1
        if r0 < 0 or c0 < 0 or r1 > GRID or c1 > GRID:
            return None
        rows, cols = np.ogrid[r0:r1, c0:c1]
        mask = ((rows - row) * cell) ** 2 + ((cols - col) * cell) ** 2 <= footprint ** 2 + 1e-9
        values = slopes[r0:r1, c0:c1][mask]
        return (float(values.max()), float(values.mean())) if values.size else None

    def route_score(offset: np.ndarray) -> tuple[float, float] | None:
        moved_start, moved_goal = start + offset, goal + offset
        start_score, goal_score = at(moved_start), at(moved_goal)
        if start_score is None or goal_score is None:
            return None
        samples = np.linspace(0.0, 1.0, 31)
        line = moved_start[None, :] + samples[:, None] * (moved_goal - moved_start)[None, :]
        cols = np.clip(np.rint((line[:, 0] + half) / cell).astype(int), 0, GRID - 1)
        rows = np.clip(np.rint((line[:, 1] + half) / cell).astype(int), 0, GRID - 1)
        path_slopes = slopes[rows, cols]
        maximum = max(float(path_slopes.max()), start_score[0], goal_score[0])
        mean = float((path_slopes.mean() + start_score[1] + goal_score[1]) / 3)
        return maximum, mean

    current_score = route_score(np.zeros(2))
    if current_score is None:
        return None
    stride = max(1, int(round(0.5 / cell)))
    candidates = []
    for row in range(0, GRID, stride):
        for col in range(0, GRID, stride):
            offset = np.array([-half + col * cell, -half + row * cell]) - start
            distance = float(np.linalg.norm(offset))
            if distance > search_radius_m or distance < cell:
                continue
            score = route_score(offset)
            if score is not None:
                candidates.append((score[1], score[0], distance, offset))
    if not candidates:
        return None
    best = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    if best[0] >= current_score[1] - 0.5:
        return None
    moved_start, moved_goal = start + best[3], goal + best[3]
    note = (f"Moved the start and goal {best[2]:.1f} m to a gentler route while keeping the same direction and "
            f"distance: average slope {best[0]:.1f}° instead of {current_score[1]:.1f}°.")
    return (tuple(round(float(v), 2) for v in moved_start),
            tuple(round(float(v), 2) for v in moved_goal), note)


def apply_gentler_route(spec: SceneSpec, search_radius_m: float = 15.0) -> str | None:
    """Apply a flatter parallel route to a single robot scene and preserve its travel vector."""
    result = find_gentler_route(spec, search_radius_m)
    if result is None:
        return None
    start, goal, note = result
    entry = spec.robots[0]
    entry.start_xy, entry.start_place = list(start), None
    entry.goal_xy, entry.goal_place = list(goal), None
    entry.waypoints, entry.navigation_grid = [], None
    spec.mission = Mission(start_xy=list(start), goal_xy=list(goal), goal_radius_m=entry.goal_radius_m)
    if note not in spec.assumptions:
        spec.assumptions.append(note)
    spec.fill_defaults()
    _refresh_default_time_limit(spec)
    _refresh_default_time_limit(spec)
    return note


def height_at(H: np.ndarray, size_m: float, x: float, y: float, radius_m: float = 0.45) -> float:
    """Highest ground within `radius_m` of (x, y): the robot spawns just above its own footprint."""
    half, cell = size_m / 2, size_m / (GRID - 1)
    c = int(round((x + half) / cell))
    r = int(round((y + half) / cell))
    k = max(1, int(math.ceil(radius_m / cell)))
    r, c = np.clip(r, 0, GRID - 1), np.clip(c, 0, GRID - 1)
    window = H[max(r - k, 0) : r + k + 1, max(c - k, 0) : c + k + 1]
    return float(window.max())


# ── Mission (start and goal) ───────────────────────────────────────────────

EDGE_MARGIN = 0.85       # keep start and goal inside 85% of the half-width, away from the edge


def place_xy(H: np.ndarray, size_m: float, place: str) -> tuple[float, float]:
    """Turn a named place into coordinates on this terrain."""
    from ..indoor import ROOM_PLACES, get_layout

    if place in ROOM_PLACES:
        return tuple(get_layout()["places"][place])
    half = size_m / 2
    lim = half * EDGE_MARGIN * 0.9
    fixed = {"start_area": default_start(size_m), "center": (0.0, 0.0),
             "north": (0.0, lim), "south": (0.0, -lim), "east": (lim, 0.0), "west": (-lim, 0.0)}
    if place in fixed:
        return fixed[place]
    # lowest / highest point, searched away from the edges, on lightly smoothed ground (not a single bump)
    k = max(1, GRID // 32)
    pad = np.pad(H, k, mode="edge")
    smooth = sum(pad[k + dr : k + dr + GRID, k + dc : k + dc + GRID] for dr in range(-k, k + 1) for dc in range(-k, k + 1))
    m = int(GRID * (1 - EDGE_MARGIN * 0.9) / 2) + 1
    inner = smooth[m:-m, m:-m]
    r, c = np.unravel_index(int(np.argmin(inner) if place == "lowest" else np.argmax(inner)), inner.shape)
    cell = size_m / (GRID - 1)
    return round(-half + (c + m) * cell, 2), round(-half + (r + m) * cell, 2)


def resolve_mission(spec: SceneSpec, H: np.ndarray) -> dict:
    """Start position, heading and (optional) goal, in metres on this terrain."""
    size, ms = spec.terrain.size_m, spec.mission
    lim = size / 2 * EDGE_MARGIN
    limits = (lim, lim)
    if spec.terrain.kind == "indoor":
        from ..indoor import get_layout

        limits = tuple(length / 2 * EDGE_MARGIN for length in get_layout(spec.terrain.preset)["size_m"])
    clamped = []

    def clamp(xy, what):
        x, y = float(np.clip(xy[0], -limits[0], limits[0])), float(np.clip(xy[1], -limits[1], limits[1]))
        if (x, y) != (float(xy[0]), float(xy[1])):
            clamped.append(what)
        return round(x, 2), round(y, 2)

    start, goal, heading = default_start(size), None, 0.0
    if ms is not None:
        if ms.start_xy is not None:
            start = clamp(ms.start_xy, "start")
        elif ms.start_place:
            start = place_xy(H, size, ms.start_place)
            if spec.terrain.kind == "indoor":
                start = clamp(start, "start")
        if ms.goal_xy is not None:
            goal = clamp(ms.goal_xy, "goal")
        elif ms.goal_place:
            goal = place_xy(H, size, ms.goal_place)
            if spec.terrain.kind == "indoor":
                goal = clamp(goal, "goal")
        elif ms.goal_structure is not None:
            goal = place_xy(H, size, "east")   # temporary anchor for placing near-goal structures
        if ms.start_heading_deg is not None:
            heading = float(ms.start_heading_deg)
        elif goal is not None:
            heading = math.degrees(math.atan2(goal[1] - start[1], goal[0] - start[0]))
    return {"start": start, "heading_deg": round(heading, 1), "goal": goal,
            "radius": ms.goal_radius_m if ms else 1.0, "clamped": clamped,
            "waypoints": list(ms.waypoints) if ms else [],
            "waypoint_radius_m": 0.4 if spec.terrain.kind == "indoor" else 1.0,
            "navigation_grid": ms.navigation_grid.model_dump() if ms and ms.navigation_grid else None}


PERTURB_M, PERTURB_DEG = 0.01, 1.0   # the start nudge for repeated runs of one condition (SceneSpec.perturb_seed)


def perturbed(spec: SceneSpec, index: int, start: tuple[float, float], heading_deg: float) -> tuple[tuple[float, float], float]:
    """Robot ``index``'s spawn point and heading, nudged by up to ±1 cm and ±1° when the scene has a perturb_seed.
    The mission (its reported start and goal) and the course stay where they are."""
    if not spec.perturb_seed:
        return start, heading_deg
    rng = np.random.default_rng([spec.perturb_seed, index])
    dx, dy = rng.uniform(-PERTURB_M, PERTURB_M, 2)
    return (start[0] + float(dx), start[1] + float(dy)), heading_deg + float(rng.uniform(-PERTURB_DEG, PERTURB_DEG))


def yaw_quat(heading_deg: float) -> str:
    h = math.radians(heading_deg) / 2
    return f"{math.cos(h):.5f} 0 0 {math.sin(h):.5f}"


def footprint_radius(rb) -> float:
    """Half the diagonal of the robot's footprint, plus a little room."""
    return 0.5 * math.hypot(rb.chassis_length + 0.2, rb.chassis_width + 0.25) + 0.05


# ── Robot ──────────────────────────────────────────────────────────────────

# Wheels and legs collide with the ground but not with each other. MuJoCo pairs geoms when
# (contype_a & conaffinity_b) or (contype_b & conaffinity_a); the ground keeps the default 1/1.
# Without this, the 6-wheel rover's neighbouring wheels (0.3 m apart, 0.15 m radius) touch and
# grind against each other, which can blow up the simulation.
LIMB_COLLISION = 'contype="2" conaffinity="1"'

WHEEL_TIMESTEP_S = 0.002
WHEEL_MAX_GAIN_RATIO = 1.5


def wheel_armature(kv: float, wheel_mass: float, radius: float) -> float:
    """Rotor inertia that keeps the explicit velocity servo stable (kv*dt/I <= 1.5; Euler diverges above 2).

    Small wheels have tiny axle inertia (I = m R^2 / 2), so without it a 0.10 m wheel blows up the
    simulation within 0.2 s. Wheels of about 0.14 m and larger get no armature, so their behaviour is unchanged."""
    inertia = 0.5 * wheel_mass * radius ** 2
    return max(0.0, kv * WHEEL_TIMESTEP_S / WHEEL_MAX_GAIN_RATIO - inertia)


def rover_xml(spec: SceneSpec, start: tuple[float, float, float], heading_deg: float = 0.0) -> tuple[str, str]:
    rb = spec.robot
    L, W, R, ww = rb.chassis_length, rb.chassis_width, rb.wheel_radius, rb.wheel_width
    chassis_h = 0.08
    wheel_mass = 0.08 * rb.mass_kg
    chassis_mass = max(rb.mass_kg - wheel_mass * rb.wheels, 0.2 * rb.mass_kg)
    xs = [L / 2, -L / 2] if rb.wheels == 4 else [L / 2, 0.0, -L / 2]
    kv = 0.6 * rb.mass_kg
    armature = wheel_armature(kv, wheel_mass, R)
    wheels, actuators = [], []
    for i, wx in enumerate(xs):
        for side, wy in (("l", W / 2 + ww / 2 + 0.01), ("r", -(W / 2 + ww / 2 + 0.01))):
            name = f"wheel_{i}{side}"
            wheels.append(
                f'''      <body name="{name}" pos="{wx:.3f} {wy:.3f} 0">
        <joint name="{name}_j" type="hinge" axis="0 1 0" damping="0.05" armature="{armature:.5f}"/>
        <geom type="cylinder" size="{R:.3f} {ww / 2:.3f}" euler="90 0 0" mass="{wheel_mass:.3f}" {LIMB_COLLISION} rgba="0.15 0.15 0.17 1"/>
      </body>'''
            )
            actuators.append(f'    <velocity joint="{name}_j" kv="{kv:.2f}" ctrlrange="-30 30"/>')
    x, y, z = start
    body = f'''    <body name="rover" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{yaw_quat(heading_deg)}">
      <freejoint/>
      <geom name="chassis" type="box" size="{L / 2 + 0.05:.3f} {W / 2:.3f} {chassis_h / 2:.3f}" mass="{chassis_mass:.3f}" rgba="0.95 0.55 0.15 1"/>
      <geom type="box" size="0.08 0.06 0.05" pos="{L / 4:.3f} 0 {chassis_h / 2 + 0.05:.3f}" mass="0.01" rgba="0.2 0.6 0.9 1"/>
{chr(10).join(wheels)}
    </body>'''
    return body, "\n".join(actuators)


KNEE_BIAS = 0.7          # rad: resting knee bend, so legs act like springs
FOOT_R = 0.035
HIP_OUTSET = 0.075       # m: hips sit outside the body edge
JOINT_DAMPING = 2.0      # tuned: lower values let the trot bounce and roll over
OUTER_BOOST = 0.3        # steering: outer legs stride at most 30% further than normal
KNEE_LIFT = 0.3          # rad: extra knee bend while a leg swings forward


def walker_layout(rb) -> list[tuple[str, float, float, float]]:
    """(name, x, y, gait phase) per leg. 4 legs trot (diagonals together); 6 legs use a tripod gait."""
    L, W = rb.chassis_length, rb.chassis_width + 2 * HIP_OUTSET   # hips sit a bit outside the body for a stable stance
    if rb.legs == 4:
        return [("fl", L / 2, W / 2, 0.0), ("fr", L / 2, -W / 2, math.pi),
                ("bl", -L / 2, W / 2, math.pi), ("br", -L / 2, -W / 2, 0.0)]
    return [("fl", L / 2, W / 2, 0.0), ("ml", 0.0, W / 2, math.pi), ("bl", -L / 2, W / 2, 0.0),
            ("fr", L / 2, -W / 2, math.pi), ("mr", 0.0, -W / 2, 0.0), ("br", -L / 2, -W / 2, math.pi)]


def walker_stand_height(rb) -> float:
    return 2 * rb.leg_length * math.cos(KNEE_BIAS / 2) + FOOT_R


def walker_xml(spec: SceneSpec, start: tuple[float, float, float], heading_deg: float = 0.0) -> tuple[str, str]:
    rb = spec.robot
    L, W, l = rb.chassis_length, rb.chassis_width, rb.leg_length
    seg_mass = 0.04 * rb.mass_kg
    torso_mass = max(rb.mass_kg - 2 * seg_mass * rb.legs, 0.3 * rb.mass_kg)
    # Stiff enough to hold the body up and move legs briskly, scaled to the robot's weight.
    kp = 6.0 * rb.mass_kg * max(spec.environment.gravity, 1.0) / rb.legs * l * 4
    legs, actuators = [], []
    for name, x, y, _ in walker_layout(rb):
        legs.append(f'''      <body name="{name}_hip" pos="{x:.3f} {y:.3f} 0">
        <joint name="{name}_hip" type="hinge" axis="0 1 0" range="-70 70" damping="{JOINT_DAMPING}"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-l:.3f}" size="0.03" mass="{seg_mass:.3f}" {LIMB_COLLISION} rgba="0.25 0.27 0.32 1"/>
        <body name="{name}_shin" pos="0 0 {-l:.3f}">
          <joint name="{name}_knee" type="hinge" axis="0 1 0" range="-10 120" damping="{JOINT_DAMPING}"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-l:.3f}" size="0.025" mass="{seg_mass:.3f}" {LIMB_COLLISION} rgba="0.35 0.37 0.42 1"/>
          <geom name="{name}_foot" type="sphere" pos="0 0 {-l:.3f}" size="{FOOT_R}" mass="0.01" {LIMB_COLLISION} rgba="0.15 0.15 0.17 1"/>
        </body>
      </body>''')
        actuators.append(f'    <position joint="{name}_hip" kp="{kp:.1f}" forcerange="{-3 * kp:.1f} {3 * kp:.1f}"/>')
        actuators.append(f'    <position joint="{name}_knee" kp="{kp:.1f}" forcerange="{-3 * kp:.1f} {3 * kp:.1f}"/>')
    x, y, z = start
    body = f'''    <body name="rover" pos="{x:.3f} {y:.3f} {z:.3f}" quat="{yaw_quat(heading_deg)}">
      <freejoint/>
      <geom name="chassis" type="box" size="{L / 2 + 0.06:.3f} {W / 2:.3f} 0.05" mass="{torso_mass:.3f}" rgba="0.2 0.6 0.9 1"/>
      <geom type="box" size="0.06 0.08 0.05" pos="{L / 2 + 0.1:.3f} 0 0.04" mass="0.01" rgba="0.95 0.55 0.15 1"/>
{chr(10).join(legs)}
    </body>'''
    return body, "\n".join(actuators)


def walker_ctrl(rb, t: float, scale: float, turn: float = 0.0) -> np.ndarray:
    """Open-loop gait. Hip angle > 0 swings the foot backward (stance); the knee lifts while it swings forward.
    `scale` (0..1) ramps the gait in smoothly from standing. `turn` > 0 turns left: the inner (left) legs take
    shorter strides (above 0.5 they step backward, turning on the spot) and the outer legs slightly longer
    ones, capped at OUTER_BOOST so no leg strides much past the tuned stride_deg."""
    a = math.radians(rb.stride_deg) * scale
    lift = KNEE_LIFT * scale
    ctrl = []
    for name, _, _, phase in walker_layout(rb):
        side = 1.0 if name.endswith("r") else -1.0
        p = 2 * math.pi * rb.gait_freq_hz * t + phase
        outer = side * turn > 0
        k = 1 + min(abs(turn), OUTER_BOOST) if outer else 1 - 2 * abs(turn)
        hip = -KNEE_BIAS / 2 + a * k * math.sin(p)
        knee = KNEE_BIAS + lift * max(0.0, -math.cos(p))
        ctrl += [hip, knee]
    return np.array(ctrl)


# ── Diagnostics (evidence for the explainer) ──────────────────────────────

SLIP_SPEED = 0.3         # m/s: touching point faster than this = sliding. The walker's feet scuff at ~0.17 m/s
                         # (median) even on firm rock with its simple gait, and ~0.75 m/s on ice, so 0.3 separates them.
FALL_TILT_DEG = 45.0     # tilt where a fall "starts" (used for its direction)
FLIP_TILT_DEG = 100.0    # tilt that counts as tipped over
HUMANOID_FLIP_TILT_DEG = 60.0   # a fallen humanoid lies near 80-90°, below FLIP_TILT_DEG; walking it stays under ~15°
# A run stops once its outcome is decided: it reached the goal, fell (past 60° for a second, or past its own flip
# tilt), or is stuck (moved less than 0.1 m in 3 s). A recorded run keeps going a moment so the viewer sees it.
TIPPED_DEG, TIPPED_HOLD_S = 60.0, 1.0
STUCK_WINDOW_S, STUCK_MOVE_M = 3.0, 0.1
SHOW_AFTER_S = 1.5        # recorded runs: how long to keep showing a decided run
PUSH_AT_S = 2.0           # the sideways push comes 2 s after the robot starts moving, once it is in its gait


def push_time(spec: SceneSpec) -> float:
    """When the push starts, in seconds after the robot starts moving (0 without a push)."""
    if spec.disturbance is None:
        return 0.0
    return max(0.0, min(PUSH_AT_S, spec.duration_s - spec.disturbance.duration_s))


class Outcome:
    """Watches one robot for the moment its outcome is decided."""

    def __init__(self) -> None:
        self.tipped_since: float | None = None
        self.window: deque = deque()
        self.decided: dict | None = None

    def tipped(self, tilt: float, t: float) -> float | None:
        """When a tilt past 60° began, once it has lasted a second (it has fallen); else None."""
        if tilt <= TIPPED_DEG:
            self.tipped_since = None
            return None
        self.tipped_since = t if self.tipped_since is None else self.tipped_since
        return self.tipped_since if t - self.tipped_since >= TIPPED_HOLD_S else None

    def update(self, t: float, xy: np.ndarray, reached_at: float | None, fell: bool,
               heading: np.ndarray | None = None) -> None:
        self.window.append((t, xy, heading.copy() if heading is not None else None))
        while len(self.window) > 1 and self.window[1][0] <= t - STUCK_WINDOW_S:
            self.window.popleft()
        if self.decided is not None:
            return
        if reached_at is not None:
            self.decided = {"reason": "reached", "at_s": round(t, 2)}
        elif fell:
            self.decided = {"reason": "fell", "at_s": round(t, 2)}
        elif self.window[0][0] <= t - STUCK_WINDOW_S and float(np.linalg.norm(xy - self.window[0][1])) < STUCK_MOVE_M \
                and (heading is None or self.window[0][2] is None
                     or float(np.linalg.norm(heading - self.window[0][2])) < 0.1):
            self.decided = {"reason": "stuck", "at_s": round(t, 2)}


def flip_tilt_deg(rb) -> float:
    """The tilt at which this robot counts as fallen over."""
    if rb.kind == "library" and LibraryRegistry().robot(rb.robot_id).kind == "humanoid":
        return HUMANOID_FLIP_TILT_DEG
    return FLIP_TILT_DEG


class TerrainProbe:
    """Reads height and slope of the heightfield at any (x, y), straight from the built model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.ok = False
        g = next((i for i in range(model.ngeom) if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_HFIELD)), None)
        if g is None:
            return
        h = int(model.geom_dataid[g])
        nrow, ncol = int(model.hfield_nrow[h]), int(model.hfield_ncol[h])
        self.hx, self.hy, ztop = (float(v) for v in model.hfield_size[h][:3])
        self.cx, self.cy = float(model.geom_pos[g][0]), float(model.geom_pos[g][1])
        adr = int(model.hfield_adr[h])
        heights = model.hfield_data[adr : adr + nrow * ncol].reshape(nrow, ncol) * ztop  # row 0 = -y, col 0 = -x
        self.heights = heights
        self.cz = float(model.geom_pos[g][2])
        self.dx, self.dy = 2 * self.hx / (ncol - 1), 2 * self.hy / (nrow - 1)
        self.gy, self.gx = np.gradient(heights, self.dy, self.dx)          # dz/dy along rows, dz/dx along columns
        self.nrow, self.ncol, self.ok = nrow, ncol, True

    def _bilinear(self, field: np.ndarray, x: float, y: float) -> float:
        c = np.clip((x - self.cx + self.hx) / self.dx, 0, self.ncol - 1.001)
        r = np.clip((y - self.cy + self.hy) / self.dy, 0, self.nrow - 1.001)
        c0, r0 = int(c), int(r)
        fc, fr = c - c0, r - r0
        top = field[r0, c0] * (1 - fc) + field[r0, c0 + 1] * fc
        bot = field[r0 + 1, c0] * (1 - fc) + field[r0 + 1, c0 + 1] * fc
        return float(top * (1 - fr) + bot * fr)

    def gradient(self, x: float, y: float) -> np.ndarray:
        return np.array([self._bilinear(self.gx, x, y), self._bilinear(self.gy, x, y)])

    def step_height_m(self, x: float, y: float, radius_m: float = 0.5) -> float:
        """Largest abrupt local height change around a point, excluding the median smooth grade."""
        if not self.ok:
            return 0.0
        col = int(np.clip(round((x - self.cx + self.hx) / self.dx), 0, self.ncol - 1))
        row = int(np.clip(round((y - self.cy + self.hy) / self.dy), 0, self.nrow - 1))
        rx, ry = max(1, math.ceil(radius_m / self.dx)), max(1, math.ceil(radius_m / self.dy))
        window = self.heights[max(0, row - ry):min(self.nrow, row + ry + 1),
                              max(0, col - rx):min(self.ncol, col + rx + 1)]
        differences = [np.abs(np.diff(window, axis=axis)).ravel() for axis in (0, 1) if window.shape[axis] > 1]
        if not differences:
            return 0.0
        values = np.concatenate(differences)
        return round(max(0.0, float(values.max() - np.median(values))), 3)

    def describe(self, x: float, y: float, heading: np.ndarray) -> dict:
        """Slope under (x, y), split relative to the robot's heading (a horizontal unit vector)."""
        g = self.gradient(x, y)
        left = np.array([-heading[1], heading[0]])
        slope = math.degrees(math.atan(float(np.hypot(*g))))
        out = {
            "slope_deg": round(slope, 1),
            "along_heading_deg": round(math.degrees(math.atan(float(g @ heading))), 1),   # + = uphill ahead
            "cross_deg": round(math.degrees(math.atan(float(g @ left))), 1),             # + = ground rises to its left
            "downhill": None,
        }
        if slope >= 2.0:
            down = -g / np.linalg.norm(g)
            parts = {"ahead": down @ heading, "behind": -(down @ heading), "left": down @ left, "right": -(down @ left)}
            out["downhill"] = max(parts, key=parts.get)
        return out


def heading_of(xmat: np.ndarray) -> np.ndarray:
    """The robot's forward direction (body x-axis) flattened onto the ground."""
    fwd = xmat.reshape(3, 3)[:2, 0]
    n = float(np.linalg.norm(fwd))
    return fwd / n if n > 1e-6 else np.array([1.0, 0.0])


def lean_direction(xmat: np.ndarray, heading: np.ndarray) -> str:
    """Which way the robot's top is leaning, relative to the heading it had while upright."""
    up = xmat.reshape(3, 3)[:2, 2]
    left = np.array([-heading[1], heading[0]])
    parts = {"forward": up @ heading, "backward": -(up @ heading), "left": up @ left, "right": -(up @ left)}
    return max(parts, key=parts.get)


def instability_count(data: mujoco.MjData) -> int:
    """How many times MuJoCo hit NaN/huge accelerations. It resets the state when that happens,
    so a finite final state alone doesn't prove the run was stable."""
    return int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)


@lru_cache(maxsize=128)
def _read_model_asset(relative_path: str) -> bytes:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"MuJoCo asset path escapes the project root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"MuJoCo asset file does not exist: {relative_path}")
    return path.read_bytes()


def _model_from_xml(xml: str) -> mujoco.MjModel:
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    asset_root = compiler.get("meshdir", "") if compiler is not None else ""
    assets = {}
    for mesh in root.findall("./asset/mesh"):   # library meshes name their file from the project root
        filename = mesh.get("file")
        if filename is None:
            continue
        relative = (Path(asset_root) / filename).as_posix() if asset_root else filename
        assets[relative] = _read_model_asset(relative)
    return mujoco.MjModel.from_xml_string(xml, assets) if assets else mujoco.MjModel.from_xml_string(xml)


def _library_robot_parts(
    spec: SceneSpec,
    start: tuple[float, float, float],
    heading_deg: float,
    prefix: str = "",
) -> tuple[str, str, str, str, str]:
    robot = spec.robot
    registry = LibraryRegistry()
    robot_card = registry.robot(robot.robot_id)
    policy_card = registry.policy(robot.policy_id)
    if policy_card.id not in {card.id for card in registry.compatible_policies(robot_card.id)}:
        raise ValueError(f"Policy {policy_card.id!r} is not compatible with robot {robot_card.id!r}.")
    if robot_card.mjcf_path is None:
        raise ValueError(f"Library robot {robot_card.id!r} does not provide an MJCF file.")

    source_path = (PROJECT_ROOT / robot_card.mjcf_path).resolve()
    try:
        source_path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Library MJCF path escapes the project root.") from exc
    source = ET.fromstring(source_path.read_text(encoding="utf-8"))
    source_compiler = source.find("compiler")
    meshdir = source_compiler.get("meshdir", "") if source_compiler is not None else ""
    asset_root = (Path(robot_card.mjcf_path).parent / meshdir).as_posix()
    asset_xml = source.find("asset")
    for mesh in asset_xml.findall("mesh") if asset_xml is not None else []:   # several robots' meshes can share a scene
        if mesh.get("file"):
            mesh.set("file", (Path(asset_root) / mesh.get("file")).as_posix())
    mesh_xml = "".join(
        ET.tostring(item, encoding="unicode") for item in asset_xml
    ) if asset_xml is not None else ""
    source_defaults = source.find("default")
    defaults_xml = "".join(
        ET.tostring(item, encoding="unicode") for item in source_defaults
    ) if source_defaults is not None else ""
    source_body = next(
        (
            body
            for body in source.findall("./worldbody/body")
            if body.get("name") == robot_card.root_body
        ),
        None,
    )
    if source_body is None:
        raise ValueError(f"Library MJCF does not contain root body {robot_card.root_body!r}.")
    x, y, z = start
    source_body.set("name", "rover")
    source_body.set("pos", f"{x:.4f} {y:.4f} {z:.4f}")
    source_body.set("quat", yaw_quat(heading_deg))
    actuator_root = source.find("actuator")
    actuators_xml = "".join(
        ET.tostring(item, encoding="unicode") for item in actuator_root
    ) if actuator_root is not None else ""
    if prefix:
        body_root = ET.fromstring(ET.tostring(source_body, encoding="unicode"))
        actuator_root = ET.fromstring(f"<actuators>{actuators_xml}</actuators>")
        asset_root_xml = ET.fromstring(f"<assets>{mesh_xml}</assets>")
        defaults_root = ET.fromstring(f"<defaults>{defaults_xml}</defaults>")
        named = {
            value
            for root in (body_root, actuator_root, asset_root_xml, defaults_root)
            for node in root.iter()
            if (value := node.get("name")) is not None
        }
        replacements = {name: f"{prefix}{name}" for name in named}
        references = {"joint", "mesh", "class", "material", "texture", "body", "site", "tendon"}
        for root in (body_root, actuator_root, asset_root_xml, defaults_root):
            for node in root.iter():
                if (name := node.get("name")) in replacements:
                    node.set("name", replacements[name])
                for attribute in references:
                    value = node.get(attribute)
                    if value in replacements:
                        node.set(attribute, replacements[value])
        source_body = body_root
        actuators_xml = "".join(ET.tostring(item, encoding="unicode") for item in actuator_root)
        mesh_xml = "".join(ET.tostring(item, encoding="unicode") for item in asset_root_xml)
        defaults_xml = "".join(ET.tostring(item, encoding="unicode") for item in defaults_root)
    if defaults_xml:   # the robot's own defaults apply to its parts only, not to other robots in the scene
        scope = f"{prefix}{robot_card.id}_defaults"
        defaults_xml = f'<default class="{scope}">{defaults_xml}</default>'
        if not source_body.get("childclass"):
            source_body.set("childclass", scope)
        scoped = ET.fromstring(f"<actuators>{actuators_xml}</actuators>")
        for node in scoped:
            node.set("class", node.get("class") or scope)
        actuators_xml = "".join(ET.tostring(item, encoding="unicode") for item in scoped)
    return (
        ET.tostring(source_body, encoding="unicode"),
        actuators_xml,
        mesh_xml,
        defaults_xml,
        asset_root,
    )


def contact_parts(model: mujoco.MjModel) -> set[int]:
    """Geoms that are meant to touch the ground: walker feet and rover wheels."""
    parts = set()
    for g in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if (
            gname.endswith("_foot")
            or bname.startswith("wheel_")
            or bname.endswith("ankle_roll_link")
            or bname.endswith("_foot")
        ):
            parts.add(g)
    return parts


def _prefix_builtin_parts(body_xml: str, actuators_xml: str, prefix: str) -> tuple[str, str]:
    bodies = ET.fromstring(f"<bodies>{body_xml}</bodies>")
    actuators = ET.fromstring(f"<actuators>{actuators_xml}</actuators>")
    for node in (*bodies.iter(), *actuators.iter()):
        if (name := node.get("name")):
            node.set("name", f"{prefix}{name}")
        if (joint := node.get("joint")):
            node.set("joint", f"{prefix}{joint}")
    root = bodies.find("body")
    if root is not None:
        root.set("name", f"{prefix}rover")
    return (
        "".join(ET.tostring(item, encoding="unicode") for item in bodies),
        "".join(ET.tostring(item, encoding="unicode") for item in actuators),
    )


def _robot_collision_masks(body_xml: str, index: int, count: int) -> str:
    root = ET.fromstring(body_xml)
    own_mask = 1 << (index + 1)
    other_masks = sum(1 << (other + 1) for other in range(count) if other != index)
    for geom in root.iter("geom"):
        if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
            continue   # a visual-only mesh stays visual
        geom.set("contype", str(own_mask))
        geom.set("conaffinity", str(1 | other_masks))
    return ET.tostring(root, encoding="unicode")


def ground_geoms(model: mujoco.MjModel) -> set[int]:
    kinds = {int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_PLANE)}
    return {geom for geom in range(model.ngeom) if int(model.geom_type[geom]) in kinds
            or (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
                ("indoor_floor", "indoor_bedroom_step", "indoor_bedroom_platform", "indoor_threshold", "indoor_ramp"))}


def sliding_contacts(model, data, parts: set[int], ground: set[int], speeds: list | None = None) -> tuple[int, int]:
    """(touching, sliding) counts for foot/wheel-ground contacts right now; appends each point's speed to `speeds`."""
    touching = sliding = 0
    vel = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        part = g1 if g1 in parts and g2 in ground else g2 if g2 in parts and g1 in ground else None
        if part is None:
            continue
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, part, vel, 0)   # world frame: [angular, linear]
        v_point = vel[3:] + np.cross(vel[:3], np.asarray(c.pos) - data.geom_xpos[part])  # velocity of the touching point
        speed = float(np.hypot(v_point[0], v_point[1]))
        if speeds is not None:
            speeds.append(speed)
        touching += 1
        sliding += int(speed > SLIP_SPEED)
    return touching, sliding


_SIMPLIFIED: dict[tuple, dict] = {}   # the same robot mesh, simplified once per process


def _simplified(vertices: np.ndarray, faces: np.ndarray, target_vertices: int) -> dict:
    """A mesh simplified for the browser, as lists; cached by its exact contents."""
    key = (hashlib.sha1(vertices.tobytes() + faces.tobytes()).hexdigest(), target_vertices)
    if key not in _SIMPLIFIED:
        if len(_SIMPLIFIED) > 512:
            _SIMPLIFIED.clear()
        small_vertices, small_faces = _decimate_mesh(vertices, faces, target_vertices)
        _SIMPLIFIED[key] = {"vertices": np.round(small_vertices, 3).reshape(-1).tolist(),
                            "faces": small_faces.reshape(-1).tolist()}
    cached = _SIMPLIFIED[key]
    return {"vertices": list(cached["vertices"]), "faces": list(cached["faces"])}


def _decimate_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) <= target_vertices:
        return vertices, faces

    minimum = vertices.min(axis=0)
    cell_low, cell_high = 0.0, float(np.ptp(vertices, axis=0).max())
    clustered_ids = None
    cluster_count = 0
    for _ in range(20):
        cell = (cell_low + cell_high) / 2
        keys = np.floor((vertices - minimum) / max(cell, 1e-9)).astype(np.int32)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        candidate_count = int(inverse.max()) + 1
        if candidate_count > target_vertices:
            cell_low = cell
        else:
            cell_high = cell
            clustered_ids = inverse
            cluster_count = candidate_count
    if clustered_ids is None:
        raise ValueError("Could not simplify MuJoCo mesh to the requested scene size.")

    counts = np.bincount(clustered_ids, minlength=cluster_count)
    simplified_vertices = np.column_stack(
        [
            np.bincount(clustered_ids, weights=vertices[:, axis], minlength=cluster_count) / counts
            for axis in range(3)
        ]
    )
    simplified_faces = clustered_ids[faces]
    distinct = (
        (simplified_faces[:, 0] != simplified_faces[:, 1])
        & (simplified_faces[:, 1] != simplified_faces[:, 2])
        & (simplified_faces[:, 0] != simplified_faces[:, 2])
    )
    simplified_faces = simplified_faces[distinct]
    if len(simplified_faces):
        _, unique = np.unique(np.sort(simplified_faces, axis=1), axis=0, return_index=True)
        simplified_faces = simplified_faces[np.sort(unique)]
        used, remapped = np.unique(simplified_faces, return_inverse=True)
        simplified_vertices = simplified_vertices[used]
        simplified_faces = remapped.reshape(-1, 3)
    return simplified_vertices, simplified_faces


class CourseTracker:
    """Follows the robot through an obstacle course: what it touched, what it got past, what stopped it."""

    def __init__(self, model: mujoco.MjModel, scene: dict, rover: int) -> None:
        self.placed = scene.get("course") or []
        self.on = bool(self.placed) and rover >= 0
        if not self.on:
            return
        ms = scene.get("mission") or {}
        self.origin = np.array(ms["start"], float)
        if ms.get("goal") is not None:
            d = np.array(ms["goal"], float) - self.origin
        else:
            h = math.radians(ms.get("heading_deg", 0.0))
            d = np.array([math.cos(h), math.sin(h)])
        self.dir = d / (np.linalg.norm(d) or 1.0)
        self.bridge_far_edges = {}
        for structure in scene.get("structures") or []:
            gap = structure.get("gap") or {}
            if structure.get("kind") == "bridge" and gap.get("center") and structure.get("dimensions", {}).get("span_m"):
                center = np.asarray(gap["center"], dtype=float)
                self.bridge_far_edges[tuple(gap["center"])] = (
                    float((center - self.origin) @ self.dir) + float(structure["dimensions"]["span_m"]) / 2
                )
        self.robot = {g for g in range(model.ngeom) if int(model.body_rootid[int(model.geom_bodyid[g])]) == rover}
        self.owner = {}
        for i, p in enumerate(self.placed):
            for gname in p["geoms"]:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
                if gid >= 0:
                    self.owner[gid] = i
        self.touched: list[float | None] = [None] * len(self.placed)
        self.max_s, self.s_stop, self.stop_xy = 0.0, 0.0, None

    def along(self, xy: np.ndarray) -> float:
        return float((xy - self.origin) @ self.dir)

    def update(self, data: mujoco.MjData, xy: np.ndarray, t: float, fallen: bool) -> None:
        if not self.on:
            return
        for k in range(data.ncon):
            g1, g2 = int(data.contact[k].geom1), int(data.contact[k].geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.owner and b in self.robot and self.touched[self.owner[a]] is None:
                    self.touched[self.owner[a]] = t
        if not fallen:                               # where it was last on its feet / wheels
            s = self.along(xy)
            self.max_s = max(self.max_s, s)
            self.s_stop, self.stop_xy = s, xy.copy()

    def summary(self, spec: SceneSpec, reached: bool) -> dict:
        if not self.on:
            return {}
        rows, stopped = [], None
        for i, p in enumerate(self.placed):
            lateral = abs(float((np.array(p["xy"]) - self.origin) @ np.array([-self.dir[1], self.dir[0]])))
            on_path = lateral < course.COURSE_WIDTH / 2 + 0.5
            far_edge = p["along"] + p["length"] / 2
            if p["kind"] == "gap":
                far_edge = max(far_edge, self.bridge_far_edges.get(tuple(p["xy"]), far_edge))
            cleared = (self.max_s > far_edge + 0.3) if on_path else None
            rows.append({"name": p["name"], "kind": p["kind"], "size": p["size"], "height_m": p["height_m"],
                         **({"width_m": p["width_m"]} if p["kind"] == "gap" else {}),
                         "along_path_m": p["along"], "on_path": on_path,
                         "touched_at_s": self.touched[i], "cleared": cleared})
        if not reached:
            # the first obstacle on the path it didn't get past, if it got close to it
            ahead = [r for r, p in zip(rows, self.placed) if r["on_path"] and not r["cleared"]
                     and p["along"] - p["length"] / 2 - 1.2 <= self.s_stop]
            if ahead:
                stopped = min(ahead, key=lambda r: r["along_path_m"])
                stopped = {k: stopped[k] for k in ("name", "kind", "size", "height_m", "width_m", "touched_at_s") if k in stopped}
        return {
            "obstacles": rows[:15],
            "obstacles_total": len(rows),
            "obstacles_cleared": sum(1 for r in rows if r["cleared"]),
            "stopped_by": stopped,
            "robot_rough_limits": course.capability(spec),
        }


def support_height_offset(robot) -> float:
    if robot.kind == "rover":
        return robot.wheel_radius
    if robot.kind == "walker":
        return walker_stand_height(robot)
    return LibraryRegistry().robot(robot.robot_id).standing_height_m


class Navigator:
    """Steers toward the goal: compares the robot's heading with the direction to the goal."""

    GAIN = 1.0
    MAX_TURN = 1.4          # beyond 1, the inner side reverses and the robot turns on the spot
    WALKER_MAX_TURN = 1.0
    STOP_FADE_S = 0.5

    def __init__(self, mission: dict | None, robot_kind: str) -> None:
        self.goal = np.array(mission["goal"]) if mission and mission.get("goal") is not None else None
        self.radius = float(mission.get("radius", 1.0)) if mission else 1.0
        self.goal_z = mission.get("goal_z") if mission and mission.get("goal_structure") else None
        self.waypoints = [np.array(point, dtype=float) for point in mission.get("waypoints", [])] if mission else []
        self.waypoint_radius = float(mission.get("waypoint_radius_m", 0.3)) if mission else 0.3
        self.navigation_grid = mission.get("navigation_grid") if mission else None
        self.reached_at: float | None = None
        self.max_turn = self.MAX_TURN if robot_kind == "rover" else self.WALKER_MAX_TURN   # walkers wobble when one side steps backward
        self.robot_kind = robot_kind

    def command(self, pos_xy: np.ndarray, heading: np.ndarray, t: float,
                surface_z: float | None = None) -> tuple[float, float]:
        """(turn, go) for this instant."""
        if self.goal is None:
            return 0.0, 1.0
        while self.waypoints and float(np.linalg.norm(self.waypoints[0] - pos_xy)) <= self.waypoint_radius:
            if self.navigation_grid:
                from ..routing import segment_cells

                target = self.waypoints[1] if len(self.waypoints) > 1 else self.goal
                cells = segment_cells(self.navigation_grid, pos_xy, target)
                allowed = self.navigation_grid["allowed"]
                if cells is None or any(not allowed[row][col] for row, col in cells):
                    break
            self.waypoints.pop(0)
        at_height = self.goal_z is None or (surface_z is not None and abs(surface_z - self.goal_z) <= 0.15)
        if not self.waypoints and self.reached_at is None and float(np.linalg.norm(self.goal - pos_xy)) <= self.radius and at_height:
            self.reached_at = t
        if self.reached_at is not None:
            return 0.0, float(np.clip(1 - (t - self.reached_at) / self.STOP_FADE_S, 0.0, 1.0))
        d = (self.waypoints[0] if self.waypoints else self.goal) - pos_xy
        err = math.atan2(d[1], d[0]) - math.atan2(heading[1], heading[0])
        err = (err + math.pi) % (2 * math.pi) - math.pi
        go = max(0.0, math.cos(err)) if self.navigation_grid else 1.0
        if self.navigation_grid and self.robot_kind == "rover" and abs(err) > 0.1:
            go = 0.0
        if self.navigation_grid and self.robot_kind == "rover" and self.waypoints:
            go *= min(1.0, float(np.linalg.norm(d)) / 0.8)
        if self.navigation_grid and self.robot_kind == "walker":
            go = max(0.2, go)
        gain = 3.0 if self.navigation_grid and self.robot_kind == "rover" else self.GAIN
        return float(np.clip(gain * err, -self.max_turn, self.max_turn)), go


# ── Backend ────────────────────────────────────────────────────────────────

class MuJoCoSimulator(Simulator):
    domain = "robot_terrain"
    name = "MuJoCo"
    language = "mjcf-xml"

    def __init__(self) -> None:
        self._summary: dict = {}
        self.mission: dict = {}
        self.missions: dict[str, dict] = {}
        self.robot_entries: dict[str, dict] = {}
        self.structure_placements: list[dict] = []
        self.structure_notes: list[str] = []

    def _resolve_structure_goals(self, spec: SceneSpec) -> None:
        targets = [entry for entry in spec.robots if entry.goal_structure is not None or entry.start_structure is not None]
        if not targets:
            return
        from ..structures import structure_goal_xy

        for entry in targets:
            mission = self.missions.get(entry.id)
            if mission is None and spec.robots and entry.id == spec.robots[0].id:
                mission = self.mission
            if mission is None:
                raise ValueError(f"No resolved mission for robot {entry.id!r}.")
            if entry.start_structure is not None:
                start = structure_goal_xy(self.structure_placements, entry.start_structure)
                entry.start_xy, entry.start_place = list(start), None
                mission["start"] = tuple(start)
            if entry.goal_structure is None:
                continue
            xy = structure_goal_xy(self.structure_placements, entry.goal_structure)
            entry.goal_xy, entry.goal_place = list(xy), None
            start = mission["start"]
            mission["goal"] = tuple(xy)
            mission["goal_structure"] = entry.goal_structure
            point = next(record["access_points"][entry.goal_structure] for record in self.structure_placements
                         if entry.goal_structure in record.get("access_points", {}))
            mission["goal_z"] = point["z"]
            if entry.goal_structure == "habitat_airlock":
                waypoints = []
                bridges = sorted((record for record in self.structure_placements if record["kind"] == "bridge"),
                                 key=lambda record: math.dist(mission["start"], [record["position"]["x"], record["position"]["y"]]))
                for bridge in bridges:
                    yaw = math.radians(bridge["orientation"]["yaw_deg"])
                    distance = bridge["dimensions"]["span_m"] / 2 + 0.8
                    for sign in (-1, 1):
                        waypoints.append([bridge["position"]["x"] + sign * math.cos(yaw) * distance,
                                          bridge["position"]["y"] + sign * math.sin(yaw) * distance])
                habitat = next(record for record in self.structure_placements if record["kind"] == "habitat")
                yaw = math.radians(habitat["orientation"]["yaw_deg"])
                distance = habitat["dimensions"]["module_length_m"] / 2 + habitat["dimensions"]["ramp"]["run_m"] + 0.8
                waypoints.append([habitat["position"]["x"] + math.cos(yaw) * distance,
                                  habitat["position"]["y"] + math.sin(yaw) * distance])
                mission["waypoints"] = waypoints
                mission["waypoint_radius_m"] = 0.3
            if entry.start_heading_deg is None:
                mission["heading_deg"] = round(math.degrees(math.atan2(xy[1] - start[1], xy[0] - start[0])), 1)

        if spec.robots:
            spec.robot = spec.robots[0].robot.model_copy(deep=True)
            spec.mission = spec.robots[0].mission()
        _refresh_default_time_limit(spec)
        self.mission = self.missions.get(spec.robots[0].id, self.mission) if spec.robots else self.mission

    def _shift_structure_heights(self, offset: float) -> None:
        for record in self.structure_placements:
            record["position"]["z"] = round(record["position"]["z"] + offset, 3)
            for point in record.get("access_points", {}).values():
                point["z"] = round(point["z"] + offset, 3)
            ramp = record.get("dimensions", {}).get("ramp", {})
            for key in ("top_z", "toe_z"):
                if key in ramp:
                    ramp[key] += offset
            for point in record.get("gap", {}).get("bearings", []):
                point["z"] += offset
        missions = [self.mission, *self.missions.values()]
        seen = set()
        for mission in missions:
            if id(mission) not in seen and "goal_z" in mission:
                mission["goal_z"] += offset
                seen.add(id(mission))

    def _multi_template(self, spec: SceneSpec) -> tuple[str, np.ndarray, float]:
        terrain = spec.terrain
        environment = spec.environment
        count = len(spec.robots)
        lane_spacing = max(1.5, max(2 * footprint_radius(entry.robot) for entry in spec.robots))
        initial_heights, _ = make_heights(spec)
        first = spec.robots[0]
        base_start = (
            tuple(first.start_xy)
            if first.start_xy is not None
            else place_xy(initial_heights, terrain.size_m, first.start_place)
            if first.start_place is not None
            else default_start(terrain.size_m)
        )
        if first.goal_xy is not None:
            target = np.asarray(first.goal_xy, dtype=float)
        elif first.goal_place is not None:
            target = np.asarray(place_xy(initial_heights, terrain.size_m, first.goal_place), dtype=float)
        elif first.start_heading_deg is not None:
            angle = math.radians(first.start_heading_deg)
            target = np.asarray(base_start) + np.array([math.cos(angle), math.sin(angle)])
        else:
            target = np.asarray(base_start) + np.array([1.0, 0.0])
        direction = target - np.asarray(base_start)
        direction /= np.linalg.norm(direction) or 1.0
        normal = np.array([-direction[1], direction[0]])
        requested_starts = []
        for index, entry in enumerate(spec.robots):
            if entry.start_xy is not None:
                start = tuple(entry.start_xy)
            elif entry.start_place is not None:
                start = place_xy(initial_heights, terrain.size_m, entry.start_place)
            else:
                offset = (index - (count - 1) / 2) * lane_spacing
                start = tuple(np.asarray(base_start) + normal * offset)
            requested_starts.append(start)

        starts = []
        moved_start = False
        for index, (entry, requested_start) in enumerate(zip(spec.robots, requested_starts)):
            avoid = [(starts[other_index], footprint_radius(spec.robots[other_index].robot))
                     for other_index in range(len(starts))]
            avoid.extend((requested_starts[other_index], footprint_radius(spec.robots[other_index].robot))
                         for other_index in range(index + 1, len(spec.robots)))
            start = requested_start
            if entry.start_xy is None:
                start, note = auto_flatter_start(spec, entry.robot, start, initial_heights, avoid=avoid)
                entry.start_xy, entry.start_place = list(start), None
                moved_start = moved_start or note is not None
            starts.append(start)
        if moved_start:
            _refresh_default_time_limit(spec)
        H, zmax = make_heights(spec, starts[0])
        self.missions = {}
        self.robot_entries = {}
        for entry, (sx, sy) in zip(spec.robots, starts):
            start_xy = (float(sx), float(sy))
            goal_xy = entry.goal_xy
            if goal_xy is None and entry.goal_place is not None:
                goal_xy = place_xy(H, terrain.size_m, entry.goal_place)
            mission_input = Mission(
                start_xy=list(start_xy),
                start_heading_deg=entry.start_heading_deg,
                goal_xy=goal_xy,
                goal_radius_m=entry.goal_radius_m,
                waypoints=entry.waypoints,
                navigation_grid=entry.navigation_grid,
            )
            robot_spec = spec.model_copy(deep=True)
            robot_spec.robot = entry.robot.model_copy(deep=True)
            robot_spec.mission = mission_input
            self.missions[entry.id] = resolve_mission(robot_spec, H)

        self.mission = self.missions[spec.robots[0].id]
        from ..structures import place_structures

        self.structure_placements, self.structure_notes = place_structures(spec, H)
        self._resolve_structure_goals(spec)
        obstacle_xml, self.course, self.course_notes = "", [], []
        from ..structures import structure_geoms

        if spec.obstacles:
            mission = self.mission
            goal = mission["goal"]
            if goal is None:
                h = math.radians(mission["heading_deg"])
                goal = (
                    mission["start"][0] + 0.7 * terrain.size_m * math.cos(h),
                    mission["start"][1] + 0.7 * terrain.size_m * math.sin(h),
                )
            slots, self.course_notes = course.plan_course(spec, H, mission["start"], goal, terrain.seed)
            if any(slot["ob"].kind == "gap" for slot in slots):
                course.cut_gaps(slots, H, terrain.size_m)
                self._shift_structure_heights(-float(H.min()))
                H -= H.min()
                zmax = max(float(H.max()), 0.01)
            obstacle_xml, self.course = course.build_geoms(slots, H, terrain.size_m, terrain.seed)

        structure_xml = structure_geoms(self.structure_placements)
        robot_xml, actuator_xml, mesh_xml, defaults_xml = [], [], [], []   # two G1s share one copy of their defaults
        compiler_assets = set()
        markers = []
        for robot_index, entry in enumerate(spec.robots):
            prefix = f"{entry.id}_"
            robot_spec = spec.model_copy(deep=True)
            robot_spec.robot = entry.robot.model_copy(deep=True)
            mission = self.missions[entry.id]
            (sx, sy), heading = perturbed(spec, robot_index, mission["start"], mission["heading_deg"])
            if entry.robot.kind == "walker":
                sz = height_at(H, terrain.size_m, sx, sy, footprint_radius(entry.robot)) + walker_stand_height(entry.robot) + 0.05
                body, actuators = walker_xml(robot_spec, (sx, sy, sz), heading)
                body, actuators = _prefix_builtin_parts(body, actuators, prefix)
            elif entry.robot.kind == "library":
                robot_card = LibraryRegistry().robot(entry.robot.robot_id)
                sz = height_at(H, terrain.size_m, sx, sy, footprint_radius(entry.robot)) + robot_card.standing_height_m
                body, actuators, meshes, defaults, asset_root = _library_robot_parts(
                    robot_spec, (sx, sy, sz), heading, prefix
                )
                compiler_assets.add(asset_root)
                mesh_xml.append(meshes)
                defaults_xml.append(defaults)
            else:
                sz = height_at(H, terrain.size_m, sx, sy, footprint_radius(entry.robot)) + entry.robot.wheel_radius + 0.1
                body, actuators = rover_xml(robot_spec, (sx, sy, sz), heading)
                body, actuators = _prefix_builtin_parts(body, actuators, prefix)
            body = _robot_collision_masks(body, robot_index, count)
            robot_xml.append(body)
            actuator_xml.append(actuators)
            self.robot_entries[entry.id] = {
                "label": entry.label,
                "robot": entry.robot.model_dump(),
                "mission": mission,
                "root_body_name": f"{prefix}rover",
            }
            if mission["goal"] is not None:
                gx, gy = mission["goal"]
                gz = mission.get("goal_z", height_at(H, terrain.size_m, gx, gy, mission["radius"]))
                goal_mark = next((marker for marker in markers if marker[0] == (gx, gy, mission["radius"])), None)
                if goal_mark is None:
                    markers.append(((gx, gy, mission["radius"]), f'''
    <geom name="{prefix}goal_zone" type="cylinder" size="{mission["radius"]:.2f} 0.02" pos="{gx:.2f} {gy:.2f} {gz + 0.02:.3f}" contype="0" conaffinity="0" rgba="0.2 0.85 0.5 0.35"/>
    <geom name="{prefix}goal_flag" type="capsule" fromto="{gx:.2f} {gy:.2f} {gz:.3f} {gx:.2f} {gy:.2f} {gz + 1.5:.3f}" size="0.03" contype="0" conaffinity="0" rgba="0.2 0.85 0.5 1"/>'''))

        compiler_xml = '<compiler angle="radian"/>' if compiler_assets else ""
        half = terrain.size_m / 2
        elevation = " ".join(f"{value:.4f}" for value in np.flipud(H / zmax).ravel())
        marker_xml = "\n".join(marker[1] for marker in markers)
        obstacle_xml = obstacle_xml + "\n" if obstacle_xml else ""
        xml = f'''<mujoco model="{spec.title}">
  {compiler_xml}
  <option timestep="0.002" gravity="0 0 {-environment.gravity:.4f}"/>
  <default><geom friction="{environment.friction:.3f} 0.005 0.0001"/>{''.join(dict.fromkeys(defaults_xml))}</default>
  <asset>
    <hfield name="terrain" nrow="{GRID}" ncol="{GRID}" size="{half:.3f} {half:.3f} {zmax:.4f} 0.5" elevation="{elevation}"/>
{''.join(mesh_xml)}
  </asset>
  <worldbody>
    <light pos="0 0 20" dir="0 0 -1"/>
    <geom name="ground" type="hfield" hfield="terrain" rgba="{' '.join(f'{c:.2f}' for c in terrain_rgb(spec))} 1"/>
{marker_xml}
{structure_xml}
{obstacle_xml}
{''.join(robot_xml)}
  </worldbody>
  <actuator>
{''.join(actuator_xml)}
  </actuator>
</mujoco>'''
        return xml, H, zmax

    def template(self, spec: SceneSpec) -> tuple[str, np.ndarray, float]:
        from ..structures import prepare_structure_terrain

        prepare_structure_terrain(spec)
        if len(spec.robots) > 1:
            return self._multi_template(spec)
        t, env = spec.terrain, spec.environment
        H, zmax = make_heights(spec)
        mission = resolve_mission(spec, H)
        if spec.mission is None or spec.mission.start_xy is None:
            H, zmax = make_heights(spec, mission["start"])
            start, _ = auto_flatter_start(spec, spec.robot, mission["start"], H)
            if start != mission["start"]:
                spec.mission = spec.mission or Mission()
                spec.mission.start_xy, spec.mission.start_place = list(start), None
                if spec.robots:
                    spec.robots[0].start_xy, spec.robots[0].start_place = list(start), None
                H, zmax = make_heights(spec, start)
                mission = resolve_mission(spec, H)
                _refresh_default_time_limit(spec)
        if mission["start"] != default_start(t.size_m):        # smooth the landing spot where the robot really starts
            H, zmax = make_heights(spec, mission["start"])
        self.mission = mission
        from ..structures import place_structures

        self.structure_placements, self.structure_notes = place_structures(spec, H)
        self._resolve_structure_goals(spec)
        self.course, self.course_notes, obstacle_xml = [], [], ""
        from ..structures import structure_geoms

        if spec.obstacles:
            goal = mission["goal"]
            if goal is None:     # no goal (e.g. one was removed): run the course straight ahead
                h = math.radians(mission["heading_deg"])
                goal = (mission["start"][0] + 0.7 * t.size_m * math.cos(h), mission["start"][1] + 0.7 * t.size_m * math.sin(h))
            slots, self.course_notes = course.plan_course(spec, H, mission["start"], goal, t.seed)
            if any(sl["ob"].kind == "gap" for sl in slots):
                course.cut_gaps(slots, H, t.size_m)
                self._shift_structure_heights(-float(H.min()))
                H -= H.min()
                zmax = max(float(H.max()), 0.01)
            obstacle_xml, self.course = course.build_geoms(slots, H, t.size_m, t.seed)
            obstacle_xml += "\n" if obstacle_xml else ""
        structure_xml = structure_geoms(self.structure_placements)
        half = t.size_m / 2
        (sx, sy), yaw = perturbed(spec, 0, mission["start"], mission["heading_deg"])
        mesh_xml, defaults_xml, compiler_xml = "", "", ""
        if spec.robot.kind == "walker":
            sz = height_at(H, t.size_m, sx, sy, footprint_radius(spec.robot)) + walker_stand_height(spec.robot) + 0.05
            robot, actuators = walker_xml(spec, (sx, sy, sz), yaw)
        elif spec.robot.kind == "library":
            robot_card = LibraryRegistry().robot(spec.robot.robot_id)
            ground_z = height_at(H, t.size_m, sx, sy, footprint_radius(spec.robot))
            sz = ground_z + robot_card.standing_height_m
            robot, actuators, mesh_xml, defaults_xml, asset_root = _library_robot_parts(
                spec, (sx, sy, sz), yaw
            )
            compiler_xml = '<compiler angle="radian"/>'
        else:
            sz = height_at(H, t.size_m, sx, sy, footprint_radius(spec.robot)) + spec.robot.wheel_radius + 0.1
            robot, actuators = rover_xml(spec, (sx, sy, sz), yaw)
        markers = ""
        if mission["goal"] is not None:     # visual only: no collisions, no mass
            gx, gy = mission["goal"]
            gz, r = mission.get("goal_z", height_at(H, t.size_m, gx, gy, mission["radius"])), mission["radius"]
            markers = (f'    <geom name="goal_zone" type="cylinder" size="{r:.2f} 0.02" pos="{gx:.2f} {gy:.2f} {gz + 0.02:.3f}" '
                       f'contype="0" conaffinity="0" rgba="0.2 0.85 0.5 0.35"/>\n'
                       f'    <geom name="goal_flag" type="capsule" fromto="{gx:.2f} {gy:.2f} {gz:.3f} {gx:.2f} {gy:.2f} {gz + 1.5:.3f}" '
                       f'size="0.03" contype="0" conaffinity="0" rgba="0.2 0.85 0.5 1"/>\n')
        elevation = " ".join(f"{v:.4f}" for v in np.flipud(H / zmax).ravel())  # MuJoCo reads top row first
        f = env.friction
        xml = f'''<mujoco model="{spec.title}">
  {compiler_xml}
  <option timestep="0.002" gravity="0 0 {-env.gravity:.4f}"/>
  <default><geom friction="{f:.3f} 0.005 0.0001"/>{defaults_xml}</default>
  <asset>
    <hfield name="terrain" nrow="{GRID}" ncol="{GRID}" size="{half:.3f} {half:.3f} {zmax:.4f} 0.5" elevation="{elevation}"/>
{mesh_xml}
  </asset>
  <worldbody>
    <light pos="0 0 20" dir="0 0 -1"/>
    <geom name="ground" type="hfield" hfield="terrain" rgba="{' '.join(f'{c:.2f}' for c in terrain_rgb(spec))} 1"/>
{markers}{structure_xml}{obstacle_xml}{robot}
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>'''
        return xml, H, zmax

    def build(self, spec: SceneSpec, code: str | None = None, view: bool = True) -> BuildResult:
        """The scene's MJCF and, with ``view``, what the browser draws (geoms, terrain, simplified meshes). A batch
        run is never drawn, so it builds without the view: simplifying a library robot's meshes is most of a
        build."""
        template_xml, _, _ = self.template(spec)      # also resolves the start and goal
        if spec.terrain.kind == "indoor":
            from ..indoor import geometry_xml
            import xml.etree.ElementTree as ET

            root = ET.fromstring(template_xml)
            ground = root.find(".//geom[@name='ground']")
            ground.set("contype", "0")
            ground.set("conaffinity", "0")
            worldbody = root.find("worldbody")
            for geom in ET.fromstring("<worldbody>" + geometry_xml(spec.terrain.preset) + "</worldbody>"):
                worldbody.append(geom)
            template_xml = ET.tostring(root, encoding="unicode")
        xml = template_xml if code is None else code
        model = _model_from_xml(xml)                  # raises with a readable error if invalid
        if view:
            fixed_event = {"type": "scene", "code": xml, "language": self.language}
            scene_budget = SCENE_EVENT_BUDGET_BYTES - len(json.dumps(fixed_event).encode("utf-8")) - 1024
            if scene_budget <= 0:
                raise ValueError("MuJoCo scene code exceeds the 5 MB event payload budget.")
            scene = self._scene(model, spec, scene_budget)
        else:
            scene = {"backend": self.name, "kind": "mujoco"}
        scene["mission"] = self.mission
        if spec.terrain.kind == "indoor":
            from ..indoor import get_layout

            scene["indoor"] = get_layout(spec.terrain.preset)
        scene["terrain_size_m"] = spec.terrain.size_m
        scene["course"] = [vars(p) for p in self.course]
        scene["course_notes"] = self.course_notes
        scene["structures"] = self.structure_placements
        scene["structure_notes"] = self.structure_notes
        bridges = [item for item in self.structure_placements if item["kind"] == "bridge"]
        if bridges:
            from ..structural.checks import evaluate_structure

            robot_loads = []
            for entry in spec.robots:
                root_name = self.robot_entries[entry.id]["root_body_name"] if len(spec.robots) > 1 else "rover"
                root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
                if root < 0:
                    raise ValueError(f"Cannot determine bridge loading for robot {entry.id!r}.")
                mass = float(model.body_mass[model.body_rootid == root].sum())
                robot_loads.append((mass, entry.label))
            mass, label = max(robot_loads, key=lambda item: item[0])
            for bridge in bridges:
                bridge["structural_check"] = evaluate_structure(
                    bridge, robot_mass_kg=mass,
                    gravity_m_s2=float(np.linalg.norm(model.opt.gravity)),
                    robot_label=label,
                )
        frames = [item for item in self.structure_placements if item["kind"] in ("habitat", "lander")]
        if frames:
            from ..structural.checks import evaluate_structure

            gravity = float(np.linalg.norm(model.opt.gravity))
            for structure in frames:
                structure["structural_check"] = evaluate_structure(structure, gravity_m_s2=gravity)
        if self.structure_placements:
            from ..next_actions import structure_actions

            for structure in self.structure_placements:
                structure["next_actions"] = structure_actions(structure)
        if len(spec.robots) > 1:
            for robot_id, info in self.robot_entries.items():
                root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info["root_body_name"])
                chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{robot_id}_chassis")
                if chassis_id < 0:
                    candidates = [
                        geom for geom in range(model.ngeom)
                        if int(model.body_rootid[int(model.geom_bodyid[geom])]) == root_id
                        and model.geom_contype[geom] != 0
                    ]
                    chassis_id = candidates[0] if candidates else -1
                info["root_body"] = root_id
                info["follow_geom"] = chassis_id
                info["mission"] = self.missions[robot_id]
            scene["robots"] = self.robot_entries
            scene["follow_geom"] = self.robot_entries[spec.robots[0].id]["follow_geom"]
            scene["robot_root_body"] = self.robot_entries[spec.robots[0].id]["root_body"]
        if not view:
            return BuildResult(code=xml, language=self.language, scene=scene)
        event_size = len(
            json.dumps(
                {"type": "scene", "scene": scene, "code": xml, "language": self.language}
            ).encode("utf-8")
        )
        if event_size > SCENE_EVENT_BUDGET_BYTES:
            raise ValueError(f"MuJoCo scene event exceeds the 5 MB budget ({event_size} bytes).")
        return BuildResult(code=xml, language=self.language, scene=scene)

    def _scene(
        self,
        model: mujoco.MjModel,
        spec: SceneSpec,
        payload_budget: int = SCENE_EVENT_BUDGET_BYTES,
    ) -> dict:
        geoms = []
        for g in range(model.ngeom):
            gtype = int(model.geom_type[g])
            item = {
                "id": g,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
                "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "",
                "type": mujoco.mjtGeom(gtype).name.replace("mjGEOM_", "").lower(),
                "size": model.geom_size[g].tolist(),
                "rgba": model.geom_rgba[g].tolist(),
            }
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                item["mesh_id"] = int(model.geom_dataid[g])
            elif gtype == mujoco.mjtGeom.mjGEOM_HFIELD:
                h = int(model.geom_dataid[g])
                item["hfield"] = {
                    "nrow": int(model.hfield_nrow[h]),
                    "ncol": int(model.hfield_ncol[h]),
                    "size": model.hfield_size[h].tolist(),
                    # stored order: row 0 at -y, col 0 at -x
                    "data": np.round(model.hfield_data[model.hfield_adr[h] : model.hfield_adr[h] + model.hfield_nrow[h] * model.hfield_ncol[h]], 4).tolist(),
                }
            geoms.append(item)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = {
            "backend": self.name,
            "kind": "mujoco",
            "world": {"body": spec.environment.body, "surface": spec.environment.surface,
                      "terrain_kind": spec.terrain.kind,
                      "terrain_preset": spec.terrain.preset, "seed": spec.terrain.seed},
            "geoms": geoms,
            "poses": self._poses(model, data),
        }
        if spec.terrain.kind == "real" and spec.terrain.preset and spec.terrain.structure_crop_size_m is None:
            if (ortho := load_ortho_image(spec.terrain.preset)) is not None:
                image, metadata = ortho
                scene["terrain_ortho"] = {
                    "data": base64.b64encode(image).decode("ascii"),
                    "mime": "image/png",
                    "credit": metadata.get("ortho_credit", metadata.get("credit", "")),
                    "source": metadata.get("ortho_source", ""),
                }
        mesh_ids = sorted(
            {
                int(model.geom_dataid[g])
                for g in range(model.ngeom)
                if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            }
        )
        for ratio in (0.4, 0.3, 0.2, 0.125, 0.08, 0.05):
            scene["mesh_assets"] = self._mesh_payload(model, mesh_ids, ratio)
            if len(json.dumps(scene).encode("utf-8")) <= payload_budget:
                break
        else:
            raise ValueError(
                f"MuJoCo mesh geometry cannot fit the {payload_budget}-byte scene budget."
            )
        follow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
        root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        if follow < 0 and root_body >= 0:
            root_geoms = [
                g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) == root_body and model.geom_contype[g] != 0
            ]
            if not root_geoms:
                root_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == root_body]
            follow = root_geoms[0] if root_geoms else -1
        scene["follow_geom"] = int(follow)
        scene["robot_root_body"] = int(root_body)
        return scene

    @staticmethod
    def _mesh_payload(model: mujoco.MjModel, mesh_ids: list[int], ratio: float) -> dict:
        meshes = {}
        for mesh_id in mesh_ids:
            vert_start = int(model.mesh_vertadr[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = model.mesh_vert[vert_start : vert_start + vertex_count].astype(np.float64)
            faces = model.mesh_face[face_start : face_start + face_count].astype(np.int32)
            meshes[str(mesh_id)] = {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or "",
                **_simplified(vertices, faces, max(4, int(vertex_count * ratio))),
            }
        return meshes

    @staticmethod
    def _poses(model: mujoco.MjModel, data: mujoco.MjData) -> list[list[float]]:
        quat, poses = np.zeros(4), []
        for g in range(model.ngeom):
            mujoco.mju_mat2Quat(quat, data.geom_xmat[g])
            poses.append([*np.round(data.geom_xpos[g], 4).tolist(), *np.round(quat, 5).tolist()])
        return poses

    def _make(self, build: BuildResult):
        model = _model_from_xml(build.code)
        data = mujoco.MjData(model)
        if "initial_qpos" in build.scene:
            positions = np.asarray(build.scene["initial_qpos"], dtype=float)
            if positions.shape != (model.nq,) or not np.isfinite(positions).all():
                raise ValueError("The saved robot start state does not match this model.")
            data.qpos[:] = positions
            mujoco.mj_forward(model, data)
        return model, data

    def update_mission_point(self, spec: SceneSpec, build: BuildResult, which: str, x: float, y: float,
                             robot_id: str | None = None):
        """Update mission/state on the existing model; never generate terrain or re-place objects."""
        import copy

        if which not in {"start", "goal"} or not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Choose a finite start or goal position.")
        model, data = self._make(build)
        probe = TerrainProbe(model)
        half_x, half_y = probe.hx, probe.hy
        if build.scene.get("indoor"):
            half_x, half_y = [length / 2 for length in build.scene["indoor"]["size_m"]]
        if not probe.ok or abs(x - probe.cx) > half_x or abs(y - probe.cy) > half_y:
            raise ValueError("The selected point is outside the physics terrain.")
        updated = spec.model_copy(deep=True)
        scene = copy.deepcopy(build.scene)
        entries = ([entry for entry in updated.robots if entry.id == (robot_id or updated.robots[0].id)]
                   if which == "start" else updated.robots)
        if not entries:
            raise ValueError("The selected robot does not exist in this scene.")
        moved = {}
        for entry in entries:
            mission = copy.deepcopy(scene.get("robots", {}).get(entry.id, {}).get("mission") or scene["mission"])
            if entry.navigation_grid is not None:
                entry.waypoints, entry.navigation_grid = [], None
                mission.pop("waypoints", None)
                mission.pop("navigation_grid", None)
            ground_z = probe._bilinear(probe.heights, x, y) + probe.cz
            if which == "start":
                radius = footprint_radius(entry.robot)
                if abs(x - probe.cx) + radius >= half_x or abs(y - probe.cy) + radius >= half_y:
                    raise ValueError("The robot's start footprint must fit inside the physics terrain.")
                ground_z = height_at(probe.heights, 2 * probe.hx, x - probe.cx, y - probe.cy, radius) + probe.cz
                entry.start_xy, entry.start_place, entry.start_structure = [x, y], None, None
                mission["start"] = [x, y]
                if entry.start_heading_deg is None and mission.get("goal") is not None:
                    goal = mission["goal"]
                    mission["heading_deg"] = math.degrees(math.atan2(goal[1] - y, goal[0] - x))
                root_name = scene.get("robots", {}).get(entry.id, {}).get("root_body_name", "rover")
                root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
                if root < 0:
                    raise ValueError("The selected robot does not exist in the compiled scene.")
                joint = int(model.body_jntadr[root])
                if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                    raise ValueError("This robot has no movable free-root start state.")
                address = int(model.jnt_qposadr[joint])
                margin = 0.1 if entry.robot.kind == "rover" else 0.05 if entry.robot.kind == "walker" else 0.0
                data.qpos[address:address + 3] = [x, y, ground_z + support_height_offset(entry.robot) + margin]
                heading = math.radians(mission["heading_deg"])
                data.qpos[address + 3:address + 7] = [math.cos(heading / 2), 0, 0, math.sin(heading / 2)]
                mujoco.mj_forward(model, data)
                poses = self._poses(model, data)
                for geom in range(model.ngeom):
                    if int(model.body_rootid[int(model.geom_bodyid[geom])]) == root:
                        scene["poses"][geom] = poses[geom]
                        moved[str(geom)] = poses[geom]
                scene["initial_qpos"] = data.qpos.tolist()
            else:
                entry.goal_xy, entry.goal_place, entry.goal_structure = [x, y], None, None
                entry.waypoints, entry.navigation_grid = [], None
                mission["goal"] = [x, y]
                for key in ("goal_structure", "goal_z", "waypoints"):
                    mission.pop(key, None)
            if entry.id == updated.robots[0].id:
                scene["mission"] = mission
            if entry.id in scene.get("robots", {}):
                scene["robots"][entry.id]["mission"] = mission
            self.missions[entry.id] = mission
        self.mission = scene["mission"]
        updated.robot = updated.robots[0].robot.model_copy(deep=True)
        updated.mission = updated.robots[0].mission()
        timing_note = None
        if not updated.time_limit_given:
            from ..timing import time_limit_from_missions

            resolved = {entry.id: scene.get("robots", {}).get(entry.id, {}).get("mission") or scene["mission"]
                        for entry in updated.robots}
            limit = time_limit_from_missions(updated, resolved)
            if limit is not None:
                updated.duration_s, timing_note = limit
                updated.assumptions = [note for note in updated.assumptions if not note.startswith("Gave the robot")]
                updated.assumptions.append(timing_note)
        result = BuildResult(code=build.code, language=build.language, scene=scene, notes=list(build.notes))
        return updated, result, {"which": which, "mission": scene["mission"], "robots": scene.get("robots", {}),
                                 "moved_poses": moved, "code_unchanged": True, "duration_s": updated.duration_s,
                                 "timing_note": timing_note}

    def _policy_runner(
        self,
        spec: SceneSpec,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        name_prefix: str = "",
    ) -> PolicyRunner | SimpleController:
        if spec.robot.kind != "library":
            raise ValueError("A library policy can only control a library robot.")
        registry = LibraryRegistry()
        policy = registry.policy(spec.robot.policy_id)
        if policy.format == "builtin":   # an uploaded robot's wheel controller, baseline gait or held pose
            return SimpleController(registry.robot(spec.robot.robot_id), policy, model, data, name_prefix=name_prefix)
        return PolicyRunner(
            policy,
            model,
            data,
            name_prefix=name_prefix,
        )

    def _multi_states(self, spec: SceneSpec, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        states = {}
        for entry in spec.robots:
            prefix = f"{entry.id}_"
            info = self.robot_entries[entry.id]
            robot = entry.robot.model_copy(deep=True)
            robot_spec = spec.model_copy(deep=True)
            robot_spec.robot = robot
            root = int(info["root_body"])
            actuator_ids = [
                actuator
                for actuator in range(model.nu)
                if int(model.body_rootid[int(model.jnt_bodyid[int(model.actuator_trnid[actuator, 0])])]) == root
            ]
            runner = self._policy_runner(robot_spec, model, data, prefix) if robot.kind == "library" else None
            mission = self.missions[entry.id]
            states[entry.id] = {
                "entry": entry,
                "robot": robot,
                "spec": robot_spec,
                "root": root,
                "actuators": actuator_ids,
                "policy": runner,
                "navigator": Navigator(mission, robot.kind),
                "heading": np.array([math.cos(math.radians(mission["heading_deg"])),
                                     math.sin(math.radians(mission["heading_deg"]))]),
                "start": data.xpos[root].copy(),
                "max_tilt": 0.0,
                "flipped_at": None,
                "flip_tilt": flip_tilt_deg(robot),
                "path_len": 0.0,
                "last_xy": None,
                "distances": [],
                "cross_contacts": 0,
            }
        return states

    def _multi_control(self, model: mujoco.MjModel, data: mujoco.MjData, states: dict, active_time: float) -> None:
        for state in states.values():
            robot, nav, root = state["robot"], state["navigator"], state["root"]
            xmat = data.xmat[root]
            if xmat[8] > 0.8:
                state["heading"] = heading_of(xmat)
            if "support_offset" not in state:
                state["support_offset"] = support_height_offset(state["robot"])
            turn, go = nav.command(data.xpos[root][:2].copy(), state["heading"], max(active_time, 0.0),
                                   float(data.xpos[root][2]) - state["support_offset"])
            policy = state["policy"]
            if policy is not None:
                if active_time <= 0.0:
                    policy.hold_default()
                else:
                    card = policy.card.command
                    if nav.goal is None:
                        commands = dict(zip(card.names, card.default))
                    elif nav.reached_at is not None:
                        commands = {name: 0.0 for name in card.names}
                    else:
                        vx_limit = card.ranges[card.names.index("vx")][1]
                        yaw_range = card.ranges[card.names.index("yaw_rate")]
                        commands = {
                            "vx": float(np.clip(robot.command_speed_m_s * (
                                go if nav.navigation_grid else math.cos(turn)), 0.0, vx_limit)),
                            "vy": 0.0,
                            "yaw_rate": float(np.clip(turn, yaw_range[0], yaw_range[1])),
                        }
                    policy.set_command(commands)
                    policy.step(active_time)
                continue
            scale = float(np.clip(active_time / 1.0, 0.0, 1.0)) * go
            if robot.kind == "walker":
                controls = walker_ctrl(robot, max(active_time, 0.0), scale, turn)
            else:
                controls = np.full(len(state["actuators"]), robot.drive_speed * scale)
                for i, actuator in enumerate(state["actuators"]):
                    joint = int(model.actuator_trnid[actuator, 0])
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
                    side = 1.0 if name.endswith("r_j") else -1.0
                    if nav.navigation_grid:
                        controls[i] = robot.drive_speed * float(np.clip(active_time / 1.0, 0.0, 1.0)) * (go + side * turn)
                    else:
                        controls[i] *= 1 + side * turn
            if len(controls) != len(state["actuators"]):
                raise ValueError(f"Controller output does not match actuator mapping for {state['entry'].id}.")
            data.ctrl[state["actuators"]] = controls

    def _ctrl(self, spec: SceneSpec, model: mujoco.MjModel, t_active: float, turn: float = 0.0, go: float = 1.0) -> np.ndarray:
        """Controller for the robot. t_active < 0 means standing still (settling).
        `turn` steers (> 0 = left); `go` (0..1) fades the gait out once the goal is reached."""
        rb = spec.robot
        if rb.kind == "library":
            raise ValueError("Library robots must be controlled by their selected policy.")
        scale = float(np.clip(t_active / 1.0, 0.0, 1.0)) * go
        if rb.kind == "walker":
            c = walker_ctrl(rb, max(t_active, 0.0), scale, turn)
            return c if len(c) == model.nu else np.zeros(model.nu)   # agent-edited models may differ
        v = np.full(model.nu, rb.drive_speed * scale)
        if turn and model.nu:
            if getattr(self, "_sides", (None,))[0] is not model:
                self._sides = (model, np.array([
                    1.0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a][0])) or "").endswith("r_j")
                    else -1.0 for a in range(model.nu)]))
            if spec.mission is not None and spec.mission.navigation_grid is not None:
                v = rb.drive_speed * float(np.clip(t_active / 1.0, 0.0, 1.0)) * (go + self._sides[1] * turn)
            else:
                v = v * (1 + self._sides[1] * turn)        # turning left: right wheels faster, left wheels slower
        return v

    def validate(self, spec: SceneSpec, build: BuildResult) -> list[Check]:
        checks = [Check("Model compiles", True, "MuJoCo accepted the scene.")]
        for index, structure in enumerate(build.scene.get("structures", [])):
            if structure["kind"] in ("bridge", "habitat"):
                report = structure["structural_check"]
                checks.append(Check(f"{structure['kind'].title()} {index + 1} structural screening", report["passed"],
                                    f"{report['summary']} {report['honesty']}. {report['assumptions']}"))
            elif structure["kind"] == "lander":
                report = structure["structural_check"]
                for case in report["cases"].values():
                    checks.append(Check(f"Lander {index + 1} {case['label']}", case["passed"],
                                        f"{case['summary']} {case['honesty']}. {case['assumptions']}",
                                        advisory=case["what_if"]))
        env = spec.environment
        model, data = self._make(build)
        if len(spec.robots) > 1:
            return self._validate_multi(spec, build, model, data, checks)
        policy_runner = None
        if spec.robot.kind == "library":
            try:
                policy_runner = self._policy_runner(spec, model, data)
                checks.append(
                    Check(
                        "Library policy loads and validates",
                        True,
                        f"Loaded {spec.robot.policy_id} with named joints and finite policy output.",
                    )
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                checks.append(Check("Library policy loads and validates", False, str(exc)))

        ref = presets.gravity_for(env.body)
        g = -float(model.opt.gravity[2])
        if ref is not None:
            ok = abs(g - ref) / ref < 0.01
            checks.append(Check("Gravity matches reference", ok, f"{g:.3f} m/s² in model vs {ref} m/s² for {env.body}."))
        lo, hi = presets.FRICTION_RANGE
        checks.append(Check("Friction is plausible", lo <= env.friction <= hi, f"{env.friction:.2f} (expected {lo}–{hi})."))
        if spec.terrain.kind == "slope":
            lo, hi = presets.SLOPE_RANGE_DEG
            checks.append(Check("Slope is plausible", lo <= spec.terrain.slope_deg <= hi, f"{spec.terrain.slope_deg}°."))
        if spec.terrain.kind == "real":
            heights, meta = scene_heights(spec.terrain), load_preset(spec.terrain.preset)[1]
            built_relief = float(model.hfield_size[0][2])
            extra = course.GAP_DEPTH if any(o.kind == "gap" for o in spec.obstacles) else 0.0   # trenches deepen it
            ok = abs(built_relief - extra - float(heights.max())) <= 0.05 * float(heights.max()) + 0.3   # bumps add a little
            resized = spec.terrain.preset_size_m is not None or spec.terrain.preset_relief_m is not None
            checks.append(Check("Real terrain matches source data", ok,
                                f"{meta['label']}: {spec.terrain.size_m:.0f} m square{' (resized for this scene)' if resized else ''}, "
                                f"relief {built_relief:.1f} m in model vs {float(heights.max()):.1f} m in the data"
                                + ("." if resized else f", typical slope {meta.get('typical_slope_deg', '?')}°.")))

        mission = build.scene.get("mission") or {}
        if mission.get("goal") is not None:
            (sx, sy), (gx, gy) = mission["start"], mission["goal"]
            gap = math.hypot(gx - sx, gy - sy)
            moved = f" ({' and '.join(mission['clamped'])} moved inside the terrain edge)" if mission.get("clamped") else ""
            checks.append(Check("Start and goal are on the terrain", True,
                                f"Start ({sx:.1f}, {sy:.1f}), goal ({gx:.1f}, {gy:.1f}) m{moved}."))
            checks.append(Check("Goal is a real distance away", gap > mission["radius"] + 0.5,
                                f"{gap:.1f} m apart; the goal counts as reached within {mission['radius']:.1f} m."))
            probe = TerrainProbe(model)
            if probe.ok and not spec.obstacles:          # with a course, the course check below covers this
                steepest = max(math.degrees(math.atan(float(np.hypot(*probe.gradient(sx + (gx - sx) * k / 40, sy + (gy - sy) * k / 40)))))
                               for k in range(41))
                limit = 20.0 if spec.robot.kind == "walker" else 25.0
                checks.append(Check("Straight path looks passable", steepest <= limit,
                                    f"Steepest ground on the straight line: {steepest:.0f}° (rough limit for this robot: {limit:.0f}°). "
                                    "A warning, not a ban: it may still find a way.", advisory=True))

        placed = build.scene.get("course") or []
        if spec.obstacles:
            kinds = {}
            for p in placed:
                n = len(p["geoms"]) if p["kind"] in ("rock", "boulder", "log") else 1
                kinds[p["kind"]] = kinds.get(p["kind"], 0) + n
            listing = ", ".join(f"{n} {k}{'s' if n > 1 and not k.endswith('s') else ''}" for k, n in kinds.items()) or "none"
            notes = " " + " ".join(build.scene.get("course_notes") or [])
            checks.append(Check("Obstacles placed clear of start and goal", bool(placed), f"Placed: {listing}.{notes}".strip()))
            lim = course.capability(spec)
            in_the_way = [p for p in placed if p.get("placement", "path") == "path"]   # scattered ones can be passed by
            hard = [p for p in in_the_way if (p["kind"] == "gap" and (p["width_m"] or 0) > lim["gap_m"])
                    or (p["kind"] in ("step", "wall", "rock", "boulder") and p["height_m"] > lim["climb_m"])
                    or (p["kind"] == "log" and p["height_m"] > 1.5 * lim["climb_m"])          # round: easier than a step
                    or (p["kind"] == "stairs" and p["height_m"] / course.STAIR_STEPS > lim["climb_m"])]
            what = "; ".join(f"{p['size']} {p['kind']} ({(str(p['width_m']) + ' m wide') if p['kind'] == 'gap' else (str(p['height_m']) + ' m high')})"
                             for p in hard)
            checks.append(Check("Course looks passable for this robot", not hard,
                                (f"Probably too much for it: {what}. " if hard else "Every obstacle is within its rough limits. ")
                                + f"Rough limits: climbs about {lim['climb_m']} m, "
                                + (f"crosses gaps up to about {lim['gap_m']} m. " if lim["gap_m"] else "can't cross gaps. ")
                                + "A warning, not a ban: run it to find out.", advisory=True))

        if spec.robot.kind == "library" and policy_runner is None:
            checks.append(
                Check("Library robot stands for 1 s at zero command", False, "The selected policy did not load.")
            )
            return checks

        # Short settle test: does the robot land on the ground and stay finite?
        ground = ground_geoms(model)
        touched_late = 0                     # most ground contacts seen in the last 0.3 s (robots may still bounce a little)
        n_steps = int(1.0 / model.opt.timestep)
        if policy_runner is not None:
            command = policy_runner.card.command
            policy_runner.set_command({name: 0.0 for name in command.names})
            settle_steps = int(0.5 / model.opt.timestep)
            for _ in range(settle_steps):
                policy_runner.hold_default()
                mujoco.mj_step(model, data)
            robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
            start_height = float(data.xpos[robot_body][2])
            tilt_peak = 0.0
            for i in range(n_steps):
                policy_runner.step(i * model.opt.timestep)
                mujoco.mj_step(model, data)
                if i >= n_steps - int(0.3 / model.opt.timestep):
                    touched_late = max(
                        touched_late,
                        sum(
                            1
                            for k in range(data.ncon)
                            if {int(data.contact[k].geom1), int(data.contact[k].geom2)} & ground
                        ),
                    )
                up_z = data.xmat[robot_body].reshape(3, 3)[2, 2]
                tilt_peak = max(tilt_peak, math.degrees(math.acos(np.clip(up_z, -1.0, 1.0))))
            robot_card = LibraryRegistry().robot(spec.robot.robot_id)
            height_delta = abs(float(data.xpos[robot_body][2]) - start_height)
            stands = (
                bool(np.all(np.isfinite(data.qpos)))
                and instability_count(data) == 0
                and tilt_peak < FALL_TILT_DEG
                and height_delta <= 0.1 * robot_card.standing_height_m
            )
            start_slope_note = ""
            mission_data = build.scene.get("mission") or {}
            if mission_data.get("start") is not None:
                probe = TerrainProbe(model)
                if probe.ok:
                    heading_rad = math.radians(mission_data.get("heading_deg", 0.0))
                    heading = np.array([math.cos(heading_rad), math.sin(heading_rad)])
                    start_slope = probe.describe(*mission_data["start"], heading)["slope_deg"]
                    start_slope_note = f" Starting footprint ground slope {start_slope:.1f}°."
            checks.append(
                Check(
                    "Library robot stands for 1 s at zero command",
                    stands,
                    f"Peak tilt {tilt_peak:.1f}°, pelvis height changed {height_delta:.3f} m "
                    f"(limit {0.1 * robot_card.standing_height_m:.3f} m).{start_slope_note}",
                )
            )
        else:
            for i in range(n_steps):
                data.ctrl[:] = self._ctrl(spec, model, -1.0)
                mujoco.mj_step(model, data)
                if i >= n_steps - int(0.3 / model.opt.timestep):
                    touched_late = max(touched_late, sum(1 for k in range(data.ncon)
                                                         if {int(data.contact[k].geom1), int(data.contact[k].geom2)} & ground))
        blowups = instability_count(data)
        finite = bool(np.all(np.isfinite(data.qpos))) and blowups == 0
        checks.append(Check("Simulation is stable", finite, "No NaN or blow-up in a 1 s settle test." if finite
                            else f"The physics became unstable {blowups or 1} time(s) in a 1 s settle test."))
        if finite and model.nbody > 1:
            part = (
                "ankle/foot"
                if spec.robot.kind == "library"
                else "foot"
                if spec.robot.kind == "walker"
                else "wheel"
            )
            checks.append(Check("Robot rests on terrain", touched_late > 0,
                                f"Up to {touched_late} {part}/ground contacts in the last 0.3 s of settling."))
        return checks

    def _validate_multi(self, spec: SceneSpec, build: BuildResult, model, data, checks: list[Check]) -> list[Check]:
        gravity = -float(model.opt.gravity[2])
        reference = presets.gravity_for(spec.environment.body)
        if reference is not None:
            checks.append(Check(
                "Gravity matches reference",
                abs(gravity - reference) / reference < 0.01,
                f"{gravity:.3f} m/s² in model vs {reference} m/s² for {spec.environment.body}.",
            ))
        lo, hi = presets.FRICTION_RANGE
        checks.append(Check(
            "Friction is plausible",
            lo <= spec.environment.friction <= hi,
            f"{spec.environment.friction:.2f} (expected {lo}–{hi}).",
        ))
        mujoco.mj_forward(model, data)
        states = self._multi_states(spec, model, data)
        entries = list(states.values())
        overlaps = []
        for index, first in enumerate(entries):
            first_xy = np.asarray(first["start"][:2])
            first_radius = footprint_radius(first["robot"])
            for second in entries[index + 1:]:
                distance = float(np.linalg.norm(first_xy - np.asarray(second["start"][:2])))
                minimum = first_radius + footprint_radius(second["robot"])
                if distance < minimum:
                    overlaps.append(f"{first['entry'].label} and {second['entry'].label} ({distance:.2f} m apart)")
        checks.append(Check(
            "Robots do not overlap at spawn",
            not overlaps,
            "; ".join(overlaps) if overlaps else "All robot spawn footprints are separated.",
        ))
        obstacle_clearance = []
        for state in entries:
            mission = state["navigator"]
            for label, point in (("start", state["start"][:2]), ("goal", mission.goal)):
                if point is None:
                    continue
                for placed in build.scene.get("course", []):
                    if np.linalg.norm(np.asarray(point) - np.asarray(placed["xy"])) < footprint_radius(state["robot"]) + placed["length"] / 2:
                        obstacle_clearance.append(f"{state['entry'].label} {label}")
                        break
        checks.append(Check(
            "Robot starts and goals clear obstacles",
            not obstacle_clearance,
            ", ".join(obstacle_clearance) if obstacle_clearance else "No start or goal intersects a placed obstacle.",
        ))
        ground = ground_geoms(model)
        policy_states = [state for state in entries if state["policy"] is not None]
        for state in policy_states:
            policy = state["policy"]
            policy.set_command({name: 0.0 for name in policy.card.command.names})

        def hold_all_still(policy_time: float | None = None) -> None:
            data.ctrl[:] = 0.0
            for state in policy_states:
                if policy_time is None:
                    state["policy"].hold_default()
                else:
                    state["policy"].step(policy_time)

        for _ in range(int(0.5 / model.opt.timestep)):
            hold_all_still()
            mujoco.mj_step(model, data)

        start_heights = {state["entry"].id: float(data.xpos[state["root"]][2]) for state in entries}
        peak_tilts = {state["entry"].id: 0.0 for state in entries}
        late_contacts = {state["entry"].id: 0 for state in entries}
        n_steps = int(1.0 / model.opt.timestep)
        contact_window = n_steps - int(0.3 / model.opt.timestep)
        for step in range(n_steps):
            hold_all_still(step * model.opt.timestep)
            mujoco.mj_step(model, data)
            for state in entries:
                root = state["root"]
                robot_id = state["entry"].id
                up_z = data.xmat[root].reshape(3, 3)[2, 2]
                peak_tilts[robot_id] = max(
                    peak_tilts[robot_id], math.degrees(math.acos(np.clip(up_z, -1.0, 1.0)))
                )
                if step >= contact_window:
                    count = sum(
                        1
                        for contact_index in range(data.ncon)
                        if (
                            int(data.contact[contact_index].geom1) in ground
                            and int(model.body_rootid[int(model.geom_bodyid[int(data.contact[contact_index].geom2)])]) == root
                        ) or (
                            int(data.contact[contact_index].geom2) in ground
                            and int(model.body_rootid[int(model.geom_bodyid[int(data.contact[contact_index].geom1)])]) == root
                        )
                    )
                    late_contacts[robot_id] = max(late_contacts[robot_id], count)

        finite = bool(np.all(np.isfinite(data.qpos))) and instability_count(data) == 0
        checks.append(Check(
            "Simulation is stable",
            finite,
            "No NaN or blow-up in a 1 s settle test." if finite else "The physics became unstable during settling.",
        ))
        contact_results = []
        for state in entries:
            entry = state["entry"]
            robot = state["robot"]
            robot_id = entry.id
            root = state["root"]
            contacted = late_contacts[robot_id] > 0
            tilt = peak_tilts[robot_id]
            height_delta = abs(float(data.xpos[root][2]) - start_heights[robot_id])
            contact_results.append((entry.label, contacted, tilt))
            if robot.kind == "library":
                robot_card = LibraryRegistry().robot(robot.robot_id)
                stands = (
                    finite
                    and contacted
                    and tilt < FALL_TILT_DEG
                    and height_delta <= 0.1 * robot_card.standing_height_m
                )
                checks.append(Check(
                    f"{entry.label} stands for 1 s at zero command",
                    stands,
                    f"Peak tilt {tilt:.1f}°, root height change {height_delta:.3f} m "
                    f"(limit {0.1 * robot_card.standing_height_m:.3f} m), "
                    f"up to {late_contacts[robot_id]} ground contacts.",
                ))
        checks.append(Check(
            "Robots rest on terrain",
            all(contacted and tilt < FALL_TILT_DEG for _, contacted, tilt in contact_results),
            "; ".join(f"{label}: {'contact' if contact else 'no contact'}, {tilt:.1f}° tilt"
                      for label, contact, tilt in contact_results),
        ))
        return checks

    def _run_multi(self, spec: SceneSpec, build: BuildResult, chunk_frames: int, record: bool = True) -> Iterator[list[dict]]:
        model, data = self._make(build)
        mujoco.mj_forward(model, data)
        states = self._multi_states(spec, model, data)
        for state in states.values():
            course_scene = dict(build.scene)
            course_scene["mission"] = {
                "start": self.missions[state["entry"].id]["start"],
                "heading_deg": self.missions[state["entry"].id]["heading_deg"],
                "goal": self.missions[state["entry"].id]["goal"],
                "radius": self.missions[state["entry"].id]["radius"],
            }
            state["track"] = CourseTracker(model, course_scene, state["root"])
        frames, cross_pairs = [], set()
        steps_per_frame = max(1, int(round(1.0 / (FPS * model.opt.timestep))))
        n_frames = int(spec.duration_s * FPS)
        settle_s = 0.5
        push_start = push_time(spec)
        settled = False
        watches, stopped = {robot_id: Outcome() for robot_id in states}, None
        for frame in range(n_frames):
            for _ in range(steps_per_frame):
                active_time = data.time - settle_s
                self._multi_control(model, data, states, active_time)
                if spec.disturbance is not None:
                    for state in states.values():
                        root = state["root"]
                        data.xfrc_applied[root, 3:5] = 0
                        if push_start <= active_time < push_start + spec.disturbance.duration_s:
                            heading = state["heading"]
                            data.xfrc_applied[root, 3:5] = spec.disturbance.push_n * np.array([-heading[1], heading[0]])
                mujoco.mj_step(model, data)
                for i in range(data.ncon):
                    first = int(model.body_rootid[int(model.geom_bodyid[int(data.contact[i].geom1)])])
                    second = int(model.body_rootid[int(model.geom_bodyid[int(data.contact[i].geom2)])])
                    if first != second and first in {state["root"] for state in states.values()} and second in {state["root"] for state in states.values()}:
                        cross_pairs.add(tuple(sorted((first, second))))
            if not settled and data.time >= settle_s:
                settled = True
                for state in states.values():
                    state["start"] = data.xpos[state["root"]].copy()
            for state in states.values():
                root = state["root"]
                xmat = data.xmat[root]
                tilt = math.degrees(math.acos(np.clip(xmat.reshape(3, 3)[2, 2], -1.0, 1.0)))
                state["max_tilt"] = max(state["max_tilt"], tilt)
                if tilt >= state["flip_tilt"] and state["flipped_at"] is None:
                    state["flipped_at"] = round(data.time - settle_s, 2)
                watch = next(w for robot_id, w in watches.items() if states[robot_id] is state)
                if settled and (tipped := watch.tipped(tilt, data.time - settle_s)) is not None and state["flipped_at"] is None:
                    state["flipped_at"] = round(tipped, 2)       # past 60° for a second: it has fallen
                xy = data.xpos[root][:2].copy()
                if state["last_xy"] is not None:
                    state["path_len"] += float(np.linalg.norm(xy - state["last_xy"]))
                state["last_xy"] = xy
                nav = state["navigator"]
                if nav.goal is not None:
                    distance = float(np.linalg.norm(nav.goal - xy))
                    state["distances"].append((round(data.time - settle_s, 2), distance, xy.copy()))
                state["track"].update(data, xy, data.time - settle_s, state["flipped_at"] is not None)
                if settled:
                    watch.update(data.time - settle_s, xy, state["navigator"].reached_at, state["flipped_at"] is not None,
                                 state["heading"] if nav.navigation_grid else None)
            if record:
                frames.append({"t": round(frame / FPS, 3), "poses": self._poses(model, data)})
                if len(frames) >= chunk_frames:
                    yield frames
                    frames = []
            if settled and frame < n_frames - 1 and all(w.decided is not None for w in watches.values()):
                last = max(w.decided["at_s"] for w in watches.values())   # every robot's outcome is decided
                if not record or frame / FPS - last >= SHOW_AFTER_S:
                    stopped = {"reason": "every robot decided", "at_s": last, "planned_s": spec.duration_s,
                               "robots": {robot_id: w.decided["reason"] for robot_id, w in watches.items()}}
                    break
        if frames:
            yield frames
        self._multi_stopped = stopped

        per_robot = {}
        for robot_id, state in states.items():
            root = state["root"]
            end = data.xpos[root].copy()
            start = state["start"]
            nav = state["navigator"]
            mission = self.missions[robot_id]
            angle = math.radians(mission["heading_deg"])
            forward = np.array([math.cos(angle), math.sin(angle)])
            lateral = np.array([-forward[1], forward[0]])
            reached = nav.reached_at is not None
            start_distance = float(np.linalg.norm(nav.goal - start[:2])) if nav.goal is not None else None
            final_distance = float(np.linalg.norm(nav.goal - end[:2])) if nav.goal is not None else None
            closest = min((row[1] for row in state["distances"]), default=start_distance)
            outcome = "reached" if reached else "fell" if state["flipped_at"] is not None else "out_of_time"
            metrics = {
                "id": robot_id,
                "label": state["entry"].label,
                "robot": state["robot"].model_dump(),
                "distance_m": round(float(np.linalg.norm(end[:2] - start[:2])), 2),
                "forward_progress_m": round(float(np.dot(end[:2] - start[:2], forward)), 2),
                "lateral_drift_m": round(float(np.dot(end[:2] - start[:2], lateral)), 2),
                "max_tilt_deg": round(state["max_tilt"], 1),
                "flipped": state["flipped_at"] is not None,
                "flipped_at_s": state["flipped_at"],
                "goal_reached": reached,
                "outcome": outcome,
                "time_to_goal_s": round(nav.reached_at, 1) if reached else None,
                "start_distance_to_goal_m": round(start_distance, 2) if start_distance is not None else None,
                "closest_distance_m": round(closest, 2) if closest is not None else None,
                "final_distance_m": round(final_distance, 2) if final_distance is not None else None,
                "progress_toward_goal_m": round(start_distance - final_distance, 2) if start_distance is not None else None,
                "path_length_m": round(state["path_len"], 2),
                "goal_xy": [round(float(v), 2) for v in nav.goal] if nav.goal is not None else None,
                "goal_radius_m": nav.radius,
                "cross_robot_contacts": len(cross_pairs),
                "obstacles": state["track"].summary(state["spec"], reached),
            }
            per_robot[robot_id] = metrics
        ordered = sorted(
            per_robot.values(),
            key=lambda row: (
                not row["goal_reached"],
                row["time_to_goal_s"] if row["time_to_goal_s"] is not None else float("inf"),
                -(row["forward_progress_m"] or 0.0),
            ),
        )
        leaderboard = []
        for rank, metrics in enumerate(ordered, 1):
            if metrics["goal_reached"]:
                result = f"arrived in {metrics['time_to_goal_s']:.1f} s"
            elif metrics["final_distance_m"] is not None:
                result = f"missed by {metrics['final_distance_m']:.1f} m - {metrics['outcome']}"
            else:
                result = f"{metrics['forward_progress_m']:.1f} m progress - {metrics['outcome']}"
            leaderboard.append({
                "rank": rank,
                "id": metrics["id"],
                "label": metrics["label"],
                "robot": metrics["robot"].get("robot_id") or metrics["robot"]["kind"],
                "brain": metrics["robot"].get("policy_id") or {
                    "rover": "wheel controller",
                    "walker": "open-loop gait",
                }.get(metrics["robot"]["kind"], "library policy"),
                "result": result,
                "goal_reached": metrics["goal_reached"],
                "time_to_goal_s": metrics["time_to_goal_s"],
            })
        self._summary = {
            "simulator": self.name,
            "duration_s": spec.duration_s,
            "gravity": spec.environment.gravity,
            "friction": spec.environment.friction,
            "terrain": spec.terrain.model_dump(),
            "robots": per_robot,
            "leaderboard": leaderboard,
            "robot_contacts": len(cross_pairs),
            "stopped_early": self._multi_stopped,
            **({"push_n": spec.disturbance.push_n, "push_at_s": push_time(spec),
                "push_duration_s": spec.disturbance.duration_s} if spec.disturbance else {}),
        }

    def run(self, spec: SceneSpec, build: BuildResult, chunk_frames: int = 30, record: bool = True,
            race_telemetry: bool = False) -> Iterator[list[dict]]:
        """Simulate, yielding chunks of frames when ``record`` (a batch run that only needs the summary doesn't
        record any: its replay re-simulates it with the same seed). Race telemetry keeps the run going to the
        time limit and records motion, slip and terrain samples."""
        if len(spec.robots) > 1:
            yield from self._run_multi(spec, build, chunk_frames, record)
            return
        model, data = self._make(build)
        rover = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
        mujoco.mj_forward(model, data)
        policy_runner = self._policy_runner(spec, model, data) if spec.robot.kind == "library" else None
        start = data.xpos[rover].copy() if rover >= 0 else np.zeros(3)
        steps_per_frame = max(1, int(round(1.0 / (FPS * model.opt.timestep))))
        n_frames = int(spec.duration_s * FPS)
        settle_s = 0.5
        settle_steps = int(settle_s / model.opt.timestep)
        max_tilt, max_z, flipped_at = 0.0, start[2], None
        flip_tilt = flip_tilt_deg(spec.robot)
        frames = []

        # Evidence for "why did it fail?"
        probe = TerrainProbe(model)
        parts, ground = contact_parts(model), ground_geoms(model)
        heading = np.array([1.0, 0.0])          # last heading seen while the robot was upright
        tilt_log: list[tuple[float, float]] = []
        slip_log: list[tuple[float, int, int]] = []   # (time, touching, sliding)
        slide_speeds: list[float] = []
        max_path_slope, fall_started_at, fall_dir, fall_ground = 0.0, None, None, None
        steepest_point = None

        nav = Navigator(build.scene.get("mission"), spec.robot.kind)
        support_offset = support_height_offset(spec.robot)
        track = CourseTracker(model, build.scene, rover)
        nav_heading = np.array([1.0, 0.0])
        dist_log: list[tuple[float, float, np.ndarray]] = []    # (time, distance to goal, position) once per frame
        race_samples: list[dict] = []
        path_len, last_xy = 0.0, None
        settled = False
        push_start = push_time(spec)
        watch, stopped = Outcome(), None
        for f in range(n_frames):
            for s in range(steps_per_frame):
                turn, go = 0.0, 1.0
                if rover >= 0 and data.time > settle_s:
                    xm = data.xmat[rover]
                    if xm[8] > 0.8:                      # only trust the heading while roughly upright
                        nav_heading = heading_of(xm)
                    turn, go = nav.command(data.xpos[rover][:2].copy(), nav_heading, data.time - settle_s,
                                           float(data.xpos[rover][2]) - support_offset)
                if policy_runner is not None:
                    if data.time <= settle_s:
                        policy_runner.hold_default()
                    else:
                        command_card = policy_runner.card.command
                        if nav.goal is None:
                            command_values = dict(zip(command_card.names, command_card.default))
                        elif nav.reached_at is not None:
                            command_values = {name: 0.0 for name in command_card.names}
                        else:
                            speed_limit = command_card.ranges[command_card.names.index("vx")][1]
                            yaw_limits = command_card.ranges[command_card.names.index("yaw_rate")]
                            command_values = {
                                "vx": float(np.clip(spec.robot.command_speed_m_s * (
                                    go if nav.navigation_grid else math.cos(turn)), 0.0, speed_limit)),
                                "vy": 0.0,
                                "yaw_rate": float(np.clip(turn, yaw_limits[0], yaw_limits[1])),
                            }
                        policy_runner.set_command(command_values)
                        policy_runner.step(data.time - settle_s)
                else:
                    data.ctrl[:] = self._ctrl(spec, model, data.time - settle_s, turn, go)
                if rover >= 0 and spec.disturbance is not None:
                    data.xfrc_applied[rover, 3:5] = 0
                    active_time = data.time - settle_s
                    if push_start <= active_time < push_start + spec.disturbance.duration_s:
                        sideways = np.array([-nav_heading[1], nav_heading[0]])
                        data.xfrc_applied[rover, 3:5] = spec.disturbance.push_n * sideways
                mujoco.mj_step(model, data)
                if settled and parts and flipped_at is None and s % 8 == 0:     # ~62 Hz slip sampling
                    touching, sliding = sliding_contacts(model, data, parts, ground, slide_speeds)
                    if touching:
                        slip_log.append((data.time, touching, sliding))
            if not settled and f * steps_per_frame > settle_steps:
                settled = True
                if rover >= 0:  # measure progress from the settled pose, not the drop-in pose
                    start = data.xpos[rover].copy()
                    max_z, max_tilt = float(start[2]), 0.0
            if rover >= 0 and settled:
                xmat = data.xmat[rover]
                up_z = xmat.reshape(3, 3)[2, 2]
                tilt = math.degrees(math.acos(np.clip(up_z, -1, 1)))
                t_now = round(f / FPS, 2)
                tilt_log.append((t_now, round(tilt, 1)))
                max_tilt = max(max_tilt, tilt)
                max_z = max(max_z, float(data.xpos[rover][2]))
                x, y = float(data.xpos[rover][0]), float(data.xpos[rover][1])
                if tilt < 20:
                    heading = heading_of(xmat)
                if probe.ok and flipped_at is None:
                    here = probe.describe(x, y, heading)
                    if here["slope_deg"] > max_path_slope:
                        max_path_slope, steepest_point = here["slope_deg"], (round(x, 1), round(y, 1))
                if tilt >= FALL_TILT_DEG and fall_started_at is None and flipped_at is None:
                    fall_started_at, fall_dir = t_now, lean_direction(xmat, heading)
                    fall_ground = probe.describe(x, y, heading) if probe.ok else None
                elif tilt < FALL_TILT_DEG * 0.6 and flipped_at is None:
                    fall_started_at, fall_dir, fall_ground = None, None, None    # it recovered; forget that wobble
                if tilt > flip_tilt and flipped_at is None:
                    flipped_at = t_now
                if (tipped := watch.tipped(tilt, t_now)) is not None and flipped_at is None:
                    flipped_at = tipped                      # past 60° for a second: it has fallen
                xy = np.array([x, y])
                track.update(data, xy, t_now, fallen=fall_started_at is not None or flipped_at is not None)
                watch.update(t_now, xy, nav.reached_at, flipped_at is not None,
                             heading if nav.navigation_grid else None)
                if last_xy is not None:
                    path_len += float(np.linalg.norm(xy - last_xy))
                last_xy = xy
                if nav.goal is not None:
                    dist_log.append((t_now, float(np.linalg.norm(nav.goal - xy)), xy))
                if race_telemetry and f % 8 == 0:
                    here = probe.describe(x, y, heading) if probe.ok else {"slope_deg": None}
                    recent = [row for row in slip_log if t_now - 1.0 <= row[0] <= t_now]
                    touching = sum(row[1] for row in recent)
                    race_samples.append({
                        "time_s": t_now, "x": round(x, 3), "y": round(y, 3),
                        "distance_to_goal_m": round(float(np.linalg.norm(nav.goal - xy)), 3)
                        if nav.goal is not None else None,
                        "local_slope_deg": here["slope_deg"],
                        "step_m": probe.step_height_m(x, y) if probe.ok else None,
                        "slip_ratio": round(sum(row[2] for row in recent) / touching, 3) if touching else None,
                    })
            if record:
                frames.append({"t": round(f / FPS, 3), "poses": self._poses(model, data)})
                if len(frames) >= chunk_frames:
                    yield frames
                    frames = []
            if not race_telemetry and watch.decided is not None and f < n_frames - 1 and \
                    (not record or f / FPS - watch.decided["at_s"] >= SHOW_AFTER_S):
                stopped = {**watch.decided, "planned_s": spec.duration_s}
                break                                    # the outcome is decided: no need to keep simulating
        if frames:
            yield frames

        end = data.xpos[rover] if rover >= 0 else start
        if flipped_at is not None and fall_ground is not None:
            ground_here = fall_ground                                       # the ground where the fall began
        elif probe.ok and rover >= 0:
            ground_here = probe.describe(float(end[0]), float(end[1]), heading)
        else:
            ground_here = {"slope_deg": None, "along_heading_deg": None, "cross_deg": None, "downhill": None}

        def ratio(rows) -> float | None:
            touching = sum(r[1] for r in rows)
            return round(sum(r[2] for r in rows) / touching, 2) if touching else None

        before_fall = [r for r in slip_log if flipped_at is not None and fall_started_at is not None
                       and fall_started_at - 1.0 <= r[0] <= fall_started_at]
        step = max(1, len(tilt_log) // 10)
        self._summary_left_terrain = bool(probe.ok and rover >= 0 and (abs(float(end[0]) - probe.cx) > probe.hx
                                                                       or abs(float(end[1]) - probe.cy) > probe.hy))
        initial_heading = math.radians(build.scene.get("mission", {}).get("heading_deg", 0.0))
        lateral_axis = np.array([-math.sin(initial_heading), math.cos(initial_heading)])
        self._summary = {
            "simulator": self.name,
            "duration_s": spec.duration_s,
            "distance_m": round(float(np.linalg.norm(end[:2] - start[:2])), 2),
            "final_xy": [round(float(end[0]), 3), round(float(end[1]), 3)] if rover >= 0 else None,
            "forward_progress_m": round(float(end[0] - start[0]), 2),
            "lateral_drift_m": round(float(np.dot(end[:2] - start[:2], lateral_axis)), 2),
            "climbed_m": round(float(max_z - start[2]), 2),
            "max_tilt_deg": round(max_tilt, 1),
            "flipped": flipped_at is not None,
            "flipped_at_s": flipped_at,
            "fall_started_at_s": fall_started_at if flipped_at is not None else None,
            "fall_direction": fall_dir if flipped_at is not None else None,
            "ground_measured_at": "where the fall began" if flipped_at is not None else "end of run",
            "local_slope_deg": ground_here["slope_deg"],
            "slope_along_heading_deg": ground_here["along_heading_deg"],
            "cross_slope_deg": ground_here["cross_deg"],
            "downhill_side": ground_here["downhill"],
            "max_slope_on_path_deg": round(max_path_slope, 1) if probe.ok else None,
            "steepest_point_xy": steepest_point,
            "slip_ratio": ratio(slip_log),
            "slip_ratio_last_1s_before_fall": ratio(before_fall) if before_fall else None,
            "typical_contact_slide_speed_m_s": round(float(np.median(slide_speeds)), 2) if slide_speeds else None,
            "unstable": instability_count(data) > 0,
            "left_terrain": self._summary_left_terrain,
            "stopped_early": stopped,
            **({"push_n": spec.disturbance.push_n, "push_at_s": push_time(spec),
                "push_duration_s": spec.disturbance.duration_s} if spec.disturbance else {}),
            "tilt_trace": tilt_log[::step][:10] + ([tilt_log[-1]] if tilt_log and tilt_log[-1] not in tilt_log[::step][:10] else []),
            **self._mission_summary(nav, dist_log, path_len, flipped_at, start),
            **track.summary(spec, nav.reached_at is not None),
            "gravity": spec.environment.gravity,
            "friction": spec.environment.friction,
            "terrain": spec.terrain.model_dump(),
            "robot": spec.robot.model_dump(),
            **({"race_telemetry": {"time_limit_s": spec.duration_s, "samples": race_samples}}
               if race_telemetry else {}),
        }

    def summary(self) -> dict:
        return self._summary

    def _mission_summary(self, nav: Navigator, dist_log, path_len: float, flipped_at, start) -> dict:
        if nav.goal is None:
            return {}
        start_dist = float(np.linalg.norm(nav.goal - start[:2]))
        final = dist_log[-1][1] if dist_log else start_dist
        closest = min((d for _, d, _ in dist_log), default=start_dist)
        reached = nav.reached_at is not None
        if reached:
            outcome = "reached"
        elif flipped_at is not None:
            outcome = "fell"
        elif self._summary_left_terrain:
            outcome = "left_terrain"
        else:
            recent = [p for t, _, p in dist_log if t >= dist_log[-1][0] - 3.0] if dist_log else []
            stuck = len(recent) > 1 and float(np.linalg.norm(recent[-1] - recent[0])) < 0.3
            outcome = "stuck" if stuck else "out_of_time"
        return {
            "goal_reached": reached,
            "outcome": outcome,
            "time_to_goal_s": round(nav.reached_at, 1) if reached else None,
            "start_distance_to_goal_m": round(start_dist, 2),
            "closest_distance_m": round(closest, 2),
            "final_distance_m": round(final, 2),
            "progress_toward_goal_m": round(start_dist - final, 2),
            "path_length_m": round(path_len, 2),
            "path_efficiency": round(float(np.clip((start_dist - final) / path_len, 0.0, 1.0)), 2) if path_len > 0.5 else None,
            "goal_xy": [round(float(v), 2) for v in nav.goal],
            "goal_radius_m": nav.radius,
        }
