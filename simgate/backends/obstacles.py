"""Obstacle courses for the MuJoCo backend.

Placement is done here, on the real terrain, so the model only says *what* to put on the course
("rocks, a gap and stairs"), never where in metres. Everything sits on the line from start to goal
unless asked otherwise, in the order given, with the start and goal kept clear.
"""
import math
from dataclasses import dataclass, field

import numpy as np

# Sizes: rocks/boulders/logs/walls/steps by height (m); stairs by rise per step; ramps by angle (deg);
# gaps by width (m, along the path).
SIZES = {
    "rock":    {"small": 0.06, "medium": 0.12, "large": 0.20},
    "boulder": {"small": 0.35, "medium": 0.55, "large": 0.85},
    "step":    {"small": 0.08, "medium": 0.15, "large": 0.25},
    "stairs":  {"small": 0.05, "medium": 0.08, "large": 0.12},
    "ramp":    {"small": 10.0, "medium": 15.0, "large": 22.0},
    "wall":    {"small": 0.25, "medium": 0.50, "large": 1.00},
    "log":     {"small": 0.10, "medium": 0.20, "large": 0.30},
    "gap":     {"small": 0.25, "medium": 0.50, "large": 1.00},   # on the grid these open to ~0.23 / 0.48 / 0.98 m
}
COURSE_WIDTH = 3.0       # m across the path for steps, stairs, ramps, walls, logs and gaps
GAP_DEPTH = 0.6          # m
STAIR_STEPS, STAIR_RUN = 4, 0.45
CLEAR_RADIUS = 1.6       # m kept free around the start and the goal
ROCK_RGB = (0.42, 0.38, 0.34)
BUILT_RGB = (0.62, 0.62, 0.66)
LOG_RGB = (0.45, 0.30, 0.18)


@dataclass
class Placed:
    """One obstacle on the course, as the backend and the summary see it."""
    name: str                 # also the prefix of its MuJoCo geom names
    kind: str
    size: str
    height_m: float           # rise of a step/wall/rock/log, total rise of stairs/ramp, depth of a gap
    xy: tuple[float, float]   # centre
    along: float              # distance along the path from the start (m)
    length: float             # extent along the path (m)
    width_m: float | None = None    # gaps: how wide the gap is
    placement: str = "path"
    geoms: list[str] = field(default_factory=list)


# ── Terrain helpers ───────────────────────────────────────────────────────

def ground(H: np.ndarray, size_m: float, x: float, y: float) -> float:
    """Bilinear terrain height at (x, y). H rows go along +y, columns along +x."""
    n = H.shape[0]
    half = size_m / 2
    c = float(np.clip((x + half) / size_m * (n - 1), 0, n - 1.001))
    r = float(np.clip((y + half) / size_m * (n - 1), 0, n - 1.001))
    c0, r0 = int(c), int(r)
    fc, fr = c - c0, r - r0
    return float((H[r0, c0] * (1 - fc) + H[r0, c0 + 1] * fc) * (1 - fr) + (H[r0 + 1, c0] * (1 - fc) + H[r0 + 1, c0 + 1] * fc) * fr)


def footprint(H, size_m, cx, cy, d, u, along_half, across_half) -> tuple[float, float]:
    """(lowest, highest) ground under a rectangle centred at (cx, cy), along direction d, across direction u."""
    zs = [ground(H, size_m, cx + d[0] * a + u[0] * b, cy + d[1] * a + u[1] * b)
          for a in np.linspace(-along_half, along_half, 7) for b in np.linspace(-across_half, across_half, 7)]
    return min(zs), max(zs)


def cut_gap(H: np.ndarray, size_m: float, cx, cy, d, u, width, depth=GAP_DEPTH) -> None:
    """Lower the terrain in a rectangle (width along d, COURSE_WIDTH across) to make a trench, in place."""
    n = H.shape[0]
    xs = np.linspace(-size_m / 2, size_m / 2, n)
    X, Y = np.meshgrid(xs, xs)
    a = (X - cx) * d[0] + (Y - cy) * d[1]
    b = (X - cx) * u[0] + (Y - cy) * u[1]
    cell = size_m / (n - 1)
    # The terrain is a grid, so the opening at the top runs from the last lip point to the next one:
    # about (interior width + one cell). Size the interior so the opening matches the width asked for.
    inside = (np.abs(a) <= max((width - cell) / 2, cell * 0.5) + 1e-6) & (np.abs(b) <= COURSE_WIDTH / 2)
    lip = float(H[inside].max()) if inside.any() else 0.0
    H[inside] = np.minimum(H[inside], lip - depth)


def measure_gap(H: np.ndarray, size_m: float, cx, cy, d) -> float:
    """Width of the opening along d, measured 5 cm below the edges (what a wheel or foot has to span)."""
    steps = np.linspace(-1.5, 1.5, 601)
    z = np.array([ground(H, size_m, cx + d[0] * t, cy + d[1] * t) for t in steps])
    lip = max(z[0], z[-1])
    low = steps[z < lip - 0.05]
    return float(low.max() - low.min()) if low.size else 0.0


# ── Placement ─────────────────────────────────────────────────────────────

def _quat_yaw_pitch(yaw: float, pitch: float) -> str:
    """Quaternion for a yaw about z, then a pitch about the new y (radians)."""
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    q = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    return " ".join(f"{v:.5f}" for v in q)


def _rgba(rgb, a=1.0) -> str:
    return " ".join(f"{v:.2f}" for v in rgb) + f" {a}"


def extent(kind: str, h: float) -> float:
    """Length along the path that an obstacle of this kind takes up (m)."""
    if kind == "stairs":
        return 2 * STAIR_STEPS * STAIR_RUN + 1.0
    if kind == "ramp":
        run = (STAIR_STEPS * 0.12) / math.tan(math.radians(h))      # rises about as high as large stairs
        return 2 * run + 1.0
    if kind == "gap":
        return h
    if kind in ("step",):
        return 1.5
    if kind in ("boulder",):
        return 2 * h
    if kind == "wall":
        return 0.25
    if kind == "log":
        return 2 * h
    return 2.5   # a rock field


def plan_course(spec, H: np.ndarray, start, goal, seed: int) -> tuple[list[dict], list[str]]:
    """Decide where every obstacle goes. Returns slots (kind, size, value, centre, direction) and notes."""
    size_m = spec.terrain.size_m
    lim = size_m / 2 * 0.9
    rng = np.random.default_rng(seed + 101)
    start, goal = np.array(start, float), np.array(goal, float)
    d = goal - start
    L = float(np.linalg.norm(d))
    d = d / L if L > 1e-6 else np.array([1.0, 0.0])
    u = np.array([-d[1], d[0]])
    notes: list[str] = []

    items = []      # expand counts: rocks/boulders/logs become one field; other kinds repeat in a row
    for ob in spec.obstacles:
        value = ob.height_m if ob.height_m is not None else SIZES[ob.kind][ob.size]
        field_kind = ob.kind in ("rock", "boulder", "log") and ob.placement != "scattered"
        for _ in range(1 if field_kind else ob.count):        # a field is one slot; everything else repeats
            items.append((ob, value, ob.count if field_kind else 1))

    on_path = [(ob, v) for ob, v, _ in items if ob.placement == "path"]
    usable = (CLEAR_RADIUS + 0.5, L - CLEAR_RADIUS - 0.5)
    need = sum(extent(ob.kind, v) for ob, v in on_path) + 0.8 * max(len(on_path) - 1, 0)
    room = usable[1] - usable[0]
    if on_path and need > room:
        notes.append(f"The course needs about {need:.0f} m but the path is {room:.0f} m long, so obstacles are packed tighter.")
    gap_between = max((room - need) / (len(on_path) + 1), 0.3) if on_path else 0
    s = usable[0] + gap_between
    slots = []
    for ob, v, n in items:
        ext = extent(ob.kind, v)
        if ob.placement == "path":
            centre_s = min(s + ext / 2, usable[1] - ext / 2) if need <= room else s + ext / 2 * min(room / need, 1)
            c = start + d * centre_s
            s = s + ext + (gap_between if need <= room else 0.3)
            slots.append(dict(ob=ob, value=v, n=n, centre=c, d=d, u=u, along=centre_s))
            continue
        if ob.placement == "place" and ob.place:
            from .mujoco_backend import place_xy
            c = np.array(place_xy(H, size_m, ob.place), float)
        else:   # scattered: try random spots away from start, goal and the edge
            c = None
            for _ in range(200):
                p = rng.uniform(-lim * 0.9, lim * 0.9, 2)
                if (np.linalg.norm(p - start) > CLEAR_RADIUS + 1 and np.linalg.norm(p - goal) > CLEAR_RADIUS + 1
                        and all(np.linalg.norm(p - o["centre"]) > extent(o["ob"].kind, o["value"]) / 2 + ext / 2 + 0.5 for o in slots)):
                    c = p
                    break
            if c is None:
                notes.append(f"No clear spot for a {ob.size} {ob.kind}; left it out.")
                continue
        if np.linalg.norm(c - start) < CLEAR_RADIUS + ext / 2 or np.linalg.norm(c - goal) < CLEAR_RADIUS + ext / 2:
            notes.append(f"A {ob.size} {ob.kind} at the {ob.place or 'chosen spot'} would block the start or goal; left it out.")
            continue
        slots.append(dict(ob=ob, value=v, n=n, centre=c, d=d, u=u, along=float((c - start) @ d)))
    return slots, notes


def cut_gaps(slots, H, size_m) -> None:
    for sl in slots:
        if sl["ob"].kind == "gap":
            cut_gap(H, size_m, *sl["centre"], sl["d"], sl["u"], sl["value"])


def build_geoms(slots, H, size_m, seed: int) -> tuple[str, list[Placed]]:
    """MJCF geoms for every obstacle (after gaps are cut and heights final) and their records."""
    rng = np.random.default_rng(seed + 202)
    xml, placed = [], []
    lim = size_m / 2 * 0.95
    for i, sl in enumerate(slots):
        ob, v, (cx, cy), d, u = sl["ob"], sl["value"], sl["centre"], sl["d"], sl["u"]
        name = f"obs{i}_{ob.kind}"
        yaw = math.atan2(d[1], d[0])
        rec = Placed(name=name, kind=ob.kind, size=ob.size, height_m=round(v, 3), xy=(round(cx, 2), round(cy, 2)),
                     along=round(sl["along"], 2), length=round(extent(ob.kind, v), 2), placement=ob.placement)
        geoms = []

        def box(gname, px, py, half_along, half_across, bottom, top, rgb=BUILT_RGB, quat=None):
            hz = max((top - bottom) / 2, 0.01)
            q = quat or _quat_yaw_pitch(yaw, 0.0)
            geoms.append(f'    <geom name="{gname}" type="box" size="{half_along:.3f} {half_across:.3f} {hz:.3f}" '
                         f'pos="{px:.3f} {py:.3f} {(top + bottom) / 2:.3f}" quat="{q}" rgba="{_rgba(rgb)}"/>')

        if ob.kind in ("rock", "boulder"):
            n = sl["n"]
            for k in range(n):
                for _ in range(30):     # spread a field along and across the path without overlaps
                    if ob.kind == "rock" and n > 1:
                        a, b = rng.uniform(-1.1, 1.1), rng.uniform(-1.3, 1.3)
                    else:
                        a, b = (k - (n - 1) / 2) * 2.6 * v, rng.uniform(-0.3, 0.3)
                    px, py = cx + d[0] * a + u[0] * b, cy + d[1] * a + u[1] * b
                    if all(math.hypot(px - q[0], py - q[1]) > 2.2 * v for q in rec_pts(geoms)):
                        break
                h = v * rng.uniform(0.75, 1.25) if n > 1 else v
                r = h * 1.15
                z = ground(H, size_m, px, py) + h - r
                rgb = tuple(np.clip(np.array(ROCK_RGB) * rng.uniform(0.85, 1.15), 0, 1))
                geoms.append(f'    <geom name="{name}_{k}" type="sphere" size="{r:.3f}" pos="{px:.3f} {py:.3f} {z:.3f}" '
                             f'rgba="{_rgba(rgb)}"/><!--pt {px:.3f} {py:.3f}-->')
            rec.length = 2.5 if n > 1 else round(2.3 * v, 2)
        elif ob.kind == "log":
            for k in range(sl["n"]):
                a = (k - (sl["n"] - 1) / 2) * max(4 * v, 0.8)
                px, py = cx + d[0] * a, cy + d[1] * a
                p1 = (px - u[0] * COURSE_WIDTH / 2, py - u[1] * COURSE_WIDTH / 2)
                p2 = (px + u[0] * COURSE_WIDTH / 2, py + u[1] * COURSE_WIDTH / 2)
                r = v / 2
                z1, z2 = ground(H, size_m, *p1) + r * 0.8, ground(H, size_m, *p2) + r * 0.8
                geoms.append(f'    <geom name="{name}_{k}" type="cylinder" size="{r:.3f}" '
                             f'fromto="{p1[0]:.3f} {p1[1]:.3f} {z1:.3f} {p2[0]:.3f} {p2[1]:.3f} {z2:.3f}" rgba="{_rgba(LOG_RGB)}"/>')
            rec.length = round(max(4 * v, 0.8) * (sl["n"] - 1) + v, 2)
        elif ob.kind in ("step", "wall"):
            half_along = 0.75 if ob.kind == "step" else 0.12
            lo, hi = footprint(H, size_m, cx, cy, d, u, half_along, COURSE_WIDTH / 2)
            box(f"{name}_0", cx, cy, half_along, COURSE_WIDTH / 2, lo - 0.3, hi + v)
        elif ob.kind == "stairs":
            # up STAIR_STEPS steps, a 1 m landing, and the same steps down
            first = -(STAIR_STEPS * STAIR_RUN + 0.5)
            lo, hi = footprint(H, size_m, cx, cy, d, u, -first, COURSE_WIDTH / 2)
            for k in range(STAIR_STEPS):
                a_up = first + (k + 0.5) * STAIR_RUN           # before the landing (toward the start)
                for a, tag in ((a_up, "up"), (-a_up, "down")):
                    box(f"{name}_{tag}{k}", cx + d[0] * a, cy + d[1] * a, STAIR_RUN / 2, COURSE_WIDTH / 2, lo - 0.3, hi + v * (k + 1))
            box(f"{name}_top", cx, cy, 0.5, COURSE_WIDTH / 2, lo - 0.3, hi + v * STAIR_STEPS)
            rec.height_m = round(v * STAIR_STEPS, 3)
        elif ob.kind == "ramp":
            rise = STAIR_STEPS * 0.12
            th = math.radians(v)
            run = rise / math.tan(th)
            slab = run / math.cos(th)
            lo, hi = footprint(H, size_m, cx, cy, d, u, run + 0.5, COURSE_WIDTH / 2)
            top = hi + rise
            box(f"{name}_top", cx, cy, 0.5, COURSE_WIDTH / 2, lo - 0.3, top)
            for sign, tag in ((1, "up"), (-1, "down")):
                mid = sign * -(0.5 + run / 2)          # up-ramp before the landing, down-ramp after
                px, py = cx + d[0] * mid, cy + d[1] * mid
                zc = hi + rise / 2 - 0.05 * math.cos(th)
                q = _quat_yaw_pitch(yaw, -th if sign == 1 else th)
                geoms.append(f'    <geom name="{name}_{tag}" type="box" size="{slab / 2 + 0.05:.3f} {COURSE_WIDTH / 2:.3f} 0.050" '
                             f'pos="{px:.3f} {py:.3f} {zc:.3f}" quat="{q}" rgba="{_rgba(BUILT_RGB)}"/>')
            rec.height_m = round(rise, 3)
            rec.length = round(2 * run + 1.0, 2)
        elif ob.kind == "gap":
            width = measure_gap(H, size_m, cx, cy, d)
            rec.height_m, rec.width_m = GAP_DEPTH, round(width, 2)
            for k, side in enumerate((-1, 1)):     # yellow edge strips so the trench is visible; no collisions
                a = side * (width / 2 + 0.03)
                px, py = cx + d[0] * a, cy + d[1] * a
                z = max(ground(H, size_m, px + u[0] * b, py + u[1] * b) for b in np.linspace(-1.5, 1.5, 9)) + 0.005
                geoms.append(f'    <geom name="{name}_edge{k}" type="box" size="0.03 {COURSE_WIDTH / 2:.3f} 0.006" '
                             f'pos="{px:.3f} {py:.3f} {z:.3f}" quat="{_quat_yaw_pitch(yaw, 0.0)}" contype="0" conaffinity="0" '
                             f'rgba="0.98 0.75 0.14 1"/>')
        if abs(cx) > lim or abs(cy) > lim:
            continue
        rec.geoms = [g.split('name="')[1].split('"')[0] for g in geoms]
        xml += [g.split("<!--")[0] for g in geoms]
        placed.append(rec)
    return "\n".join(xml), placed


def rec_pts(geoms: list[str]) -> list[tuple[float, float]]:
    """Centres of rocks already placed in this field (stored in a trailing comment)."""
    pts = []
    for g in geoms:
        if "<!--pt " in g:
            x, y = g.split("<!--pt ")[1].rstrip("->").split()[:2]
            pts.append((float(x), float(y)))
    return pts


def capability(spec) -> dict:
    """Rough limits for the built-in robots with default settings, measured on flat Earth ground with single
    steps, rocks and gaps (sweeps in the dev notes; the tests pin the key cases):
    - 4-wheel rover: steps up to 0.09 m (0.6 × wheel radius), gaps whose opening is 0.23 m but not 0.48 m.
    - 6-wheel rover: single steps up to 0.13 m reliably and often 0.2 m (the rear wheels push the front ones up),
      but it can get its belly caught on a step's edge in a longer course, so the estimate is 0.9 × wheel radius.
    - 6-legged walker: steps up to 0.05 m; 4-legged walker: about 0.03 m. The open-loop gait doesn't look where
      it steps, so walkers drop a foot into even the smallest gap and get stuck.
    Scaled by wheel radius / leg length so a changed robot gets a sensible estimate."""
    rb = spec.robot
    if rb.kind == "rover":
        climb = (0.9 if rb.wheels == 6 else 0.6) * rb.wheel_radius
        return {"climb_m": round(climb, 2), "gap_m": round(1.6 * rb.wheel_radius, 2)}
    if rb.kind == "library":   # an uploaded wheeled robot: the 4-wheel rover's rule with its own wheel radius
        from ..library import LibraryRegistry

        card = LibraryRegistry().robot(rb.robot_id)
        if card.wheels:
            radius = min(wheel.radius_m for wheel in card.wheels)
            return {"climb_m": round(0.6 * radius, 2), "gap_m": round(1.6 * radius, 2)}
    climb = (0.25 if rb.legs == 6 else 0.15) * rb.leg_length
    return {"climb_m": round(climb, 2), "gap_m": 0.0}


def course_length_needed(obstacles) -> float:
    """Path length a list of obstacles needs, including the clear zones at the start and goal."""
    on_path = [ob for ob in obstacles if ob.placement == "path"]
    total = 0.0
    for ob in on_path:
        v = ob.height_m if ob.height_m is not None else SIZES[ob.kind][ob.size]
        copies = 1 if ob.kind in ("rock", "boulder", "log") else ob.count
        total += copies * (extent(ob.kind, v) + 0.8)
    return total + 2 * (CLEAR_RADIUS + 0.5) + 1.0
