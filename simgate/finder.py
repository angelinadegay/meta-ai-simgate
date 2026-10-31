"""Find where it breaks: search the world's conditions (never the robot's design) for where one robot starts to
fail, then map the two tightest conditions together. Every run goes through the batch runner.

Phase 1 binary-searches each condition, one change at a time, from the scene as it is toward the harsh end of its
range (at most 5 runs per condition, the conditions in parallel). Phase 2 maps the two conditions whose limits are
closest to the scene as it is on a 4×4 grid, running the cells nearest the boundary first and filling in the rest
by assuming that harsher on both never helps (shown as inferred). Phase 3, the Hunter, proposes combinations where
every condition stays below its own single limit but together might fail; the rules check each value before it runs.
The confirmation re-runs every failure found and the phase-1 boundary points on 3 seeds, keeps only what reproduces
and runs up to 2 inferred map cells for real. The limits, the failures ranked from the closest to normal, the map,
the runs behind them and the cost are reported with an honesty line written in code, and the Explainer's numbers are
checked against them. Each phase has a share of the quick (30 runs) or thorough (60) budget; what one leaves
unused rolls forward to the next.
"""

import re
from collections import Counter
from itertools import combinations
from typing import Any, Callable

from . import config, cost, llm, reliability
from .batch import run_batch
from .diagnose import robot_metrics
from .spec import SceneSpec, Terrain

StepCallback = Callable[[dict], None]

# World conditions only: the scene's own value is where each search starts, "harsh" is where it ends.
KNOBS: dict[str, dict[str, Any]] = {
    "slope_deg": {"name": "slope", "harsh": 30.0},
    "friction": {"name": "friction", "harsh": 0.1},
    "gravity": {"name": "gravity", "harsh": 1.62},
    "push_n": {"name": "sideways push", "harsh": 80.0},
    "payload": {"name": "payload", "harsh": 0.5},              # a share of the robot's mass
    "obstacle_scale": {"name": "obstacle height", "harsh": 2.0},  # times the robot's measured climb limit
    "roughness": {"name": "roughness", "harsh": 1.0},
    "speed": {"name": "commanded speed", "harsh": 1.5},       # times the scene's speed (+50%)
    # legged and library robots are also searched slower (-50%): a second half-search of the same condition
    "speed_slower": {"name": "slower commanded speed", "harsh": 0.5},
}
RULE_ORDER = ["slope_deg", "obstacle_scale", "friction", "roughness", "push_n", "payload", "gravity", "speed"]
BOTH_WAYS = ("walker", "library")   # robots whose speed is searched in both directions
# every phase gets a share of the budget; what a phase leaves unused rolls forward to the next one
SHARES = {"quick": {"search": 14, "map": 6, "hunt": 5, "confirm": 5},
          "thorough": {"search": 24, "map": 12, "hunt": 12, "confirm": 12}}
PHASES = ("search", "map", "hunt", "confirm")
BUDGETS = {budget: sum(shares.values()) for budget, shares in SHARES.items()}   # quick 30, thorough 60
KNOB_COUNT = {"quick": 3, "thorough": 4}
SEARCH_RUNS = 5          # per condition at most: its harsh end, then 4 halvings (fewer when the search share is tight)
GRID = 4
MAP_BATCH = 6            # map cells run together
TIPPED_DEG = 60.0
FAILURES = {"task": "missed the goal", "safety": f"fell or tilted past {TIPPED_DEG:g}°"}


# ── Conditions ───────────────────────────────────────────────────────────────

def show(knob: str, value: float | None) -> str:
    """A condition's value as the card and the Explainer write it."""
    if value is None:
        return "–"
    if knob == "slope_deg":
        return f"{value:.1f}°"
    if knob in ("friction", "roughness"):
        return f"{value:.2f}"
    if knob == "gravity":
        return f"{value:.2f} m/s²"
    if knob == "push_n":
        return f"{value:.0f} N"
    if knob in ("payload", "speed", "speed_slower"):
        return f"{value * 100:.0f}%" + (" of its mass" if knob == "payload" else " of its speed")
    return f"{value:.2f}× its climb limit"


def _entry(spec: SceneSpec, robot_id: str | None):
    entries = spec.robots
    if robot_id:
        word = robot_id.strip().lower()
        for entry in entries:
            if word in (entry.id, entry.label.lower(), entry.robot.kind, (entry.robot.robot_id or "").lower()):
                return entry
        for entry in entries:
            if word in entry.label.lower():
                return entry
    return entries[0]


def _obstacle_context(spec: SceneSpec, robot) -> tuple[float, float] | None:
    """(the tallest obstacle that isn't a gap, the robot's measured climb limit), or None when there is none."""
    from .backends.obstacles import SIZES, capability

    heights = [o.height_m if o.height_m is not None else SIZES[o.kind][o.size] for o in spec.obstacles if o.kind != "gap"]
    if not heights:
        return None
    probe = spec.model_copy(deep=True)
    probe.robot = robot.model_copy(deep=True)
    climb = capability(probe)["climb_m"]
    return (max(heights), climb) if climb > 0 else None


def scene_knobs(spec: SceneSpec, robot, gravity_asked: bool = False) -> tuple[dict[str, float], dict[str, str]]:
    """The conditions that fit this scene with the scene's own value of each, and why the others don't. Gravity is
    searched on Earth, or on any world when the request asks about it."""
    terrain, env = spec.terrain, spec.environment
    fits: dict[str, float] = {}
    skipped: dict[str, str] = {}
    if terrain is not None and terrain.kind in ("flat", "slope") and (terrain.slope_deg if terrain.kind == "slope" else 0) < 30:
        fits["slope_deg"] = terrain.slope_deg if terrain.kind == "slope" else 0.0
    else:
        skipped["slope_deg"] = "the ground isn't flat or a slope, so a slope angle would replace it"
    if (env.friction or 0.8) > 0.1:
        fits["friction"] = env.friction or 0.8
    else:
        skipped["friction"] = "friction is already at the bottom of its range"
    if not gravity_asked and (env.body != "earth" or (terrain is not None and terrain.kind == "real")):
        skipped["gravity"] = f"the scene is on {env.body.title()}, so its gravity stays"
    elif (env.gravity or 9.81) <= KNOBS["gravity"]["harsh"]:
        skipped["gravity"] = "gravity is already at the bottom of its range"
    else:
        fits["gravity"] = env.gravity or 9.81
    if spec.disturbance is None or spec.disturbance.push_n < 80:
        fits["push_n"] = spec.disturbance.push_n if spec.disturbance else 0.0
    else:
        skipped["push_n"] = "the push is already at the top of its range"
    if robot.kind != "library":
        fits["payload"] = 0.0
    else:
        skipped["payload"] = "a library robot's trained policy expects its exact body"
    if (context := _obstacle_context(spec, robot)) is not None:
        tallest, climb = context
        if tallest / climb < 2.0:
            fits["obstacle_scale"] = round(tallest / climb, 3)
        else:
            skipped["obstacle_scale"] = "the obstacles are already twice the robot's climb limit"
    else:
        skipped["obstacle_scale"] = "there are no obstacles it could climb"
    if terrain is not None and terrain.kind not in ("flat",) and terrain.roughness < 1.0:
        fits["roughness"] = terrain.roughness
    else:
        skipped["roughness"] = "flat ground has no bumps to roughen"
    fits["speed"] = 1.0
    if robot.kind in BOTH_WAYS:   # a gait or a policy can fail too slow as well as too fast; a rover only too fast
        fits["speed_slower"] = 1.0
    return fits, skipped


def choose_knobs(fits: dict[str, float], budget: str, named: list[str] | None) -> tuple[list[str], list[str]]:
    """2–4 conditions: the ones named (if they fit), else the rules' order for this kind of scene. Returns the
    chosen keys and notes about named ones that were dropped. Speed searched both ways is one condition: its slower
    half-search rides along and doesn't count toward the 2–4."""
    chosen, notes = _choose(fits, budget, named)
    if "speed" in chosen and "speed_slower" in fits:
        chosen.insert(chosen.index("speed") + 1, "speed_slower")
    return chosen, notes


def _choose(fits: dict[str, float], budget: str, named: list[str] | None) -> tuple[list[str], list[str]]:
    notes = []
    if named:
        chosen = []
        for knob in named:
            if knob not in KNOBS or knob == "speed_slower":
                notes.append(f"{knob} isn't a world condition I can search.")
            elif knob not in fits:
                notes.append(f"{KNOBS[knob]['name']} doesn't fit this scene.")
            elif knob not in chosen:
                chosen.append(knob)
        if len(chosen) >= 2:
            return chosen[:KNOB_COUNT[budget]], notes   # quick: at most 3, thorough: 4
        extra = [knob for knob in RULE_ORDER if knob in fits and knob not in chosen]
        return (chosen + extra)[:2], notes   # fewer than 2 named: one more from the rules
    return [knob for knob in RULE_ORDER if knob in fits][:KNOB_COUNT[budget]], notes


def change(knob: str, value: float, context: dict) -> dict:
    """The batch variation that sets one condition (empty at the scene's own value)."""
    if abs(value - context["base"][knob]) < 1e-9 and knob in ("push_n", "payload", "speed", "speed_slower"):
        return {}
    if knob == "slope_deg":
        return {"slope_deg": round(value, 3)}
    if knob == "friction":
        return {"friction": round(value, 4)}
    if knob == "gravity":
        return {"gravity": round(value, 4)}
    if knob == "push_n":
        return {"push_n": round(value, 2)} if value > 0 else {}
    if knob == "payload":
        return {"payload_kg": round(value * context["mass"], 3), "robot_id": context["robot_id"]}
    if knob == "obstacle_scale":
        tallest, climb = context["obstacles"]
        return {"obstacle_scale": round(value * climb / tallest, 5)}
    if knob == "roughness":
        return {"roughness": round(value, 4)}
    return {"speed_scale": round(value, 4), "robot_id": context["robot_id"]}


def harshness(knob: str, value: float, base: float) -> float:
    """0 at the scene's value, 1 at the harsh end."""
    harsh = KNOBS[knob]["harsh"]
    return (value - base) / (harsh - base) if harsh != base else 0.0


# ── Runs ─────────────────────────────────────────────────────────────────────

class Runner:
    """Runs batches of variations, numbers every run as evidence and keeps to the budget."""

    def __init__(self, scene: SceneSpec, robot_id: str, failure: str, budget: int) -> None:
        self.scene, self.robot_id, self.failure, self.budget = scene, robot_id, failure, budget
        self.cap = budget      # the end of the current phase's share
        self.evidence: list[dict] = []

    @property
    def left(self) -> int:
        """Runs left in the current phase's share."""
        return min(self.budget, self.cap) - len(self.evidence)

    def phase(self, allowance: int) -> int:
        """Start a phase with ``allowance`` runs; returns it."""
        self.cap = len(self.evidence) + allowance
        return allowance

    def run(self, jobs: list[tuple[dict, dict]]) -> list[dict]:
        """jobs = (what it tests, the variation); returns one judged evidence row per job."""
        jobs = jobs[: max(0, self.left)]
        if not jobs:
            return []
        seed = self.scene.terrain.seed if self.scene.terrain is not None else 0
        variations = [{"label": f"run {len(self.evidence) + i + 1}", **(variation or {"seed": seed})}
                      for i, (_, variation) in enumerate(jobs)]
        rows = []
        for (setting, variation), result in zip(jobs, run_batch(self.scene, variations, baseline=False, record=False)):
            row = {"id": f"r{len(self.evidence) + 1}", **setting, "change": variation, **self.judge(result)}
            self.evidence.append(row)
            rows.append(row)
        return rows

    def judge(self, result: dict) -> dict:
        if result.get("error"):
            return {"broke": None, "failure": None, "error": result["error"][:200]}
        m = robot_metrics(result.get("summary") or {}, self.robot_id)
        tilt = m.get("max_tilt_deg") or 0.0
        if self.failure == "safety":
            broke = bool(m.get("flipped")) or tilt > TIPPED_DEG
            kind = "fell" if m.get("flipped") else f"tilted {tilt:.0f}°" if broke else None
        else:
            broke = not m["passed"]
            kind = ({"fell": "fell", "stuck": "got stuck", "out_of_time": "ran out of time",
                     "left_terrain": "went off the edge"}.get(m.get("outcome") or "", "missed the goal")
                    if broke else None)
        return {"broke": broke, "failure": kind, "max_tilt_deg": round(tilt, 1),
                "distance_left_m": m.get("final_distance_m")}


# ── Phase 1: each condition's breaking point ─────────────────────────────────

FIRST_RUNS = 2           # every condition's harsh end and one halving, before the extra runs are given out


def allot(state: dict[str, dict], base: dict[str, float], left: int) -> dict[str, int]:
    """Each condition's run cap once all have had FIRST_RUNS: the runs left go to the tightest first (the one whose
    bracket is closest to the scene as it is), up to SEARCH_RUNS each, so its limit gets the finest bracket."""
    caps = {knob: s["runs"] for knob, s in state.items()}
    open_ = [knob for knob, s in state.items() if not s["done"] and s["fail"] is not None]
    for knob in sorted(open_, key=lambda k: harshness(k, (state[k]["ok"] + state[k]["fail"]) / 2, base[k])):
        extra = min(SEARCH_RUNS - state[knob]["runs"], left)
        caps[knob] += extra
        left -= extra
    return caps


def search(runner: Runner, knobs: list[str], context: dict, on_step: StepCallback | None) -> list[dict]:
    """Binary-search every condition at once (one change per run): its harsh end first, then halve the bracket
    between the last value that was OK and the first that failed. Each gets FIRST_RUNS runs, then the tightest gets
    the extra runs of the search's share first (at most SEARCH_RUNS per condition)."""
    base = context["base"]
    state = {knob: {"ok": base[knob], "fail": None, "ok_run": "r1", "fail_run": None, "failure": None, "runs": 0,
                    "done": False, "crashed": 0} for knob in knobs}
    caps = {knob: FIRST_RUNS for knob in knobs}
    for step in range(SEARCH_RUNS):
        if step == FIRST_RUNS:
            caps = allot(state, base, runner.left)
        active = [knob for knob in knobs if not state[knob]["done"] and state[knob]["runs"] < caps[knob]]
        if not active:
            break
        jobs = []
        for knob in active:
            s = state[knob]
            value = KNOBS[knob]["harsh"] if s["fail"] is None else (s["ok"] + s["fail"]) / 2
            jobs.append(({"phase": 1, "knob": knob, "value": value}, change(knob, value, context)))
        _emit(on_step, "Batch runner", "search",
              f"Search step {step + 1}: " + ", ".join(f"{KNOBS[j[0]['knob']]['name']} {show(j[0]['knob'], j[0]['value'])}"
                                                      for j in jobs) + ".", "", "working")
        rows = runner.run(jobs)
        for row in rows:
            s = state[row["knob"]]
            s["runs"] += 1
            if row["broke"] is None:          # a crash says nothing about the physics: stop searching this one
                s["crashed"] += 1
                s["done"] = True
            elif row["broke"]:
                s["fail"], s["fail_run"], s["failure"] = row["value"], row["id"], row["failure"]
            else:
                s["ok"], s["ok_run"] = row["value"], row["id"]
                if s["fail"] is None:          # it held at the harsh end: nothing to find in this range
                    s["done"] = True
        _emit(on_step, "Batch runner", "search", "Search step results.",
              "; ".join(f"{KNOBS[r['knob']]['name']} {show(r['knob'], r['value'])}: "
                        f"{'crashed' if r['broke'] is None else r['failure'] if r['broke'] else 'OK'}" for r in rows))
        if len(rows) < len(jobs):
            break                              # out of budget
    limits = []
    for knob in knobs:
        s = state[knob]
        never = s["fail"] is None
        step = None if never else abs(s["fail"] - s["ok"])
        limits.append({
            "knob": knob, "name": KNOBS[knob]["name"], "baseline": base[knob], "harsh": KNOBS[knob]["harsh"],
            "last_ok": s["ok"], "first_fail": s["fail"], "failure": s["failure"], "never_broke": never,
            "step": step, "ok_run": s["ok_run"], "fail_run": s["fail_run"], "runs": s["runs"], "crashed": s["crashed"],
            "distance": 1.0 + (1 - harshness(knob, s["ok"], base[knob])) if never
            else round(harshness(knob, (s["ok"] + s["fail"]) / 2, base[knob]), 4),
            "shown": {"baseline": show(knob, base[knob]), "last_ok": show(knob, s["ok"]),
                      "first_fail": show(knob, s["fail"]), "harsh": show(knob, KNOBS[knob]["harsh"]),
                      "step": "–" if step is None else "±" + show(knob, step).replace(" of its mass", "").replace(" of its speed", "")},
        })
    return sorted(limits, key=lambda limit: limit["distance"])


# ── Phase 2: the two tightest conditions together ────────────────────────────

def _axis(limit: dict) -> list[float]:
    """4 values from the scene's own one toward the harsh end, reaching past the limit so the boundary is inside."""
    base, harsh = limit["baseline"], limit["harsh"]
    if limit["never_broke"]:
        end = harsh
    else:
        end = base + min(1.0, 1.5 * harshness(limit["knob"], limit["first_fail"], base)) * (harsh - base)
    return [base + (end - base) * i / (GRID - 1) for i in range(GRID)]


def _from_search(limit: dict, value: float) -> str | None:
    """What phase 1 already says about this value of one condition with everything else as it is."""
    base, knob = limit["baseline"], limit["knob"]
    h = harshness(knob, value, base)
    if h <= harshness(knob, limit["last_ok"], base) + 1e-9:
        return "pass"
    if not limit["never_broke"] and h >= harshness(knob, limit["first_fail"], base) - 1e-9:
        return "fail"
    return None


def grid_map(runner: Runner, first: dict, second: dict, context: dict, on_step: StepCallback | None) -> dict:
    """Within the map's share of the budget."""
    xs, ys = _axis(first), _axis(second)
    stop_at = 0
    cells: list[list[dict]] = [[{"state": None, "how": None, "run": None} for _ in xs] for _ in ys]
    # The edges, where one condition stays as it is, follow from phase 1's bracket; like every cell without a run of
    # its own they are inferred, not measured.
    for i, x in enumerate(xs):
        if (state := _from_search(first, x)) is not None:
            cells[0][i] = {"state": state, "how": "inferred", "run": None, "from": "search"}
    for j, y in enumerate(ys):
        if (state := _from_search(second, y)) is not None:
            cells[j][0] = {"state": state, "how": "inferred", "run": None, "from": "search"}
    _infer(cells)

    def boundary(limit: dict) -> float:
        return 1e9 if limit["never_broke"] else max(harshness(limit["knob"], (limit["last_ok"] + limit["first_fail"]) / 2,
                                                              limit["baseline"]), 1e-6)

    bx, by = boundary(first), boundary(second)
    rounds = 0
    while runner.left > stop_at:
        unknown = [(j, i) for j in range(GRID) for i in range(GRID) if cells[j][i]["state"] is None]
        if not unknown:
            break
        unknown.sort(key=lambda c: abs(harshness(first["knob"], xs[c[1]], first["baseline"]) / bx
                                       + harshness(second["knob"], ys[c[0]], second["baseline"]) / by - 1))
        picked = unknown[: min(MAP_BATCH, runner.left - stop_at)]
        rounds += 1
        _emit(on_step, "Batch runner", "map", f"Map round {rounds}: {len(picked)} combinations near the boundary.", "", "working")
        jobs = [({"phase": 2, "knob": f"{first['knob']}+{second['knob']}", "value": [xs[i], ys[j]], "cell": [j, i]},
                 {**change(first["knob"], xs[i], context), **change(second["knob"], ys[j], context)}) for j, i in picked]
        rows = runner.run(jobs)
        for row in rows:
            j, i = row["cell"]
            cells[j][i] = ({"state": "error", "how": "run", "run": row["id"]} if row["broke"] is None else
                           {"state": "fail" if row["broke"] else "pass", "how": "run", "run": row["id"]})
        _infer(cells)
        _emit(on_step, "Batch runner", "map", "Map round results.",
              f"{sum(1 for r in rows if r['broke'] is False)} passed, {sum(1 for r in rows if r['broke'])} failed.")
        if len(rows) < len(jobs):
            break
    grid = {"x": {"knob": first["knob"], "name": first["name"], "values": xs, "shown": [show(first["knob"], v) for v in xs]},
            "y": {"knob": second["knob"], "name": second["name"], "values": ys, "shown": [show(second["knob"], v) for v in ys]},
            "cells": cells, "checked": {"checked": 0, "wrong": 0}}
    recount(grid)
    return grid


def recount(grid: dict) -> None:
    """measured: the cells that ran; inferred: the rest that were filled in; unknown: neither."""
    flat = [cell for row in grid["cells"] for cell in row]
    run = [c for c in flat if c["how"] == "run"]
    inferred = [c for c in flat if c["how"] == "inferred"]
    grid["counts"] = {"run": len(run), "pass": sum(c["state"] == "pass" for c in run),
                      "fail": sum(c["state"] == "fail" for c in run), "error": sum(c["state"] == "error" for c in run),
                      "inferred": len(inferred), "unknown": sum(1 for c in flat if c["state"] is None)}
    grid["inferred"] = len(inferred)


def _infer(cells: list[list[dict]]) -> None:
    """Harsher on both never helps: a pass makes every milder cell pass, a fail every harsher cell fail."""
    changed = True
    while changed:
        changed = False
        for j in range(GRID):
            for i in range(GRID):
                state = cells[j][i]["state"]
                if state not in ("pass", "fail"):
                    continue
                for jj in range(GRID):
                    for ii in range(GRID):
                        milder = jj <= j and ii <= i
                        harsher = jj >= j and ii >= i
                        if cells[jj][ii]["state"] is None and ((state == "pass" and milder) or (state == "fail" and harsher)):
                            cells[jj][ii] = {"state": state, "how": "inferred", "run": None}
                            changed = True


# ── Phase 3: the Hunter ──────────────────────────────────────────────────────

HUNT_FIRST, HUNT_MORE = 6, 4          # combinations in the first round, and in the one more round it may ask for
HUNT_LOW, HUNT_HIGH = 0.7, 0.9        # each condition at 70–90% of its own single limit
NEVER_HIGH = 0.9                      # a condition that never broke: up to 90% of its range
HUNT_TOLERANCE = 0.01                 # in harshness, for values the model rounded
UNITS = {"slope_deg": "degrees", "friction": "friction coefficient", "gravity": "m/s²", "push_n": "newtons sideways",
         "payload": "share of the robot's mass", "obstacle_scale": "times its climb limit", "roughness": "0 to 1",
         "speed": "times the scene's speed", "speed_slower": "times the scene's speed"}
SIBLINGS = {frozenset({"speed", "speed_slower"})}   # one physical condition: never both in a combination


def raw(value: float) -> str:
    """A value in its own units, as FACTS for the Hunter write it."""
    return f"{value:.4g}"


def value_at(knob: str, h: float, base: float) -> float:
    return base + h * (KNOBS[knob]["harsh"] - base)


def hunt_ranges(limits: list[dict]) -> dict[str, dict]:
    """The conditions the Hunter may set and each one's window, in harshness: 70–90% of its single limit and never
    past the last value that was OK alone; up to 90% of the range for one that never broke. A condition whose search
    crashed, or whose limit is too close to the scene to leave a window, is left out."""
    ranges = {}
    for limit in limits:
        knob, base = limit["knob"], limit["baseline"]
        if limit["never_broke"]:
            if limit["crashed"]:
                continue
            low, high = 0.0, NEVER_HIGH
        else:
            fail_h = harshness(knob, limit["first_fail"], base)
            low, high = HUNT_LOW * fail_h, min(HUNT_HIGH * fail_h, harshness(knob, limit["last_ok"], base))
            if high <= 0 or high < low:
                continue
        ends = sorted([value_at(knob, low, base), value_at(knob, high, base)])
        ranges[knob] = {"limit": limit, "low": low, "high": high, "window": ends}
    return ranges


def combo_text(values: dict[str, float]) -> str:
    return " and ".join(f"{KNOBS[knob]['name']} {show(knob, value)}" for knob, value in values.items())


def distance(values: dict[str, float], base: dict[str, float]) -> float:
    """How far from normal: each condition's harshness (0 as it is, 1 at the harsh end), added in quadrature."""
    return round(sum(harshness(knob, value, base[knob]) ** 2 for knob, value in values.items()) ** 0.5, 4)


def below_limits(values: dict[str, float], limits: list[dict]) -> bool:
    """Every condition is milder than the first value that failed alone (or it never failed alone)."""
    by_knob = {limit["knob"]: limit for limit in limits}
    for knob, value in values.items():
        limit = by_knob.get(knob)
        if limit is None or (limit["never_broke"] and limit["crashed"]):
            return False
        if not limit["never_broke"] and harshness(knob, value, limit["baseline"]) >= harshness(
                knob, limit["first_fail"], limit["baseline"]) - 1e-9:
            return False
    return True


def _spread_line(tried: list[dict[str, float]], ranges: dict[str, dict]) -> str:
    pair, count = most_tested(tried, ranges)
    if pair is None:
        return ""
    return (f"\nMost tested so far: {' and '.join(sorted(pair))} ({_plural(count, 'run')}); at least half your "
            "proposals must use other conditions.")


def hunter_facts(report: dict, ranges: dict[str, dict], most: int) -> str:
    lines = [f"- robot: {report['robot_label']}; failure counted: {FAILURES[report['failure']]}; "
             "the scene as it is passes (run r1)",
             "CONDITIONS you may set (key, its units: what one change at a time measured; your window):"]
    for knob, r in ranges.items():
        limit, (low, high) = r["limit"], r["window"]
        if limit["never_broke"]:
            alone = f"never broke up to {raw(limit['harsh'])}"
        else:
            alone = (f"OK alone at {raw(limit['last_ok'])} (run {limit['ok_run']}), failed alone at "
                     f"{raw(limit['first_fail'])} ({limit['failure']}, run {limit['fail_run']})")
        lines.append(f"- {knob} ({KNOBS[knob]['name']}, {UNITS[knob]}): now {raw(limit['baseline'])}; {alone}; "
                     f"window {raw(low)} to {raw(high)}")
    if (grid := report.get("map")) is not None:
        measured = [(j, i, cell) for j, row in enumerate(grid["cells"]) for i, cell in enumerate(row) if cell["how"] == "run"]
        if measured:
            lines.append(f"MEASURED combinations of {grid['x']['knob']} and {grid['y']['knob']}:")
            lines += [f"- {grid['x']['knob']} {raw(grid['x']['values'][i])}, {grid['y']['knob']} {raw(grid['y']['values'][j])}: "
                      f"{cell['state']} (run {cell['run']})" for j, i, cell in measured]
    lines.append(f"Runs left for hunting: {most}; propose at most {most} combinations.")
    return "FACTS:\n" + "\n".join(lines)


HUNTER_SYSTEM = """You are the Hunter. A search changed one world condition at a time and found where a robot starts to
fail on each. Your job: find COMBINATIONS that fail although every condition in them is still fine on its own.
Reply with JSON only:
{"combinations": [{"values": {"<key>": <number>, "<key>": <number>}, "reason": "<why, citing the measured evidence>"}],
 "decision": "hunt more" or "stop", "reason": "<why>"}
Rules:
- Use only the condition keys listed in FACTS, in their units, each inside its window: 70-90% of its own single
  breaking point (a condition that never broke: up to 90% of its range). Never at or past its single limit.
- Each combination sets 2 or more conditions. speed and speed_slower are one condition: never both.
- Each reason cites the measured results from FACTS (the values that were OK or failed alone, or measured
  combinations), not the windows, for example "falls backward at 12° and slips at friction 0.3: 9° at friction 0.45
  may combine both".
- Spread out: a proposal closer than 15% of each condition's window to another proposal, or to a combination
  already run, is a duplicate; and at least half the proposals use other conditions than the pair tested most.
- decision and its reason are only needed when RESULTS are given: "hunt more" proposes at most 4 new combinations,
  "stop" proposes none."""


def measured_values(report: dict, ranges: dict[str, dict], results: list[dict] | None = None) -> list[float]:
    """The measured results a reason may cite: each condition's last OK value and first failure (or the harsh end
    it held at), the measured map cells and the hunt's own results. Not the windows: they are a rule, not evidence."""
    values = []
    for r in ranges.values():
        limit = r["limit"]
        values += [limit["harsh"]] if limit["never_broke"] else [limit["last_ok"], limit["first_fail"]]
    if (grid := report.get("map")) is not None:
        for j, row in enumerate(grid["cells"]):
            for i, cell in enumerate(row):
                if cell["how"] == "run":
                    values += [grid["x"]["values"][i], grid["y"]["values"][j]]
    for c in results or []:
        values += list(c["values"].values())
    return values


def _cites(reason: str, values: list[float]) -> bool:
    return any(any(abs(float(token) - value) <= 0.5 * 10 ** -(len(token.split(".")[1]) if "." in token else 0) + 1e-9
                   for value in values) for token in _NUMBER.findall(reason))


SPREAD = 0.15   # of each condition's window: a proposal closer than this to another one (or to a run) is a duplicate


def _width(r: dict) -> float:
    return max(r["high"] - r["low"], 1e-6)


def too_close(a: dict[str, float], b: dict[str, float], ranges: dict[str, dict]) -> bool:
    """The same conditions, each within 15% of its window of the other's value."""
    if set(a) != set(b) or not set(a) <= set(ranges):
        return False
    return all(abs(harshness(k, a[k], ranges[k]["limit"]["baseline"]) - harshness(k, b[k], ranges[k]["limit"]["baseline"]))
               < SPREAD * _width(ranges[k]) for k in a)


def tried_so_far(report: dict, results: list[dict]) -> list[dict[str, float]]:
    """The combinations already run: the measured map cells and the hunt's own."""
    tried = []
    if (grid := report.get("map")) is not None:
        for j, row in enumerate(grid["cells"]):
            for i, cell in enumerate(row):
                if cell["how"] == "run":
                    tried.append({grid["x"]["knob"]: grid["x"]["values"][i], grid["y"]["knob"]: grid["y"]["values"][j]})
    return tried + [c["values"] for c in results if c.get("run")]


def most_tested(tried: list[dict[str, float]], ranges: dict[str, dict]) -> tuple[frozenset | None, int]:
    """The set of conditions run together most so far, and how often, when proposals could use another set."""
    counts = Counter(frozenset(values) for values in tried)
    if not counts:
        return None, 0
    pair, n = counts.most_common(1)[0]
    others = [frozenset(two) for two in combinations(ranges, 2) if frozenset(two) != pair and frozenset(two) not in SIBLINGS]
    return (pair, n) if others else (None, 0)


def spread_problems(proposals: list[dict], ranges: dict[str, dict], tried: list[dict[str, float]]) -> list[str]:
    """Duplicates (of each other or of a combination already run) and too many on the most tested pair."""
    issues = []
    for n, combo in enumerate(proposals, 1):
        values = combo["values"]
        if (twin := next((m for m, other in enumerate(proposals[:n - 1], 1) if too_close(values, other["values"], ranges)),
                         None)) is not None:
            issues.append(f"Combination {combo.get('number', n)} is a duplicate of combination "
                          f"{proposals[twin - 1].get('number', twin)}: each value is within {SPREAD * 100:.0f}% of its "
                          "window of the other's; spread the proposals out.")
        elif (ran := next((t for t in tried if too_close(values, t, ranges)), None)) is not None:
            issues.append(f"Combination {combo.get('number', n)} duplicates a combination already run "
                          f"({', '.join(f'{k} {raw(v)}' for k, v in ran.items())}): within {SPREAD * 100:.0f}% of each "
                          "window; try somewhere new.")
    pair, count = most_tested(tried, ranges)
    if pair is not None and proposals:
        same = sum(1 for combo in proposals if frozenset(combo["values"]) == pair)
        need = -(-len(proposals) // 2)
        if len(proposals) - same < need:
            issues.append(f"At least {need} of the {len(proposals)} proposals must use other conditions than "
                          f"{' and '.join(sorted(pair))}, the most tested so far ({_plural(count, 'run')}).")
    return issues


def spread_out(candidates: list[dict], ranges: dict[str, dict], tried: list[dict[str, float]], most: int) -> list[dict]:
    """The rules' proposals under the same spread: no duplicates, and at least half off the most tested pair."""
    pair, _ = most_tested(tried, ranges)
    kept: list[dict] = []
    for c in candidates:
        if not any(too_close(c["values"], t, ranges) for t in tried + [k["values"] for k in kept]):
            kept.append(c)
    other = [c for c in kept if pair is None or frozenset(c["values"]) != pair]
    same = [c for c in kept if c not in other]
    chosen = other[:most]
    chosen += same[:min(most - len(chosen), len(chosen) if pair is not None else most)]
    return sorted(chosen, key=kept.index)


def parse_hunt(data: dict, ranges: dict[str, dict], most: int, evidence: list[float], decide: bool,
               tried: list[dict[str, float]] | None = None) -> dict:
    """Validate the Hunter's JSON; raises ValueError naming every problem, so one retry can fix them all."""
    if not isinstance(data, dict):
        raise ValueError("Reply with one JSON object.")
    combos = data.get("combinations") or []
    if not isinstance(combos, list):
        raise ValueError("combinations must be a list.")
    errors, out = [], []
    if len(combos) > most:
        errors.append(f"{len(combos)} combinations proposed; at most {most}.")
    for n, combo in enumerate(combos[:most], 1):
        if not isinstance(combo, dict) or not isinstance(combo.get("values"), dict):
            errors.append(f"Combination {n}: needs a values object.")
            continue
        values, reason, problems, clean = combo["values"], str(combo.get("reason") or "").strip(), [], {}
        if len(values) < 2:
            problems.append("set at least 2 conditions")
        if any(pair <= set(values) for pair in SIBLINGS):
            problems.append("speed and speed_slower are one condition; use one")
        for knob, value in values.items():
            if knob not in ranges:
                problems.append(f"{knob} isn't a condition you may set (allowed: {', '.join(ranges)})")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{knob} must be a number")
                continue
            r = ranges[knob]
            limit, (low, high) = r["limit"], r["window"]
            h = harshness(knob, float(value), limit["baseline"])
            window = f"keep {knob} between {raw(low)} and {raw(high)}"
            if h < -1e-9 or h > 1 + 1e-9:
                problems.append(f"{knob} {raw(value)} is outside its range ({raw(limit['baseline'])} to {raw(limit['harsh'])})")
            elif h > r["high"] + HUNT_TOLERANCE:
                problems.append(f"{knob} {raw(value)} is " + (f"above 90% of its range; {window}" if limit["never_broke"] else
                                f"at or past its single limit (OK alone at {raw(limit['last_ok'])}, failed alone at "
                                f"{raw(limit['first_fail'])}); {window}"))
            elif h < r["low"] - HUNT_TOLERANCE or h <= HUNT_TOLERANCE:
                problems.append(f"{knob} {raw(value)} is too mild to test anything; {window}")
            else:
                clean[knob] = float(value)
        if not reason or not _cites(reason, evidence):
            problems.append("its reason must cite a measured result from FACTS (a value that was OK or failed alone, "
                            "or a measured combination), not only the windows")
        if problems:
            errors.append(f"Combination {n}: " + "; ".join(problems) + ".")
        else:
            out.append({"values": clean, "reason": reason, "number": n})
    errors += spread_problems(out, ranges, tried or [])
    out = [{"values": c["values"], "reason": c["reason"]} for c in out]
    decision, why = data.get("decision"), str(data.get("reason") or "").strip()
    if decide:
        if decision not in ("hunt more", "stop"):
            errors.append('decision must be "hunt more" or "stop".')
        elif not why:
            errors.append("The decision needs a reason.")
        elif decision == "hunt more" and not out and not errors:
            errors.append('"hunt more" needs at least one combination.')
    elif not out and not errors:
        errors.append("Propose at least one combination.")
    if errors:
        raise ValueError(" ".join(errors))
    return {"combinations": out if decision != "stop" else [], "decision": decision if decide else None, "reason": why}


def _at(r: dict, share: float) -> float:
    """A condition at ``share`` of its single limit (of its range when it never broke), inside its window."""
    limit = r["limit"]
    top = 1.0 if limit["never_broke"] else harshness(limit["knob"], limit["first_fail"], limit["baseline"])
    h = min(max(share * top, r["low"]), r["high"])
    return round(value_at(limit["knob"], h, limit["baseline"]), 4)


def _alone(r: dict) -> str:
    limit = r["limit"]
    if limit["never_broke"]:
        return f"{limit['name']} never broke up to {raw(limit['harsh'])}"
    return f"{limit['name']} is OK alone at {raw(limit['last_ok'])} and fails at {raw(limit['first_fail'])} ({limit['failure']})"


def _pair(ranges: dict[str, dict], a: str, b: str, share: float) -> dict:
    return {"values": {a: _at(ranges[a], share), b: _at(ranges[b], share)},
            "reason": f"{_alone(ranges[a])}; {_alone(ranges[b])}: at {share * 100:.0f}% of each, together they may fail"}


def _pairs(ranges: dict[str, dict]) -> list[tuple[str, str]]:
    """Every pair that can be combined, the most sensitive first (the limits come sorted from the tightest); the
    most sensitive condition with a push before its other partners."""
    order = list(ranges)
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1:] if frozenset({a, b}) not in SIBLINGS]
    if pairs and "push_n" in order[1:]:
        top = pairs[0][0]
        with_push = [(a, b) for a, b in pairs if a == top and b == "push_n"]
        pairs = pairs[:1] + with_push + [p for p in pairs[1:] if p not in with_push]
    return pairs


def rules_hunt(ranges: dict[str, dict], most: int, share: float = 0.8, tried: list[dict] | None = None) -> list[dict]:
    """The two most sensitive conditions at 80% of their limits, the most sensitive one with a push (or, when the
    push can't be set, with the next condition), then the other pairs; spread out like the Hunter's."""
    return spread_out([_pair(ranges, a, b, share) for a, b in _pairs(ranges)], ranges, tried or [], most)


def rules_decide(results: list[dict], ranges: dict[str, dict], most: int, tried: list[dict] | None = None) -> dict:
    failed = [c for c in results if c.get("broke")]
    if most <= 0:
        return {"decision": "stop", "reason": "No runs are left for another round.", "combinations": []}
    if not failed:
        return {"decision": "stop", "reason": "Nothing combined failed at 80% of the single limits, so milder "
                                              "combinations are unlikely to.", "combinations": []}
    milder = [{"values": {k: _at(ranges[k], 0.7) for k in c["values"]},
               "reason": f"{combo_text(c['values'])} failed together; trying 70% of each limit to see how mild it gets"}
              for c in failed]
    others = [_pair(ranges, a, b, 0.8) for a, b in _pairs(ranges)]   # pairs not tried yet, to keep the spread
    milder = spread_out(milder + others, ranges, tried if tried is not None else [r["values"] for r in results], most)
    if not milder:
        return {"decision": "stop", "reason": "The failing combinations are already at the bottom of their windows.",
                "combinations": []}
    return {"decision": "hunt more", "reason": f"{len(failed)} combination{'s' if len(failed) != 1 else ''} failed at 80% "
                                               "of the single limits; testing milder ones.", "combinations": milder[:most]}


def _ask_hunter(messages: list[dict], ranges: dict, most: int, evidence: list[float], decide: bool,
                calls: list[dict], tried: list[dict[str, float]]) -> dict | None:
    parsed, attempts = llm.chat_json(messages, config.PLANNER_MODEL,
                                     lambda data: parse_hunt(data, ranges, most, evidence, decide, tried))
    calls.append({"agent": "Hunter", "purpose": "decide" if decide else "propose", "attempts": attempts})
    return parsed


def _hunt_round(runner: Runner, combos: list[dict], context: dict, number: int, on_step: StepCallback | None) -> list[dict]:
    for combo in combos:
        _emit(on_step, "Hunter", "propose", f"Hunter: {combo['reason']}", f"Testing {combo_text(combo['values'])} together.")
    jobs = []
    for combo in combos:
        variation: dict = {}
        for knob, value in combo["values"].items():
            variation.update(change(knob, value, context))
        jobs.append(({"phase": 3, "knob": "+".join(combo["values"]), "value": list(combo["values"].values()),
                      "values": combo["values"], "round": number}, variation))
    _emit(on_step, "Batch runner", "hunt", f"Hunt round {number}: {len(jobs)} combinations.", "", "working")
    rows = runner.run(jobs)
    results = []
    for combo, row in zip(combos, rows):
        results.append({**combo, "round": number, "run": row["id"], "broke": row["broke"], "failure": row["failure"]})
    _emit(on_step, "Batch runner", "hunt", "Hunt round results.",
          "; ".join(f"{combo_text(c['values'])}: {'crashed' if c['broke'] is None else c['failure'] if c['broke'] else 'OK'}"
                    for c in results) + (f" ({len(combos) - len(rows)} not run: out of budget)" if len(rows) < len(combos) else ""))
    return results


def hunt(runner: Runner, report: dict, ranges: dict[str, dict], context: dict, on_step: StepCallback | None) -> dict:
    """Up to 6 combinations below every single limit, then the Hunter decides "hunt more" (one round, up to 4) or
    "stop" whenever its share of the budget has runs left; only a spent share ends the hunt without its decision.
    Each decision records who made it: the model, the rules (mock mode or a failed reply) or the budget."""
    info: dict[str, Any] = {"by": "rules" if config.MOCK_MODE else "model", "results": [], "decisions": [],
                            "share": runner.left}
    if len(ranges) < 2:
        why = "Fewer than 2 conditions have room below their single limits, so there is nothing to combine."
        info["decisions"].append({"decision": "stop", "reason": why, "by": "rules"})
        _emit(on_step, "Hunter", "decide", "Hunter: looking for combinations.", why)
        return info
    most = min(HUNT_FIRST, runner.left)
    if most <= 0:
        why = "The hunt's share of the budget is spent, so the hunt didn't start."
        info["decisions"].append({"decision": "stop", "reason": why, "by": "budget"})
        _emit(on_step, "Hunter", "decide", "Hunter: looking for combinations.", why, "warning")
        return info
    tried = tried_so_far(report, [])
    facts = hunter_facts(report, ranges, most) + _spread_line(tried, ranges)
    proposal = None
    if not config.MOCK_MODE:
        proposal = _ask_hunter([{"role": "system", "content": HUNTER_SYSTEM}, {"role": "user", "content": facts}],
                               ranges, most, measured_values(report, ranges), False, report["model_calls"], tried)
        if proposal is None:
            info["by"] = "rules"
            _emit(on_step, "Hunter", "decide", "Hunter: proposing combinations.",
                  "The Hunter's reply failed validation twice, so the rules' combinations are tested instead.", "warning")
    combos = proposal["combinations"] if proposal else rules_hunt(ranges, most, tried=tried)
    info["results"] += _hunt_round(runner, combos, context, 1, on_step)

    more = min(HUNT_MORE, runner.left)
    decision, by = None, "rules"
    if more <= 0:   # the share is spent: the budget ends the hunt, not the Hunter
        decision, by = {"decision": "stop", "combinations": [],
                        "reason": f"The hunt's share of the budget ({_plural(info['share'], 'run')}) is spent, so the hunt "
                                  "stops here."}, "budget"
    elif info["by"] == "model":
        lines = [f"- {combo_text(c['values'])} ({', '.join(f'{k} {raw(v)}' for k, v in c['values'].items())}): "
                 + ("crashed" if c["broke"] is None else f"FAILED ({c['failure']})" if c["broke"] else "OK")
                 + f" (run {c['run']})" for c in info["results"]]
        tried = tried_so_far(report, info["results"])
        facts_now = facts.rsplit("\nRuns left", 1)[0] + f"\nRuns left for hunting: {more}." + _spread_line(tried, ranges)
        decision = _ask_hunter([{"role": "system", "content": HUNTER_SYSTEM},
                                {"role": "user", "content": facts_now + "\nRESULTS of round 1:\n" + "\n".join(lines)
                                 + f"\nDecide: \"hunt more\" (at most {more} new combinations) or \"stop\", with a reason."}],
                               ranges, more, measured_values(report, ranges, info["results"]), True, report["model_calls"],
                               tried)
        by = "model" if decision is not None else "rules"
    decision = decision or rules_decide(info["results"], ranges, more, tried_so_far(report, info["results"]))
    info["decisions"].append({"decision": decision["decision"], "reason": decision["reason"], "by": by})
    _emit(on_step, "Hunter", "decide", "The budget ends the hunt." if by == "budget" else
          f"Hunter decides: {decision['decision']}.", decision["reason"])
    if decision["decision"] == "hunt more":
        fresh = decision["combinations"][:more]
        if fresh:
            info["results"] += _hunt_round(runner, fresh, context, 2, on_step)
    return info


# ── Confirmation: the same points on 3 seeds ─────────────────────────────────

CONFIRM_SEEDS = 3        # seed 0 is the run already made; seeds 1 and 2 are new
INFERRED_CHECKS = 2      # inferred map cells run for real
CONFIRM_BATCH = 12


def _verdict(rows: list[dict], wanted: str) -> dict:
    """``rows`` are one point on each seed. A failure reproduces when another seed fails too; a pass when another
    seed passes. Labels follow reliability: reliable fail / borderline / reliable pass."""
    ran = [row for row in rows if row["broke"] is not None]
    passed = sum(1 for row in ran if not row["broke"])
    failed = len(ran) - passed
    result = {"runs": [row["id"] for row in rows], "passed": passed, "total": len(ran),
              "text": reliability.ci_text(passed, len(ran))}
    if len(ran) < 2:
        result["label"] = "not confirmed"
    elif wanted == "fail":
        result["label"] = "didn't reproduce" if failed <= 1 else reliability.label(passed, len(ran))
    else:
        result["label"] = "failed on other seeds" if passed <= 1 else reliability.label(passed, len(ran))
    return result


def confirm(runner: Runner, report: dict, context: dict, on_step: StepCallback | None) -> None:
    """Re-run every failure found and the phase-1 boundary points on seeds 1 and 2 (seed 0 is the run already made),
    and run up to 2 inferred map cells for real. In order of what matters most, while the budget lasts: the hunt's
    failures; then each limit from the tightest, its first failure and its last OK value; then the inferred cells."""
    evidence = {row["id"]: row for row in runner.evidence}
    base = context["base"]
    items = []   # (what, the original row, wanted)
    for c in report["hunt"]["results"]:
        if c["broke"]:
            items.append(({"kind": "combined", "combo": c}, evidence[c["run"]], "fail"))
    for limit in report["limits"]:
        if not limit["never_broke"]:
            items.append(({"kind": "limit", "limit": limit, "side": "fails_at"}, evidence[limit["fail_run"]], "fail"))
            if limit["ok_run"] in evidence:
                items.append(({"kind": "limit", "limit": limit, "side": "ok_at"}, evidence[limit["ok_run"]], "pass"))
    checks = _inferred_to_check(report.get("map"))

    jobs, planned, unconfirmed, left, rerun = [], [], [], runner.left, set()
    for item, original, wanted in items + [("check", cell, None) for cell in checks]:
        if item != "check" and original["id"] in rerun:   # the same run behind two points (the scene as it is)
            planned.append((item, original, wanted))
            continue
        need = 1 if item == "check" else CONFIRM_SEEDS - 1
        if left < need:
            unconfirmed.append(item)
            continue
        left -= need
        if item == "check":
            j, i = original
            grid = report["map"]
            variation = {**change(grid["x"]["knob"], grid["x"]["values"][i], context),
                         **change(grid["y"]["knob"], grid["y"]["values"][j], context)}
            jobs.append(({"phase": 4, "knob": f"{grid['x']['knob']}+{grid['y']['knob']}", "checks": [j, i],
                          "value": [grid["x"]["values"][i], grid["y"]["values"][j]]}, variation))
        else:
            rerun.add(original["id"])
            for k in range(1, CONFIRM_SEEDS):
                jobs.append(({"phase": 4, "knob": original.get("knob"), "value": original.get("value"),
                              "confirms": original["id"], "seed": k},
                             reliability.seeded(runner.scene, original["change"], k)))
        planned.append((item, original, wanted))
    if jobs:
        _emit(on_step, "Batch runner", "confirm",
              f"Confirming {sum(1 for p in planned if p[0] != 'check')} points on {CONFIRM_SEEDS} seeds"
              + (f" and running {sum(1 for p in planned if p[0] == 'check')} inferred map cells for real" if checks else "") + ".",
              "", "working")
    rows = []
    for start in range(0, len(jobs), CONFIRM_BATCH):
        rows += runner.run(jobs[start:start + CONFIRM_BATCH])
    by_original: dict[str, list[dict]] = {}
    checked: dict[tuple[int, int], dict] = {}
    for row in rows:
        if "confirms" in row:
            by_original.setdefault(row["confirms"], []).append(row)
        else:
            checked[tuple(row["checks"])] = row

    failures, dropped = [], []
    for item, original, wanted in planned:
        if item == "check":
            continue
        verdict = _verdict([original] + by_original.get(original["id"], []), wanted)
        if item["kind"] == "limit":
            item["limit"].setdefault("confirm", {})[item["side"]] = verdict
            if item["side"] == "fails_at":
                values = {item["limit"]["knob"]: item["limit"]["first_fail"]}
                entry = {"kind": "single", "values": values, "failure": item["limit"]["failure"], "run": original["id"]}
            else:
                continue
        else:
            c = item["combo"]
            entry = {"kind": "combined", "values": c["values"], "failure": c["failure"], "run": c["run"],
                     "reason": c["reason"], "round": c["round"]}
        entry.update(text=combo_text(entry["values"]), distance=distance(entry["values"], base), confirm=verdict)
        (dropped if verdict["label"] == "didn't reproduce" else failures).append(entry)
    for entry in failures + dropped:   # once every limit has its badges
        entry["surprising"], entry["surprise_note"] = surprise(entry, report["limits"])
    report["failures"] = rank([f for f in failures if f["confirm"]["label"] in ("reliable fail", "borderline")])
    report["dropped"] = dropped
    report["unconfirmed"] = [_unconfirmed_text(item) for item in unconfirmed] + \
                            [f"{f['text']} (its other seeds crashed)" for f in failures if f["confirm"]["label"] == "not confirmed"]

    wrong = 0
    if (grid := report.get("map")) is not None:
        for (j, i), row in checked.items():
            if row["broke"] is None:
                continue
            cell = grid["cells"][j][i]
            state = "fail" if row["broke"] else "pass"
            grid["cells"][j][i] = {"state": state, "how": "run", "run": row["id"], "checked": True,
                                   "inferred_as": cell["state"], "corrected": state != cell["state"]}
            wrong += state != cell["state"]
        grid["checked"] = {"checked": sum(1 for row in checked.values() if row["broke"] is not None), "wrong": wrong}
        recount(grid)
    _emit(on_step, "Finder", "confirm", "Keeping only what reproduces.", confirm_text(report) or "Nothing to confirm.")


def surprise(entry: dict, limits: list[dict]) -> tuple[bool, str | None]:
    """A combined failure is surprising when each condition in it is fine alone, and that is known on 3 seeds: its
    value is at or below the condition's last OK value, and that OK value passed on every seed when rechecked (a
    condition that never broke held at the harsh end). Below single-seed limits only, it is a possible surprise, with
    the reason it isn't confirmed."""
    if entry["kind"] != "combined" or not below_limits(entry["values"], limits):
        return False, None
    by_knob = {limit["knob"]: limit for limit in limits}
    unsure = []
    for knob, value in entry["values"].items():
        limit = by_knob[knob]
        if limit["never_broke"]:
            continue
        ok = limit.get("confirm", {}).get("ok_at")
        if harshness(knob, value, limit["baseline"]) > harshness(knob, limit["last_ok"], limit["baseline"]) + 1e-9:
            unsure.append(f"{limit['name']} {show(knob, value)} is past its last OK value alone")
        elif ok is None:
            unsure.append(f"{limit['name']}'s OK value alone ({limit['shown']['last_ok']}) wasn't rechecked")
        elif ok["label"] != "reliable pass":
            unsure.append(f"{limit['name']}'s OK value alone ({limit['shown']['last_ok']}) {ok['text']}")
    if unsure:
        return False, "Possibly surprising, but " + "; ".join(unsure) + "."
    return True, None


def _inferred_to_check(grid: dict | None) -> list[tuple[int, int]]:
    """Up to 2 inferred cells where the guess is least sure: not an edge the search already decided, and nearest the
    diagonal the boundary crosses."""
    if grid is None:
        return []
    cells = [(j, i) for j, row in enumerate(grid["cells"]) for i, cell in enumerate(row) if cell["how"] == "inferred"]
    cells.sort(key=lambda c: (grid["cells"][c[0]][c[1]].get("from") == "search", abs(c[0] + c[1] - (GRID - 1)), -min(c)))
    return cells[:INFERRED_CHECKS]


def _unconfirmed_text(item) -> str:
    if item == "check":
        return "an inferred map cell"
    if item["kind"] == "combined":
        return f"{combo_text(item['combo']['values'])} (failed once)"
    limit = item["limit"]
    side = "first failure" if item["side"] == "fails_at" else "last OK value"
    return f"the {limit['name']} limit's {side}"


def rank(failures: list[dict]) -> list[dict]:
    """Closest to normal first; at the same distance, fewer conditions first."""
    return sorted(failures, key=lambda f: (f["distance"], len(f["values"])))


def _plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else many or word + 's'}"


def confirm_text(report: dict) -> str:
    """What the confirmation kept, dropped, corrected and couldn't get to."""
    parts = []
    failures = report.get("failures") or []
    if failures:
        reliable = sum(1 for f in failures if f["confirm"]["label"] == "reliable fail")
        parts.append(f"{_plural(len(failures), 'failure')} kept on {CONFIRM_SEEDS} seeds ({reliable} reliable, "
                     f"{len(failures) - reliable} borderline)")
    if (dropped := report.get("dropped")):
        parts.append(f"{_plural(len(dropped), 'apparent failure')} didn't reproduce")
    if (grid := report.get("map")) is not None and grid["checked"]["checked"]:
        wrong = grid["checked"]["wrong"]
        parts.append(f"{_plural(grid['checked']['checked'], 'inferred cell')} checked by running "
                     + ("it" if grid["checked"]["checked"] == 1 else "them") + ", "
                     + (f"{wrong} {'was' if wrong == 1 else 'were'} wrong; corrected" if wrong else "none wrong"))
    if (unconfirmed := report.get("unconfirmed")):
        parts.append("not confirmed (out of budget): " + ", ".join(unconfirmed))
    return "; ".join(parts) + "." if parts else ""


# ── Explainer ────────────────────────────────────────────────────────────────

def honesty(report: dict) -> str:
    """Written in code: how many runs, how precise each limit is, and that it is simulation only."""
    steps = [f"{limit['name']} {limit['shown']['step']}" for limit in report["limits"] if not limit["never_broke"]]
    precision = f"limits accurate to one search step ({', '.join(steps)})" if steps else "no limit found in the tested ranges"
    inferred = (f"; {report['map']['inferred']} of {GRID * GRID} map cells inferred, not run"
                if report.get("map") is not None else "")
    line = f"{report['runs']} runs, {precision}, simulation only (search on one seed; one robot judged{inferred})."
    if (confirmed := confirm_text(report)):
        line += f" Confirmation: {confirmed}"
    return line


def _seeds(verdict: dict) -> str:
    return f"{verdict['label']}, failed on {verdict['total'] - verdict['passed']} of {verdict['total']} seeds"


def _rechecked(limit: dict) -> str:
    """How sure a limit is: rechecked on 3 seeds, or a single run each."""
    confirm = limit.get("confirm", {})
    if (again := confirm.get("fails_at")) is None:
        return "from one run each, not rechecked"
    ok = confirm.get("ok_at")
    return (f"rechecked on {CONFIRM_SEEDS} seeds: the failure {_seeds(again)}"
            + (f"; the OK value {ok['label']}, passed on {ok['passed']} of {ok['total']} seeds" if ok else
               "; the OK value not rechecked"))


def _push_note(report: dict) -> str:
    push = report.get("push")
    return f" (at {push['at_s']:.1f} s, for {push['for_s']:.1f} s)" if push else ""


def limit_sentences(report: dict) -> str:
    """The limits, written in code from one template, tightest first: "Sideways push (at 2.0 s, for 0.3 s): OK at
    30 N, fell at 40 N (single runs, not rechecked)." The Explainer only writes what follows."""
    sentences = []
    for limit in report["limits"]:
        shown = limit["shown"]
        name = limit["name"].capitalize() + (_push_note(report) if limit["knob"] == "push_n" else "")
        if limit["never_broke"]:
            sentences.append(f"{name}: never broke, up to {shown['harsh']} (a single run).")
            continue
        confirm = limit.get("confirm", {})
        fail, ok = confirm.get("fails_at"), confirm.get("ok_at")
        if fail is None:
            sure = "single runs, not rechecked"
        else:
            sure = (f"rechecked on {CONFIRM_SEEDS} seeds: " + (f"OK {ok['passed']}/{ok['total']}, " if ok else "")
                    + f"failed {fail['total'] - fail['passed']}/{fail['total']}" + ("" if ok else "; the OK value a single run"))
        sentences.append(f"{name}: OK at {shown['last_ok']}, {limit['failure']} at {shown['first_fail']} ({sure}).")
    return " ".join(sentences)


def facts_text(report: dict) -> str:
    lines = [f"- robot: {report['robot_label']}; failure counted: {FAILURES[report['failure']]}; runs: {report['runs']}"]
    for limit in report["limits"]:
        shown = limit["shown"]
        name = limit["name"] + (_push_note(report) if limit["knob"] == "push_n" else "")
        if limit["never_broke"]:
            lines.append(f"- {name} (now {shown['baseline']}): never broke up to {shown['harsh']}")
        else:
            lines.append(f"- {name} (now {shown['baseline']}): last OK at {shown['last_ok']}, first failure at "
                         f"{shown['first_fail']} ({limit['failure']}), accurate to {shown['step']}; {_rechecked(limit)}")
    for f in report.get("failures") or []:   # confirmed only: what didn't reproduce is counted, not described
        if f["kind"] == "combined":
            lines.append(f"- together: {f['text']} failed ({f['failure']})"
                         + (", although each condition is fine alone (its OK value rechecked on 3 seeds)" if f["surprising"]
                            else ", below each condition's single limit, but those limits come from one run" if f["surprise_note"]
                            else "")
                         + f"; {_seeds(f['confirm'])}")
    if (grid := report.get("map")) is not None:   # only the combinations that ran: inferred cells aren't measurements
        counts = grid["counts"]
        lines.append(f"- {grid['x']['name']} and {grid['y']['name']} changed together: {counts['run']} combinations run, "
                     f"{counts['pass']} passed, {counts['fail']} failed")
        if grid["checked"]["checked"]:
            lines.append(f"- {grid['checked']['checked']} more map cells were run to check the map's guesses; "
                         f"{grid['checked']['wrong']} guesses were wrong and were corrected")
    if (dropped := report.get("dropped")):
        lines.append(f"- {len(dropped)} apparent failures didn't reproduce on other seeds and were dropped")
    if (unconfirmed := report.get("unconfirmed")):
        lines.append(f"- not confirmed for lack of budget: {len(unconfirmed)}")
    return ("FACTS (the limits are ordered from the tightest; use these numbers as written):\n" + "\n".join(lines)
            + "\n\nALREADY WRITTEN for the reader, just before your text:\n" + limit_sentences(report))


def rules_text(report: dict) -> str:
    """The interpretation after the limit sentences, by rules: what failed together, the map, or that nothing broke."""
    sentences = []
    combined = [f for f in report.get("failures") or [] if f["kind"] == "combined"]
    if combined:
        f = next((f for f in combined if f["surprising"]), combined[0])
        sentences.append(f"Together, {f['text']} failed ({f['failure']})"
                         + (", although each was fine on its own" if f["surprising"] else
                            ", below each single limit found in one run" if f["surprise_note"] else "")
                         + f"; it was {_seeds(f['confirm'])}.")
    if not any(not limit["never_broke"] for limit in report["limits"]):
        sentences.append("It never broke within the tested ranges.")
    if (grid := report.get("map")) is not None and grid["counts"]["run"]:
        counts = grid["counts"]
        sentences.append(f"With {grid['x']['name']} and {grid['y']['name']} changed together, {counts['run']} combinations "
                         f"were run: {counts['pass']} passed and {counts['fail']} failed.")
    if not sentences:
        sentences.append(f"This comes from {report['runs']} simulation runs.")
    return " ".join(sentences[:3])


EXPLAINER_SYSTEM = """You are the Explainer. A search found where a robot starts to fail as the world gets harsher.
The limits are already written for the reader, by code, just before your text (see ALREADY WRITTEN). Write 1 to 3
plain sentences for a non-expert that interpret them: which matters most for this robot, what the combined failures
show, and how sure the results are. Do not repeat the limit values; refer to the conditions by name. Use only
numbers that appear in FACTS, written as they are there; never estimate, round differently or add a number. Do not
suggest changing the robot. No headings or lists."""

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def _matches(token: str, values: list[float]) -> bool:
    decimals = len(token.split(".")[1]) if "." in token else 0
    return any(abs(float(token) - value) <= 0.5 * 10 ** -decimals + 1e-9 for value in values)


def _values(limit: dict, key: str) -> list[float]:
    """A limit's value in its own units and as shown (a percentage, say)."""
    return [float(limit[key])] + [float(n) for n in _NUMBER.findall(limit["shown"][key])]


def allowed_numbers(report: dict) -> list[float]:
    """Every number the envelope states: values in their own units and as shown (percentages), steps, counts."""
    numbers = [float(report["runs"]), float(GRID), float(GRID * GRID), TIPPED_DEG, float(len(report["limits"]))]
    if (push := report.get("push")):
        numbers += [float(push["at_s"]), float(push["for_s"])]
    for limit in report["limits"]:
        for key in ("baseline", "harsh", "last_ok", "first_fail", "step"):
            if limit[key] is not None:
                numbers.append(float(limit[key]))
        for shown in limit["shown"].values():
            numbers += [float(n) for n in _NUMBER.findall(shown)]
    if (grid := report.get("map")) is not None:   # measured counts only: the inferred ones are not facts to state
        numbers += [float(grid["counts"][key]) for key in ("run", "pass", "fail")]
        for axis in (grid["x"], grid["y"]):
            numbers += [float(n) for shown in axis["shown"] for n in _NUMBER.findall(shown)]
        numbers += [float(grid["checked"]["checked"]), float(grid["checked"]["wrong"])]
    # the confirmation: seeds, counts, and each confirmed failure's values and seed counts
    verdicts = [verdict for limit in report["limits"] for verdict in limit.get("confirm", {}).values()]
    numbers += [float(CONFIRM_SEEDS), float(len(report.get("failures") or [])), float(len(report.get("dropped") or [])),
                float(len(report.get("unconfirmed") or []))]
    for f in report.get("failures") or []:
        verdicts.append(f["confirm"])
        if f["kind"] == "combined":
            numbers += [float(v) for v in f["values"].values()] + [float(n) for n in _NUMBER.findall(f["text"])]
    for verdict in verdicts:
        numbers += [float(verdict["passed"]), float(verdict["total"]), float(verdict["total"] - verdict["passed"])]
    return numbers


def number_problems(text: str, report: dict) -> list[str]:
    """The interpretation's numbers: each one the envelope states (as written, or that value rounded as written),
    and none of them a limit's value, which the code's sentences already state."""
    allowed = allowed_numbers(report)
    edges = [value for limit in report["limits"] if not limit["never_broke"]
             for key in ("last_ok", "first_fail") for value in _values(limit, key)]
    # a number that is also a count, or a value of a combined failure (one may sit at a last OK value), may be said
    verdicts = [v for limit in report["limits"] for v in limit.get("confirm", {}).values()] + \
               [f["confirm"] for f in report.get("failures") or []]
    other = [float(report["runs"]), float(CONFIRM_SEEDS), float(GRID), float(GRID * GRID), float(len(report["limits"])),
             float(len(report.get("failures") or [])), float(len(report.get("dropped") or []))]
    other += [float(n) for v in verdicts for n in (v["passed"], v["total"], v["total"] - v["passed"])]
    other += [float(v) for f in report.get("failures") or [] if f["kind"] == "combined" for v in f["values"].values()]
    other += [float(n) for f in report.get("failures") or [] if f["kind"] == "combined" for n in _NUMBER.findall(f["text"])]
    if (grid := report.get("map")) is not None:
        other += [float(n) for n in (*grid["counts"].values(), *grid["checked"].values())]
    issues = []
    for token in _NUMBER.findall(text):
        if not _matches(token, allowed):
            issues.append(f"It states {token}, which isn't a number in FACTS.")
        elif _matches(token, edges) and not _matches(token, other):
            issues.append(f"It repeats {token}, a limit value the sentences before it already state.")
    return issues


def explain(report: dict, calls: list[dict]) -> str:
    """The limit sentences (code), then the Explainer's interpretation, checked number by number against the
    envelope (one retry with the exact problems, then the rules' interpretation)."""
    limits = limit_sentences(report)
    if config.MOCK_MODE:
        return f"{limits} {rules_text(report)}".strip()
    messages = [{"role": "system", "content": EXPLAINER_SYSTEM}, {"role": "user", "content": facts_text(report)}]
    attempts = []
    for attempt in range(2):
        text = llm.chat(messages, temperature=0.3).strip()
        issues = number_problems(text, report)
        attempts.append({"attempt": attempt + 1, "valid": not issues, "problems": issues})
        if not issues:
            calls.append({"agent": "Explainer", "purpose": "envelope", "attempts": attempts})
            return f"{limits} {text}".strip()
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Your reply has problems: " + " ".join(issues)
                      + " Rewrite it using only the numbers in FACTS, without repeating the limit values."}]
    calls.append({"agent": "Explainer", "purpose": "envelope", "attempts": attempts})
    return f"{limits} {rules_text(report)}".strip()


# ── The search ───────────────────────────────────────────────────────────────

def _emit(on_step: StepCallback | None, agent: str, tool: str, thought: str, result: str, status: str = "done") -> None:
    if on_step is not None:
        on_step({"agent": agent, "tool": tool, "thought": thought, "result": result, "status": status})


def prepare(spec: SceneSpec, knobs: list[str]) -> SceneSpec:
    """The scene the search runs on. Flat ground searched for a slope becomes a 0° slope without bumps: the same
    ground, written as a slope so its angle can change."""
    scene = spec.model_copy(deep=True)
    if "slope_deg" in knobs and scene.terrain is not None and scene.terrain.kind == "flat":
        scene.terrain = Terrain(**{**scene.terrain.model_dump(), "kind": "slope", "slope_deg": 0.0, "roughness": 0.0})
    scene.assumptions = []
    return scene.fill_defaults()


def find_failures(spec: SceneSpec, robot_id: str | None = None, failure: str = "task", budget: str = "quick",
                  knobs: list[str] | None = None, on_step: StepCallback | None = None) -> dict:
    """Where one robot starts to fail as the world gets harsher: limits, a map of the two tightest conditions,
    the runs behind them, what it cost and an honesty line."""
    with cost.track() as spent:
        report = _find(spec, robot_id, failure, budget, knobs, on_step)
        report["cost"] = spent.snapshot()
    return report


def _find(spec: SceneSpec, robot_id: str | None, failure: str, budget: str, named: list[str] | None,
          on_step: StepCallback | None) -> dict:
    if spec.domain != "robot_terrain" or not spec.robots:
        raise ValueError("Finding where it breaks needs a robot scene.")
    entry = _entry(spec, robot_id)
    # a named "gravity" comes from the user's own words (the Director drops it otherwise), so any world may search it
    fits, skipped = scene_knobs(spec, entry.robot, gravity_asked="gravity" in (named or []))
    chosen, notes = choose_knobs(fits, budget, named)
    report: dict[str, Any] = {"type": "envelope", "robot_id": entry.id, "robot_label": entry.label, "failure": failure,
                              "budget": budget, "budget_runs": BUDGETS[budget], "knobs": chosen, "notes": notes,
                              "skipped": {k: v for k, v in skipped.items() if named and k in named}, "limits": [],
                              "map": None, "evidence": [], "runs": 0, "model_calls": [], "failures": [], "dropped": [],
                              "unconfirmed": [], "hunt": {"by": "rules", "results": [], "decisions": []}}
    if "push_n" in chosen:   # when the push comes, for the limit sentence and FACTS
        from .backends.mujoco_backend import push_time
        from .spec import Disturbance

        pushed = spec.model_copy(update={"disturbance": spec.disturbance or Disturbance(push_n=1.0)})
        report["push"] = {"at_s": push_time(pushed), "for_s": pushed.disturbance.duration_s}
    others = f" The other robots stay as they are and only {entry.label} is judged." if len(spec.robots) > 1 else ""
    _emit(on_step, "Finder", "conditions", "Choosing the world conditions to search (never the robot's design).",
          f"Searching {', '.join(KNOBS[k]['name'] for k in chosen)} for where {entry.label} "
          f"{FAILURES[failure]}; budget {budget} (at most {BUDGETS[budget]} runs).{others}"
          + (" " + " ".join(notes) if notes else ""))
    scene = prepare(spec, chosen)
    context = {"base": fits, "robot_id": entry.id, "mass": entry.robot.mass_kg,
               "obstacles": _obstacle_context(spec, entry.robot)}
    runner = Runner(scene, entry.id, failure, BUDGETS[budget])
    shares, carry = SHARES[budget], 0
    report["shares"] = {}

    def start(phase: str) -> int:
        """The phase's share plus whatever the earlier phases left unused."""
        nonlocal carry
        allowance = runner.phase(shares[phase] + carry)
        report["shares"][phase] = {"share": shares[phase], "allowed": allowance, "from": len(runner.evidence)}
        return allowance

    def finish(phase: str) -> None:
        nonlocal carry
        entry_ = report["shares"][phase]
        entry_["used"] = len(runner.evidence) - entry_["from"]
        carry = entry_["allowed"] - entry_["used"]

    start("search")   # the scene as it is counts toward the search
    _emit(on_step, "Batch runner", "baseline", "Running the scene as it is first.", "", "working")
    first = runner.run([({"phase": 0, "knob": None, "value": None}, {})])[0]
    if first["broke"] is not False:
        message = ("The scene as it is crashed, so nothing could be searched." if first["broke"] is None else
                   f"It already fails as it is ({first['failure']}), so there is no breaking point to search for: "
                   "ask why it failed instead.")
        _emit(on_step, "Batch runner", "baseline", "Running the scene as it is.", message, "warning")
        report.update(evidence=runner.evidence, runs=len(runner.evidence), stopped=message, explanation=message)
        report["honesty"] = f"{report['runs']} run, simulation only."
        return report
    _emit(on_step, "Batch runner", "baseline", "Running the scene as it is.", "It passes as it is (run r1).")

    report["limits"] = search(runner, chosen, context, on_step)
    finish("search")
    pair = report["limits"][:2]
    start("map")
    if len(pair) == 2 and runner.left > 0:
        report["map"] = grid_map(runner, pair[0], pair[1], context, on_step)
    finish("map")
    start("hunt")
    report["hunt"] = hunt(runner, report, hunt_ranges(report["limits"]), context, on_step)
    finish("hunt")
    start("confirm")
    confirm(runner, report, context, on_step)
    finish("confirm")
    report["evidence"], report["runs"] = runner.evidence, len(runner.evidence)
    report["honesty"] = honesty(report)
    report["explanation"] = explain(report, report["model_calls"])
    _emit(on_step, "Explainer", "envelope", "Explaining the limits from the measured envelope only.", report["explanation"])
    return report


def describe(report: dict) -> str:
    """The chat reply: the explanation, then the honesty line."""
    return f"{report['explanation']} {report.get('honesty', '')}".strip()
