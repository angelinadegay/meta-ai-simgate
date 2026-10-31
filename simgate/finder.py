"""Find where it breaks: search the world's conditions (never the robot's design) for where one robot starts to
fail, then map the two tightest conditions together. Every run goes through the batch runner.

Phase 1 binary-searches each condition, one change at a time, from the scene as it is toward the harsh end of its
range (at most 5 runs per condition, the conditions in parallel). Phase 2 maps the two conditions whose limits are
closest to the scene as it is on a 4×4 grid, running the cells nearest the boundary first and filling in the rest
by assuming that harsher on both never helps (shown as inferred). The limits, the map, the runs behind them and the
cost are reported with an honesty line written in code, and the Explainer's numbers are checked against them.
"""

import re
from typing import Any, Callable

from . import config, cost, llm
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
BUDGETS = {"quick": 25, "thorough": 60}
KNOB_COUNT = {"quick": 3, "thorough": 4}
MAX_KNOBS = 4
SEARCH_RUNS = 5          # per condition: its harsh end, then 4 halvings
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
            return chosen[:MAX_KNOBS], notes
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
        self.evidence: list[dict] = []

    @property
    def left(self) -> int:
        return self.budget - len(self.evidence)

    def run(self, jobs: list[tuple[dict, dict]]) -> list[dict]:
        """jobs = (what it tests, the variation); returns one judged evidence row per job."""
        jobs = jobs[: max(0, self.left)]
        if not jobs:
            return []
        seed = self.scene.terrain.seed if self.scene.terrain is not None else 0
        variations = [{"label": f"run {len(self.evidence) + i + 1}", **(variation or {"seed": seed})}
                      for i, (_, variation) in enumerate(jobs)]
        rows = []
        for (setting, variation), result in zip(jobs, run_batch(self.scene, variations, baseline=False)):
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

def search(runner: Runner, knobs: list[str], context: dict, on_step: StepCallback | None) -> list[dict]:
    """Binary-search every condition at once (one change per run): its harsh end first, then halve the bracket
    between the last value that was OK and the first that failed. At most SEARCH_RUNS runs per condition."""
    base = context["base"]
    state = {knob: {"ok": base[knob], "fail": None, "ok_run": "r1", "fail_run": None, "failure": None, "runs": 0,
                    "done": False, "crashed": 0} for knob in knobs}
    for step in range(SEARCH_RUNS):
        active = [knob for knob in knobs if not state[knob]["done"] and state[knob]["runs"] < SEARCH_RUNS]
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
    xs, ys = _axis(first), _axis(second)
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
    while runner.left > 0:
        unknown = [(j, i) for j in range(GRID) for i in range(GRID) if cells[j][i]["state"] is None]
        if not unknown:
            break
        unknown.sort(key=lambda c: abs(harshness(first["knob"], xs[c[1]], first["baseline"]) / bx
                                       + harshness(second["knob"], ys[c[0]], second["baseline"]) / by - 1))
        picked = unknown[: min(MAP_BATCH, runner.left)]
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
    flat = [cell for row in cells for cell in row]
    run = [c for c in flat if c["how"] == "run"]
    inferred = [c for c in flat if c["how"] == "inferred"]
    return {"x": {"knob": first["knob"], "name": first["name"], "values": xs, "shown": [show(first["knob"], v) for v in xs]},
            "y": {"knob": second["knob"], "name": second["name"], "values": ys, "shown": [show(second["knob"], v) for v in ys]},
            "cells": cells,
            # measured: the cells that ran; inferred: the rest that were filled in; unknown: neither
            "counts": {"run": len(run), "pass": sum(c["state"] == "pass" for c in run),
                       "fail": sum(c["state"] == "fail" for c in run), "error": sum(c["state"] == "error" for c in run),
                       "inferred": len(inferred), "unknown": sum(1 for c in flat if c["state"] is None)},
            "inferred": len(inferred)}


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


# ── Explainer ────────────────────────────────────────────────────────────────

def honesty(report: dict) -> str:
    """Written in code: how many runs, how precise each limit is, and that it is simulation only."""
    steps = [f"{limit['name']} {limit['shown']['step']}" for limit in report["limits"] if not limit["never_broke"]]
    precision = f"limits accurate to one search step ({', '.join(steps)})" if steps else "no limit found in the tested ranges"
    inferred = (f"; {report['map']['inferred']} of {GRID * GRID} map cells inferred, not run"
                if report.get("map") is not None else "")
    return f"{report['runs']} runs, {precision}, simulation only (one seed; one robot judged{inferred})."


def facts_text(report: dict) -> str:
    lines = [f"- robot: {report['robot_label']}; failure counted: {FAILURES[report['failure']]}; runs: {report['runs']}"]
    for limit in report["limits"]:
        shown = limit["shown"]
        if limit["never_broke"]:
            lines.append(f"- {limit['name']} (now {shown['baseline']}): never broke up to {shown['harsh']}")
        else:
            lines.append(f"- {limit['name']} (now {shown['baseline']}): OK at {shown['last_ok']}, failed at "
                         f"{shown['first_fail']} ({limit['failure']}); accurate to {shown['step']}")
    if (grid := report.get("map")) is not None:   # only the combinations that ran: inferred cells aren't measurements
        counts = grid["counts"]
        lines.append(f"- {grid['x']['name']} and {grid['y']['name']} changed together: {counts['run']} combinations run, "
                     f"{counts['pass']} passed, {counts['fail']} failed")
    return "FACTS (the limits are ordered from the tightest; use these numbers as written):\n" + "\n".join(lines)


def rules_text(report: dict) -> str:
    broke = [limit for limit in report["limits"] if not limit["never_broke"]]
    held = [limit for limit in report["limits"] if limit["never_broke"]]
    sentences = []
    if broke:
        top = broke[0]
        sentences.append(f"The tightest limit is {top['name']}: it was OK at {top['shown']['last_ok']} and failed at "
                         f"{top['shown']['first_fail']} ({top['failure']}), against {top['shown']['baseline']} now.")
        for limit in broke[1:]:
            sentences.append(f"{limit['name'].capitalize()} was OK at {limit['shown']['last_ok']} and failed at "
                             f"{limit['shown']['first_fail']} ({limit['failure']}).")
    else:
        sentences.append("It never broke within the tested ranges.")
    for limit in held:
        sentences.append(f"It never broke on {limit['name']}, up to {limit['shown']['harsh']}.")
    if (grid := report.get("map")) is not None and grid["counts"]["run"]:
        counts = grid["counts"]
        sentences.append(f"With {grid['x']['name']} and {grid['y']['name']} changed together, {counts['run']} combinations "
                         f"were run: {counts['pass']} passed and {counts['fail']} failed.")
    if len(sentences) < 3:
        sentences.append(f"This comes from {report['runs']} simulation runs.")
    return " ".join(sentences[:5])


EXPLAINER_SYSTEM = """You are the Explainer. A search found where a robot starts to fail as the world gets harsher.
Write 3 to 5 plain sentences for a non-expert. Start with the most important limit: the first one in FACTS. Use only
numbers that appear in FACTS, written as they are there; never estimate, round differently or add a number. Do not
suggest changing the robot. No headings or lists."""

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def allowed_numbers(report: dict) -> list[float]:
    """Every number the envelope states: values in their own units and as shown (percentages), steps, counts."""
    numbers = [float(report["runs"]), float(GRID), float(GRID * GRID), TIPPED_DEG, float(len(report["limits"]))]
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
    return numbers


def number_problems(text: str, report: dict) -> list[str]:
    """Each number in the text must be one the envelope states (as written, or that value rounded as written);
    and the text must start with the tightest limit."""
    allowed = allowed_numbers(report)
    issues = []
    for token in _NUMBER.findall(text):
        decimals = len(token.split(".")[1]) if "." in token else 0
        stated = float(token)
        if not any(abs(stated - value) <= 0.5 * 10 ** -decimals + 1e-9 for value in allowed):
            issues.append(f'It states {token}, which isn\'t a number in FACTS.')
    broke = [limit for limit in report["limits"] if not limit["never_broke"]]
    if broke:
        first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
        if broke[0]["name"].split()[-1].lower() not in first.lower():
            issues.append(f"It doesn't start with the tightest limit, {broke[0]['name']}.")
    return issues


def explain(report: dict, calls: list[dict]) -> str:
    """The Explainer's text, checked number by number against the envelope (one retry with the exact problems,
    then the rules text)."""
    if config.MOCK_MODE:
        return rules_text(report)
    messages = [{"role": "system", "content": EXPLAINER_SYSTEM}, {"role": "user", "content": facts_text(report)}]
    attempts = []
    for attempt in range(2):
        text = llm.chat(messages, temperature=0.3).strip()
        issues = number_problems(text, report)
        attempts.append({"attempt": attempt + 1, "valid": not issues, "problems": issues})
        if not issues:
            calls.append({"agent": "Explainer", "purpose": "envelope", "attempts": attempts})
            return text
        messages += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Your reply has problems: " + " ".join(issues)
                      + " Rewrite it using only the numbers in FACTS, starting with the tightest limit."}]
    calls.append({"agent": "Explainer", "purpose": "envelope", "attempts": attempts})
    return rules_text(report)


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
                              "map": None, "evidence": [], "runs": 0, "model_calls": []}
    others = f" The other robots stay as they are and only {entry.label} is judged." if len(spec.robots) > 1 else ""
    _emit(on_step, "Finder", "conditions", "Choosing the world conditions to search (never the robot's design).",
          f"Searching {', '.join(KNOBS[k]['name'] for k in chosen)} for where {entry.label} "
          f"{FAILURES[failure]}; budget {budget} (at most {BUDGETS[budget]} runs).{others}"
          + (" " + " ".join(notes) if notes else ""))
    scene = prepare(spec, chosen)
    context = {"base": fits, "robot_id": entry.id, "mass": entry.robot.mass_kg,
               "obstacles": _obstacle_context(spec, entry.robot)}
    runner = Runner(scene, entry.id, failure, BUDGETS[budget])

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
    pair = report["limits"][:2]
    if len(pair) == 2 and runner.left > 0:
        report["map"] = grid_map(runner, pair[0], pair[1], context, on_step)
    report["evidence"], report["runs"] = runner.evidence, len(runner.evidence)
    report["honesty"] = honesty(report)
    report["explanation"] = explain(report, report["model_calls"])
    _emit(on_step, "Explainer", "envelope", "Explaining the limits from the measured envelope only.", report["explanation"])
    return report


def describe(report: dict) -> str:
    """The chat reply: the explanation, then the honesty line."""
    return f"{report['explanation']} {report.get('honesty', '')}".strip()
