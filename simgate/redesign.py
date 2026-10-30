"""Make it pass: change the ROBOT's design (never the world) over a few rounds of parallel runs until it passes,
pick the smallest change that works and check that it holds up. Every run goes through the batch runner."""

import json
import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import config, cost, llm
from .batch import _validate_variation, crash_report, run_batch
from .diagnose import (SYSTEM_PROBLEM, describe_numbers, diagnose, key_numbers, pin_task, result_metrics,
                       robot_metrics)
from .reliability import ci_text, wilson
from .spec import Robot, SceneSpec

MAX_ROUNDS = 3
CANDIDATES_PER_ROUND = 4
ROBUSTNESS_RUNS = 3
MIN_ROBUST_PASSES = 2   # a design is the winner only if it passes at least 2 of the 3 robustness checks
TESTED_GAIT = {"gait_freq_hz": 2.0, "stride_deg": 15.0}
StepCallback = Callable[[dict], None]

# Hard bounds per robot kind. Two-value tuples of ints are choices; floats are ranges.
BOUNDS: dict[str, dict[str, tuple]] = {
    "rover": {"wheel_radius": (0.08, 0.35), "wheels": (4, 6), "drive_speed": (2.0, 12.0),
              "chassis_length": (0.4, 1.2), "chassis_width": (0.3, 0.9), "mass_kg": (3.0, 40.0)},
    "walker": {"legs": (4, 6), "leg_length": (0.15, 0.35), "gait_freq_hz": (1.2, 2.5), "stride_deg": (8.0, 22.0)},
    "library": {"command_speed_m_s": (0.2, 1.0)},
}
CHOICES = {"wheels", "legs"}
UNITS = {"wheel_radius": " m", "drive_speed": " rad/s", "chassis_length": " m", "chassis_width": " m",
         "mass_kg": " kg", "leg_length": " m", "gait_freq_hz": " Hz", "stride_deg": "°", "command_speed_m_s": " m/s"}
NAMES = {"wheel_radius": "wheel radius", "drive_speed": "wheel speed", "chassis_length": "chassis length",
         "chassis_width": "chassis width", "mass_kg": "mass", "leg_length": "leg length",
         "gait_freq_hz": "gait frequency", "stride_deg": "stride", "command_speed_m_s": "walking speed",
         "wheels": "wheels", "legs": "legs", "policy_id": "policy"}
GROUPS = {
    "wheel size": {"wheel_radius"}, "wheels": {"wheel_radius", "wheels"}, "legs": {"legs", "leg_length"},
    "speed": {"drive_speed", "gait_freq_hz", "command_speed_m_s"}, "gait": {"gait_freq_hz", "stride_deg"},
    "chassis": {"chassis_length", "chassis_width"}, "size": {"chassis_length", "chassis_width"},
    "body": {"chassis_length", "chassis_width"}, "mass": {"mass_kg"}, "weight": {"mass_kg"},
    "policy": {"policy_id"},
}
COUNT_WORDS = {"4": 4, "four": 4, "6": 6, "six": 6}


# ── Design space and constraints ─────────────────────────────────────────────

def kind_of(robot: Robot) -> str:
    return robot.kind if robot.kind in BOUNDS else "rover"


class DesignSpace:
    """The knobs and ranges for one robot after the user's constraints, which are never violated."""

    def __init__(self, robot: Robot, constraints: str | None) -> None:
        self.robot = robot
        self.kind = kind_of(robot)
        self.bounds = dict(BOUNDS[self.kind])
        self.policies: list[str] = []
        if self.kind == "library" and robot.robot_id:
            from .library import LibraryRegistry

            self.policies = [card.id for card in LibraryRegistry().compatible_policies(robot.robot_id)]
            if len(self.policies) > 1:
                self.bounds["policy_id"] = tuple(self.policies)
        self.fixed: dict[str, Any] = {}
        self.notes: list[str] = []
        self.text = (constraints or "").strip()
        self._parse(self.text.lower())

    def _parse(self, text: str) -> None:
        number = r"(\d+(?:\.\d+)?)"
        if m := re.search(rf"(?:under|below|at most|no more than|less than|max(?:imum)?(?: of)?|lighter than)\s*{number}\s*kg", text):
            self._narrow("mass_kg", high=float(m.group(1)))
        if m := re.search(rf"(?:over|above|at least|more than|min(?:imum)?(?: of)?|heavier than)\s*{number}\s*kg", text):
            self._narrow("mass_kg", low=float(m.group(1)))
        for count, part in re.findall(r"keep\s+(?:it\s+(?:at|to|on)\s+|the\s+)?(4|four|6|six)\s+(wheels|legs)", text):
            self._fix(part, COUNT_WORDS[count])
        for group in re.findall(r"(?:don'?t|do not|never)\s+change\s+(?:the\s+)?([a-z ]+?)(?:[,.;]|\s+and\b|$)", text):
            for name in self._group(group):
                self._fix(name, getattr(self.robot, name, None))
        for group in re.findall(r"keep\s+(?:the\s+)?([a-z ]+?)\s+(?:the same|as (?:it is|is)|unchanged)", text):
            for name in self._group(group):
                self._fix(name, getattr(self.robot, name, None))
        if m := re.search(r"only\s+(?:change|adjust|touch|modify)\s+(?:the\s+)?([a-z ]+?)(?:[,.;]|\s+and\b|$)", text):
            allowed = self._group(m.group(1))
            if allowed:
                for name in list(self.bounds):
                    if name not in allowed:
                        self._fix(name, getattr(self.robot, name, None), quiet=True)
                self.notes.append(f"Only {m.group(1).strip()} may change.")

    def _group(self, words: str) -> set[str]:
        words = words.strip()
        for key, names in GROUPS.items():
            if key in words:
                return {name for name in names if name in self.bounds}
        return set()

    def _narrow(self, name: str, low: float | None = None, high: float | None = None) -> None:
        if name not in self.bounds:
            return
        lo, hi = self.bounds[name]
        lo, hi = max(lo, low) if low is not None else lo, min(hi, high) if high is not None else hi
        self.bounds[name] = (lo, hi)
        self.notes.append(f"{NAMES[name].capitalize()} between {lo:g} and {hi:g}{UNITS.get(name, '')}.")

    def _fix(self, name: str, value: Any, quiet: bool = False) -> None:
        if name not in self.bounds or value is None:
            return
        self.fixed[name] = value
        if not quiet:
            self.notes.append(f"{NAMES[name].capitalize()} stays at {value:g}{UNITS.get(name, '')}." if isinstance(value, float)
                              else f"{NAMES[name].capitalize()} stays at {value}.")

    @property
    def knobs(self) -> list[str]:
        return [name for name in self.bounds if name not in self.fixed]

    def original(self) -> dict[str, Any]:
        return {name: getattr(self.robot, name) for name in self.bounds}

    def clean(self, changes: dict[str, Any], decisions: list[dict] | None = None, who: str = "Designer") -> tuple[dict, list[str]]:
        """Keep only allowed knobs, clamp to bounds and constraints, drop no-ops. Returns (changes, notes)."""
        kept, notes = {}, []
        for name, value in changes.items():
            if name not in self.bounds:
                notes.append(f"{name} isn't a design knob for this robot")
                continue
            if name in self.fixed:
                notes.append(f"{NAMES[name]} is fixed by your constraints")
                continue
            if name in CHOICES or name == "policy_id":
                options = self.bounds[name]
                if value not in options:
                    notes.append(f"{NAMES[name]} must be one of {', '.join(map(str, options))}")
                    continue
            else:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    notes.append(f"{NAMES[name]} needs a number")
                    continue
                lo, hi = self.bounds[name]
                clamped = min(max(number, lo), hi)
                if clamped != number:
                    notes.append(f"{NAMES[name]} {number:g} clamped to {clamped:g}{UNITS.get(name, '')}")
                value = round(clamped, 3)
            if value != getattr(self.robot, name):
                kept[name] = value
        if decisions is not None and notes:
            decisions.append({"agent": who, "decision": "adjusted", "reason": "; ".join(notes) + "."})
        return kept, notes

    def distance(self, changes: dict[str, Any]) -> float:
        """How big a change is: each numeric knob as a share of its full range; a choice counts 0.5."""
        total = 0.0
        for name, value in changes.items():
            if name in CHOICES or name == "policy_id":
                total += 0.5
            else:
                lo, hi = BOUNDS[self.kind][name]
                total += abs(float(value) - float(getattr(self.robot, name))) / (hi - lo)
        return round(total, 4)

    def at_bound(self, changes: dict[str, Any]) -> list[str]:
        limits = []
        for name, value in changes.items():
            if name in CHOICES or name == "policy_id":
                continue
            lo, hi = self.bounds[name]
            if value >= hi - 1e-9:
                limits.append(f"{NAMES[name]} already at the {hi:g}{UNITS.get(name, '')} maximum")
            elif value <= lo + 1e-9:
                limits.append(f"{NAMES[name]} already at the {lo:g}{UNITS.get(name, '')} minimum")
        return limits


def describe_design(changes: dict[str, Any], original: dict[str, Any]) -> str:
    parts = []
    for name, value in changes.items():
        before = original.get(name)
        unit = UNITS.get(name, "")
        fmt = (lambda v: f"{v:g}{unit}") if isinstance(value, (int, float)) and not isinstance(value, bool) else str
        parts.append(f"{NAMES.get(name, name)} {fmt(before)} → {fmt(value)}")
    return ", ".join(parts) or "no change"


# ── Candidates ───────────────────────────────────────────────────────────────

def _problem(base: dict, diagnosis: dict | None) -> str:
    causes = {h["cause"] for h in (diagnosis or {}).get("hypotheses", []) if h.get("verdict") in ("confirmed", "partly")}
    stop = base.get("stopped_by") or {}
    if stop.get("kind") == "gap":
        return "gap"
    if stop or "obstacle" in causes:
        return "climb"
    if "slipping" in causes or "low gravity" in causes or (base.get("slip_ratio") or 0) > 0.4:
        return "slip"
    if causes & {"steep ground", "side slope", "too fast"} or base.get("flipped"):
        return "tip"
    return "general"


def rule_candidates(space: DesignSpace, problem: str) -> list[dict]:
    """Mock Designer, round 1: the changes the failure points to."""
    r = space.robot
    if space.kind == "rover":
        plans = {
            "climb": [{"wheel_radius": r.wheel_radius * 1.3}, {"wheels": 6},
                      {"wheels": 6, "wheel_radius": r.wheel_radius * 1.3}, {"wheel_radius": r.wheel_radius * 1.6}],
            "gap": [{"wheel_radius": r.wheel_radius * 1.3}, {"wheel_radius": r.wheel_radius * 1.6},
                    {"wheels": 6, "wheel_radius": r.wheel_radius * 1.6}, {"wheel_radius": r.wheel_radius * 2.0}],
            "slip": [{"wheels": 6}, {"drive_speed": r.drive_speed * 0.7}, {"mass_kg": r.mass_kg * 0.7},
                     {"wheels": 6, "drive_speed": r.drive_speed * 0.7}],
            "tip": [{"chassis_width": r.chassis_width * 1.3}, {"drive_speed": r.drive_speed * 0.7},
                    {"chassis_width": r.chassis_width * 1.3, "drive_speed": r.drive_speed * 0.7},
                    {"chassis_width": r.chassis_width * 1.6}],
        }
        plans["general"] = [{"wheel_radius": r.wheel_radius * 1.3}, {"wheels": 6},
                            {"drive_speed": r.drive_speed * 0.7}, {"chassis_width": r.chassis_width * 1.3}]
    elif space.kind == "walker":
        legs = [{"legs": 6}, {"leg_length": r.leg_length * 1.25}, {"legs": 6, "leg_length": r.leg_length * 1.25},
                {"leg_length": r.leg_length * 1.5}]
        plans = {"climb": legs, "gap": legs, "general": legs,
                 "slip": [{"legs": 6}, {"gait_freq_hz": r.gait_freq_hz * 0.8}, {"stride_deg": r.stride_deg * 0.75},
                          {"legs": 6, "gait_freq_hz": r.gait_freq_hz * 0.8}],
                 "tip": [{"legs": 6}, {"stride_deg": r.stride_deg * 0.75}, {"gait_freq_hz": r.gait_freq_hz * 0.8},
                         {"legs": 6, "stride_deg": r.stride_deg * 0.75}]}
    else:
        speed = r.command_speed_m_s
        options = [{"command_speed_m_s": speed * 0.6}, {"command_speed_m_s": speed * 0.4},
                   {"command_speed_m_s": speed * 1.4}]
        options += [{"policy_id": policy} for policy in space.policies if policy != r.policy_id]
        plans = {key: options for key in ("climb", "gap", "slip", "tip", "general")}
    return plans[problem]


def _further(space: DesignSpace, changes: dict, factor: float) -> dict:
    """The same knobs, moved `factor` times as far from the original."""
    out = {}
    for name, value in changes.items():
        if name in CHOICES or name == "policy_id":
            out[name] = value
        else:
            before = float(getattr(space.robot, name))
            out[name] = before + (float(value) - before) * factor
    return out


def rule_strategy(space: DesignSpace, rounds: list[dict], tried: set[str]) -> tuple[str, str, list[dict]]:
    """Mock Designer between rounds: decide from how close the best run got."""
    scored = [c for rnd in rounds for c in rnd["candidates"] if c.get("closeness") is not None]
    if not scored:
        return "stop", "No candidate could run, so more rounds can't help.", []
    best = max(scored, key=lambda c: c["closeness"])
    previous_best = max((c["closeness"] for rnd in rounds[:-1] for c in rnd["candidates"]
                         if c.get("closeness") is not None), default=-1.0)
    limits = space.at_bound(best["changes"])
    used = {name for c in scored for name in c["changes"]}
    unused = [name for name in space.knobs if name not in used]
    if limits or (len(rounds) > 1 and best["closeness"] <= previous_best + 0.05):
        why = limits[0] if limits else f"{describe_design(best['changes'], space.original())} isn't helping more"
        fresh = []
        for name in unused + [n for n in space.knobs if n in used]:
            for change in _fresh_change(space, name):
                merged = {**{k: v for k, v in best["changes"].items() if k not in limits}, **change}
                fresh += [change, merged]
        return "try a different knob", f"{why}; switching to {NAMES.get(unused[0], unused[0]) if unused else 'other knobs'}.", fresh
    if best["closeness"] >= 0.6:
        return "refine", f"{describe_design(best['changes'], space.original())} nearly passed; pushing it a little further.", [
            _further(space, best["changes"], 1.3), _further(space, best["changes"], 1.6),
            _further(space, best["changes"], 2.0), _further(space, best["changes"], 2.5)]
    pushed: dict[str, float] = {}   # every numeric knob tried so far, at its limit in the direction it was moved
    for candidate in sorted(scored, key=lambda c: -c["closeness"]):
        for name, value in candidate["changes"].items():
            if name not in CHOICES and name != "policy_id" and name not in pushed:
                pushed[name] = space.bounds[name][1] if float(value) > float(getattr(space.robot, name)) else space.bounds[name][0]
    at_limit = [{name: value} for name, value in pushed.items()]
    choices = {name: value for name, value in best["changes"].items() if name in CHOICES or name == "policy_id"}
    return "bigger change", "Nothing got close, so trying bigger changes, up to the limits.", [
        {**choices, **change} for change in at_limit] + at_limit + [
        _further(space, c["changes"], 2.0) for c in sorted(scored, key=lambda c: -c["closeness"])[:CANDIDATES_PER_ROUND]]


def _fresh_change(space: DesignSpace, name: str) -> list[dict]:
    if name in CHOICES:
        return [{name: 6 if getattr(space.robot, name) == 4 else 4}]
    if name == "policy_id":
        return [{name: policy} for policy in space.policies if policy != space.robot.policy_id]
    lo, hi = space.bounds[name]
    before = float(getattr(space.robot, name))
    lower = name in ("drive_speed", "mass_kg", "gait_freq_hz", "stride_deg")    # slower and lighter usually help
    return [{name: lo + (before - lo) * 0.5}] if lower else [{name: before + (hi - before) * 0.5}, {name: hi}]


class CandidateIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    changes: dict[str, Any]
    why: str = Field(default="", max_length=300)


class DesignReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategy: Literal["bigger change", "refine", "try a different knob", "stop"]
    reason: str = Field(min_length=1, max_length=400)
    candidates: list[CandidateIn] = Field(default_factory=list, max_length=8)


DESIGNER_SYSTEM = """You are the Designer. Change ONLY the robot's design so it passes the same world (the world,
course, start and goal never change). Before each round choose a strategy and say why:
- "bigger change": nothing is close yet
- "refine": something nearly passed
- "try a different knob": the current knob is at its bound or isn't helping
- "stop": a robust winner exists or more rounds can't help (no candidates then)
"refine" moves further in a direction that helped; it doesn't undo a change that helped. Propose up to {count}
candidates; each changes one or a few knobs. Allowed knobs and ranges (never outside them; fixed knobs must not
appear): {bounds}
What the knobs do (measured on this simulator): {effects}
The user message gives the robot's measured rough limits ("robot_limits"); compare them with what stopped it.
"facts" describes the terrain and tilt in plain words: in your reasons, describe them only with those words.
Prefer the smallest change that could work, and don't repeat a design already tried. Reply with JSON only, in this shape (the values are placeholders
showing the format, not suggestions):
{{"strategy":"<one of the four>","reason":"<why>","candidates":[{{"changes":{{"<knob>":<value>}},"why":"<reason>"}}]}}"""


KNOB_EFFECTS = {
    "wheel_radius": "bigger wheels climb taller steps and cross wider gaps (a rover climbs about 0.6 x wheel radius "
                    "with 4 wheels, 0.9 x with 6) and cross gaps up to about 1.6 x wheel radius",
    "wheels": "6 wheels let the rear wheels push the front ones up a step, and spread the load for grip",
    "drive_speed": "slower wheels slip and tip less on steep or loose ground; faster ones reach the goal sooner",
    "chassis_length": "a longer chassis bridges dips but can catch on a step's edge",
    "chassis_width": "a wider chassis resists tipping on side slopes",
    "mass_kg": "a lighter robot needs less grip to climb; a heavier one presses wheels down harder",
    "legs": "6 legs (tripod gait) are steadier and step a little higher than 4",
    "leg_length": "longer legs step over taller obstacles",
    "gait_freq_hz": "a slower gait slips and tips less (tuned at 2.0 Hz)",
    "stride_deg": "a shorter stride is steadier, a longer one covers more ground (tuned at 15°)",
    "command_speed_m_s": "the walking speed asked of the trained policy; slower is usually steadier",
    "policy_id": "another trained policy that fits this exact robot",
}


def model_round(space: DesignSpace, context: dict, decisions: list[dict], calls: list[dict]) -> DesignReply | None:
    bounds = {name: (list(space.bounds[name]) if name not in space.fixed else f"fixed at {space.fixed[name]}")
              for name in space.bounds}
    effects = json.dumps({name: KNOB_EFFECTS[name] for name in space.bounds if name in KNOB_EFFECTS})
    messages = [
        {"role": "system", "content": DESIGNER_SYSTEM.format(count=CANDIDATES_PER_ROUND, bounds=json.dumps(bounds),
                                                             effects=effects)},
        {"role": "user", "content": json.dumps(context, default=str)},
    ]
    reply, attempts = llm.chat_json(messages, config.BUILDER_MODEL, DesignReply.model_validate)
    calls.append({"agent": "Designer", "purpose": f"round {context.get('round')}", "attempts": attempts})
    return reply


# ── The search ───────────────────────────────────────────────────────────────

def _emit(on_step: StepCallback | None, agent: str, tool: str, thought: str, result: str, status: str = "done") -> None:
    if on_step is not None:
        on_step({"agent": agent, "tool": tool, "thought": thought, "result": result, "status": status})


def closeness(base: dict, after: dict) -> float | None:
    """0 = no better than the original failure, 1 = passed."""
    if after.get("error"):
        return None
    if after["passed"]:
        return 1.0
    if base.get("final_distance_m") and after.get("final_distance_m") is not None:
        score = 1 - after["final_distance_m"] / base["final_distance_m"]
    else:
        before, now = base.get("forward_progress_m") or 0.0, after.get("forward_progress_m") or 0.0
        score = (now - before) / max(1.0 - before, 1.0)
    if after.get("flipped"):
        score -= 0.2
    return round(max(min(score, 0.99), -1.0), 3)


def _variation(changes: dict, robot_id: str | None, label: str) -> dict:
    variation = {"label": label, "robot": changes}
    if robot_id is not None:
        variation["robot_id"] = robot_id
    return variation


def _world(spec: SceneSpec, robot_id: str | None = None) -> dict:
    """Everything that must not change: environment, terrain, obstacles, starts, goals, conditions and, in a race,
    every other robot's design."""
    return {
        "other_robots": [e.robot.model_dump() for e in spec.robots if robot_id is not None and e.id != robot_id],
        "environment": spec.environment.model_dump(), "terrain": spec.terrain.model_dump() if spec.terrain else None,
        "obstacles": [o.model_dump() for o in spec.obstacles], "disturbance": spec.disturbance,
        "duration_s": spec.duration_s,
        "places": [(e.start_xy, e.start_place, e.start_heading_deg, e.goal_xy, e.goal_place) for e in spec.robots],
    }


def robustness(spec: SceneSpec, changes: dict, robot_id: str | None) -> dict:
    """The winner on 3 nearby conditions inside the same world: two other terrain/obstacle seeds, 10% less grip."""
    _, designed, _ = _validate_variation(spec, {"robot": changes, **({"robot_id": robot_id} if robot_id else {})})
    seed = designed.terrain.seed if designed.terrain else 0
    friction = designed.environment.friction or 0.8
    checks = [{"label": f"seed {seed + 1}", "seed": seed + 1}, {"label": f"seed {seed + 2}", "seed": seed + 2},
              {"label": "friction -10%", "friction": round(friction * 0.9, 3)}]
    results = run_batch(designed, checks, baseline=False)
    outcomes = [(check["label"], result_metrics(result, robot_id)) for check, result in zip(checks, results)
                if not result.get("error")]   # a crashed check is a system problem, not a robustness result
    failed = [label for label, metrics in outcomes if not metrics["passed"]]
    low, high = wilson(len(outcomes) - len(failed), len(outcomes))
    return {"passed": len(outcomes) - len(failed), "total": len(outcomes), "failed": failed,
            "ci95": [round(low, 3), round(high, 3)],
            **({"crashed": crash_report(results)} if crash_report(results) else {})}


def redesign(spec: SceneSpec, summary: dict, constraints: str | None = None, diagnosis: dict | None = None,
             robot_id: str | None = None, on_step: StepCallback | None = None) -> dict:
    """Search robot designs (never the world) for the smallest change that makes the failed run pass, and report
    what it cost."""
    with cost.track() as spent:
        report = _redesign(spec, summary, constraints, diagnosis, robot_id, on_step)
        report["cost"] = spent.snapshot()
    return report


def _redesign(spec: SceneSpec, summary: dict, constraints: str | None, diagnosis: dict | None,
              robot_id: str | None, on_step: StepCallback | None) -> dict:
    original_metrics = robot_metrics(summary, robot_id)
    robot_id = original_metrics["robot_id"]
    entry = next((e for e in spec.robots if e.id == robot_id), spec.robots[0] if spec.robots else None)
    robot = entry.robot if entry is not None else spec.robot
    space = DesignSpace(robot, constraints)
    report: dict[str, Any] = {
        "type": "redesign", "robot_id": robot_id, "kind": space.kind, "original": space.original(),
        "constraints": {"text": space.text, "notes": space.notes, "fixed": space.fixed,
                        "bounds": {name: list(space.bounds[name]) for name in space.bounds}},
        "winner": None, "rounds": [], "decisions": [], "model_calls": [], "tried_count": 0, "robustness_runs": 0,
        "notes": [],
    }
    if space.kind == "library":
        report["notes"].append("Only the command speed and the choice of compatible policy can change: a library "
                               "robot's controller expects its exact body, so its body is never edited.")
    if original_metrics["passed"]:
        report["message"] = (f"{original_metrics['label']} already passes, so there is nothing to redesign."
                             if original_metrics.get("label") else "This run already passes, so there is nothing to redesign.")
        return report
    if not space.knobs:
        report["message"] = "Your constraints leave no design knob free to change."
        return report

    if diagnosis is None:
        diagnosis = diagnose(spec, summary, robot_id, on_step)
    report["diagnosis"] = {k: diagnosis.get(k) for k in ("reproduced", "conclusion", "hypotheses")}
    if diagnosis.get("reproduced") is False:
        report["message"] = "The failure didn't reproduce, so there is nothing to fix."
        return report
    world = pin_task(spec, summary)
    from .facts import facts, facts_block

    facts_text = facts_block(facts(world, summary))   # plain-language terrain and tilt labels, built in code
    base = diagnosis.get("reproduced_metrics") or original_metrics
    report["before"] = (f"{base.get('outcome') or 'failed'}" + (f", {base['final_distance_m']:.1f} m from the goal"
                        if base.get("final_distance_m") is not None else ""))
    problem = _problem(base, diagnosis)
    tried: set[str] = set()
    passing: list[dict] = []
    limits = None
    if space.kind != "library":
        from .backends.obstacles import capability

        probe = world.model_copy(deep=True)
        probe.robot = robot.model_copy(deep=True)
        limits = capability(probe)

    for number in range(1, MAX_ROUNDS + 1):
        context = {
            "round": number, "original_design": space.original(), "fixed_by_user": space.fixed,
            "facts": facts_text, "robot_limits": limits,
            "failure": {k: base.get(k) for k in ("outcome", "final_distance_m", "forward_progress_m", "flipped",
                                                   "slip_ratio", "stopped_by")},
            "diagnosis": [{"cause": h["cause"], "verdict": h.get("verdict")} for h in diagnosis.get("hypotheses", [])],
            "previous_rounds": [{"strategy": r["strategy"], "results": [
                {"changes": c["changes"], "passed": c["passed"], "result": c["result"]} for c in r["candidates"]]}
                for r in report["rounds"]],
        }
        def rules_round() -> tuple[str, str, list]:
            if number == 1:
                return ("aim at the diagnosis", f"The failure points to {problem if problem != 'general' else 'no single cause'}.",
                        [(c, "") for c in rule_candidates(space, problem)])
            strategy, reason, proposals = rule_strategy(space, report["rounds"], tried)
            return strategy, reason, [(c, "") for c in proposals]

        def take(raw: list) -> list[dict]:
            fresh = []
            for changes, why in raw:
                cleaned, _ = space.clean(changes, report["decisions"])
                key = json.dumps(cleaned, sort_keys=True)
                if cleaned and key not in tried:
                    tried.add(key)
                    fresh.append({"changes": cleaned, "why": why})
                if len(fresh) == CANDIDATES_PER_ROUND:
                    break
            return fresh

        recorded = len(report["decisions"])
        reply = None if config.MOCK_MODE else model_round(space, context, report["decisions"], report["model_calls"])
        if reply is not None:
            strategy, reason, raw = reply.strategy, reply.reason, [(c.changes, c.why) for c in reply.candidates]
        else:
            strategy, reason, raw = rules_round()
            if not config.MOCK_MODE:
                report["decisions"].append({"agent": "Designer", "decision": "fallback",
                                            "reason": "The Designer's reply didn't validate twice; using the design rules."})
        candidates = take(raw)
        if reply is not None and strategy != "stop" and not candidates:   # all repeats or out of bounds
            strategy, reason, raw = rules_round()
            report["decisions"].append({"agent": "Designer", "decision": "fallback",
                                        "reason": "The Designer's candidates were all tried already or not allowed; "
                                                  "using the design rules for this round."})
            candidates = take(raw)
        for note in report["decisions"][recorded:]:
            _emit(on_step, "Designer", "candidates", "Checking the proposed designs against the bounds.",
                  f"{note['decision'].capitalize()}: {note['reason']}", "warning")
        if strategy == "stop" or not candidates:
            reason = reason if strategy == "stop" else f"{reason} No new design within the allowed ranges is left to try."
            report["decisions"].append({"agent": "Designer", "decision": "stop", "reason": reason})
            _emit(on_step, "Designer", "strategy", "Choosing how to continue.", f"Stop: {reason}")
            break
        report["decisions"].append({"agent": "Designer", "decision": strategy, "reason": reason})
        _emit(on_step, "Designer", "strategy", f"Round {number}: {strategy}.",
              f"{reason} Trying: " + "; ".join(describe_design(c["changes"], space.original()) for c in candidates))

        _emit(on_step, "Batch runner", "designs", f"Running {len(candidates)} designs in parallel.", "", "working")
        results = run_batch(world, [_variation(c["changes"], robot_id, f"Design {i}")
                                    for i, c in enumerate(candidates, 1)], baseline=False)
        report["tried_count"] += len(results)
        for candidate, result in zip(candidates, results):
            after = result_metrics(result, robot_id)
            candidate["passed"] = after["passed"]
            candidate["closeness"] = closeness(base, after)
            candidate["numbers"] = key_numbers(base, after)
            candidate["result"] = "crashed (not a physics result)" if result.get("error") else describe_numbers(candidate["numbers"])
            candidate["distance"] = space.distance(candidate["changes"])
            if space.kind == "walker" and any(candidate["changes"].get(k, TESTED_GAIT[k]) != TESTED_GAIT[k] for k in TESTED_GAIT):
                candidate["label"] = "untested gait"
            if after["passed"]:
                passing.append(candidate)
        _emit(on_step, "Batch runner", "designs", "Design runs.",
              "; ".join(f"{describe_design(c['changes'], space.original())}: "
                        f"{'passed' if c['passed'] else c['result']}" for c in candidates))
        unchanged = all(_world(_validate_variation(world, _variation(c["changes"], robot_id, "check"))[1], robot_id)
                        == _world(world, robot_id) for c in candidates)
        _emit(on_step, "Checker", "designs", "Checking every design kept the world and stayed in bounds.",
              ("World unchanged; every change within the allowed ranges." if unchanged
               else "A design changed the world; its result is not used.") +
              f" {sum(c['passed'] for c in candidates)} of {len(candidates)} passed.",
              "done" if unchanged else "warning")
        if not unchanged:
            passing = [c for c in passing if c not in candidates]
        report["rounds"].append({"round": number, "strategy": strategy, "reason": reason, "candidates": candidates})
        if (crashed := crash_report(results)):   # a system problem: crashed designs are left out of the search
            report["crashed"] = crashed
            _emit(on_step, "Batch runner", "crashed", "Checking every design run completed.",
                  f"{crashed}. {SYSTEM_PROBLEM}; crashed designs are left out.", "warning")
            if not any(c.get("closeness") is not None for c in candidates):
                report["decisions"].append({"agent": "Designer", "decision": "stop",
                                            "reason": "Every design run crashed, so the search stopped."})
                break

        if passing:
            winner = _pick_winner(world, space, passing, robot_id, report, on_step)
            if winner is not None:
                report["winner"] = winner
                report["decisions"].append({"agent": "Designer", "decision": "stop",
                                            "reason": "A passing design held up in the robustness checks."})
                _emit(on_step, "Designer", "strategy", "Choosing how to continue.",
                      "Stop: a passing design held up in the robustness checks.")
                break

    if report["winner"] is None and passing:   # passed once but not reliably: shown as fragile, never the winner
        checked_designs = [c for c in passing if c.get("robustness")] or passing
        best = max(checked_designs, key=lambda c: ((c.get("robustness") or {}).get("passed", 0), -c["distance"]))
        report["fragile"] = {**_winner_fields(best, space), "robustness": best.get("robustness"), "fragile": True}
    report["message"] = _message(report, space)
    return report


def _winner_fields(candidate: dict, space: DesignSpace) -> dict:
    original = space.original()
    rows = [{"knob": NAMES.get(name, name), "before": _shown(name, original.get(name)), "after": _shown(name, value)}
            for name, value in candidate["changes"].items()]
    return {"changes": candidate["changes"], "summary": candidate["result"], "numbers": candidate["numbers"],
            "distance": candidate["distance"], "label": candidate.get("label"), "rows": rows,
            "description": describe_design(candidate["changes"], original)}


def _shown(name: str, value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}{UNITS.get(name, '')}"
    return str(value)


def _pick_winner(world: SceneSpec, space: DesignSpace, passing: list[dict], robot_id: str | None,
                 report: dict, on_step: StepCallback | None) -> dict | None:
    """The smallest passing change that passes at least 2 of 3 robustness checks; when one doesn't, the next-best
    passing design is tried."""
    ranked = sorted(passing, key=lambda c: (c["distance"], len(c["changes"])))
    unchecked = [candidate for candidate in ranked if candidate.get("robustness") is None]
    for attempt, candidate in enumerate(unchecked):
        _emit(on_step, "Batch runner", "robustness", "Checking the smallest passing design on 3 nearby conditions.", "", "working")
        candidate["robustness"] = robustness(world, candidate["changes"], robot_id)
        report["robustness_runs"] += candidate["robustness"]["total"]
        held = candidate["robustness"]["passed"] >= MIN_ROBUST_PASSES
        _emit(on_step, "Batch runner", "robustness", "Robustness checks.",
              f"{describe_design(candidate['changes'], space.original())}: "
              f"{ci_text(candidate['robustness']['passed'], candidate['robustness']['total'])}"
              + (f" (failed: {', '.join(candidate['robustness']['failed'])})" if candidate["robustness"]["failed"] else "")
              + (f"; {candidate['robustness']['crashed']} (not counted)" if candidate["robustness"].get("crashed") else ""),
              "done" if held else "warning")
        if held:
            return {**_winner_fields(candidate, space), "robustness": candidate["robustness"]}
        if attempt + 1 < len(unchecked):
            report["decisions"].append({"agent": "Designer", "decision": "next-best",
                                        "reason": f"{describe_design(candidate['changes'], space.original())} passed only "
                                                  f"{candidate['robustness']['passed']}/{candidate['robustness']['total']} "
                                                  "robustness checks; trying the next-best passing design."})
    return None


def _message(report: dict, space: DesignSpace) -> str:
    winner = report["winner"]
    tried = report["tried_count"]
    if winner is not None:
        robust = winner.get("robustness") or {}
        held = f" Robustness: {ci_text(robust['passed'], robust['total'])}" + (
            f" (failed: {', '.join(robust['failed'])})." if robust.get("failed") else ".") if robust else ""
        held += f" {robust['crashed']}, not counted." if robust.get("crashed") else ""
        gait = " Untested gait: outside the tuned 2.0 Hz / 15° settings." if winner.get("label") == "untested gait" else ""
        return (f"Smallest change that passes: {winner['description']}.{held}{gait} Tried {tried} designs. "
                "Say \"use this design\" to apply it.")
    if (fragile := report.get("fragile")) is not None:
        robust = fragile.get("robustness") or {}
        score = f" ({robust['passed']}/{robust['total']})" if robust else ""
        return (f"Found a design that passed once but isn't reliable{score}: {fragile['description']}. It's shown as "
                f"fragile, not as a winner. Tried {tried} designs. Say \"use fragile design\" to apply it anyway.")
    scored = [c for r in report["rounds"] for c in r["candidates"] if c.get("closeness") is not None]
    if not scored:
        if report.get("crashed"):
            return f"{report['crashed']}. {SYSTEM_PROBLEM}, so no design could be judged."
        return report.get("message") or f"No design passed in {tried} runs."
    best = max(scored, key=lambda c: c["closeness"])
    limits = [limit for c in sorted(scored, key=lambda c: -c["closeness"]) for limit in space.at_bound(c["changes"])]
    report["best_attempt"] = {"changes": best["changes"], "summary": best["result"],
                              "description": describe_design(best["changes"], space.original())}
    report["limit"] = limits[0] if limits else "no design within the allowed ranges got it through"
    return (f"No design passed in {tried} runs. Best attempt: {report['best_attempt']['description']} "
            f"({best['result']}). Limit: {report['limit']}.")
