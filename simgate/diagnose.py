"""Prove why a run failed: reproduce it, test one-change hypotheses in parallel, and report what the physics
confirmed or ruled out. Every rerun goes through the batch runner."""

import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from . import config, llm
from .batch import _validate_variation, run_batch
from .spec import SceneSpec

MAX_RERUNS = 5          # reproduction + every experiment, in total
MAX_HYPOTHESES = 4
MAX_DECISION_ROUNDS = 2
MAX_FOLLOW_UPS = 2
StepCallback = Callable[[dict], None]

# One change per experiment, using batch knobs only. Values are the allowed mappings, not suggestions.
ALLOWED_KNOBS = {
    "surface": "slipping: a grippier surface, normally \"rock\"",
    "friction": "slipping: friction 50% higher than now",
    "gravity": "low gravity: Earth gravity, 9.81",
    "slope_deg": "steep ground on slope terrain: 10° less than now",
    "flatten_terrain": "steep ground on hills, rocky or crater terrain: true (same scene on flat ground)",
    "remove_obstacle": "an obstacle: its name from the evidence, e.g. obs1_step",
    "obstacle_scale": "an obstacle: 0.5 to halve every obstacle's height",
    "no_push": "a push: true to remove it",
    "speed_scale": "too fast: 0.6",
    "face_uphill": "tipping on a side slope: true (slope terrain)",
}


# ── Evidence and verdicts ────────────────────────────────────────────────────

def robot_metrics(summary: dict, robot_id: str | None = None) -> dict:
    """Pass/fail and the key numbers for one robot, from a single- or multi-robot run summary."""
    robots = summary.get("robots") if isinstance(summary.get("robots"), dict) else None
    if robots:
        robot_id = robot_id if robot_id in robots else next(iter(robots))
        m = robots[robot_id]
        stopped_by = (m.get("obstacles") or {}).get("stopped_by")
    else:
        m = summary
        stopped_by = m.get("stopped_by")
    has_goal = m.get("goal_reached") is not None and m.get("final_distance_m") is not None
    passed = bool(m.get("goal_reached")) if has_goal else (
        not m.get("flipped", False) and (m.get("forward_progress_m") or 0) >= 1.0)
    return {
        "robot_id": robot_id,
        "passed": passed,
        "has_goal": has_goal,
        "final_distance_m": m.get("final_distance_m"),
        "forward_progress_m": m.get("forward_progress_m"),
        "flipped": bool(m.get("flipped")),
        "slip_ratio": m.get("slip_ratio"),
        "max_tilt_deg": m.get("max_tilt_deg"),
        "local_slope_deg": m.get("local_slope_deg"),
        "cross_slope_deg": m.get("cross_slope_deg"),
        "outcome": m.get("outcome"),
        "stopped_by": stopped_by,
    }


def result_metrics(result: dict, robot_id: str | None) -> dict:
    if result.get("error"):
        return {"passed": False, "error": result["error"]}
    return robot_metrics(result.get("summary") or {}, robot_id)


def verdict(base: dict, after: dict) -> str:
    """"confirmed" passes with only this change; "partly" clearly improves the key metric; else "ruled out"."""
    if after.get("error"):
        return "not tested"
    if after["passed"]:
        return "confirmed"
    closer = (base.get("final_distance_m") or 0) > 0 and after.get("final_distance_m") is not None \
        and after["final_distance_m"] <= 0.7 * base["final_distance_m"]
    fall_avoided = base.get("flipped") and not after.get("flipped")
    less_slip = (base.get("slip_ratio") or 0) > 0.05 and after.get("slip_ratio") is not None \
        and after["slip_ratio"] <= 0.5 * base["slip_ratio"]
    return "partly" if closer or fall_avoided or less_slip else "ruled out"


def key_numbers(base: dict, after: dict) -> dict:
    if after.get("error"):
        return {"error": after["error"][:200]}
    numbers: dict[str, Any] = {"passed": [base["passed"], after["passed"]]}
    if base.get("final_distance_m") is not None:
        numbers["distance_left_m"] = [base["final_distance_m"], after.get("final_distance_m")]
    elif base.get("forward_progress_m") is not None:
        numbers["progress_m"] = [base["forward_progress_m"], after.get("forward_progress_m")]
    if base.get("flipped") or after.get("flipped"):
        numbers["fell"] = [base["flipped"], after.get("flipped")]
    if base.get("slip_ratio") is not None:
        numbers["slip"] = [base["slip_ratio"], after.get("slip_ratio")]
    return numbers


def describe_numbers(numbers: dict) -> str:
    if "error" in numbers:
        return f"run failed: {numbers['error']}"
    parts = []
    if "distance_left_m" in numbers:
        before, after = numbers["distance_left_m"]
        parts.append(f"distance left {before:.1f} → {after:.1f} m" if after is not None else f"distance left {before:.1f} m")
    if "progress_m" in numbers:
        before, after = numbers["progress_m"]
        parts.append(f"progress {before:.1f} → {after:.1f} m" if after is not None else f"progress {before:.1f} m")
    if "fell" in numbers:
        parts.append("fall avoided" if numbers["fell"][0] and not numbers["fell"][1] else
                     "still fell" if numbers["fell"][1] else "no fall")
    if "slip" in numbers and numbers["slip"][1] is not None and numbers["slip"][0] is not None:
        parts.append(f"slip {numbers['slip'][0]:.0%} → {numbers['slip'][1]:.0%}")
    return ", ".join(parts)


def describe_change(intervention: dict) -> str:
    parts = [_describe_one(knob, value) for knob, value in intervention.items() if knob != "robot_id"]
    return " + ".join(parts)


def _describe_one(knob: str, value: Any) -> str:
    return {
        "surface": f"surface → {value}",
        "friction": f"friction → {value:g}" if isinstance(value, (int, float)) else f"friction → {value}",
        "gravity": f"gravity → {value:g} m/s²" if isinstance(value, (int, float)) else f"gravity → {value}",
        "slope_deg": f"slope → {value:g}°" if isinstance(value, (int, float)) else f"slope → {value}",
        "flatten_terrain": "same scene on flat ground",
        "remove_obstacle": f"without {value}",
        "obstacle_scale": f"obstacles × {value:g}" if isinstance(value, (int, float)) else f"obstacles × {value}",
        "no_push": "without the push",
        "speed_scale": f"speed × {value:g}" if isinstance(value, (int, float)) else f"speed × {value}",
        "face_uphill": "facing straight uphill",
    }.get(knob, f"{knob} → {value}")


def pin_task(spec: SceneSpec, summary: dict) -> SceneSpec:
    """The scene with the original run's resolved goals and duration written in, so an experiment that removes
    an obstacle (and with it a course's automatic finish line) still asks the same task of the robot."""
    pinned = spec.model_copy(deep=True)
    robots = summary.get("robots") if isinstance(summary.get("robots"), dict) else None
    for entry in pinned.robots:
        goal = (robots or {}).get(entry.id, {}).get("goal_xy") if robots else summary.get("goal_xy")
        if goal is not None and entry.goal_xy is None and entry.goal_place is None:
            entry.goal_xy = list(goal)
    if not robots and pinned.robots and pinned.robots[0].goal_xy is not None:
        mission = pinned.mission
        if mission is None or (mission.goal_xy is None and mission.goal_place is None):
            from .spec import Mission

            first = pinned.robots[0]
            pinned.mission = (mission or Mission()).model_copy(update={
                "goal_xy": first.goal_xy, "start_xy": first.start_xy, "start_place": first.start_place,
                "start_heading_deg": first.start_heading_deg})
    if summary.get("duration_s"):
        pinned.duration_s = float(summary["duration_s"])
    pinned.assumptions = []
    return pinned.fill_defaults()   # the batch runner fills defaults again; ending here keeps the scene stable


# ── Hypotheses ───────────────────────────────────────────────────────────────

class HypothesisIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cause: str = Field(min_length=1, max_length=80)
    why: str = Field(min_length=1, max_length=400)
    intervention: dict[str, Any]
    control: bool = False


class HypothesesReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hypotheses: list[HypothesisIn] = Field(min_length=1, max_length=8)


def _applicable(spec: SceneSpec, intervention: dict) -> str | None:
    """None if the batch runner accepts this single change for the scene, otherwise the reason it doesn't."""
    try:
        _, changed, _ = _validate_variation(spec, intervention)
    except Exception as exc:  # noqa: BLE001 - any rejection means the experiment can't run
        return str(exc).splitlines()[0][:200]
    if changed.model_dump(exclude={"assumptions"}) == spec.model_dump(exclude={"assumptions"}):
        return "it would not change anything in this scene"
    return None


def _slipping(spec: SceneSpec) -> dict:
    if spec.environment.surface != "rock":
        return {"surface": "rock"}
    return {"friction": round((spec.environment.friction or 0.8) * 1.5, 2)}


def rule_hypotheses(spec: SceneSpec, m: dict) -> list[dict]:
    """Mock Explainer: causes the measured evidence points to, plus at least one control it doesn't."""
    env, terrain = spec.environment, spec.terrain
    found: list[dict] = []
    if (m.get("slip_ratio") or 0) > 0.4:
        found.append({"cause": "slipping", "why": f"wheels or feet slid {m['slip_ratio']:.0%} of the contact time",
                      "intervention": _slipping(spec)})
    if m.get("stopped_by"):
        stop = m["stopped_by"]
        found.append({"cause": "obstacle",
                      "why": f"stopped at the {stop.get('size', '')} {stop.get('kind', 'obstacle')}"
                             f" ({stop.get('height_m', '?')} m)".replace("  ", " "),
                      "intervention": {"remove_obstacle": stop["name"]}})
    if m.get("flipped") and (m.get("local_slope_deg") or 0) > 12:
        steeper = {"slope_deg": max(0.0, terrain.slope_deg - 10)} if terrain and terrain.kind == "slope" else {"flatten_terrain": True}
        found.append({"cause": "steep ground", "why": f"fell where the ground sloped {m['local_slope_deg']:.0f}°",
                      "intervention": steeper})
    if (env.gravity or 9.81) < 5:
        found.append({"cause": "low gravity", "why": f"gravity is only {env.gravity:g} m/s², so there is less grip",
                      "intervention": {"gravity": 9.81}})
    if abs(m.get("cross_slope_deg") or 0) > 10 and (m.get("flipped") or (m.get("max_tilt_deg") or 0) > 25):
        found.append({"cause": "side slope", "why": f"tilted to {m.get('max_tilt_deg', 0):.0f}° with "
                                                     f"{abs(m['cross_slope_deg']):.0f}° of sideways slope",
                      "intervention": {"face_uphill": True}})
    if spec.disturbance is not None:
        found.append({"cause": "push", "why": f"a {spec.disturbance.push_n:g} N push hit it mid-run",
                      "intervention": {"no_push": True}})
    found = [h for h in found if _applicable(spec, h["intervention"]) is None]
    named = {h["cause"] for h in found}
    controls = []
    if "slipping" not in named:
        controls.append({"cause": "slipping", "why": f"slip was only {m['slip_ratio']:.0%}" if m.get("slip_ratio") is not None
                         else "no slip evidence", "intervention": _slipping(spec)})
    if "low gravity" not in named and (env.gravity or 9.81) < 9.8:
        controls.append({"cause": "low gravity", "why": f"gravity {env.gravity:g} m/s² is not unusually low",
                         "intervention": {"gravity": 9.81}})
    controls.append({"cause": "too fast", "why": "nothing measured points to speed", "intervention": {"speed_scale": 0.6}})
    controls = [c for c in controls if _applicable(spec, c["intervention"]) is None]
    # Two evidence-backed causes and a control leave room in the budget for one follow-up experiment.
    hypotheses = found[:2]
    for control in controls:
        if any(h.get("control") for h in hypotheses) and len(hypotheses) >= 2:
            break
        hypotheses.append({**control, "control": True})
    return hypotheses


HYPOTHESES_SYSTEM = """You are the Explainer in a physics sandbox. A robot's run failed. Propose 2 to 4 candidate
causes. Each one is tested by rerunning the same scene with exactly ONE change, so each intervention has exactly
one key from this list (the right side says what the change means; never invent other keys):
{knobs}
"why" must cite a measured number from the evidence. Include at least one control: a cause the evidence
suggests is NOT responsible, marked "control": true, so the experiments can rule something out.
Reply with JSON only, in this shape (the values below are placeholders showing the format, not suggestions):
{{"hypotheses":[{{"cause":"<short name>","why":"<evidence with its number>","intervention":{{"<knob>":<value>}},"control":false}}]}}"""


def _evidence(spec: SceneSpec, m: dict) -> str:
    return json.dumps({
        "robot": m,
        "world": {"body": spec.environment.body, "gravity": spec.environment.gravity,
                  "surface": spec.environment.surface, "friction": spec.environment.friction,
                  "terrain": spec.terrain.model_dump(include={"kind", "slope_deg", "roughness"}) if spec.terrain else None,
                  "push_n": spec.disturbance.push_n if spec.disturbance else None},
        "obstacles": [o.model_dump(exclude_none=True) for o in spec.obstacles],
    }, default=str)


def model_hypotheses(spec: SceneSpec, m: dict, decisions: list[dict], calls: list[dict]) -> list[dict] | None:
    knobs = "\n".join(f"- {name}: {meaning}" for name, meaning in ALLOWED_KNOBS.items())
    messages = [
        {"role": "system", "content": HYPOTHESES_SYSTEM.format(knobs=knobs)},
        {"role": "user", "content": f"Evidence from the failed run:\n{_evidence(spec, m)}"},
    ]
    reply, attempts = llm.chat_json(messages, config.PLANNER_MODEL, HypothesesReply.model_validate)
    calls.append({"agent": "Explainer", "purpose": "hypotheses", "attempts": attempts})
    if reply is None:
        return None
    kept = []
    for item in reply.hypotheses:
        allowed = {key: value for key, value in item.intervention.items() if key in ALLOWED_KNOBS}
        dropped = sorted(set(item.intervention) - set(allowed))
        if dropped:
            decisions.append({"agent": "Explainer", "decision": "dropped",
                              "reason": f"{item.cause}: {', '.join(dropped)} is not an allowed experiment."})
        if len(allowed) > 1:
            first = next(iter(allowed))
            decisions.append({"agent": "Explainer", "decision": "dropped",
                              "reason": f"{item.cause}: one change per experiment, kept only {first}."})
            allowed = {first: allowed[first]}
        if not allowed:
            continue
        problem = _applicable(spec, allowed)
        if problem:
            decisions.append({"agent": "Explainer", "decision": "dropped", "reason": f"{item.cause}: {problem}."})
            continue
        if any(h["intervention"] == allowed for h in kept):
            continue
        kept.append({"cause": item.cause, "why": item.why, "intervention": allowed, "control": item.control})
    return kept


# ── The experiment ───────────────────────────────────────────────────────────

def _emit(on_step: StepCallback | None, agent: str, tool: str, thought: str, result: str, status: str = "done") -> None:
    if on_step is not None:
        on_step({"agent": agent, "tool": tool, "thought": thought, "result": result, "status": status})


def _run(spec: SceneSpec, interventions: list[dict], robot_id: str | None, baseline: bool) -> list[dict]:
    variations = []
    for index, intervention in enumerate(interventions, 1):
        variation = {"label": f"Experiment {index}", **intervention}
        if robot_id is not None and any(k in intervention for k in ("speed_scale", "face_uphill")):
            variation["robot_id"] = robot_id
        variations.append(variation)
    return run_batch(spec, variations, baseline=baseline)


# ── Decisions between batches ────────────────────────────────────────────────

class FollowUpIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cause: str = Field(min_length=1, max_length=80)
    intervention: dict[str, Any]


class DecisionReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision: Literal["conclude", "test more", "can't tell"]
    reason: str = Field(min_length=1, max_length=400)
    next: list[FollowUpIn] = Field(default_factory=list, max_length=4)


DECISION_SYSTEM = """You are the Explainer deciding what to do after a batch of single-change experiments.
Choose one:
- "conclude": the evidence is enough
- "test more": up to 2 follow-up experiments, e.g. the top two causes together (two keys in one
  intervention) or a stronger version of one cause; each intervention uses only these keys: {knobs}
- "can't tell": the results contradict each other
You have {remaining} rerun(s) left. Reply with JSON only, in this shape (the values are placeholders showing the
format, not suggestions):
{{"decision":"<conclude|test more|can't tell>","reason":"<one sentence>","next":[{{"cause":"<a + b>","intervention":{{"<knob>":<value>}}}}]}}"""


def improvement(base: dict, after: dict) -> float:
    """How much better a run got than the reproduced failure (1 = passed)."""
    if after.get("error"):
        return -1.0
    if after.get("passed"):
        return 1.0
    score = 0.0
    if (base.get("final_distance_m") or 0) > 0 and after.get("final_distance_m") is not None:
        score = 1 - after["final_distance_m"] / base["final_distance_m"]
    if base.get("flipped") and not after.get("flipped"):
        score += 0.3
    if (base.get("slip_ratio") or 0) > 0.05 and after.get("slip_ratio") is not None:
        score += 0.3 * (1 - after["slip_ratio"] / base["slip_ratio"])
    return round(score, 3)


def stronger(spec: SceneSpec, intervention: dict) -> dict | None:
    """The same single change, pushed further (None when it can't go further)."""
    knob, value = next(iter(intervention.items()))
    if knob == "slope_deg" and value > 0:
        return {"slope_deg": max(0.0, value - 10)}
    if knob == "speed_scale" and value > 0.3:
        return {"speed_scale": round(value * 0.6, 2)}
    if knob == "obstacle_scale" and value > 0.2:
        return {"obstacle_scale": round(value * 0.5, 3)}
    if knob == "friction":
        return {"friction": round(min(value * 1.5, 2.0), 2)} if value < 2.0 else None
    if knob == "surface":
        return {"friction": round(min((spec.environment.friction or 0.8) * 2.0, 2.0), 2)}
    return None


def rule_decision(spec: SceneSpec, hypotheses: list[dict], remaining: int) -> dict:
    """Mock Explainer: test more only when the best result is "partly"; otherwise conclude."""
    if any(h["verdict"] == "confirmed" for h in hypotheses):
        confirmed = [h["cause"] for h in hypotheses if h["verdict"] == "confirmed"]
        return {"decision": "conclude", "reason": f"{_join(confirmed)} passed on {'its' if len(confirmed) == 1 else 'their'} own.", "next": []}
    ranked = sorted(hypotheses, key=lambda h: -h.get("improvement", 0))
    partly = [h for h in ranked if h["verdict"] == "partly"]
    if not partly:
        return {"decision": "conclude", "reason": "Nothing improved clearly, so more of the same won't help.", "next": []}
    if remaining <= 0:
        return {"decision": "conclude", "reason": "The rerun budget is used up.", "next": []}
    top = partly[0]
    tested = [h["intervention"] for h in hypotheses]
    if len(partly) >= 2:
        second = partly[1]
        combined = {**top["intervention"], **second["intervention"]}
        if combined not in tested and len(combined) == 2:
            return {"decision": "test more",
                    "reason": f"{top['cause']} and {second['cause']} each only partly explain it; testing them together.",
                    "next": [{"cause": f"{top['cause']} + {second['cause']}", "intervention": combined}]}
    push = stronger(spec, top["intervention"])
    if push is not None and push not in tested and _applicable(spec, push) is None:
        return {"decision": "test more", "reason": f"{top['cause']} only partly explains it; testing a stronger version.",
                "next": [{"cause": f"{top['cause']} (stronger)", "intervention": push}]}
    other = next((h for h in ranked if h is not top and h["cause"] not in top["cause"]), None)
    if other is not None:
        combined = {**top["intervention"], **other["intervention"]}
        if combined not in tested and len(combined) == 2 and _applicable(spec, combined) is None:
            return {"decision": "test more",
                    "reason": f"{top['cause']} only partly explains it; testing {top['cause']} + {other['cause']} together.",
                    "next": [{"cause": f"{top['cause']} + {other['cause']}", "intervention": combined}]}
    return {"decision": "conclude", "reason": f"{top['cause']} only partly explains it and can't be pushed further.", "next": []}


def model_decision(spec: SceneSpec, hypotheses: list[dict], remaining: int, decisions: list[dict],
                   calls: list[dict]) -> dict | None:
    knobs = ", ".join(ALLOWED_KNOBS)
    results = [{"cause": h["cause"], "change": h["intervention"], "verdict": h["verdict"],
                "numbers": describe_numbers(h["key_numbers"])} for h in hypotheses]
    messages = [{"role": "system", "content": DECISION_SYSTEM.format(knobs=knobs, remaining=remaining)},
                {"role": "user", "content": json.dumps(results)}]
    reply, attempts = llm.chat_json(messages, config.PLANNER_MODEL, DecisionReply.model_validate)
    calls.append({"agent": "Explainer", "purpose": "decision", "attempts": attempts})
    if reply is None:
        return None
    follow = []
    for item in reply.next:
        allowed = {k: v for k, v in item.intervention.items() if k in ALLOWED_KNOBS}
        problem = None if allowed and len(allowed) <= 2 else "needs one or two allowed changes"
        problem = problem or _applicable(spec, allowed)
        if problem or allowed in [h["intervention"] for h in hypotheses]:
            decisions.append({"agent": "Explainer", "decision": "dropped",
                              "reason": f"{item.cause}: {problem or 'already tested'}."})
            continue
        follow.append({"cause": item.cause, "intervention": allowed})
    return {"decision": reply.decision, "reason": reply.reason, "next": follow}


def mock_conclusion(hypotheses: list[dict], outcome: str = "conclude") -> str:
    confirmed = [h["cause"] for h in hypotheses if h["verdict"] == "confirmed"]
    partly = [h["cause"] for h in hypotheses if h["verdict"] == "partly"]
    ruled_out = [h["cause"] for h in hypotheses if h["verdict"] == "ruled out"]
    combined = any("+" in h["cause"] for h in hypotheses)
    sentences = ["The results contradict each other, so I can't tell for sure."] if outcome == "can't tell" else []
    if confirmed:
        sentences.append(f"Confirmed: {_join(confirmed)} — removing {'it' if len(confirmed) == 1 else 'each one'} "
                         "alone was enough for the run to pass.")
    else:
        sentences.append("None of the tested causes explains it on its own.")
    if partly:
        sentences.append(f"{_join(partly).capitalize()} partly explain{'s' if len(partly) == 1 else ''} it: the run "
                         "got clearly better but still failed.")
    if ruled_out:
        sentences.append(f"Ruled out: {_join(ruled_out)} — changing {'it' if len(ruled_out) == 1 else 'each'} made no clear difference.")
    if not combined:
        sentences.append("Each cause was tested on its own, so causes that only matter together could be missed.")
    return " ".join(sentences)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


CONCLUSION_SYSTEM = """You are the Explainer. Write 3 to 5 plain sentences for a non-expert, using ONLY the verdicts
given. Name the confirmed causes; say what was ruled out; mention causes that only partly explain it. If nothing
is confirmed, say "none of the tested causes explains it on its own" and do not invent a cause. Say that each
cause was tested on its own, so causes that only matter together could be missed, unless a combined test is
listed. No headings or lists."""


def conclusion(hypotheses: list[dict], calls: list[dict], outcome: str = "conclude") -> str:
    if config.MOCK_MODE:
        return mock_conclusion(hypotheses, outcome)
    verdicts = {"outcome": outcome, "results": [
        {"cause": h["cause"], "change_tested": describe_change(h["intervention"]), "verdict": h["verdict"],
         "numbers": describe_numbers(h["key_numbers"])} for h in hypotheses]}
    text = llm.chat([{"role": "system", "content": CONCLUSION_SYSTEM},
                     {"role": "user", "content": json.dumps(verdicts)}], temperature=0.3).strip()
    calls.append({"agent": "Explainer", "purpose": "conclusion", "attempts": [{"attempt": 1, "valid": bool(text)}]})
    return text or mock_conclusion(hypotheses, outcome)


def _experiment(spec: SceneSpec, items: list[dict], base: dict, robot_id: str | None, report: dict,
                on_step: StepCallback | None, thought: str) -> None:
    _emit(on_step, "Batch runner", "experiments", thought.format(n=len(items)), "", "working")
    results = _run(spec, [h["intervention"] for h in items], robot_id, baseline=False)
    report["runs"] += len(results)
    for item, result in zip(items, results):
        after = result_metrics(result, robot_id)
        item["verdict"] = verdict(base, after)
        item["key_numbers"] = key_numbers(base, after)
        item["improvement"] = improvement(base, after)
        item["change"] = describe_change(item["intervention"])
    report["hypotheses"] += items
    _emit(on_step, "Batch runner", "experiments", "Experiments.",
          "; ".join(f"{h['cause']}: {h['verdict']} ({describe_numbers(h['key_numbers'])})" for h in items))


def diagnose(spec: SceneSpec, summary: dict, robot_id: str | None = None,
             on_step: StepCallback | None = None) -> dict:
    """Reproduce a failed run, test 2–4 single-change hypotheses in parallel and report verdicts."""
    original = robot_metrics(summary, robot_id)
    robot_id = original["robot_id"]
    report: dict[str, Any] = {"type": "diagnosis", "robot_id": robot_id, "original": original,
                              "hypotheses": [], "decisions": [], "model_calls": [], "runs": 0}
    if original["passed"]:
        report.update(reproduced=None, conclusion="This run passed, so there is no failure to explain. "
                                                  "Ask \"why did it fail?\" after a run that fails.")
        return report

    spec = pin_task(spec, summary)
    _emit(on_step, "Batch runner", "reproduce", "Rerunning the unchanged scene to check the failure repeats.", "", "working")
    base_result = run_batch(spec, [])[0]
    report["runs"] = 1
    base = result_metrics(base_result, robot_id)
    report["reproduced_metrics"] = base
    if base.get("error") or base["passed"]:
        message = (f"The rerun could not run: {base['error']}" if base.get("error")
                   else "The failure didn't reproduce: the unchanged scene passed when rerun, so I stopped there.")
        _emit(on_step, "Batch runner", "reproduce", "Rerunning the unchanged scene.", message, "warning")
        report.update(reproduced=False, conclusion=message)
        return report
    report["reproduced"] = True
    _emit(on_step, "Batch runner", "reproduce", "Rerunning the unchanged scene.",
          f"The failure reproduced ({base.get('outcome') or 'failed'}).")

    hypotheses = None if config.MOCK_MODE else model_hypotheses(spec, base, report["decisions"], report["model_calls"])
    for dropped in report["decisions"]:
        _emit(on_step, dropped["agent"], "hypotheses", "Checking the proposed experiments.",
              f"Dropped: {dropped['reason']}", "warning")
    if hypotheses is None or len(hypotheses) < 2:
        rules = rule_hypotheses(spec, base)
        hypotheses = (hypotheses or []) + [h for h in rules if h["cause"] not in {x["cause"] for x in hypotheses or []}]
    if not any(h.get("control") for h in hypotheses):
        extra = [h for h in rule_hypotheses(spec, base) if h.get("control")]
        hypotheses += [h for h in extra if h["intervention"] not in [x["intervention"] for x in hypotheses]][:1]
    hypotheses = hypotheses[: min(MAX_HYPOTHESES, MAX_RERUNS - report["runs"])]
    _emit(on_step, "Explainer", "hypotheses", "Proposing causes, each tested by changing one thing.",
          "; ".join(f"{h['cause']}{' (control)' if h.get('control') else ''}: {describe_change(h['intervention'])}"
                    for h in hypotheses))

    _experiment(spec, hypotheses, base, robot_id, report, on_step, "Running {n} single-change experiments in parallel.")
    outcome = "conclude"
    for _ in range(MAX_DECISION_ROUNDS):
        remaining = MAX_RERUNS - report["runs"]
        recorded = len(report["decisions"])
        decided = None if config.MOCK_MODE else model_decision(
            spec, report["hypotheses"], remaining, report["decisions"], report["model_calls"])
        for dropped in report["decisions"][recorded:]:
            _emit(on_step, dropped["agent"], "decision", "Checking the follow-up experiments.",
                  f"Dropped: {dropped['reason']}", "warning")
        decided = decided or rule_decision(spec, report["hypotheses"], remaining)
        follow = decided["next"][: min(MAX_FOLLOW_UPS, remaining)] if decided["decision"] == "test more" else []
        outcome = decided["decision"] if decided["decision"] != "test more" or follow else "conclude"
        reason = decided["reason"] + ("" if follow or decided["decision"] != "test more" else " No reruns are left, so concluding.")
        report["decisions"].append({"agent": "Explainer", "decision": outcome if not follow else "test more", "reason": reason})
        _emit(on_step, "Explainer", "decision", "Deciding whether the evidence is enough.",
              f"{'Test more' if follow else outcome.capitalize()}: {reason}")
        if not follow:
            break
        for item in follow:
            item["follow_up"] = True
        _experiment(spec, follow, base, robot_id, report, on_step, "Running {n} follow-up experiment(s).")
    report["outcome"] = outcome
    report["conclusion"] = conclusion(report["hypotheses"], report["model_calls"], outcome)
    _emit(on_step, "Explainer", "conclusion", "Writing the conclusion from the verdicts only.", report["conclusion"])
    return report
