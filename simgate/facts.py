"""FACTS for the Explainers: plain-language labels built in code from the scene and the measured numbers, which
the model must use as given, and a check that a reply doesn't contradict them."""

import json
import re

from .spec import SceneSpec

STEADY_DEG, LEANING_DEG, NEARLY_TIPPED_DEG = 15.0, 35.0, 60.0
FACTS_RULE = ("Describe the terrain and tilt only with the words given in FACTS; never call a tilt above "
              f"{LEANING_DEG:g}° stable or steady. A tip may change only a setting FACTS list under \"can change\".")

ADJUSTABLE = {   # the settings a tip may change, per kind of robot
    "rover": ("wheels", "wheel_radius", "drive_speed"),
    "walker": ("legs", "leg_length", "gait_freq_hz", "stride_deg"),
    "library": ("command_speed_m_s",),   # a library robot's body comes from its model file
}
SETTING_NAMES = {"wheels": "number of wheels", "wheel_radius": "wheel size", "drive_speed": "wheel speed",
                 "legs": "number of legs", "leg_length": "leg length", "gait_freq_hz": "gait frequency",
                 "stride_deg": "stride", "command_speed_m_s": "walking speed"}
_SETTING_WORDS = {   # how a tip names each setting
    "wheels": r"\b(?:four|six|4|6)[\s-]*wheel(?:s|ed)?\b|\bmore wheels\b",
    "wheel_radius": r"\bwheel[_ ](?:radius|size)\b|\b(?:bigger|larger|smaller) wheels\b",
    "drive_speed": r"\bdrive[_ ]speed\b|\bwheel speed\b",
    "legs": r"\b(?:four|six|4|6)[\s-]*leg(?:s|ged)?\b|\bmore legs\b",
    "leg_length": r"\bleg[_ ]length\b|\b(?:longer|shorter) legs\b",
    "gait_freq_hz": r"\bgait\b|\bgait[_ ]freq",
    "stride_deg": r"\bstride",
    "command_speed_m_s": r"\bcommand[_ ]speed|\bwalking speed\b",
}
_TIP_CUES = re.compile(r"\b(?:try|next|you (?:could|can|might)|consider|adjust|set|increase|decrease|reduce|raise|lower"
                       r"|change|tweak|use|give)\b", re.I)


def terrain_label(spec: SceneSpec) -> str:
    terrain = spec.terrain
    if terrain is None:
        return "flat ground"
    if terrain.kind == "real":
        try:
            from .terrain_data import load_preset

            site = load_preset(terrain.preset)[1]["label"].split(",")[0].strip()
        except Exception:  # noqa: BLE001 - a missing preset still gets a truthful label
            site = terrain.preset or "a real site"
        return f"real {spec.environment.body.title()} terrain: {site}"
    if terrain.kind == "slope":
        return f"a {terrain.slope_deg:g}° slope"
    return {"flat": "flat ground", "hills": "hills", "rocky": "rocky ground", "craters": "craters"}.get(terrain.kind, terrain.kind)


def tilt_label(degrees: float | None) -> str | None:
    if degrees is None:
        return None
    if degrees < STEADY_DEG:
        return "steady"
    if degrees < LEANING_DEG:
        return "leaning"
    if degrees <= NEARLY_TIPPED_DEG:
        return "nearly tipped"
    return "tipped over"


def _result(m: dict) -> str:
    if m.get("goal_reached"):
        return f"reached the goal in {m.get('time_to_goal_s')} s"
    if m.get("final_distance_m") is not None:
        return f"missed the goal by {m['final_distance_m']} m"
    if m.get("flipped"):
        return "fell over"
    return f"moved {m.get('forward_progress_m', 0)} m forward"


def _reason(m: dict) -> str:
    reason = {"reached": "reached the goal", "fell": "fell over", "stuck": "got stuck", "out_of_time": "ran out of time",
              "left_terrain": "went off the edge"}.get(m.get("outcome"), m.get("outcome") or "")
    course = m.get("obstacles")   # a single robot's summary lists them; a racer's metrics summarise them
    stop = m.get("stopped_by") or (course.get("stopped_by") if isinstance(course, dict) else None)
    if stop:
        reason += f", stopped by the {stop.get('height_m', '')} m {stop.get('kind', 'obstacle')}".replace(" ,", ",")
    return reason


def _can_change(robot: dict) -> dict:
    """The settings a tip may change for this robot."""
    return {key: SETTING_NAMES[key] for key in ADJUSTABLE.get(robot.get("kind"), ())}


def _robot_facts(label: str, m: dict) -> dict:
    from .agents import robot_view

    robot = robot_view(m.get("robot") or {})
    return {"robot": label, "kind": robot.get("kind"), "result": _result(m), "reason": _reason(m),
            "max_tilt_deg": m.get("max_tilt_deg"), "tilt": tilt_label(m.get("max_tilt_deg")), "properties": robot,
            "can_change": _can_change(m.get("robot") or {})}


def facts(spec: SceneSpec, metrics: dict) -> dict:
    """Terrain, the steepest slope crossed and, per robot, kind, result, reason, tilt label and its properties."""
    out = {"terrain": terrain_label(spec)}
    if metrics.get("max_slope_on_path_deg") is not None:
        out["steepest_slope_crossed_deg"] = metrics["max_slope_on_path_deg"]
    if isinstance(metrics.get("robots"), dict):
        labels = {entry.id: entry.label for entry in spec.robots}
        out["robots"] = [_robot_facts(labels.get(robot_id, m.get("label") or robot_id), m)
                         for robot_id, m in metrics["robots"].items()]
    elif metrics.get("simulator") == "MuJoCo" or "max_tilt_deg" in metrics:
        out["robots"] = [_robot_facts(spec.robots[0].label if spec.robots else "Robot", metrics)]
    return out


def facts_block(found: dict) -> str:
    lines = [f"- terrain: {found['terrain']}"]
    if "steepest_slope_crossed_deg" in found:
        lines.append(f"- steepest slope crossed: {found['steepest_slope_crossed_deg']}°")
    for robot in found.get("robots", []):
        tilt = f"; tilt: {robot['tilt']} ({robot['max_tilt_deg']}° max)" if robot.get("tilt") else ""
        properties = ", ".join(f"{k}={v}" for k, v in robot["properties"].items() if k != "kind")
        can_change = ", ".join(f"{key} ({name})" for key, name in robot.get("can_change", {}).items()) or "nothing"
        lines.append(f"- {robot['robot']} ({robot['kind']}): {robot['result']}; {robot['reason']}{tilt}; has: {properties}; "
                     f"can change: {can_change}")
    return "FACTS (use these words as-is):\n" + "\n".join(lines)


_EXPLAINER_MEASUREMENTS = {
    "goal_reached": "reached_goal",
    "time_to_goal_s": "goal_time_seconds",
    "distance_m": "distance_travelled_metres",
    "final_distance_m": "distance_remaining_metres",
    "closest_distance_m": "closest_approach_metres",
    "path_length_m": "path_length_metres",
    "path_efficiency": "path_efficiency",
    "forward_progress_m": "forward_progress_metres",
    "climbed_m": "climb_height_metres",
    "max_tilt_deg": "maximum_tilt_degrees",
    "flipped": "flipped",
    "fall_direction": "fall_direction",
    "flipped_at_s": "fall_time_seconds",
    "local_slope_deg": "ground_angle_at_event_degrees",
    "slope_along_heading_deg": "ground_angle_ahead_degrees",
    "cross_slope_deg": "side_slope_degrees",
    "max_slope_on_path_deg": "maximum_path_angle_degrees",
    "slip_ratio": "sliding_contact_fraction",
    "slip_ratio_last_1s_before_fall": "sliding_before_fall_fraction",
    "left_terrain": "outside_terrain",
    "obstacles_cleared": "obstacles_cleared",
    "obstacles_total": "obstacles_total",
    "robot_contacts": "robot_contact_pairs",
    "max_roof_displacement_cm": "maximum_roof_sway_centimetres",
    "max_story_drift_pct": "maximum_story_drift_percent",
    "stories": "story_count",
    "height_m": "frame_height_metres",
    "period_s": "natural_period_seconds",
    "code_period_estimate_s": "reference_period_seconds",
    "building_freq_hz": "building_frequency_hertz",
    "shaking_freq_hz": "shaking_frequency_hertz",
    "pga_g": "peak_ground_acceleration_g",
    "damage_estimate": "damage_estimate",
    "model_limits": "simulation_scope",
}
_OUTCOME_VALUES = {
    "reached": "reached",
    "fell": "fell",
    "stuck": "stuck",
    "out_of_time": "ran out of time",
    "left_terrain": "left the terrain",
}


def _explainer_measurements(metrics: dict) -> dict:
    values = {label: metrics[key] for key, label in _EXPLAINER_MEASUREMENTS.items()
              if metrics.get(key) is not None}
    if metrics.get("outcome") is not None:
        values["run_ending"] = _OUTCOME_VALUES.get(metrics["outcome"], metrics["outcome"])
    if metrics.get("stopped_by"):
        stop = metrics["stopped_by"]
        values["stopping_obstacle"] = {
            "kind": stop.get("kind"),
            "height_or_width_metres": stop.get("width_m") if stop.get("kind") == "gap" else stop.get("height_m"),
            "touched_at_seconds": stop.get("touched_at_s"),
        }
    if metrics.get("obstacles"):
        values["course_obstacles"] = [
            {
                "kind": obstacle.get("kind"),
                "height_or_gap_width_metres": obstacle.get("width_m")
                    if obstacle.get("kind") == "gap" else obstacle.get("height_m"),
                "contact_time_seconds": obstacle.get("touched_at_s"),
                "cleared": obstacle.get("cleared"),
            }
            for obstacle in metrics["obstacles"]
        ]
    if metrics.get("robot_rough_limits"):
        limits = metrics["robot_rough_limits"]
        values["usual_robot_limits_metres"] = {
            "step_height": limits.get("climb_m"),
            "gap_width": limits.get("gap_m"),
        }
    return values


def explainer_values(spec: SceneSpec, metrics: dict) -> dict:
    """Measured scene and outcome values only; omit prose labels, assumptions, and adjustable robot settings."""
    terrain = spec.terrain
    scene_values = {
        "world": spec.environment.body,
        "terrain": terrain.kind if terrain is not None else None,
        "surface": spec.environment.surface,
        "gravity_metres_per_second_squared": spec.environment.gravity,
        "surface_friction": spec.environment.friction,
    }
    if terrain is not None and terrain.kind == "slope":
        scene_values["terrain_angle_degrees"] = terrain.slope_deg

    robot_rows = []
    metric_rows = metrics.get("robots") if isinstance(metrics.get("robots"), dict) else None
    if metric_rows:
        labels = {entry.id: entry.label for entry in spec.robots}
        kinds = {entry.id: entry.robot.kind for entry in spec.robots}
        for robot_id, row in metric_rows.items():
            robot_rows.append({"name": labels.get(robot_id, row.get("label", robot_id)),
                               "kind": kinds.get(robot_id, row.get("robot", {}).get("kind")),
                               "measurements": _explainer_measurements(row)})
    elif metrics:
        entry = spec.robots[0] if spec.robots else None
        robot_name = entry.label if entry else (
            "G1" if spec.robot and spec.robot.robot_id == "g1" else "Robot")
        robot_kind = entry.robot.kind if entry else spec.robot.kind if spec.robot else None
        robot_rows.append({"name": robot_name, "kind": robot_kind,
                           "measurements": _explainer_measurements(metrics)})

    values = {"scene": scene_values, "robots": robot_rows}
    if metrics.get("leaderboard"):
        values["arrival_order"] = [row.get("label") for row in metrics["leaderboard"] if row.get("label")]
    return values


def explainer_values_block(spec: SceneSpec, metrics: dict) -> str:
    return "FACTS_VALUES_JSON (measured values; object keys are internal):\n" + json.dumps(
        explainer_values(spec, metrics), separators=(",", ":"), ensure_ascii=True,
    )


_RAW_IDENTIFIER = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", re.I)
_LEAKED_FACT_LABEL = re.compile(
    r"\bsteepest\s+slope\s+crossed\b|\b(?:tilt|stability)\s*:\s*(?:steady|leaning|nearly tipped|tipped over)\b",
    re.I,
)
_TIP_SENTENCE = re.compile(
    r"^(?:try\b|next\s*,?\s*try\b|you\s+(?:could|can|might)\s+try\b|consider\s+trying\b|type\b)", re.I,
)
_REPEATED_OUTCOME = re.compile(
    r"\b(reached the goal|missed the goal|ran out of time|fell over|got stuck|went off the edge)\b", re.I,
)


def explainer_output_problems(text: str, found: dict) -> list[str]:
    """Reject copied field labels, internal identifiers, advice, and repeated outcome claims."""
    issues = []
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    for sentence in sentences:
        if _LEAKED_FACT_LABEL.search(sentence):
            issues.append("It repeats a FACTS label instead of describing the value.")
        if _RAW_IDENTIFIER.search(sentence):
            issues.append("It includes an internal field or setting name.")
        if _TIP_SENTENCE.search(sentence):
            issues.append("It suggests a next action; actions are supplied separately.")
    robots = found.get("robots") or []
    outcomes_by_subject: dict[tuple[str, str | None], int] = {}
    labels = [str(robot.get("robot", "")).lower() for robot in robots if robot.get("robot")]
    for match in _REPEATED_OUTCOME.finditer(text):
        sentence_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?\n") + 1
        prefix = text[sentence_start:match.start()].lower()
        mentioned = [(prefix.rfind(label), label) for label in labels if label in prefix]
        subject = max(mentioned)[1] if mentioned else None
        if len(robots) <= 1:
            subject = labels[0] if labels else None
        if subject is not None:
            key = (match.group(1).lower(), subject)
            outcomes_by_subject[key] = outcomes_by_subject.get(key, 0) + 1
    if any(count > 1 for count in outcomes_by_subject.values()):
        issues.append("It repeats the same robot outcome; state that outcome once.")
    return list(dict.fromkeys(issues))


def strip_explainer_artifacts(text: str, found: dict) -> str:
    """Strip any remaining prompt labels, raw names, or action tips from the displayed explanation."""
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
        sentence = sentence.strip()
        if not sentence or _LEAKED_FACT_LABEL.search(sentence) or _RAW_IDENTIFIER.search(sentence) \
                or _TIP_SENTENCE.search(sentence):
            continue
        kept.append(sentence)
    return " ".join(kept) or "The simulation is complete; see the measured results for details."


def drop_invalid_tips(text: str, found: dict) -> tuple[str, list[str]]:
    """Remove each tip sentence that names a setting no robot in FACTS can change (gait for the G1, wheels for a
    walker). Returns the text and the settings that were named."""
    allowed = {key for robot in found.get("robots", []) for key in robot.get("can_change", {})}
    if not found.get("robots"):
        return text, []
    named: list[str] = []
    kept = []
    for part in re.split(r"(?<=[.!?])(\s+)", text):
        if part.strip() and _TIP_CUES.search(part):
            wrong = [key for key, pattern in _SETTING_WORDS.items() if key not in allowed and re.search(pattern, part, re.I)]
            if wrong:
                named += wrong
                if kept and not kept[-1].strip():
                    kept.pop()   # the space before the removed sentence
                continue
        kept.append(part)
    return "".join(kept).strip(), named


_FLAT = re.compile(r"\bflat\b", re.I)
_RELIEF = re.compile(r"\bhill(?:s|y)?\b|\bcraters?\b|\bslopes? of \d|\bsteep (?:hill|slope)", re.I)
_CALM = re.compile(r"\b(?:stable|steady|stability)\b", re.I)
_DEGREES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?)", re.I)


def problems(text: str, found: dict) -> list[str]:
    """Where a reply contradicts FACTS: a terrain word for different ground, or a tilt above 35° called stable."""
    issues = []
    terrain = found.get("terrain")
    if terrain and terrain != "flat ground" and _FLAT.search(text):
        issues.append(f'It calls the terrain flat, but FACTS say the terrain is "{terrain}".')
    if terrain == "flat ground" and (match := _RELIEF.search(text)):
        issues.append(f'It says "{match.group(0)}", but FACTS say the terrain is "flat ground".')
    tilts = [robot["max_tilt_deg"] for robot in found.get("robots", []) if robot.get("max_tilt_deg") is not None]
    big = [t for t in tilts if t > LEANING_DEG]
    for sentence in re.split(r"(?<=[.!?])\s+|[,;]\s*|\s+(?:while|whereas|but|and)\s+", text):   # clause by clause
        if not _CALM.search(sentence):
            continue
        named = [float(value) for value in _DEGREES.findall(sentence)]
        named_big = any(value > LEANING_DEG and any(abs(value - t) <= 1.0 for t in big) for value in named)
        if named_big or (big and len(big) == len(tilts) and not named):   # unnamed: only if every robot tilted that far
            tilt = max(big)
            issues.append(f'It calls a {tilt:g}° tilt stable or steady; FACTS label it "{tilt_label(tilt)}".')
            break
    return issues


def checked(ask, found: dict, fallback, extra=None) -> tuple[str, list[str]]:
    """ask(feedback) returns the model's text; one retry with the exact problems, then the rules text.
    ``extra(text)`` adds caller-specific problems (e.g. a conclusion contradicting its verdicts)."""
    def issues_of(text: str) -> list[str]:
        return problems(text, found) + (extra(text) if extra else [])

    from .llm import ModelUnavailable

    try:
        text = ask(None)
    except ModelUnavailable as exc:   # no model to answer: the rules text, said so
        return fallback(), [f"model unavailable: {exc}"]
    issues = issues_of(text)
    if not issues:
        return text, []
    try:
        retry = ask("Your reply has problems: " + " ".join(issues) + " Rewrite it so none of these remain.")
    except ModelUnavailable as exc:
        return fallback(), issues + [f"model unavailable: {exc}"]
    if not issues_of(retry):
        return retry, issues
    return fallback(), issues + issues_of(retry)
