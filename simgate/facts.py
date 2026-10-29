"""FACTS for the Explainers: plain-language labels built in code from the scene and the measured numbers, which
the model must use as given, and a check that a reply doesn't contradict them."""

import re

from .spec import SceneSpec

STEADY_DEG, LEANING_DEG, NEARLY_TIPPED_DEG = 15.0, 35.0, 60.0
FACTS_RULE = ("Describe the terrain and tilt only with the words given in FACTS; never call a tilt above "
              f"{LEANING_DEG:g}° stable or steady. A tip may change only a setting FACTS list under \"can change\".")

ADJUSTABLE = {   # the settings a tip may change, per kind of robot
    "rover": ("wheels", "wheel_radius", "drive_speed"),
    "walker": ("legs", "leg_length", "gait_freq_hz", "stride_deg"),
    "library": ("command_speed_m_s", "policy_id"),   # a trained policy expects its exact body
}
SETTING_NAMES = {"wheels": "number of wheels", "wheel_radius": "wheel size", "drive_speed": "wheel speed",
                 "legs": "number of legs", "leg_length": "leg length", "gait_freq_hz": "gait frequency",
                 "stride_deg": "stride", "command_speed_m_s": "walking speed", "policy_id": "policy"}
_SETTING_WORDS = {   # how a tip names each setting
    "wheels": r"\b(?:four|six|4|6)[\s-]*wheel(?:s|ed)?\b|\bmore wheels\b",
    "wheel_radius": r"\bwheel[_ ](?:radius|size)\b|\b(?:bigger|larger|smaller) wheels\b",
    "drive_speed": r"\bdrive[_ ]speed\b|\bwheel speed\b",
    "legs": r"\b(?:four|six|4|6)[\s-]*leg(?:s|ged)?\b|\bmore legs\b",
    "leg_length": r"\bleg[_ ]length\b|\b(?:longer|shorter) legs\b",
    "gait_freq_hz": r"\bgait\b|\bgait[_ ]freq",
    "stride_deg": r"\bstride",
    "command_speed_m_s": r"\bcommand[_ ]speed|\bwalking speed\b",
    "policy_id": r"\bpolic(?:y|ies)\b|\bpolicy_id\b",
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
    """The settings a tip may change for this robot; a library robot's policy only to a compatible one."""
    settings = {key: SETTING_NAMES[key] for key in ADJUSTABLE.get(robot.get("kind"), ())}
    if "policy_id" in settings and robot.get("robot_id"):
        from .library import LibraryRegistry

        compatible = [policy.id for policy in LibraryRegistry().compatible_policies(robot["robot_id"])]
        settings["policy_id"] = f"policy: {', '.join(compatible)}"
    return settings


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

    text = ask(None)
    issues = issues_of(text)
    if not issues:
        return text, []
    retry = ask("Your reply has problems: " + " ".join(issues) + " Rewrite it so none of these remain.")
    if not issues_of(retry):
        return retry, issues
    return fallback(), issues + issues_of(retry)
