"""Code-selected, plain-language reasons and follow-up actions for each simulation outcome."""

from __future__ import annotations

from typing import Any

from .spec import SceneSpec


def _action(label: str, message: str | None = None, action: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label}
    if message is not None:
        result["message"] = message
    if action is not None:
        result["action"] = action
    return result


def _entries(scene: SceneSpec | None) -> list[Any]:
    if scene is None:
        return []
    if scene.robots:
        return scene.robots
    if scene.robot is not None:
        return [type("RobotEntryView", (), {"label": "Robot", "robot": scene.robot})()]
    return []


def _robot(scene: SceneSpec | None) -> Any | None:
    entries = _entries(scene)
    return entries[0] if entries else None


def _robot_name(scene: SceneSpec | None, facts: dict[str, Any]) -> str:
    known = facts.get("robots") or []
    if known and known[0].get("robot"):
        return str(known[0]["robot"])
    entry = _robot(scene)
    if entry is None:
        return "The robot"
    robot = entry.robot
    if robot.kind == "library" and robot.robot_id == "g1":
        return "The G1"
    if entry.label.lower() not in {"robot", "library"}:
        return entry.label
    if robot.robot_id == "g1":
        return "The G1"
    if robot.kind == "library":
        return f"The {robot.robot_id or 'library'} robot"
    if robot.kind == "walker":
        return f"The {robot.legs}-legged walker"
    return f"The {robot.wheels}-wheel rover"


def _other_robot(scene: SceneSpec | None) -> tuple[str, str]:
    entry = _robot(scene)
    if entry is None:
        return "a rover", "race a rover"
    robot = entry.robot
    if robot.kind == "library":
        return "a rover", "race the G1 against a rover"
    if robot.kind == "walker":
        return "a rover", "race the walker against a rover"
    return "a 6-legged walker", "race the rover against a 6-legged walker"


def _gentler_terrain(scene: SceneSpec | None) -> str:
    terrain = scene.terrain.kind if scene and scene.terrain else "hills"
    return "flat" if terrain in {"flat", "hills"} else "hills"


def _different_robot(scene: SceneSpec | None) -> tuple[str, str]:
    entry = _robot(scene)
    if entry is None:
        return "Use a 6-legged walker", "use a 6-legged walker"
    robot = entry.robot
    if robot.kind == "library":
        return "Use a 6-legged walker", "use a 6-legged walker"
    if robot.kind == "walker":
        return "Use a 4-wheel rover", "use a rover"
    if robot.wheels == 4:
        return "Use a 6-wheel rover", "use six wheels"
    return "Use a 6-legged walker", "use a 6-legged walker"


def _plain_check_reason(outcome: dict[str, Any], facts: dict[str, Any], scene: SceneSpec | None) -> str:
    checks = outcome.get("checks") or []
    failed = []
    for check in checks:
        passed = check.get("passed", True) if isinstance(check, dict) else getattr(check, "passed", True)
        advisory = check.get("advisory", False) if isinstance(check, dict) else getattr(check, "advisory", False)
        if not passed and not advisory:
            failed.append(check)
    name = failed[0].get("name", "Physics check") if failed and isinstance(failed[0], dict) else (
        getattr(failed[0], "name", "Physics check") if failed else "Physics check")
    detail = failed[0].get("detail", "") if failed and isinstance(failed[0], dict) else (
        getattr(failed[0], "detail", "") if failed else outcome.get("detail", ""))
    evidence = f"{name} {detail}".lower()
    robot_name = _robot_name(scene, facts)
    slope = facts.get("start_slope_deg") or outcome.get("start_slope_deg")
    entry = _robot(scene)
    trained_walker = bool(entry and entry.robot.kind == "library")
    if "unit" in evidence:
        return "The robot file uses inconsistent units, so its size and movement cannot be checked safely."
    if "rest" in evidence or "stand" in evidence or "spawn" in evidence:
        if slope is not None:
            policy = " and its trained walking policy is made for flat ground" if trained_walker else ""
            return f"{robot_name} can't stand where it starts: the ground there is about {slope:g}° steep{policy}."
        return f"{robot_name} can't stand securely where it starts, so the physics check stopped the run."
    if "slope" in evidence and slope is not None:
        return f"The starting ground is about {slope:g}° steep, beyond what this robot can safely start on."
    if failed:
        return f"The {str(name).lower()} physics check failed: {str(detail).strip().rstrip('.')} .".replace(" .", ".")
    return "The scene did not pass its physics checks, so it was not run."


def _failed_run_reason(outcome: dict[str, Any], facts: dict[str, Any], scene: SceneSpec | None) -> str:
    if outcome.get("reason"):
        return str(outcome["reason"]).strip().rstrip(".") + "."
    robots = facts.get("robots") or []
    if robots:
        robot = next((entry for entry in robots if entry.get("result") != "reached the goal"), robots[0])
        result = robot.get("result") or "missed the goal"
        reason = robot.get("reason")
        return f"{robot.get('robot', _robot_name(scene, facts))} {result}{f' because it {reason}' if reason and reason not in result else ''}."
    metrics = outcome.get("metrics") or outcome
    name = _robot_name(scene, facts)
    result = metrics.get("outcome") or "missed the goal"
    distance = metrics.get("final_distance_m")
    ending = f"{result}, {distance:g} m short" if distance is not None else result
    return f"{name} did not reach the goal: it {ending}."


def _diagnosis_actions(report: dict[str, Any], scene: SceneSpec | None, facts: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses = report.get("hypotheses") or []
    if report.get("reproduced") is None:
        failed = next((robot for robot in facts.get("robots", [])
                   if robot.get("reason") != "reached the goal"), None)
        if failed:
            label = str(failed.get("robot") or _robot_name(scene, facts))
            return [_action(f"Why did the {label} fail?", f"why did the {label} fail?"),
                    _action(f"Make the {label} pass", f"make the {label} pass"),
                    _action("Find where it breaks", "find where it breaks")]
        return [_action("Run again", "run again"), _action("Find where it breaks", "find where it breaks")]
    confirmed = next((hypothesis for hypothesis in hypotheses if hypothesis.get("verdict") == "confirmed"), None)
    cause = (confirmed or next((hypothesis for hypothesis in hypotheses if hypothesis.get("verdict") == "partly"), {})).get("cause", "")
    if cause in {"steep ground", "side slope"}:
        world_action = _action("Find a gentler route", "find a gentler route")
    elif cause == "slipping":
        world_action = _action("Use a grippier surface", "change surface to rock")
    elif cause == "obstacle":
        world_action = _action("Make the obstacle smaller", "make the step smaller")
    elif cause == "low gravity":
        world_action = _action("Try Earth gravity", "change gravity to Earth")
    else:
        world_action = _action(f"Try gentler terrain ({_gentler_terrain(scene)})", f"change terrain to {_gentler_terrain(scene)}")
    return [_action("Make it pass", "make it pass"), world_action, _action("Replay the failure", "run again")]


def _diagnosis_reason(report: dict[str, Any]) -> str:
    hypotheses = report.get("hypotheses") or []
    if report.get("reproduced") is None and report.get("conclusion"):
        return _first_sentence(report["conclusion"], "That robot passed, so there is no failure to explain.")
    confirmed = [item.get("cause") for item in hypotheses if item.get("verdict") == "confirmed"]
    partial = [item.get("cause") for item in hypotheses if item.get("verdict") == "partly"]
    if confirmed:
        return f"The tests confirmed {', '.join(confirmed)} as a cause of the failed run."
    if partial:
        return f"The tests found that {', '.join(partial)} partly explains the failed run."
    if report.get("reproduced") is False:
        return "The failure did not repeat, so there is not enough evidence to name a cause."
    return "No tested change explained the failure on its own."


def _first_sentence(text: str, fallback: str) -> str:
    sentence = str(text or "").strip().split("\n", 1)[0].strip()
    if not sentence:
        return fallback
    first = sentence.split(". ", 1)[0].strip()
    return first if first.endswith((".", "!", "?")) else first + "."


def next_actions(workflow: str, outcome: dict[str, Any] | str, facts: dict[str, Any] | None,
                 scene: SceneSpec | None) -> dict[str, Any]:
    """Return one plain-language reason and 2–4 labeled actions for a completed or blocked outcome.

    Each action contains the exact chat message it sends, or an explicit UI action for operations that are not
    ordinary chat (opening an upload form or replaying a stored result).
    """
    data = outcome if isinstance(outcome, dict) else {"kind": str(outcome)}
    facts = facts or {}
    kind = str(data.get("kind") or data.get("status") or "").lower().replace(" ", "_")
    report = data.get("report") or data

    if workflow in {"scene", "build"} and kind in {"ready", "scene_ready", ""}:
        return {"reason": "The scene passed its physics checks and is ready to run.", "actions": [
            _action("Run", "run"),
            _action("Find where it breaks", "find where it breaks"),
            _action("Stress test", "stress test"),
        ]}

    if workflow in {"checks", "check_failed"} or kind in {"check_failed", "checks_failed"}:
        terrain = _gentler_terrain(scene)
        robot_label, robot_message = _different_robot(scene)
        return {"reason": _plain_check_reason(data, facts, scene), "actions": [
            _action("Start on flatter ground", "start on flatter ground"),
            _action(f"Try gentler terrain ({terrain})", f"change terrain to {terrain}"),
            _action(robot_label, robot_message),
            _action("Run anyway", "run anyway"),
        ]}

    if workflow in {"run", "metrics"} or kind in {"run_passed", "run_failed", "passed", "failed"}:
        metrics = data.get("metrics") or data
        if "passed" in data:
            passed = bool(data["passed"])
        elif "goal_reached" in metrics:
            passed = bool(metrics["goal_reached"])
        elif isinstance(metrics.get("robots"), dict) and metrics["robots"]:
            passed = all(bool(robot.get("goal_reached")) for robot in metrics["robots"].values())
        else:
            passed = False
        robot = _robot(scene)
        robot_label = _robot_name(scene, facts)
        if passed:
            time_to_goal = metrics.get("time_to_goal_s")
            reason = (f"{robot_label} reached the goal in {time_to_goal:g} seconds."
                      if time_to_goal is not None else f"{robot_label} reached the goal.")
            harder = "make the obstacles bigger" if scene and scene.obstacles else (
                "make the slope steeper" if scene and scene.terrain and scene.terrain.kind == "slope" else "make the terrain rougher")
            other_label, race_message = _other_robot(scene)
            return {"reason": reason, "actions": [
                _action("Find where it breaks", "find where it breaks"),
                _action("Make it harder", harder),
                _action(f"Race it against {other_label}", race_message),
            ]}
        return {"reason": _failed_run_reason(data, facts, scene), "actions": [
            _action("Why did it fail?", "why did it fail?"),
            _action("Make it pass", "make it pass"),
            _action("Find where it breaks", "find where it breaks"),
        ]}

    if workflow in {"diagnose", "diagnosis"}:
        return {"reason": _diagnosis_reason(report), "actions": _diagnosis_actions(report, scene, facts)}

    if workflow in {"redesign", "design"}:
        winner = report.get("winner") or report.get("fragile")
        if winner:
            return {"reason": f"A design was found that {('passed some checks but is not reliable' if winner.get('fragile') else 'passed the run')}: {winner.get('description', 'review the proposed changes')}.",
                    "actions": [
                        _action("Use fragile design" if winner.get("fragile") else "Use this design",
                                "use fragile design" if winner.get("fragile") else "use this design"),
                        _action("Re-check the limits", "find where it breaks again"),
                        _action("Show runs", action={"type": "show_design"}),
                    ]}
        diagnosis = report.get("diagnosis") or {}
        conclusion = str(diagnosis.get("conclusion") or "").strip()
        best = report.get("best_attempt") or {}
        details = []
        if scene and scene.robot and scene.robot.kind == "library":
            details.append("the trained robot body is fixed, so only its speed or a compatible policy can change")
        if conclusion:
            details.append(_first_sentence(conclusion, "" ).rstrip("."))
        if best.get("description"):
            details.append(f"the best attempt was {best['description']}")
        reason = "No tested robot design made the run pass"
        if details:
            reason += ": " + "; ".join(details)
        reason += "."
        other_label, race_message = _other_robot(scene)
        actions = [
            _action("Find a gentler route", "find a gentler route"),
            _action(f"Try gentler terrain ({_gentler_terrain(scene)})", f"change terrain to {_gentler_terrain(scene)}"),
            _action(f"Race it against {other_label}", race_message),
        ]
        if scene and scene.robot and scene.robot.kind == "library":
            actions.append(_action("Upload a better policy", action={"type": "open_upload", "kind": "policy"}))
        else:
            actions.append(_action("See where it breaks", "find where it breaks"))
        return {"reason": reason, "actions": actions[:4]}

    if workflow in {"finder", "find_failures", "envelope"}:
        limits = report.get("limits") or []
        if limits:
            closest = limits[0]
            limit_name = closest.get("name") or closest.get("knob_name") or "a world condition"
            value = closest.get("first_fail")
            reason = (f"The search found where {limit_name} first caused a failure at {value:g}."
                      if isinstance(value, (int, float)) else f"The search found a measured limit for {limit_name}.")
        else:
            reason = _first_sentence(report.get("stopped") or report.get("explanation"),
                                     "The condition search is complete; its measured results are ready to review.")
        return {"reason": reason, "actions": [
            _action("Replay the weakest spot", "replay the weakest spot"),
            _action("Why?", "why did it fail?"),
            _action("Make it pass", "make it pass"),
        ]}

    if workflow in {"ambiguous", "clarify"}:
        options = data.get("options") or []
        actions = []
        for option in options[:4]:
            if isinstance(option, dict):
                actions.append(_action(str(option.get("label", "Choose")), option.get("message"), option.get("action")))
            else:
                actions.append(_action(str(option), str(option)))
        if len(actions) < 2:
            actions = [_action("Change the terrain", "change the terrain"), _action("Change the robot", "change the robot")]
        return {"reason": str(data.get("reason") or "Which part of the scene would you like to change?"), "actions": actions}

    if workflow in {"planner_unrecognized", "unrecognized"}:
        options = data.get("options") or [
            {"label": "Change the terrain", "message": "change the terrain"},
            {"label": "Change the robot", "message": "change the robot"},
            {"label": "Set a goal", "message": "set a goal"},
        ]
        actions = [_action(str(option.get("label", "Try this")), option.get("message"), option.get("action"))
                   for option in options[:3]]
        return {"reason": "I didn't catch that; did you mean one of these?", "actions": actions}

    return {"reason": "The scene is ready for another request.", "actions": [
        _action("Run", "run"), _action("Change the scene", "change the scene")
    ]}