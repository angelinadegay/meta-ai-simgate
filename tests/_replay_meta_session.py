import asyncio
import json
import time
from typing import Any

from simgate import config
from simgate.orchestrator import Session


PROMPTS = [
    "G1 humanoid walking on Mars hills to the summit, go",
    "race the G1 against a rover",
    "run",
    "find where it breaks",
]


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = ("goal_reached", "outcome", "time_to_goal_s", "final_distance_m", "max_tilt_deg",
            "forward_progress_m", "flipped", "left_terrain")
    result = {key: metrics[key] for key in keys if key in metrics}
    if isinstance(metrics.get("robots"), dict):
        result["robots"] = {
            robot_id: {key: row[key] for key in keys if key in row}
            for robot_id, row in metrics["robots"].items()
        }
    return result


def visible(event: dict[str, Any]) -> dict[str, Any] | None:
    kind = event.get("type")
    if kind == "agent":
        return {"type": kind, "agent": event.get("agent"), "status": event.get("status"),
                "message": event.get("message")}
    if kind == "step":
        return {"type": kind, "agent": event.get("agent"), "model": event.get("model"),
                "tool": event.get("tool"), "thought": event.get("thought"), "result": event.get("result"),
                "status": event.get("status")}
    if kind == "spec":
        spec = event.get("spec", {})
        return {"type": kind, "title": spec.get("title"), "world": spec.get("environment", {}).get("body"),
                "terrain": spec.get("terrain"),
                "robots": [{"id": robot.get("id"), "label": robot.get("label"),
                            "kind": (robot.get("robot") or {}).get("kind"),
                            "robot_id": (robot.get("robot") or {}).get("robot_id")}
                           for robot in spec.get("robots", [])]}
    if kind == "scene":
        scene = event.get("scene", {})
        return {"type": kind, "kind": scene.get("kind"), "robot_ids": list((scene.get("robots") or {}).keys())}
    if kind == "checks":
        return {"type": kind, "checks": [{"name": row.get("name"), "passed": row.get("passed"),
                                            "detail": row.get("detail")} for row in event.get("checks", [])]}
    if kind == "question":
        return {"type": kind, "message": event.get("message"), "note": event.get("note"),
                "actions": [action.get("label") for action in event.get("actions", [])]}
    if kind == "explanation":
        return {"type": kind, "message": event.get("message"), "reason": event.get("reason"),
                "actions": [action.get("label") for action in event.get("actions", [])]}
    if kind == "run_start":
        return {"type": kind, "duration_s": event.get("duration_s")}
    if kind == "frames":
        frames = event.get("frames") or []
        return {"type": kind, "count": len(frames), "first_t": frames[0].get("t") if frames else None,
                "last_t": frames[-1].get("t") if frames else None}
    if kind == "metrics":
        return {"type": kind, "metrics": compact_metrics(event.get("metrics") or {})}
    if kind == "envelope":
        report = event.get("report", {})
        return {"type": kind, "robot": report.get("robot_label"), "robot_id": report.get("robot_id"),
                "headline": report.get("headline"),
                "surprising_failure_line": report.get("surprising_failure_line"),
                "honesty": report.get("honesty"), "runs": report.get("runs"),
                "limits": [{"name": row.get("name"), "ok": row.get("shown", {}).get("last_ok"),
                            "fails_at": row.get("shown", {}).get("first_fail"),
                            "confirmation": row.get("confirm")} for row in report.get("limits", [])],
                "failures": [{"kind": row.get("kind"), "values": row.get("values"),
                              "failure": row.get("failure"), "surprising": row.get("surprising"),
                              "confirmation": row.get("confirm")} for row in report.get("failures", [])]}
    if kind == "cost":
        cost = event.get("cost", {})
        return {"type": kind, "text": cost.get("text"), "runs": cost.get("runs"),
                "model_calls": cost.get("model_calls"), "wall_s": cost.get("wall_s")}
    if kind == "error":
        return {"type": kind, "message": event.get("message")}
    return None


async def replay() -> None:
    if config.MOCK_MODE:
        raise RuntimeError("Configured model mode is unavailable; refusing to run a Mock replay.")
    configured_builder_mode = config.BUILDER_MODE
    config.BUILDER_MODE = "llm"  # use Muse Wonder for this replay without changing .env or the app's normal mode

    events = []

    async def emit(event: dict[str, Any]) -> None:
        events.append(event)

    session = Session(emit)
    turns = []
    for prompt in PROMPTS:
        start = len(events)
        began = time.perf_counter()
        await session.handle(prompt)
        current = events[start:]
        turns.append({"request": prompt, "elapsed_s": round(time.perf_counter() - began, 2),
                      "events": [item for event in current if (item := visible(event)) is not None]})

    report = session.last_finder_report
    if not report:
        turns.append({"request": "replay the weakest spot", "events": [],
                      "blocked": "Finder did not produce a report."})
    else:
        target = report.get("robot_id")
        start = len(events)
        began = time.perf_counter()
        await session.handle_workflow_action("replay_weakest_spot", target)
        current = events[start:]
        turns.append({"request": "replay the weakest spot", "target_robot_id": target,
                      "elapsed_s": round(time.perf_counter() - began, 2),
                      "events": [item for event in current if (item := visible(event)) is not None]})

    print(json.dumps({
        "mode": {"mock": config.MOCK_MODE, "planner": config.PLANNER_MODEL,
                 "builder": config.BUILDER_MODEL, "configured_builder_mode": configured_builder_mode,
                 "replay_builder_mode": config.BUILDER_MODE},
        "turns": turns,
        "total_cost": {"model_calls": sum((e.get("cost") or {}).get("model_calls", 0)
                                           for e in events if e.get("type") == "cost"),
                       "runs": sum((e.get("cost") or {}).get("runs", 0)
                                   for e in events if e.get("type") == "cost")},
    }, ensure_ascii=False, separators=(",", ":"), default=str))


if __name__ == "__main__":
    asyncio.run(replay())
