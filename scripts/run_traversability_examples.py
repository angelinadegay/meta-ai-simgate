"""Run the requested examples using live configured Meta models and save measured reports.

python scripts/run_traversability_examples.py [name ...]   (default: every example)
"""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simgate import config
from simgate.orchestrator import Session


# (name, prompt, start a new session first)
PROMPTS = [
    ("mars-rover", "Where on Gusev can the rover drive safely? Plan a safe route to the summit and drive it.", True),
    ("apartment-rover", "In the apartment, can the rover get from the kitchen to the bedroom?", True),
    ("apartment-walker", "Same question for the walker.", False),
    ("apartment-rover-two-leg", "In the apartment, can the rover go from the kitchen to the living room, then to the bedroom?",
     True),
]
# The standing-only G1 (no trained policy): a scene on Gusev, then the standing question (Finder's standing preset).
STANDING = ("mars-g1-standing", "Put the G1 humanoid on Gusev Crater.", "Can the humanoid stand safely on this Mars slope?")


async def run(output, names=None):
    if config.MOCK_MODE or not config.PLANNER_MODEL.startswith("muse-"):
        raise RuntimeError("These examples require the configured live Meta models; Mock mode is refused.")
    output.mkdir(parents=True, exist_ok=True)
    events = []

    async def emit(event):
        events.append(event)
        if event["type"] in ("error", "cost", "traversability"):
            print(event.get("message") or event.get("cost") or event["report"]["summary"], flush=True)

    summary_path = output / "summary.json"
    summaries = {row["name"]: row for row in json.loads(summary_path.read_text(encoding="utf-8"))} \
        if summary_path.exists() else {}
    needed = set(names or (name for name, _, _ in PROMPTS))
    for index, (name, _, fresh) in enumerate(PROMPTS):   # a "same question" example needs the one before it
        if name in needed and not fresh:
            needed.add(PROMPTS[index - 1][0])
    session = Session(emit)
    for name, prompt, fresh in PROMPTS:
        if fresh:
            session = Session(emit)
        if name not in needed:
            continue
        events.clear()
        print(f"RUN {name}: {prompt}", flush=True)
        await session.handle(prompt)
        if names and name not in names:
            continue
        folder = output / name
        folder.mkdir(exist_ok=True)
        (folder / "events.json").write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
        report = session.last_map_report
        if report is None:
            raise RuntimeError(f"No map produced for {name}: {[e for e in events if e['type'] == 'error']}")
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        costs = [event["cost"] for event in events if event["type"] == "cost"]
        replay = {
            "spec": session.spec.model_dump(mode="json"), "scene": session.build.scene,
            "code": session.build.code, "report": report, "cost": costs[-1] if costs else report["cost"],
        }
        (folder / "replay.json").write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
        # the route runs, recorded once (scene, frames, metrics) so a saved demo replays them with zero simulation
        route_runs = {}
        for route_run in report["comparison"]["runs"]:
            events.clear()
            await session.replay_map_run(route_run["id"])
            route_runs[route_run["id"]] = [event for event in events if event["type"] != "cost"]
        (folder / "route-runs.json").write_text(json.dumps(route_runs, ensure_ascii=False), encoding="utf-8")
        summaries[name] = {
            "name": name, "prompt": prompt, "summary": report["summary"], "costs": costs,
            "planner_model": config.PLANNER_MODEL, "builder_model": config.BUILDER_MODEL,
            "builder_mode": config.BUILDER_MODE, "mock_mode": config.MOCK_MODE,
            "percentages": report["percentages"], "agreement": report["agreement"],
            "corrections": [spot for spot in report["spot_checks"] if spot["corrected"]],
            "routes": [{"label": leg.get("label", "Start to goal"), "text": leg["text"]}
                       for leg in report["comparison"].get("legs", [report["comparison"]])],
        }
        summary_path.write_text(json.dumps(list(summaries.values()), indent=2, ensure_ascii=False), encoding="utf-8")
    if not names or STANDING[0] in names:
        await run_standing(output, emit, events)


async def run_standing(output, emit, events):
    name, build_prompt, question = STANDING
    session = Session(emit)
    events.clear()
    print(f"RUN {name}: {build_prompt} / {question}", flush=True)
    await session.handle(build_prompt)
    await session.handle(question)
    report = session.last_finder_report
    if report is None:
        raise RuntimeError(f"No standing limits for {name}: {[e for e in events if e['type'] == 'error']}")
    folder = output / name
    folder.mkdir(exist_ok=True)
    (folder / "events.json").write_text(json.dumps(events, ensure_ascii=False, default=str), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    (folder / "spec.json").write_text(json.dumps(session.spec.model_dump(mode="json"), ensure_ascii=False),
                                      encoding="utf-8")
    print(report.get("headline"), flush=True)


if __name__ == "__main__":
    asyncio.run(run(Path("reports") / "traversability-live", set(sys.argv[1:]) or None))
