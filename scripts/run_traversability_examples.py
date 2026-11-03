"""Run the requested examples using live configured Meta models and save measured reports."""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simgate import config
from simgate.orchestrator import Session


PROMPTS = [
    ("mars-g1", "Where on Gusev can the G1 walk safely? Plan a safe route to the summit and walk it."),
    ("apartment-rover", "In the apartment, can the rover get from the kitchen to the bedroom? Show where it can go."),
    ("apartment-g1", "Same question for the G1."),
    ("apartment-rover-two-leg", "In the apartment, can the rover go from the kitchen to the living room, then to the bedroom?"),
]


async def run(output):
    if config.MOCK_MODE or not config.PLANNER_MODEL.startswith("muse-"):
        raise RuntimeError("These examples require the configured live Meta models; Mock mode is refused.")
    output.mkdir(parents=True, exist_ok=True)
    events = []

    async def emit(event):
        events.append(event)
        if event["type"] in ("error", "cost", "traversability"):
            print(event.get("message") or event.get("cost") or event["report"]["summary"], flush=True)

    session = Session(emit)
    summaries = []
    for index, (name, prompt) in enumerate(PROMPTS):
        if index in (1, 3):
            session = Session(emit)
        events.clear()
        print(f"RUN {name}: {prompt}", flush=True)
        await session.handle(prompt)
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
        summaries.append({
            "name": name, "prompt": prompt, "summary": report["summary"], "costs": costs,
            "planner_model": config.PLANNER_MODEL, "builder_model": config.BUILDER_MODEL,
            "builder_mode": config.BUILDER_MODE, "mock_mode": config.MOCK_MODE,
            "percentages": report["percentages"], "agreement": report["agreement"],
            "corrections": [spot for spot in report["spot_checks"] if spot["corrected"]],
            "routes": [{"label": leg.get("label", "Start to goal"), "text": leg["text"]}
                       for leg in report["comparison"].get("legs", [report["comparison"]])],
        })
        (output / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(run(Path("reports") / "traversability-live"))
