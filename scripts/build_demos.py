#!/usr/bin/env python3
"""Build the saved demo runs (demos/*.json.gz and demos/index.json) from the live reports in reports/.

python scripts/build_demos.py

Each demo is a recording made once with the Meta models; the app replays it with zero simulations and zero model
calls, so it works in mock mode too. Rebuild after rerunning a demo script:
  scripts/run_traversability_examples.py mars-rover apartment-rover   (Gusev safe route, apartment)
  scripts/run_evolution_example.py                                     (crater evolution)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "demos"
TRAVERSABILITY = [   # (demo id, title, saved example name)
    ("gusev-safe-route", "Gusev safe route", "mars-rover"),
    ("apartment-traversability", "Apartment: kitchen to bedroom", "apartment-rover"),
]
EVOLUTION = ("crater-evolution", "Crater evolution", "evolution-e8-live")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(demo_id: str, payload: dict) -> tuple[str, int]:
    OUT.mkdir(exist_ok=True)
    name = f"{demo_id}.json.gz"
    data = gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9, mtime=0)
    (OUT / name).write_bytes(data)
    return name, len(data)


def traversability_demo(demo_id: str, title: str, example: str) -> dict:
    folder = REPORTS / "traversability-live" / example
    summary = next(row for row in _read(REPORTS / "traversability-live" / "summary.json") if row["name"] == example)
    if summary.get("mock_mode"):
        raise RuntimeError(f"{example} was recorded in mock mode; demos are recorded with the Meta models.")
    route_runs = folder / "route-runs.json"
    if not route_runs.is_file():
        raise RuntimeError(f"{example} has no recorded route runs; rerun scripts/run_traversability_examples.py.")
    events = [event for event in _read(folder / "events.json") if event["type"] != "cost"]
    file, size = _write(demo_id, {"events": events, "route_runs": _read(route_runs)})
    return {"id": demo_id, "title": title, "kind": "traversability", "prompt": summary["prompt"], "file": file,
            "size_bytes": size, "recorded": {"planner_model": summary["planner_model"],
                                             "builder_model": summary["builder_model"], "mock_mode": False,
                                             "cost": summary["costs"][-1]}}


def evolution_demo(demo_id: str, title: str, folder_name: str) -> dict:
    report = _read(REPORTS / folder_name / "report.json")
    demo = report["demo"]
    if demo.get("mock_mode"):
        raise RuntimeError(f"{folder_name} was recorded in mock mode; demos are recorded with the Meta models.")
    for generation in report["generations"]:   # each generation repeats the race scene; the replay falls back to it
        if generation.get("replay_scene") == report.get("race_scene"):
            generation.pop("replay_scene")
    file, size = _write(demo_id, {"report": report})
    return {"id": demo_id, "title": title, "kind": "evolution", "prompt": demo["prompt"], "file": file,
            "size_bytes": size, "recorded": {"planner_model": demo["planner_model"],
                                             "builder_model": demo["builder_model"], "mock_mode": False,
                                             "cost": report["cost"]}}


def main() -> int:
    index = [evolution_demo(*EVOLUTION)] + [traversability_demo(*demo) for demo in TRAVERSABILITY]
    (OUT / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for demo in index:
        print(f"{demo['id']}: {demo['size_bytes'] / 1e6:.2f} MB ({demo['prompt']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
