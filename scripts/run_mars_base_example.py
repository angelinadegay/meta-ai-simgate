"""Run the documented Mars base with configured models and save its real evidence."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import math
import socket
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simgate import config
from simgate.orchestrator import Session
from simgate.spec import Robot
from simgate.backends.mujoco_backend import support_height_offset

PROMPT = ("Build a Mars base on Jezero: a habitat, a lander, and a bridge over a trench. "
          "Can the rover drive from the lander to the habitat?")


async def run_example(output: Path) -> dict:
    if config.MOCK_MODE:
        raise RuntimeError("Configure live models first; this example refuses Mock mode.")
    output.mkdir(parents=True, exist_ok=True)
    events, frames, costs = [], [], []

    async def emit(event):
        if event["type"] == "frames":
            frames.extend(event["frames"])
        elif event["type"] not in {"scene", "spec"}:
            events.append(event)
        if event["type"] == "cost":
            costs.append(event["cost"])
        if event["type"] in {"question", "error"}:
            print(event.get("message", ""), flush=True)
        if event["type"] == "agent" and event.get("agent") == "Simulator":
            print(event["message"], flush=True)

    session = Session(emit)
    started = time.perf_counter()
    print(f"Live planner: {config.PLANNER_MODEL}; builder mode: {config.BUILDER_MODE}", flush=True)
    await session.handle(PROMPT)
    requests = [PROMPT]
    if session.build is not None and session.last_summary is None:
        requests.append("run")
        await session.handle("run")
    report = {
        "prompt": PROMPT, "requests": requests, "mock_mode": config.MOCK_MODE,
        "planner_model": config.PLANNER_MODEL, "builder_mode": config.BUILDER_MODE,
        "configured_builder_model": config.BUILDER_MODEL,
        "wall_s": round(time.perf_counter() - started, 2), "costs": costs,
        "model_calls": sum(item["model_calls"] for item in costs),
        "runs": sum(item["runs"] for item in costs),
        "checks": [check.__dict__ for check in session.checks],
        "structures": session.build.scene.get("structures", []) if session.build else [],
        "metrics": session.last_summary, "frame_count": len(frames),
        "errors": [event["message"] for event in events if event["type"] == "error"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    if session.build is not None:
        replay = {"spec": session.spec.model_dump(mode="json"), "scene": session.build.scene,
                  "code": session.build.code, "frames": frames, "metrics": session.last_summary}
        (output / "replay.json").write_text(json.dumps(replay), encoding="utf-8")
    print(json.dumps({"model_calls": report["model_calls"], "runs": report["runs"],
                      "wall_s": report["wall_s"], "metrics": report["metrics"],
                      "structures": [{"kind": record["kind"], "passed": record["structural_check"]["passed"],
                                       "utilisation_pct": record["structural_check"]["utilisation_pct"]}
                                      for record in report["structures"]], "output": str(output)}, indent=2), flush=True)
    return report


def screenshot_frames(replay: dict) -> dict[str, int]:
    scene, frames = replay["scene"], replay["frames"]
    body = scene["follow_geom"]
    bridge = next(record for record in scene["structures"] if record["kind"] == "bridge")
    centre = [bridge["position"]["x"], bridge["position"]["y"]]
    crossing = min(range(len(frames)), key=lambda index: math.dist(frames[index]["poses"][body][:2], centre))
    arrived = replay["metrics"].get("time_to_goal_s")
    if not replay["metrics"].get("goal_reached") or arrived is None:
        raise RuntimeError("No recorded airlock arrival exists; an arrival screenshot must not be fabricated.")
    arrival = min(range(len(frames)), key=lambda index: abs(frames[index]["t"] - arrived))
    offset = support_height_offset(Robot.model_validate(replay["spec"]["robot"]))
    pose = frames[crossing]["poses"][body]
    yaw = math.radians(bridge["orientation"].get("yaw_deg", 0))
    dx, dy = pose[0] - centre[0], pose[1] - centre[1]
    along, across = dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)
    if abs(along) > bridge["dimensions"]["span_m"] / 2 or abs(across) > bridge["dimensions"]["deck_width_m"] / 2:
        raise RuntimeError("No recorded rover-on-bridge pose exists.")
    if abs(pose[2] - offset - (bridge["position"]["z"] + 0.03)) > 0.2:
        raise RuntimeError("The bridge screenshot pose is not at deck height.")
    goal_z = scene["mission"].get("goal_z")
    if goal_z is not None and abs(frames[arrival]["poses"][body][2] - offset - goal_z) > 0.17:
        raise RuntimeError("The reported airlock arrival pose is below or above the raised goal.")
    return {"overview": 0, "rover-on-bridge": crossing, "arriving-at-airlock": arrival}


def capture_screenshots(output: Path) -> None:
    from playwright.sync_api import sync_playwright

    replay = json.loads((output / "replay.json").read_text(encoding="utf-8"))
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    selected = screenshot_frames(replay)
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1",
                                "--port", str(port)], cwd=Path(__file__).resolve().parents[1],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in process.stdout:
            if "Uvicorn running" in line:
                break
            if "ERROR" in line:
                raise RuntimeError(line)
        with sync_playwright() as driver:
            browser = driver.chromium.launch(channel="msedge", headless=True,
                                             args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            for name, index in selected.items():
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                page = context.new_page()
                page.set_default_timeout(60_000)
                routes = []
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))

                def connect(route):
                    routes.append(route)
                    route.send(json.dumps({"type": "spec", "spec": replay["spec"]}))
                    route.send(json.dumps({"type": "scene", "scene": replay["scene"], "code": replay["code"]}))
                    route.send(json.dumps({"type": "checks", "checks": report["checks"]}))

                page.route_web_socket("**/ws", connect)
                page.goto(f"http://127.0.0.1:{port}/")
                page.wait_for_function("document.querySelectorAll('#structureCards article').length === 3")
                if any(record["dimensions"].get("visual_model") == "nasa_insight" for record in replay["scene"]["structures"]):
                    page.wait_for_function("Array.from(document.querySelectorAll('#structureCards a')).some(a=>a.textContent.includes('NASA InSight'))")
                if name != "overview":
                    routes[0].send(json.dumps({"type": "run_start"}))
                    page.wait_for_function("!document.getElementById('player').hidden")
                    page.locator("#play").click()
                    routes[0].send(json.dumps({"type": "frames", "frames": replay["frames"][:index + 1]}))
                    page.wait_for_function("Number(document.getElementById('scrub').max) > 0")
                    page.locator("#scrub").evaluate("(element,value)=>{element.value=value;element.dispatchEvent(new Event('input',{bubbles:true}));}", index)
                    page.evaluate("() => new Promise(resolve=>{let count=0;function frame(){if(++count===180)resolve();else requestAnimationFrame(frame);}requestAnimationFrame(frame);})")
                page.locator("#view").screenshot(path=str(output / f"{name}.png"))
                if errors:
                    raise RuntimeError("Browser errors: " + "; ".join(errors))
                context.close()
            browser.close()
        captures = {name: {"frame_index": index, "simulation_time_s": replay["frames"][index]["t"],
                          "rover_xyz_m": replay["frames"][index]["poses"][replay["scene"]["follow_geom"]][:3]}
                    for name, index in selected.items()}
        (output / "screenshots.json").write_text(json.dumps(captures, indent=2), encoding="utf-8")
        print(json.dumps(captures, indent=2), flush=True)
    finally:
        process.terminate()
        process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tests/screenshots/mars-base-live"))
    parser.add_argument("--screenshots-only", action="store_true", help="Render saved frames; no model calls or simulation reruns.")
    args = parser.parse_args()
    if args.screenshots_only:
        capture_screenshots(args.output)
        return 0
    report = asyncio.run(run_example(args.output))
    return 0 if report["metrics"] is not None and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())