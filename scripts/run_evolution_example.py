"""Run the Victoria Crater lander-limited evolution demo and capture its recorded 3D browser views."""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate_challenge import calibrate
from simgate import config, evolve
from simgate.orchestrator import Session


PROMPT = "Design a robot under 12 kg that can climb out of Victoria Crater."


async def run_example(output: Path, calibration_path: Path | None = None) -> dict:
    if config.MOCK_MODE or not config.BUILDER_MODEL.startswith("muse-"):
        raise RuntimeError("Configure the live Meta Designer first; this example refuses Mock mode.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path else calibrate(seed=0)
    )
    chosen = calibration["default_row"]
    evolve.DEFAULT_CHALLENGE_CONFIG.update({
        "rock_size_m": chosen["rock_size_m"], "rock_count": chosen["rock_count"],
        "friction": chosen["friction"],
        "design_constraints": chosen["design_constraints"],
    })
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print("| Trial | Gen 0 | Rock size | Density | Friction | Mass cap | Stowed size | Time limit | Reached | Best progress | Legal design climbs out? | Qualifies |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|:---:|")
    for row in calibration["rows"]:
        limits = row["design_constraints"]
        check = row.get("feasibility")
        feasible = "not checked" if check is None else ("yes" if check["feasible"] else "no")
        print(
            f"| {row['trial']} | {row['generation_zero']} | {row['rock_size_m']:.2f} m | "
            f"{row['rock_count']} rocks | {row['friction']:.2f} | ≤{limits['max_mass_kg']:g} kg "
            f"| {limits['max_size_m'][0]:g} × {limits['max_size_m'][1]:g} m | {row['duration_s']:.1f} s "
            f"| {'yes' if row['any_reached'] else 'no'} | {row['best_progress_percent']:.1f}% "
            f"| {feasible} | {'yes' if row['qualifies'] else 'no'} |",
            flush=True,
        )
    if not (calibration["qualified"] or calibration.get("rule_based_row")):
        raise RuntimeError("No feasible setting kept generation zero inside the no-reach, 20–80% target.")
    if not calibration["qualified"]:
        print("Model-designed generation zero already climbs out on the calibrated course; "
              "evolution will escalate the rocks to find its edge.", flush=True)
    attempts, reports = [], []
    for prompt in (PROMPT,):
        events = []

        async def emit(event):
            events.append(event)
            if event["type"] == "error":
                print(event["message"], flush=True)
            elif event["type"] == "agent" and event.get("agent") == "Simulator":
                print(event["message"], flush=True)

        session = Session(emit)
        print(f"Designer model: {config.BUILDER_MODEL}; prompt: {prompt}", flush=True)
        await session.handle(prompt)
        if session.last_evolution is None:
            errors = [event["message"] for event in events if event["type"] == "error"]
            raise RuntimeError("Evolution did not produce a report: " + "; ".join(errors))
        report = session.last_evolution
        report["session_cost"] = next(
            (event["cost"] for event in reversed(events) if event["type"] == "cost"), None
        )
        reports.append(report)
        attempts.append({
            "prompt": prompt, "challenge": report["challenge"],
            "success": report["success"], "generations": len(report["generations"]),
            "winner": (report.get("winner") or {}).get("id"),
            "model_calls": (report["session_cost"] or {}).get("model_calls", 0),
            "runs": (report["session_cost"] or {}).get("runs", 0),
        })
        slug = "hadley" if "Hadley" in report["challenge"] else "calibrated-default"
        attempt_dir = output / "attempts" / f"{len(attempts):02d}-{slug}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (attempt_dir / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        attempts[-1]["report_path"] = str((attempt_dir / "report.json").relative_to(output))
        attempts[-1]["events_path"] = str((attempt_dir / "events.json").relative_to(output))
        break
    report = reports[-1]
    report["attempted_challenges"] = attempts
    report["calibration"] = calibration
    report["total_demo_cost"] = {
        "model_calls": sum(attempt["model_calls"] for attempt in attempts),
        "runs": sum(attempt["runs"] for attempt in attempts) + calibration["simulations"],
        "calibration_runs": calibration["simulations"],
    }
    report["demo"] = {
        "prompt": PROMPT, "mock_mode": config.MOCK_MODE,
        "planner_model": config.PLANNER_MODEL, "builder_model": config.BUILDER_MODEL,
        "builder_mode": config.BUILDER_MODE, "wall_s": round(time.perf_counter() - started, 2),
        "attempts": attempts, "tree_generations": len(report["generations"]),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": report["report"], "success": report["success"], "generations": len(report["generations"]),
        "family_tree": report["family_tree"], "winner": report.get("winner"),
        "verification": report.get("verification"), "margin": report.get("margin"),
        "model_calls": report["total_demo_cost"]["model_calls"],
        "runs": report["total_demo_cost"]["runs"], "wall_s": report["demo"]["wall_s"],
        "attempted_challenges": attempts,
        "output": str(output),
    }, indent=2), flush=True)
    return report


def capture_screenshots(output: Path) -> None:
    from playwright.sync_api import sync_playwright

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    if not report.get("generations") or not report.get("challenge_scene") or not report.get("race_scene"):
        raise RuntimeError("The saved evolution report lacks race results or its replay scenes.")
    replay_generation = next(
        (generation for generation in reversed(report["generations"]) if generation.get("race_replay")),
        None,
    )
    if replay_generation is None:
        raise RuntimeError("The saved evolution report lacks a 3D race track.")
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in process.stdout:
            if "Uvicorn running" in line:
                break
            if "ERROR" in line:
                raise RuntimeError(line)
        output.mkdir(parents=True, exist_ok=True)
        errors = []
        with sync_playwright() as driver:
            browser = driver.chromium.launch(
                channel="msedge", headless=True,
                args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(f"{error}\n{error.stack}"))
            routes = []
            page.route_web_socket("**/ws", lambda route: routes.append(route))
            page.goto(f"http://127.0.0.1:{port}/")
            page.wait_for_function("document.querySelector('#log') !== null")
            page.wait_for_timeout(250)
            if not routes:
                raise RuntimeError("The browser did not open its replay websocket.")
            route = routes[0]
            route.send(json.dumps({"type": "spec", "spec": report["challenge_spec"]}))
            route.send(json.dumps({
                "type": "scene", "scene": report["challenge_scene"], "code": report.get("challenge_code", ""),
            }))
            route.send(json.dumps({
                "type": "agent", "agent": "Simulator", "status": "working",
                "message": f"Racing generation {replay_generation['generation']} on the challenge…",
            }))
            route.send(json.dumps({
                "type": "evolution_race_scene",
                "scene": replay_generation.get("replay_scene") or report["race_scene"],
                "generation": replay_generation["generation"],
            }))
            route.send(json.dumps({
                "type": "evolution_generation", "generation": replay_generation,
            }))
            page.locator("#evolutionCard").wait_for(state="visible")
            page.locator("#evolutionGenerations svg").wait_for(state="visible")
            page.locator("#player").wait_for(state="visible")
            page.locator("#scrub").evaluate(
                "(slider) => { slider.value = String(Math.floor(Number(slider.max) / 2)); "
                "slider.dispatchEvent(new Event('input', { bubbles: true })); }"
            )
            page.wait_for_timeout(500)
            clean_capture = page.add_style_tag(content="#topright, #robotCameras { visibility: hidden !important; }")
            page.locator("#view canvas").screenshot(path=str(output / "race-3d-midrun.png"))
            clean_capture.evaluate("(style) => style.remove()")
            page.locator("#evolutionCard").screenshot(path=str(output / "generation-race.png"))

            has_children = bool(report.get("family_tree"))
            children = [design for generation in report["generations"] for design in generation["results"]]
            if has_children:
                route.send(json.dumps({
                    "type": "evolution_tree", "family_tree": report["family_tree"], "designs": children,
                }))
                page.locator("#evolutionTree .evolution-node").first.wait_for(state="visible")
            else:
                route.send(json.dumps({"type": "evolution", "report": report}))
                page.wait_for_function(
                    "document.querySelector('#evolutionTree')?.textContent.includes('no children were needed')"
                )
            page.locator("#evolutionTree").screenshot(path=str(output / "family-tree.png"))

            if has_children:
                route.send(json.dumps({"type": "evolution", "report": report}))
            if report.get("winner"):
                page.locator("#evolutionWinner .winner").wait_for(state="visible")
            else:
                page.locator("#evolutionWinner p").wait_for(state="visible")
            card_only = page.add_style_tag(content=(
                "#evolutionGenerations, #evolutionTree, #evolutionCard h4, #evolutionLive "
                "{ display: none !important; }"
            ))
            page.locator("#evolutionCard").evaluate("(card) => card.scrollIntoView({block: 'start'})")
            page.locator("#evolutionCard").screenshot(path=str(output / "winner-card.png"))
            card_only.evaluate("(style) => style.remove()")
            page.wait_for_function("localStorage.getItem('simgate.savedEvolution') !== null")
            page.locator("#replayEvolution").click()
            page.wait_for_function("document.querySelector('#statusText')?.textContent.includes('Saved replay')")
            page.locator("#evolutionGenerations button").last.click()
            page.locator("#player").wait_for(state="visible")
            page.wait_for_timeout(300)
            if errors:
                raise RuntimeError("Browser errors: " + "; ".join(errors))
            browser.close()
        (output / "screenshots.json").write_text(json.dumps({
            "generation_3d": "race-3d-midrun.png", "generation": "generation-race.png",
            "family_tree": "family-tree.png",
            "winner": "winner-card.png", "telemetry_source": "report.json",
            "tree_generations": len(report.get("generations", [])),
        }, indent=2), encoding="utf-8")
    finally:
        process.terminate()
        process.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/evolution-e8-live"))
    parser.add_argument("--calibration", type=Path,
                        help="Reuse a saved calibration JSON instead of rerunning its simulations.")
    parser.add_argument("--screenshots-only", action="store_true",
                        help="Render saved race telemetry; do not call models or run simulations.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.screenshots_only:
        capture_screenshots(args.output)
    else:
        asyncio.run(run_example(args.output, args.calibration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
