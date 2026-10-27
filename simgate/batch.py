"""Parallel execution of robot-scene variations."""
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import os
import time
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .spec import Disturbance, SceneSpec

RUN_TIMEOUT_SECONDS = 120.0
MAX_BATCH_RUNS = 24
Surface = Literal["rock", "regolith", "gravel", "sand", "mud", "ice"]


class Variation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1, max_length=80)
    gravity: float | None = Field(default=None, gt=0, le=30)
    surface: Surface | None = None
    friction: float | None = Field(default=None, ge=0, le=2)
    payload_kg: float | None = Field(default=None, ge=0, le=1000)
    slope_deg: float | None = Field(default=None, ge=-89, le=89)
    obstacle_scale: float | None = Field(default=None, gt=0, le=10)
    push_n: float | None = Field(default=None, gt=0, le=1000)
    seed: int | None = Field(default=None, ge=0)


def _validate_variation(base_spec: SceneSpec, raw: dict) -> tuple[Variation, SceneSpec, dict]:
    variation = Variation.model_validate(raw)
    knobs = variation.model_dump(exclude={"label"}, exclude_unset=True)
    if not knobs:
        raise ValueError("A variation must change at least one condition.")
    spec = base_spec.model_copy(deep=True)
    if spec.domain != "robot_terrain" or spec.robot is None:
        raise ValueError("Stress tests currently support robot scenes only.")
    if "surface" in knobs:
        spec.environment.surface = knobs["surface"]
        spec.environment.friction = None
    if "gravity" in knobs:
        spec.environment.gravity = knobs["gravity"]
    if "friction" in knobs:
        spec.environment.friction = knobs["friction"]
    if "payload_kg" in knobs:
        spec.robot.mass_kg += knobs["payload_kg"]
    if "slope_deg" in knobs:
        if spec.terrain is None or spec.terrain.kind != "slope":
            raise ValueError("slope_deg only applies to slope terrain.")
        spec.terrain.slope_deg = knobs["slope_deg"]
    if "obstacle_scale" in knobs:
        if not spec.obstacles:
            raise ValueError("obstacle_scale requires at least one obstacle.")
        from .backends.obstacles import SIZES

        for obstacle in spec.obstacles:
            height = obstacle.height_m if obstacle.height_m is not None else SIZES[obstacle.kind][obstacle.size]
            obstacle.height_m = height * knobs["obstacle_scale"]
    if "seed" in knobs:
        if spec.terrain is None:
            raise ValueError("seed requires terrain.")
        spec.terrain.seed = knobs["seed"]
    if "push_n" in knobs:
        spec.disturbance = Disturbance(push_n=knobs["push_n"])
    spec.assumptions = []
    spec.fill_defaults()
    changes = {key: value for key, value in knobs.items()}
    return variation, spec, changes


def _result_error(run_id: int, label: str, changes: dict, message: str) -> dict:
    return {
        "id": f"run-{run_id}",
        "label": label,
        "changes": changes,
        "summary": {},
        "passed": False,
        "reason": "error",
        "ghost": None,
        "error": message,
    }


def _run_worker(run_id: int, label: str, base_spec_json: str, raw_variation: dict | None) -> dict:
    changes = {key: value for key, value in raw_variation.items() if key != "label"} if raw_variation else {}
    try:
        from . import agents

        base_spec = SceneSpec.model_validate_json(base_spec_json)
        if raw_variation is None:
            spec = base_spec.model_copy(deep=True)
            spec.assumptions = []
            spec.fill_defaults()
        else:
            _, spec, changes = _validate_variation(base_spec, raw_variation)
        if spec.domain != "robot_terrain":
            raise ValueError("Stress tests currently support robot scenes only.")
        sim = agents.route(spec)
        build = sim.build(spec)
        for _ in sim.run(spec, build):
            pass
        summary = sim.summary()
        has_goal = spec.mission is not None and spec.mission.has_goal()
        passed = bool(summary.get("goal_reached")) if has_goal else (
            not summary.get("flipped", False) and summary.get("forward_progress_m", 0) >= 1.0
        )
        reason = summary.get("outcome") or (
            "fell" if summary.get("flipped") else
            "made progress" if passed else "made little progress"
        )
        if summary.get("stopped_by"):
            reason += f"; stopped by {summary['stopped_by'].get('kind', 'obstacle')}"
        return {
            "id": f"run-{run_id}",
            "label": label,
            "changes": changes,
            "summary": summary,
            "passed": passed,
            "reason": reason,
            "ghost": None,
        }
    except Exception as exc:
        return _result_error(run_id, label, changes, f"{type(exc).__name__}: {exc}")


def _notify(on_result: Callable[[dict], None] | None, result: dict) -> None:
    if on_result is not None:
        on_result(result)


def run_batch(
    base_spec: SceneSpec,
    variations: list[dict],
    on_result: Callable[[dict], None] | None = None,
    timeout: float = RUN_TIMEOUT_SECONDS,
) -> list[dict]:
    """Run an unchanged baseline and each valid variation in separate worker processes.

    Results are returned in baseline/input order and also sent to ``on_result`` as
    workers finish. Invalid variations become error rows without aborting the batch.
    """
    raw_variations = list(variations)
    overflow = len(raw_variations) + 1 > MAX_BATCH_RUNS
    if overflow:
        raw_variations = raw_variations[: MAX_BATCH_RUNS - 2]

    results: dict[int, dict] = {}
    baseline_label = "Baseline"
    base_json = base_spec.model_dump_json()
    executor: ProcessPoolExecutor | None = None
    running: dict[Future, tuple[int, str, float]] = {}
    requested_changes: dict[int, dict] = {}

    def store(result: dict) -> None:
        run_id = int(result["id"].removeprefix("run-"))
        results[run_id] = result
        _notify(on_result, result)

    try:
        worker_count = min(
            len(raw_variations) + 1,
            max(1, (os.cpu_count() or 1) - 1),
        )
        executor = ProcessPoolExecutor(max_workers=worker_count)
        jobs = [(0, baseline_label, None)]
        for run_id, raw in enumerate(raw_variations, start=1):
            label = raw.get("label") if isinstance(raw, dict) else None
            changes = {key: value for key, value in raw.items() if key != "label"} if isinstance(raw, dict) else {}
            requested_changes[run_id] = changes
            if not isinstance(label, str) or not label.strip():
                label = f"Variation {run_id}"
            try:
                Variation.model_validate(raw)
                jobs.append((run_id, label, raw))
            except (ValidationError, TypeError, ValueError) as exc:
                store(_result_error(run_id, label, changes, f"Invalid variation: {exc}"))

        for run_id, label, raw in jobs:
            future = executor.submit(_run_worker, run_id, label, base_json, raw)
            running[future] = (run_id, label, time.monotonic() + timeout)

        while running:
            now = time.monotonic()
            next_deadline = min(deadline for _, _, deadline in running.values())
            completed, _ = wait(
                running,
                timeout=max(0.0, next_deadline - now),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                run_id, label, _ = running.pop(future)
                try:
                    store(future.result())
                except Exception as exc:
                    store(_result_error(run_id, label, requested_changes.get(run_id, {}),
                                        f"{type(exc).__name__}: {exc}"))

            now = time.monotonic()
            expired = [
                (future, run_id, label)
                for future, (run_id, label, deadline) in running.items()
                if deadline <= now
            ]
            if expired:
                for future, run_id, label in expired:
                    running.pop(future)
                    store(_result_error(run_id, label, requested_changes.get(run_id, {}),
                                        f"Timed out after {timeout:g} seconds."))
                for future, (run_id, label, _) in list(running.items()):
                    running.pop(future)
                    store(_result_error(run_id, label, requested_changes.get(run_id, {}),
                                        "Worker stopped after another run timed out."))
                    future.cancel()
                processes = list((executor._processes or {}).values())
                for process in processes:
                    process.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
                executor = None
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if overflow:
        cap_id = MAX_BATCH_RUNS - 1
        store(_result_error(cap_id, "Run limit", {}, f"Batch capped at {MAX_BATCH_RUNS} runs; extra variations were not started."))

    return [results[key] for key in sorted(results)]
