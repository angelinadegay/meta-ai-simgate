"""Parallel execution of robot-scene variations."""
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import base64
import json
import os
import time
from typing import Any, Callable, Literal
import zlib

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .spec import Disturbance, SceneSpec

RUN_TIMEOUT_SECONDS = 120.0
MAX_BATCH_RUNS = 24
MAX_BATCH_GHOST_BYTES = 4 * 1024 * 1024
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
    # Single-change experiments (Prove why) and design changes (Make it pass).
    remove_obstacle: str | None = Field(default=None, pattern=r"^obs\d+_[a-z]+$")
    no_push: bool | None = None
    speed_scale: float | None = Field(default=None, gt=0, le=2)
    face_uphill: bool | None = None
    flatten_terrain: bool | None = None
    robot: dict[str, Any] | None = None
    robot_id: str | None = None      # robot-level knobs apply to this robot only (default: every robot)


GEOMETRY_KNOBS = ("slope_deg", "obstacle_scale", "seed", "remove_obstacle", "flatten_terrain", "robot")


def obstacle_entry_index(spec: SceneSpec, name: str) -> int:
    """The spec.obstacles entry behind a placed obstacle name such as "obs1_step" (same expansion as the course
    planner: a rock/boulder/log field is one slot; other kinds repeat once per count)."""
    slot = int(name.removeprefix("obs").split("_", 1)[0])
    seen = 0
    for index, obstacle in enumerate(spec.obstacles):
        field = obstacle.kind in ("rock", "boulder", "log") and obstacle.placement != "scattered"
        seen += 1 if field else obstacle.count
        if slot < seen:
            return index
    raise ValueError(f"No obstacle named {name} in this scene.")


def _target_entries(spec: SceneSpec, robot_id: str | None) -> list:
    entries = [entry for entry in spec.robots if robot_id is None or entry.id == robot_id]
    if not entries:
        raise ValueError(f"Unknown robot id: {robot_id}")
    return entries


def _update_robots(spec: SceneSpec, robot_id: str | None, change) -> None:
    """Apply change(robot) -> robot to the targeted entries, keeping the legacy single robot in step."""
    for entry in _target_entries(spec, robot_id):
        entry.robot = change(entry.robot)
        if spec.robot is not None and entry is spec.robots[0]:
            spec.robot = entry.robot.model_copy(deep=True)


def _validate_variation(base_spec: SceneSpec, raw: dict) -> tuple[Variation, SceneSpec, dict]:
    variation = Variation.model_validate(raw)
    knobs = variation.model_dump(exclude={"label"}, exclude_unset=True)
    target = knobs.get("robot_id")
    if not set(knobs) - {"robot_id"}:
        raise ValueError("A variation must change at least one condition.")
    spec = base_spec.model_copy(deep=True)
    if spec.domain != "robot_terrain" or spec.robot is None:
        raise ValueError("Stress tests currently support robot scenes only.")
    if target is not None:
        _target_entries(spec, target)
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
    if knobs.get("no_push"):
        if spec.disturbance is None:
            raise ValueError("no_push requires a push in the scene.")
        spec.disturbance = None
    if "remove_obstacle" in knobs:
        index = obstacle_entry_index(spec, knobs["remove_obstacle"])
        obstacle = spec.obstacles[index]
        field = obstacle.kind in ("rock", "boulder", "log") and obstacle.placement != "scattered"
        if obstacle.count > 1 and not field:
            obstacle.count -= 1
        else:
            del spec.obstacles[index]
    if knobs.get("flatten_terrain"):
        if spec.terrain is None or spec.terrain.kind not in ("hills", "rocky", "craters"):
            raise ValueError("flatten_terrain applies to generated hills, rocky or crater terrain.")
        spec.terrain.kind = "flat"
    if "speed_scale" in knobs:
        scale = knobs["speed_scale"]

        def slower(robot):
            if robot.kind == "rover":
                return robot.model_copy(update={"drive_speed": robot.drive_speed * scale})
            if robot.kind == "walker":
                return robot.model_copy(update={"gait_freq_hz": min(max(robot.gait_freq_hz * scale, 0.8), 3.0)})
            return robot.model_copy(update={"command_speed_m_s": min(max(robot.command_speed_m_s * scale, 0.1), 1.0)})

        _update_robots(spec, target, slower)
    if "robot" in knobs:
        from .spec import Robot

        changes = knobs["robot"]
        not_allowed = (set(changes) - set(Robot.model_fields)) | ({"kind", "robot_id"} & set(changes))
        if not_allowed:
            raise ValueError(f"Design changes must be robot settings; not allowed: {', '.join(sorted(not_allowed))}")
        _update_robots(spec, target, lambda robot: Robot.model_validate({**robot.model_dump(), **changes}))
    if knobs.get("face_uphill"):
        _face_uphill(spec, target)
    spec.assumptions = []
    spec.fill_defaults()
    changes = {key: value for key, value in knobs.items()}
    return variation, spec, changes


def _face_uphill(spec: SceneSpec, robot_id: str | None) -> None:
    """Point the robot straight up a slope (uphill is +x); a goal moves to straight uphill at the same distance."""
    import math

    from .backends.mujoco_backend import default_start, make_heights, place_xy

    if spec.terrain is None or spec.terrain.kind != "slope":
        raise ValueError("face_uphill applies to slope terrain.")
    heights = None
    for entry in _target_entries(spec, robot_id):
        if entry.start_place is not None:
            heights = heights if heights is not None else make_heights(spec)[0]
            start = place_xy(heights, spec.terrain.size_m, entry.start_place)
        else:
            start = tuple(entry.start_xy) if entry.start_xy is not None else default_start(spec.terrain.size_m)
        if entry.goal_xy is not None or entry.goal_place is not None:
            if entry.goal_xy is not None:
                goal = entry.goal_xy
            else:
                heights = heights if heights is not None else make_heights(spec)[0]
                goal = place_xy(heights, spec.terrain.size_m, entry.goal_place)
            distance = math.dist(start, goal)
            limit = spec.terrain.size_m / 2 * 0.9
            entry.goal_xy, entry.goal_place = [round(min(start[0] + distance, limit), 2), round(start[1], 2)], None
        entry.start_xy, entry.start_place, entry.start_heading_deg = [round(start[0], 2), round(start[1], 2)], None, 0.0
    if spec.mission is not None and spec.robots:
        first = spec.robots[0]
        spec.mission.start_xy, spec.mission.start_place = first.start_xy, None
        spec.mission.start_heading_deg = first.start_heading_deg
        if first.goal_xy is not None:
            spec.mission.goal_xy, spec.mission.goal_place = first.goal_xy, None


def _result_error(run_id: int, label: str, changes: dict, message: str) -> dict:
    geometry_changed = any(key in changes for key in GEOMETRY_KNOBS)
    return {
        "id": f"run-{run_id}",
        "label": label,
        "changes": changes,
        "summary": {},
        "passed": False,
        "reason": "error",
        "ghost": None,
        "ghost_status": "not in ghost view" if geometry_changed else "unavailable",
        "error": message,
    }


def _robot_geom_ids(code: str) -> tuple[list[int], int]:
    import mujoco

    model = mujoco.MjModel.from_xml_string(code)
    robot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rover")
    if robot < 0:
        raise ValueError("The simulation has no robot body named 'rover'.")
    geom_ids = [
        geom for geom in range(model.ngeom)
        if int(model.body_rootid[int(model.geom_bodyid[geom])]) == robot
    ]
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
    if not geom_ids or chassis not in geom_ids:
        raise ValueError("The simulation has no robot geoms to record.")
    return geom_ids, geom_ids.index(chassis)


def _stopped_time(frames: list[list], chassis_index: int, outcome: str | None) -> float | None:
    if outcome != "stuck" or len(frames) < 2:
        return None
    positions = [frame[1][chassis_index][:2] for frame in frames]
    window = 45
    for start in range(min(8, len(frames) - 1), max(1, len(frames) - window)):
        end = min(start + window, len(frames) - 1)
        if end - start >= 30 and sum((a - b) ** 2 for a, b in zip(positions[end], positions[start])) ** 0.5 < 0.3:
            return frames[start][0]
    return max(0.0, frames[-1][0] - 3.0)


def _encode_ghost(geom_ids: list[int], frames: list[list]) -> dict:
    raw = json.dumps(frames, separators=(",", ":"), allow_nan=False).encode("utf-8")
    packed = zlib.compress(raw, level=9)
    return {
        "encoding": "deflate-base64",
        "geom_ids": geom_ids,
        "frame_count": len(frames),
        "data": base64.b64encode(packed).decode("ascii"),
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
        ghost_allowed = not any(key in changes for key in GEOMETRY_KNOBS)
        ghost_ids, chassis_index = _robot_geom_ids(build.code) if ghost_allowed else ([], -1)
        ghost_frames = []
        frame_number = 0
        for chunk in sim.run(spec, build):
            for frame in chunk:
                if ghost_allowed and frame_number % 2 == 0:
                    robot_poses = [
                        [round(float(value), 3) for value in frame["poses"][geom_id]]
                        for geom_id in ghost_ids
                    ]
                    ghost_frames.append([round(float(frame["t"]), 3), robot_poses])
                frame_number += 1
        summary = sim.summary()
        stopped_at = _stopped_time(ghost_frames, chassis_index, summary.get("outcome")) if ghost_allowed else None
        if stopped_at is not None:
            summary["stopped_making_progress_at_s"] = stopped_at
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
            "ghost": _encode_ghost(ghost_ids, ghost_frames) if ghost_allowed else None,
            "ghost_status": "available" if ghost_allowed else "not in ghost view",
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
    baseline: bool = True,
) -> list[dict]:
    """Run an unchanged baseline (unless ``baseline`` is False) and each valid variation in separate
    worker processes.

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
    ghost_bytes = 0

    def store(result: dict) -> None:
        nonlocal ghost_bytes
        if result.get("ghost") is not None:
            size = len(result["ghost"]["data"].encode("ascii")) + 256
            if ghost_bytes + size > MAX_BATCH_GHOST_BYTES:
                result["ghost"] = None
                result["ghost_status"] = "not in ghost view (batch payload limit)"
            else:
                ghost_bytes += size
        run_id = int(result["id"].removeprefix("run-"))
        results[run_id] = result
        _notify(on_result, result)

    try:
        worker_count = min(
            max(len(raw_variations) + int(baseline), 1),
            max(1, (os.cpu_count() or 1) - 1),
        )
        executor = ProcessPoolExecutor(max_workers=worker_count)
        jobs = [(0, baseline_label, None)] if baseline else []
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
