"""Bounded generation-zero robot designs and same-challenge generation races."""

import json
import logging
import math
from typing import Any

import numpy as np

from . import config, llm
from .spec import Environment, Mission, Robot, SceneSpec, Terrain

logger = logging.getLogger(__name__)

DESIGN_SPACE = {
    "rover": {
        "wheels": (4, 6), "wheel_radius": (0.10, 0.40), "wheel_width": (0.05, 0.20),
        "chassis_length": (0.4, 1.2), "chassis_width": (0.3, 0.9),
        "mass_kg": (5.0, 40.0), "drive_speed": (3.0, 12.0),
    },
    "walker": {
        "legs": (4, 6), "leg_length": (0.12, 0.40), "gait_freq_hz": (1.0, 3.0),
        "stride_deg": (8.0, 25.0), "chassis_length": (0.4, 1.2),
        "chassis_width": (0.3, 0.9), "mass_kg": (5.0, 40.0),
    },
}
COLORS = ("#40d8a0", "#f2bc55", "#718cff", "#ef829f",
          "#65cad4", "#bc8bea", "#e59354", "#9eb962")
RACE_STUCK_WINDOW_S = 5.0
RACE_NO_PROGRESS_M = 0.1
RACE_HIGH_SLIP_RATIO = 0.5

SYSTEM = """You are the Designer proposing generation-zero procedural robots for a place.
Propose exactly the requested number, including at least one rover and one walker.
G1 and all library robots are NOT designable. Change only robot parameters; never the place,
gravity, friction, terrain, start or goal. These are untested proposals, not measured winners.
Use FACTS only: null rock sizes mean unknown, not zero; do not invent terrain measurements.
DESIGN_SPACE lists numeric ranges, except wheels and legs which are discrete choices 4 or 6.
Use only the knobs for that kind. Vary designs rather than repeating them.
Return JSON only: {"designs":[{"name":"short name","reason":"one-line design rationale",
"robot":{"kind":"rover or walker","<allowed knob>":0}}]}."""


def place_facts(challenge: SceneSpec) -> dict[str, Any]:
    """Measure relief/slope without running a robot; read rock sizes only from terrain metadata."""
    from .backends.mujoco_backend import make_heights
    from .terrain_data import load_preset

    scene = challenge.model_copy(deep=True).fill_defaults()
    heights, _ = make_heights(scene)
    spacing = float(scene.terrain.size_m) / (heights.shape[0] - 1)
    dy, dx = np.gradient(heights, spacing, spacing)
    slopes = np.degrees(np.arctan(np.hypot(dx, dy)))
    meta = load_preset(scene.terrain.preset)[1] if scene.terrain.kind == "real" else {}
    rocks = meta.get("rock_sizes_m")
    if rocks is not None and (not isinstance(rocks, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or value < 0 for value in rocks
    )):
        raise ValueError("Terrain card rock_sizes_m must be a list of finite nonnegative metres.")
    if not np.isfinite(heights).all():
        raise ValueError("The challenge terrain contains non-finite heights.")
    return {
        "place": meta.get("label", scene.terrain.preset or scene.terrain.kind),
        "body": scene.environment.body, "gravity_m_s2": scene.environment.gravity,
        "surface": scene.environment.surface, "friction": scene.environment.friction,
        "relief_m": float(np.ptp(heights)), "p95_slope_deg": float(np.percentile(slopes, 95)),
        "height_facts_source": "sampled scene terrain",
        "rock_sizes_m": rocks,
        "rock_sizes_source": "terrain card" if rocks is not None else "not measured in terrain card",
    }


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or len(value.splitlines()) != 1:
        raise ValueError(f"{label} must be a nonempty single line of at most {maximum} characters.")
    return value.strip()


def _validate_designs(reply: dict, n: int) -> list[dict]:
    rows = reply.get("designs")
    if not isinstance(rows, list) or len(rows) != n:
        raise ValueError(f"Designer must return exactly {n} designs.")
    designs, seen, kinds = [], set(), set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("robot"), dict):
            raise ValueError("Each design requires a robot object.")
        raw = row["robot"]
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in DESIGN_SPACE:
            raise ValueError("Only procedural rover and walker designs are allowed; G1 is not designable.")
        bounds = DESIGN_SPACE[kind]
        extra = set(raw) - {"kind", *bounds}
        if extra:
            raise ValueError(f"Not design knobs for {kind}: {', '.join(sorted(extra))}.")
        values, notes = {}, []
        defaults = Robot(kind=kind)
        for key, (low, high) in bounds.items():
            value = raw.get(key, getattr(defaults, key))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number.")
            if key in ("wheels", "legs"):
                if value not in (low, high):
                    raise ValueError(f"{key} must be 4 or 6.")
                values[key] = int(value)
            else:
                values[key] = min(high, max(low, float(value)))
                if values[key] != value:
                    notes.append(f"{key} clamped from {value:g} to {values[key]:g}")
        fingerprint = (kind, *sorted(values.items()))
        if fingerprint in seen:
            raise ValueError("Duplicate robot design after clamping/defaults.")
        seen.add(fingerprint)
        kinds.add(kind)
        index = len(designs)
        designs.append({
            "id": f"G0-{chr(ord('A') + index)}", "color": COLORS[index],
            "name": _text(row.get("name"), "Design name", 40),
            "reason": _text(row.get("reason"), "Design reason", 200),
            "robot": Robot(kind=kind, **values).model_dump(mode="json"),
            "adjustments": notes,
        })
    if kinds != {"rover", "walker"}:
        raise ValueError("Generation zero requires at least one rover and one walker.")
    return designs


def _rules(n: int) -> list[dict]:
    rows = [
        ("Trail Four", "Four medium wheels offer a compact starting design.",
         {"kind": "rover", "wheel_radius": 0.22, "wheel_width": 0.12, "drive_speed": 5.0}),
        ("Steady Six", "Six legs and a short stride propose more support contacts.",
         {"kind": "walker", "legs": 6, "leg_length": 0.28, "gait_freq_hz": 1.5, "stride_deg": 12.0}),
        ("Wide Six", "Six large wide wheels propose load spreading and obstacle clearance.",
         {"kind": "rover", "wheels": 6, "wheel_radius": 0.34, "wheel_width": 0.18,
          "chassis_length": 0.9, "chassis_width": 0.7, "mass_kg": 18.0, "drive_speed": 4.0}),
        ("Light Four", "Four longer legs and low mass propose a lighter stepping design.",
         {"kind": "walker", "legs": 4, "leg_length": 0.34, "gait_freq_hz": 2.0,
          "stride_deg": 18.0, "chassis_width": 0.5, "mass_kg": 7.0}),
        ("Low Four", "A wide chassis and slower wheels propose a lower-speed alternative.",
         {"kind": "rover", "wheel_radius": 0.15, "chassis_width": 0.8, "drive_speed": 3.0}),
        ("Compact Six", "Six short legs propose a compact support pattern.",
         {"kind": "walker", "legs": 6, "leg_length": 0.18, "chassis_length": 0.4, "mass_kg": 5.0}),
        ("Long Six", "A longer six-wheel chassis proposes spanning small terrain dips.",
         {"kind": "rover", "wheels": 6, "chassis_length": 1.1, "wheel_radius": 0.28, "mass_kg": 24.0}),
        ("Long Stride", "A larger four-leg stride proposes a distinct gait for later testing.",
         {"kind": "walker", "leg_length": 0.30, "gait_freq_hz": 2.5, "stride_deg": 24.0}),
    ]
    return _validate_designs({"designs": [
        {"name": name, "reason": reason, "robot": robot} for name, reason, robot in rows[:n]
    ]}, n)


def propose_generation0(challenge: SceneSpec, n: int = 4) -> list[dict]:
    """Return 2-8 JSON-ready untested designs for a SceneSpec's place and mission.

    Missing/invalid model output uses fixed diverse rules, disclosed on every design and in logs.
    Library robots in the challenge remain untouched and are never design candidates.
    """
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(COLORS):
        raise ValueError("Generation zero needs 2-8 designs to include both robot kinds.")
    if not isinstance(challenge, SceneSpec) or challenge.domain != "robot_terrain":
        raise ValueError("A generation-zero challenge must be a robot-terrain SceneSpec.")
    scene = challenge.model_copy(deep=True).fill_defaults()
    found = place_facts(scene)
    fallback_reason = "Model unavailable: MOCK_MODE; using fixed design rules." if config.MOCK_MODE else None
    designs = None
    if not config.MOCK_MODE:
        from openai import OpenAIError

        messages = [
            {"role": "system", "content": SYSTEM + "\nDESIGN_SPACE: " + json.dumps(DESIGN_SPACE)},
            {"role": "user", "content": json.dumps({
                "challenge": {"title": scene.title, "mission": scene.mission.model_dump(mode="json") if scene.mission else None},
                "n": n, "FACTS": found,
            })},
        ]
        try:
            designs, attempts = llm.chat_json(messages, config.BUILDER_MODEL, lambda reply: _validate_designs(reply, n))
        except OpenAIError as error:
            logger.warning("Generation-zero Designer request failed (%s); using fixed design rules.", type(error).__name__)
            fallback_reason = f"Designer request failed ({type(error).__name__}); using fixed design rules."
        else:
            if designs is None:
                fallback_reason = "Designer JSON failed validation; using fixed design rules."
                logger.warning("%s Validation: %s", fallback_reason, attempts)
    if designs is None:
        if config.MOCK_MODE:
            logger.info("%s", fallback_reason)
        designs = _rules(n)
    for design in designs:
        design.update(source="rules" if fallback_reason else "model", fallback_reason=fallback_reason,
                      tested=False)
    return designs


def challenge_preset(preset_id: str) -> SceneSpec:
    """Resolve a built-in robot-design challenge by its stable preset id."""
    if preset_id != "victoria-climb-out":
        raise ValueError(f"Unknown design challenge preset: {preset_id}")
    return SceneSpec(
        domain="robot_terrain",
        title="Victoria Crater floor to rim",
        environment=Environment(body="mars", surface="regolith"),
        terrain=Terrain(kind="real", preset="victoria-opportunity", roughness=0.0),
        robot=Robot(),
        mission=Mission(start_place="lowest", goal_place="highest", goal_radius_m=2.0),
        duration_s=90.0,
        time_limit_given=True,
    ).fill_defaults()


def _validate_race_designs(designs: list[dict]) -> list[dict]:
    if not isinstance(designs, list) or not 2 <= len(designs) <= len(COLORS):
        raise ValueError("A generation race needs 2-8 procedural designs.")
    validated, seen, kinds = [], set(), set()
    for design in designs:
        if not isinstance(design, dict) or not isinstance(design.get("robot"), dict):
            raise ValueError("Each race candidate must contain a robot object.")
        raw = design["robot"]
        extra = set(raw) - set(Robot.model_fields)
        if extra:
            raise ValueError(f"Unknown robot settings: {', '.join(sorted(extra))}.")
        kind = raw.get("kind")
        if not isinstance(kind, str) or kind not in DESIGN_SPACE:
            raise ValueError("Only procedural rover and walker designs can race; library robots are excluded.")
        if any(isinstance(raw.get(key), bool) for key in DESIGN_SPACE[kind]):
            raise ValueError(f"{design.get('id', 'Design')} contains boolean design parameters.")
        name = _text(design.get("name"), "Design name", 40)
        design_id = _text(design.get("id"), "Design id", 20)
        robot = Robot.model_validate(design["robot"])
        if robot.kind not in DESIGN_SPACE:
            raise ValueError("Only procedural rover and walker designs can race; library robots are excluded.")
        if design_id in seen:
            raise ValueError(f"Duplicate design id: {design_id}.")
        seen.add(design_id)
        kinds.add(robot.kind)
        for key, (low, high) in DESIGN_SPACE[robot.kind].items():
            value = getattr(robot, key)
            if not low <= value <= high:
                raise ValueError(f"{design_id} has {key} outside the {robot.kind} design space.")
        validated.append({**design, "id": design_id, "name": name, "robot": robot.model_dump(mode="json")})
    if kinds != {"rover", "walker"}:
        raise ValueError("A generation race requires at least one rover and one walker.")
    return validated


def _race_event(samples: list[dict], index: int, time_s: float | None = None) -> dict:
    if not samples:
        return {"time_s": time_s, "x": None, "y": None, "local_slope_deg": None, "step_m": None}
    sample = min(samples, key=lambda row: abs(row["time_s"] - time_s)) if time_s is not None else samples[index]
    return {
        "time_s": round(float(sample["time_s"] if time_s is None else time_s), 2),
        "x": sample.get("x"), "y": sample.get("y"),
        "local_slope_deg": sample.get("local_slope_deg"), "step_m": sample.get("step_m"),
    }


def _score_race_result(design: dict, run: dict, time_limit_s: float) -> dict:
    summary = run.get("summary") or {}
    telemetry = summary.get("race_telemetry") or {}
    samples = telemetry.get("samples") or []
    if run.get("error"):
        raise RuntimeError(f"Race run for {design['id']} failed: {run['error']}")
    reached = bool(summary.get("goal_reached"))
    start_distance = summary.get("start_distance_to_goal_m")
    closest = summary.get("closest_distance_m")
    progress = 100.0 if reached else (
        max(0.0, min(100.0, 100.0 * (start_distance - closest) / start_distance))
        if isinstance(start_distance, (int, float)) and start_distance > 0
        and isinstance(closest, (int, float)) else 0.0
    )
    event_time = summary.get("time_to_goal_s") if reached else None
    if reached:
        outcome = "reached"
    elif summary.get("flipped"):
        outcome = "flipped"
        event_time = summary.get("flipped_at_s")
    else:
        stuck_window = None
        for end_index, end in enumerate(samples):
            start_index = next((i for i in range(end_index, -1, -1)
                                if end["time_s"] - samples[i]["time_s"] >= RACE_STUCK_WINDOW_S), None)
            if start_index is None:
                continue
            start, end_position = samples[start_index], end
            displacement = math.hypot(end_position["x"] - start["x"], end_position["y"] - start["y"])
            if displacement < RACE_NO_PROGRESS_M:
                stuck_window = samples[start_index:end_index + 1]
                event_time = end["time_s"]
                break
        if stuck_window is not None:
            slip_samples = [row["slip_ratio"] for row in stuck_window if row.get("slip_ratio") is not None]
            outcome = ("slipped" if slip_samples and
                       sum(slip_samples) / len(slip_samples) >= RACE_HIGH_SLIP_RATIO else "stuck")
        else:
            outcome = "too slow"
            event_time = time_limit_s
    event_index = min(range(len(samples)), key=lambda i: abs(samples[i]["time_s"] - event_time)) \
        if samples and event_time is not None else max(0, len(samples) - 1)
    event = _race_event(samples, event_index, event_time)
    if event["x"] is None and summary.get("final_xy") is not None:
        event["x"], event["y"] = summary["final_xy"]
    return {
        "id": design["id"], "name": design["name"], "color": design.get("color"),
        "kind": design["robot"]["kind"], "reached_goal": reached,
        "progress_percent": round(progress, 1),
        "time_s": round(float(event_time if event_time is not None else time_limit_s), 2),
        "outcome": outcome, "event": event, "reason": design.get("reason"),
        "run_id": run.get("id"),
    }


def race_generation(challenge: SceneSpec, designs: list[dict], seed: int) -> dict:
    """Run each candidate in a parallel worker on the same cloned challenge and terrain seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Race seed must be a nonnegative integer.")
    if not isinstance(challenge, SceneSpec) or challenge.domain != "robot_terrain":
        raise ValueError("A generation race requires a robot-terrain SceneSpec.")
    candidates = _validate_race_designs(designs)
    base = challenge.model_copy(deep=True).fill_defaults()
    if (base.terrain is None or base.mission is None or not base.mission.has_goal()
            or (base.mission.start_xy is None and base.mission.start_place is None)):
        raise ValueError("A generation race challenge must define both a start and a goal.")
    if len(base.robots) != 1:
        raise ValueError("A generation race challenge must start with one robot.")
    base.terrain.seed = seed
    time_limit_s = base.duration_s

    from . import batch

    variations = []
    for candidate in candidates:
        kind = candidate["robot"]["kind"]
        robot = {key: value for key, value in candidate["robot"].items()
                 if key not in {"kind", "robot_id", "policy_id"}}
        variations.append({
            "label": f"{candidate['id']} {candidate['name']}",
            "seed": seed,
            "robot_kind": kind,
            "robot": robot,
        })
    runs = batch.run_batch(base, variations, baseline=False, record=False, race_telemetry=True)
    by_label = {run.get("label"): run for run in runs}
    missing = [f"{candidate['id']} {candidate['name']}" for candidate in candidates
               if f"{candidate['id']} {candidate['name']}" not in by_label]
    if missing:
        raise RuntimeError(f"Race worker returned no result for: {', '.join(missing)}.")
    results = [
        _score_race_result(candidate, by_label.get(f"{candidate['id']} {candidate['name']}", {}), time_limit_s)
        for candidate in candidates
    ]
    results.sort(key=lambda row: (
        not row["reached_goal"],
        row["time_s"] if row["reached_goal"] else -row["progress_percent"],
        row["id"],
    ))
    for rank, row in enumerate(results, 1):
        row["rank"] = rank
    return {
        "challenge": base.title, "seed": seed, "time_limit_s": time_limit_s,
        "terrain": base.terrain.model_dump(mode="json"),
        "start": {"place": base.mission.start_place, "xy": base.mission.start_xy},
        "goal": {"place": base.mission.goal_place, "xy": base.mission.goal_xy,
                 "radius_m": base.mission.goal_radius_m},
        "results": results,
    }
