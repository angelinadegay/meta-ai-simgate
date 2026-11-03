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
MAX_EVOLUTION_RUNS = 16

SYSTEM = """You are the Designer proposing generation-zero procedural robots for a place.
Propose exactly the requested number, including at least one rover and one walker.
G1 and all library robots are NOT designable. Change only robot parameters; never the place,
gravity, friction, terrain, start or goal. These are untested proposals, not measured winners.
Use FACTS only: null rock sizes mean unknown, not zero; do not invent terrain measurements.
DESIGN_SPACE lists numeric ranges, except wheels and legs which are discrete choices 4 or 6.
Use only the knobs for that kind. Vary designs rather than repeating them.
Return JSON only: {"designs":[{"name":"short name","reason":"one-line design rationale",
"robot":{"kind":"rover or walker","<allowed knob>":0}}]}."""

MUTATION_SYSTEM = """You are the Designer improving already-raced procedural robots.
The best design is kept unchanged; propose exactly the requested number of children.
Each child must name one parent_id and change one or two allowed parameters of that parent's
kind only. Return only actual parameter differences in `changes`; never change kind or the
world. Tie the one-line reason to that parent's measured outcome and failure location from FACTS.
Use allowed numeric ranges and discrete values only. Return JSON:
{"children":[{"parent_id":"G0-A","name":"short name","reason":"failure-linked reason",
"changes":{"wheel_radius":0.28}}]}"""


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
        "robot": design["robot"],
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


def _result_rows(results: dict | list[dict]) -> list[dict]:
    rows = results.get("results") if isinstance(results, dict) else results
    if not isinstance(rows, list) or not rows:
        raise ValueError("Generation results must include at least one raced design.")
    return rows


def _fingerprint(robot: dict) -> tuple:
    kind = robot["kind"]
    return (kind, *sorted((key, robot[key]) for key in DESIGN_SPACE[kind]))


def _failure_fact(parent: dict) -> str:
    event = parent.get("event") or {}
    where = []
    if event.get("step_m") is not None and event["step_m"] > 0.01:
        where.append(f"a {event['step_m']:.2f} m local step")
    if event.get("local_slope_deg") is not None:
        where.append(f"{event['local_slope_deg']:g}° slope")
    if event.get("x") is not None and event.get("y") is not None:
        where.append(f"({event['x']:g}, {event['y']:g})")
    return " at " + ", ".join(where) if where else " at an unmeasured location"


def _change_description(parent: dict, changes: dict) -> str:
    return ", ".join("{} {} -> {}".format(key, parent["robot"][key], value)
                     for key, value in changes.items())


def _bounded_changes(parent: dict, changes: dict) -> tuple[dict, dict]:
    if not isinstance(changes, dict) or not 1 <= len(changes) <= 2:
        raise ValueError("Each child must change one or two parameters.")
    kind = parent["robot"]["kind"]
    allowed = DESIGN_SPACE[kind]
    if set(changes) - set(allowed):
        raise ValueError(f"Child changes must use {kind} design parameters only.")
    robot = dict(parent["robot"])
    applied = {}
    for key, raw in changes.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(f"{key} must be a finite number.")
        low, high = allowed[key]
        if key in ("wheels", "legs"):
            if raw not in (low, high):
                raise ValueError(f"{key} must be 4 or 6.")
            value = int(raw)
        else:
            value = min(high, max(low, float(raw)))
        if value != robot[key]:
            robot[key] = value
            applied[key] = value
    if not applied:
        raise ValueError("A child must differ from its parent after clamping.")
    return Robot.model_validate(robot).model_dump(mode="json"), applied


def _validate_children(
    suggestions: list[dict],
    parents: list[dict],
    elite: dict,
    n_children: int,
    generation: int,
    raced_fingerprints: set[tuple],
) -> list[dict]:
    if not isinstance(suggestions, list) or len(suggestions) != n_children:
        raise ValueError(f"Designer must propose exactly {n_children} children.")
    by_id = {row["id"]: row for row in parents}
    seen = set(raced_fingerprints)
    seen.add(_fingerprint(elite["robot"]))
    children = []
    child_colors = [color for color in COLORS if color != elite.get("color")]
    for index, suggestion in enumerate(suggestions):
        if not isinstance(suggestion, dict) or suggestion.get("parent_id") not in by_id:
            raise ValueError("Each child must name a raced parent_id.")
        parent = by_id[suggestion["parent_id"]]
        reason = _text(suggestion.get("reason"), "Child reason", 200)
        name = _text(suggestion.get("name"), "Child name", 40)
        robot, changes = _bounded_changes(parent, suggestion.get("changes"))
        fingerprint = _fingerprint(robot)
        if fingerprint in seen:
            raise ValueError("Child duplicates a design already raced.")
        seen.add(fingerprint)
        children.append({
            "id": f"G{generation}-{chr(ord('A') + index)}",
            "color": child_colors[index], "name": name, "reason": reason,
            "robot": robot, "parent_id": parent["id"],
            "changes": changes, "changed_parameters": list(changes),
            "parameter_changes": {
                key: {"from": parent["robot"][key], "to": value} for key, value in changes.items()
            },
            "source": "model", "tested": False,
        })
    if {elite["robot"]["kind"], *(row["robot"]["kind"] for row in children)} != {"rover", "walker"}:
        raise ValueError("The next generation must include at least one rover and one walker.")
    return children


def _fallback_change(parent: dict, outcome: str, attempt: int) -> tuple[dict, str]:
    robot = parent["robot"]
    kind = robot["kind"]
    space = DESIGN_SPACE[kind]
    if outcome == "stuck":
        if kind == "rover":
            changes = {"wheel_radius": robot["wheel_radius"] + 0.05 + 0.02 * attempt,
                       "wheel_width": robot["wheel_width"] + 0.02 * attempt}
            why = "try larger wheels and more contact width"
        else:
            changes = {"leg_length": robot["leg_length"] + 0.04 + 0.02 * attempt,
                       "stride_deg": robot["stride_deg"] + 1.0 * attempt}
            why = "try longer legs and stride"
    elif outcome == "flipped":
        lower = "wheel_radius" if kind == "rover" else "leg_length"
        lower_delta = 0.02 + 0.01 * attempt
        if attempt % 3 == 2:
            changes = {"chassis_width": robot["chassis_width"] + 0.06,
                       "mass_kg": robot["mass_kg"] + 2.0 + attempt}
            why = "try a wider, heavier stance; chassis height is not a design parameter"
        elif attempt % 3 == 1:
            changes = {"mass_kg": robot["mass_kg"] + 2.0 + attempt,
                       lower: robot[lower] - lower_delta}
            why = "add mass and lower the chassis with smaller wheels or shorter legs"
        else:
            changes = {"chassis_width": robot["chassis_width"] + 0.06 + 0.02 * (attempt // 3),
                       lower: robot[lower] - lower_delta}
            why = "widen the stance and lower the chassis with smaller wheels or shorter legs"
    elif outcome == "slipped":
        if kind == "rover":
            changes = {"drive_speed": robot["drive_speed"] * (0.8 - 0.05 * attempt)}
            if robot["wheels"] < 6:
                changes["wheels"] = 6
                why = "increase to six wheel contacts and slow down"
            else:
                changes["wheel_width"] = robot["wheel_width"] + 0.02 + 0.01 * attempt
                why = "widen the tires and slow down; the design already has six wheels"
        else:
            changes = {"gait_freq_hz": robot["gait_freq_hz"] * (0.8 - 0.05 * attempt)}
            if robot["legs"] < 6:
                changes["legs"] = 6
                why = "increase to six leg contacts and slow down"
            else:
                changes["chassis_width"] = robot["chassis_width"] + 0.02 + 0.01 * attempt
                why = "widen the stance and slow down; the design already has six legs"
    else:
        speed = "drive_speed" if kind == "rover" else "gait_freq_hz"
        changes = {speed: robot[speed] * (1.15 + 0.05 * attempt)}
        second = "wheel_radius" if kind == "rover" else "stride_deg"
        changes[second] = robot[second] + (0.02 if second == "wheel_radius" else 1.0) * (attempt + 1)
        why = "try a faster drive and larger step response"
    clamped = {}
    for key, value in changes.items():
        low, high = space[key]
        if key in ("wheels", "legs"):
            clamped[key] = value
        else:
            clamped[key] = min(high, max(low, value))
    return clamped, why


def _fallback_children(
    parents: list[dict], elite: dict, n_children: int, generation: int,
    raced_fingerprints: set[tuple], fallback_reason: str,
) -> list[dict]:
    seen = set(raced_fingerprints)
    seen.add(_fingerprint(elite["robot"]))
    failures = [row for row in parents if not row.get("reached_goal")]
    if not failures:
        failures = parents
    children = []
    child_colors = [color for color in COLORS if color != elite.get("color")]
    for index in range(n_children):
        target_kind = "walker" if elite["robot"]["kind"] == "rover" else "rover"
        candidates = sorted(failures, key=lambda row: (
            row["robot"]["kind"] != target_kind,
            -float(row.get("progress_percent", 0)),
            row["id"],
        ))
        if candidates:
            offset = index % len(candidates)
            candidates = candidates[offset:] + candidates[:offset]
        accepted = None
        for parent in candidates:
            for attempt in range(8):
                outcome = parent.get("outcome", "too slow")
                changes, mutation_reason = _fallback_change(parent, outcome, attempt)
                try:
                    robot, applied = _bounded_changes(parent, changes)
                except ValueError:
                    continue
                fingerprint = _fingerprint(robot)
                if fingerprint in seen or robot["kind"] == elite["robot"]["kind"] and \
                        not any(row["robot"]["kind"] != elite["robot"]["kind"] for row in children) \
                        and index == n_children - 1:
                    continue
                seen.add(fingerprint)
                accepted = {
                    "id": f"G{generation}-{chr(ord('A') + index)}",
                    "color": child_colors[index], "name": f"{parent['name']} {index + 1}",
                    "reason": f"{parent['id']} was {outcome}{_failure_fact(parent)}; {mutation_reason} "
                              f"({_change_description(parent, applied)}).",
                    "robot": robot, "parent_id": parent["id"],
                    "changes": applied, "changed_parameters": list(applied),
                    "parameter_changes": {
                        key: {"from": parent["robot"][key], "to": value} for key, value in applied.items()
                    },
                    "source": "rules", "fallback_reason": fallback_reason, "tested": False,
                }
                break
            if accepted:
                break
        if accepted is None:
            raise ValueError("No unique bounded fallback mutation remains for this generation.")
        children.append(accepted)
    if {elite["robot"]["kind"], *(row["robot"]["kind"] for row in children)} != {"rover", "walker"}:
        raise ValueError("The next generation must include at least one rover and one walker.")
    return children


def next_generation(results: dict | list[dict], n: int = 4) -> list[dict]:
    """Keep the ranked elite and request n-1 bounded, failure-guided children."""
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(COLORS):
        raise ValueError("A generation needs 2-8 designs.")
    rows = _result_rows(results)
    parents = _validate_race_designs(rows)
    ordered = sorted(parents, key=lambda row: (
        not row.get("reached_goal", False),
        row.get("time_s", float("inf")) if row.get("reached_goal") else
        -float(row.get("progress_percent", 0)),
        row["id"],
    ))
    elite = {**ordered[0], "tested": True}
    generation = int(results.get("generation", 0)) + 1 if isinstance(results, dict) else 1
    if not 1 <= generation <= 8:
        raise ValueError("Generation IDs are limited to G1-G8.")
    raced_designs = results.get("raced_designs", rows) if isinstance(results, dict) else rows
    raced_fingerprints = {
        _fingerprint(row["robot"]) for row in raced_designs
        if isinstance(row, dict) and isinstance(row.get("robot"), dict)
        and row["robot"].get("kind") in DESIGN_SPACE
    }
    fallback_reason = "Designer unavailable or child validation failed; used bounded rule mutations."
    children = None
    if not config.MOCK_MODE:
        from openai import OpenAIError

        payload = [{
            "id": row["id"], "name": row["name"], "robot": row["robot"],
            "outcome": row.get("outcome"), "progress_percent": row.get("progress_percent"),
            "event": row.get("event"), "reason": row.get("reason"),
            "goal_reached": row.get("reached_goal"),
        } for row in parents]
        messages = [
            {"role": "system", "content": MUTATION_SYSTEM + "\nDESIGN_SPACE: " + json.dumps(DESIGN_SPACE)},
            {"role": "user", "content": json.dumps({
                "elite_id": elite["id"], "n_children": n - 1,
                "generation": generation, "FACTS": payload,
            })},
        ]
        try:
            children, attempts = llm.chat_json(
                messages, config.BUILDER_MODEL,
                lambda reply: _validate_children(reply.get("children"), parents, elite,
                                                 n - 1, generation, raced_fingerprints),
            )
        except OpenAIError as error:
            fallback_reason = f"Designer request failed ({type(error).__name__}); used bounded rule mutations."
            logger.warning("%s", fallback_reason)
        else:
            if children is None:
                logger.warning("Generation %s child proposals failed validation: %s", generation, attempts)
    if children is None:
        children = _fallback_children(parents, elite, n - 1, generation,
                                      raced_fingerprints, fallback_reason)
    else:
        child_colors = [color for color in COLORS if color != elite.get("color")]
        for index, child in enumerate(children):
            child["color"] = child_colors[index]
    return [elite, *children]


def _failure_report(row: dict) -> str:
    event = row.get("event") or {}
    location = []
    if event.get("x") is not None and event.get("y") is not None:
        location.append(f"near ({event['x']:g}, {event['y']:g})")
    if event.get("local_slope_deg") is not None:
        location.append(f"at a measured {event['local_slope_deg']:g}° slope")
    if event.get("step_m") is not None and event["step_m"] > 0.01:
        location.append(f"at a {event['step_m']:.2f} m local step")
    where = " " + " ".join(location) if location else " at an unmeasured location"
    return f"closest: {row['id']}, {row['progress_percent']:g}% of the way, {row['outcome']}{where}."


def evolve(challenge: SceneSpec, max_generations: int = 4, n: int = 4, seed: int = 0) -> dict:
    """Propose, race, and mutate generations until success or the bounded race budget is spent."""
    if isinstance(max_generations, bool) or not isinstance(max_generations, int) or not 1 <= max_generations <= 8:
        raise ValueError("max_generations must be an integer from 1 to 8.")
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(COLORS):
        raise ValueError("A generation needs 2-8 designs.")
    if max_generations * n > MAX_EVOLUTION_RUNS:
        raise ValueError(f"Evolution is limited to {MAX_EVOLUTION_RUNS} races.")
    from . import cost

    with cost.track() as spent:
        designs = propose_generation0(challenge, n)
        generations, family_tree, all_raced = [], [], []
        success = False
        for generation in range(max_generations):
            race = race_generation(challenge, designs, seed)
            race["generation"] = generation
            generations.append(race)
            current_by_id = {design["id"]: design for design in designs}
            for result in race["results"]:
                source = current_by_id[result["id"]]
                result.update(
                    robot=source["robot"], reason=source.get("reason"),
                    color=source.get("color"), source=source.get("source"),
                )
                all_raced.append({**result, "generation": generation})
            if any(row["reached_goal"] for row in race["results"]):
                success = True
                break
            if generation + 1 < max_generations:
                next_designs = next_generation({
                    "generation": generation,
                    "results": race["results"],
                    "raced_designs": all_raced,
                }, n)
                for child in next_designs[1:]:
                    family_tree.append({
                        "parent_id": child["parent_id"], "child_id": child["id"],
                        "generation": generation + 1, "changes": child["parameter_changes"],
                        "reason": child["reason"], "source": child.get("source"),
                        "result": None,
                    })
                designs = next_designs
        by_id = {row["id"]: row for row in all_raced}
        for edge in family_tree:
            edge["result"] = {
                key: by_id[edge["child_id"]].get(key)
                for key in ("outcome", "progress_percent", "reached_goal", "event")
            } if edge["child_id"] in by_id else None
        best = max(all_raced, key=lambda row: (
            bool(row["reached_goal"]), float(row.get("progress_percent", 0)),
            -float(row.get("time_s", float("inf"))) if row["reached_goal"] else 0,
        ))
        report = (f"Goal reached by {next(row['id'] for row in all_raced if row['reached_goal'])}."
                  if success else _failure_report(best))
        return {
            "challenge": challenge.title, "success": success,
            "generations": generations, "family_tree": family_tree,
            "best": best, "report": report,
            "cost": spent.snapshot(),
        }
