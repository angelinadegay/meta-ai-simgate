"""Bounded generation-zero robot designs and same-challenge generation races."""

import json
import logging
import math
import base64
import zlib
from typing import Any, Callable

import numpy as np

from . import config, llm
from .spec import Environment, Mission, Obstacle, Robot, RobotDesignConstraints, SceneSpec, Terrain

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
MAX_EVOLUTION_RUNS = 20
MAX_EVOLUTION_GENERATIONS = 5
VERIFIED_PER_GENERATION = 2
DEFAULT_CHALLENGE = "victoria-rim-boulders"
DEFAULT_CHALLENGE_DURATION_S = 90.0
DEFAULT_ROCK_SIZE_M = 0.20
DEFAULT_ROCK_COUNT = 12
DEFAULT_LOOSE_SOIL_FRICTION = 0.65
# Calibrated from the spec's 0.6 × 0.5 m: at 0.6 m no legal rover fits wheels big enough to climb out.
DEFAULT_DESIGN_CONSTRAINTS = {"max_mass_kg": 12.0, "max_size_m": (0.8, 0.5)}
ROCK_SIZE_RANGE_M = (0.15, 0.35)
ROCK_COUNT_RANGE = (12, 30)
LOOSE_SOIL_FRICTION_RANGE = (0.45, 0.65)
DEFAULT_CHALLENGE_CONFIG = {
    "rock_size_m": DEFAULT_ROCK_SIZE_M,
    "rock_count": DEFAULT_ROCK_COUNT,
    "friction": DEFAULT_LOOSE_SOIL_FRICTION,
    "start_offset_m": 0.0,
    "design_constraints": DEFAULT_DESIGN_CONSTRAINTS.copy(),
}

SYSTEM = """You are the Designer proposing generation-zero procedural robots for a place.
Propose exactly the requested number, including at least one rover and one walker.
G1 and all library robots are NOT designable. Change only robot parameters; never the place,
gravity, friction, terrain, start or goal. These are untested proposals, not measured winners.
Every design must satisfy `FACTS.design_constraints`; satisfy the mass, stowed-size and power limits
as well as the design-space bounds. Use FACTS only: null rock sizes mean unknown, not zero; do not invent terrain measurements.
DESIGN_SPACE lists numeric ranges, except wheels and legs which are discrete choices 4 or 6.
Use only the knobs for that kind. Vary designs rather than repeating them.
Return JSON only: {"designs":[{"name":"short name","reason":"one-line design rationale",
"robot":{"kind":"rover or walker","<allowed knob>":0}}]}."""

MUTATION_SYSTEM = """You are the Designer improving already-raced procedural robots.
The best design is kept unchanged; propose exactly the requested number of children.
Each child must name one parent_id and change one or two allowed parameters of that parent's
kind only. Return only actual parameter differences in `changes`; never change kind or the
world. Respect the challenge's design constraints: compute each child's stowed size with
`design_constraints.stowed_size_rule` and stay within each parent's `stowed` headroom (growing wheels
or legs may require shrinking the chassis). Tie the one-line reason to that parent's measured outcome and failure location from FACTS.
Use allowed numeric ranges and discrete values only. Return JSON:
{"children":[{"parent_id":"G0-A","name":"short name","reason":"failure-linked reason",
"changes":{"wheel_radius":0.28}}]}"""


def _design_dimensions(robot: Robot) -> tuple[float, float]:
    if robot.kind == "rover":
        return robot.chassis_length + 2 * robot.wheel_radius, robot.chassis_width + 2 * robot.wheel_width + 0.02
    return robot.chassis_length, robot.chassis_width + 0.15


STOWED_SIZE_RULE = (
    "rover stowed length = chassis_length + 2*wheel_radius; rover stowed width = chassis_width + "
    "2*wheel_width + 0.02; walker stowed length = chassis_length; walker stowed width = chassis_width + 0.15"
)


def _constraint_facts(constraints: RobotDesignConstraints | dict | None) -> dict | None:
    if constraints is None:
        return None
    limits = (constraints.model_dump(mode="json") if isinstance(constraints, RobotDesignConstraints)
              else dict(constraints))
    return {**limits, "stowed_size_rule": STOWED_SIZE_RULE,
            "power_proxy_rule": "drive_speed * mass_kg"}


def _stowed_facts(robot: dict, constraints: RobotDesignConstraints | dict | None) -> dict:
    length, width = _design_dimensions(Robot.model_validate(robot))
    facts = {"stowed_length_m": round(length, 3), "stowed_width_m": round(width, 3)}
    if constraints is not None:
        limits = (constraints.model_dump() if isinstance(constraints, RobotDesignConstraints) else constraints)
        facts["mass_headroom_kg"] = round(float(limits["max_mass_kg"]) - float(robot["mass_kg"]), 3)
        if limits.get("max_size_m") is not None:
            facts["length_headroom_m"] = round(float(limits["max_size_m"][0]) - length, 3)
            facts["width_headroom_m"] = round(float(limits["max_size_m"][1]) - width, 3)
    return facts


def _constraint_violations(robot: Robot, constraints: RobotDesignConstraints | dict | None) -> list[str]:
    if constraints is None:
        return []
    limits = (constraints.model_dump() if isinstance(constraints, RobotDesignConstraints) else constraints)
    violations = []
    if robot.mass_kg > limits["max_mass_kg"] + 1e-9:
        violations.append(f"mass {robot.mass_kg:g} kg exceeds {limits['max_mass_kg']:g} kg")
    if limits.get("max_size_m") is not None:
        length, width = _design_dimensions(robot)
        max_length, max_width = limits["max_size_m"]
        if length > max_length + 1e-9 or width > max_width + 1e-9:
            violations.append(
                f"stowed size {length:.3f} × {width:.3f} m exceeds {max_length:g} × {max_width:g} m "
                f"({STOWED_SIZE_RULE})"
            )
    if limits.get("max_power_proxy") is not None:
        proxy = robot.drive_speed * robot.mass_kg
        if proxy > limits["max_power_proxy"] + 1e-9:
            violations.append(
                f"power proxy {proxy:g} exceeds {limits['max_power_proxy']:g}"
            )
    return violations


def _constrain_design(design: dict, constraints: RobotDesignConstraints | dict | None) -> dict:
    if constraints is None:
        return design
    limits = constraints.model_dump() if isinstance(constraints, RobotDesignConstraints) else constraints
    robot = Robot.model_validate(design["robot"])
    space = DESIGN_SPACE[robot.kind]
    robot.mass_kg = min(robot.mass_kg, float(limits["max_mass_kg"]))
    size = limits.get("max_size_m")
    if size is not None:
        max_length, max_width = map(float, size)
        if robot.kind == "rover":
            robot.chassis_length = min(robot.chassis_length, max_length - 2 * space["wheel_radius"][0])
            robot.wheel_radius = min(robot.wheel_radius, (max_length - robot.chassis_length) / 2)
            robot.chassis_width = min(
                robot.chassis_width, max_width - 2 * space["wheel_width"][0] - 0.02,
            )
            robot.wheel_width = min(robot.wheel_width, (max_width - robot.chassis_width - 0.02) / 2)
        else:
            robot.chassis_length = min(robot.chassis_length, max_length)
            robot.chassis_width = min(robot.chassis_width, max_width - 0.15)
    power = limits.get("max_power_proxy")
    if power is not None:
        max_speed = float(power) / robot.mass_kg
        if robot.kind == "rover" and max_speed < space["drive_speed"][0]:
            robot.mass_kg = min(robot.mass_kg, float(power) / space["drive_speed"][0])
            max_speed = float(power) / robot.mass_kg
        if robot.kind == "rover":
            robot.drive_speed = min(robot.drive_speed, max_speed)
        else:
            robot.mass_kg = min(robot.mass_kg, float(power) / robot.drive_speed)
    for key, (low, high) in space.items():
        value = getattr(robot, key)
        if value < low - 1e-9 or value > high + 1e-9:
            raise ValueError(f"{design.get('id', 'Design')} cannot fit the challenge design constraints.")
        if abs(value - low) <= 1e-9:
            setattr(robot, key, low)
        elif abs(value - high) <= 1e-9:
            setattr(robot, key, high)
    fitted = Robot.model_validate(robot.model_dump())
    violations = _constraint_violations(fitted, limits)
    if violations:
        raise ValueError(f"{design.get('id', 'Design')} violates design constraints: {'; '.join(violations)}.")
    result = {**design, "robot": fitted.model_dump(mode="json")}
    if fitted.model_dump() != Robot.model_validate(design["robot"]).model_dump():
        result["constraint_adjustments"] = [
            f"{key} adjusted to {getattr(fitted, key):g} to meet the mission constraints"
            for key in DESIGN_SPACE[fitted.kind]
            if getattr(fitted, key) != getattr(Robot.model_validate(design["robot"]), key)
        ]
    return result


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


def _validate_designs(
    reply: dict, n: int, constraints: RobotDesignConstraints | dict | None = None,
) -> list[dict]:
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
        kinds.add(kind)
        index = len(designs)
        design = {
            "id": f"G0-{chr(ord('A') + index)}", "color": COLORS[index],
            "name": _text(row.get("name"), "Design name", 40),
            "reason": _text(row.get("reason"), "Design reason", 200),
            "robot": Robot(kind=kind, **values).model_dump(mode="json"),
            "adjustments": notes,
        }
        design = _constrain_design(design, constraints)
        fingerprint = _fingerprint(design["robot"])
        if fingerprint in seen:
            raise ValueError("Duplicate robot design after clamping/defaults/mission constraints.")
        seen.add(fingerprint)
        designs.append(design)
    if kinds != {"rover", "walker"}:
        raise ValueError("Generation zero requires at least one rover and one walker.")
    return designs


def _rules(n: int, constraints: RobotDesignConstraints | dict | None = None) -> list[dict]:
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
    ]}, n, constraints)


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
    found["design_constraints"] = _constraint_facts(scene.design_constraints)
    found["challenge_obstacles"] = [
        {"kind": obstacle.kind, "height_m": obstacle.height_m, "count": obstacle.count,
         "placement": obstacle.placement}
        for obstacle in scene.obstacles
    ]
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
            designs, attempts = llm.chat_json(
                messages, config.BUILDER_MODEL,
                lambda reply: _validate_designs(reply, n, scene.design_constraints),
            )
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
        designs = _rules(n, scene.design_constraints)
    for design in designs:
        design.update(source="rules" if fallback_reason else "model", fallback_reason=fallback_reason,
                      tested=False)
    return designs


def steepest_climb_points(
    challenge: SceneSpec, span_m: float = 20.0,
    start_height_percentile: float | None = None,
    goal_height_percentile: float | None = None,
) -> dict:
    """Find the steepest uphill segment of a fixed length on the sampled terrain grid."""
    from scipy.ndimage import gaussian_filter, map_coordinates
    from .backends.mujoco_backend import make_heights

    scene = challenge.model_copy(deep=True).fill_defaults()
    if scene.terrain is None or scene.terrain.kind != "real":
        raise ValueError("A steepest climb requires real elevation terrain.")
    if isinstance(span_m, bool) or not isinstance(span_m, (int, float)) or not 8 <= span_m <= 60:
        raise ValueError("A climb segment must be between 8 and 60 metres.")
    heights, _ = make_heights(scene)
    spacing = float(scene.terrain.size_m) / (heights.shape[0] - 1)
    smooth = gaussian_filter(heights, sigma=max(0.7, 0.8 / spacing))
    dy, dx = np.gradient(smooth, spacing, spacing)
    magnitude = np.hypot(dx, dy)
    ux = np.divide(dx, magnitude, out=np.zeros_like(dx), where=magnitude > 1e-9)
    uy = np.divide(dy, magnitude, out=np.zeros_like(dy), where=magnitude > 1e-9)
    half = float(scene.terrain.size_m) / 2
    coordinates = np.linspace(-half, half, heights.shape[0])
    X, Y = np.meshgrid(coordinates, coordinates)
    start_x, start_y = X - ux * span_m / 2, Y - uy * span_m / 2
    goal_x, goal_y = X + ux * span_m / 2, Y + uy * span_m / 2
    edge = half - spacing
    valid = ((np.abs(start_x) <= edge) & (np.abs(start_y) <= edge) &
             (np.abs(goal_x) <= edge) & (np.abs(goal_y) <= edge) & (magnitude > 1e-6))
    rows, cols = np.where(valid)
    if not len(rows):
        raise ValueError("No uphill segment of the requested length fits inside this terrain.")
    row_scale = (heights.shape[0] - 1) / (2 * half)
    start_h = map_coordinates(smooth, np.array([
        (start_y[rows, cols] + half) * row_scale,
        (start_x[rows, cols] + half) * row_scale,
    ]), order=1)
    goal_h = map_coordinates(smooth, np.array([
        (goal_y[rows, cols] + half) * row_scale,
        (goal_x[rows, cols] + half) * row_scale,
    ]), order=1)
    rises = goal_h - start_h
    candidates = np.ones(len(rises), dtype=bool)
    if start_height_percentile is not None:
        candidates &= start_h <= np.percentile(smooth, start_height_percentile)
    if goal_height_percentile is not None:
        candidates &= goal_h >= np.percentile(smooth, goal_height_percentile)
    candidates &= rises > 0
    if not candidates.any():
        raise ValueError("No uphill segment fits the requested floor-to-rim elevation bands.")
    best = int(np.argmax(np.where(candidates, rises, -np.inf)))
    rise = float(rises[best])
    start = [round(float(start_x[rows[best], cols[best]]), 2),
             round(float(start_y[rows[best], cols[best]]), 2)]
    goal = [round(float(goal_x[rows[best], cols[best]]), 2),
            round(float(goal_y[rows[best], cols[best]]), 2)]
    return {
        "start_xy": start, "goal_xy": goal, "span_m": float(span_m),
        "rise_m": round(rise, 3),
        "slope_deg": round(math.degrees(math.atan2(rise, span_m)), 2),
    }


def challenge_preset(
    preset_id: str, span_m: float = 20.0, duration_s: float = 90.0,
    rock_size_m: float | None = None, rock_count: int | None = None,
    friction: float | None = None, start_offset_m: float | None = None,
    design_constraints: RobotDesignConstraints | dict | None = None,
) -> SceneSpec:
    """Resolve a climb challenge; Hadley and Meteor use the steepest measured uphill segment."""
    presets = {
        "victoria-climb-out": {
            "title": "Victoria Crater floor to rim", "terrain": "victoria-opportunity",
            "body": "mars", "surface": "regolith", "endpoints": "lowest-highest",
        },
        "hadley-climb-out": {
            "title": "Apollo 15 Hadley Rille floor to rim", "terrain": "apollo15-hadley-rille",
            "body": "moon", "surface": "regolith", "endpoints": "steepest",
        },
        "meteor-crater-climb": {
            "title": "Meteor Crater steep rim climb", "terrain": "meteor-crater-rim",
            "body": "earth", "surface": "rock", "endpoints": "steepest",
        },
        "victoria-rim-boulders": {
            "title": "Victoria Crater steep rim climb through a loose boulder field",
            "terrain": "victoria-opportunity", "body": "mars", "surface": "regolith",
            "endpoints": "steepest",
        },
    }
    preset = presets.get(preset_id)
    if preset is None:
        raise ValueError(f"Unknown design challenge preset: {preset_id}")
    if preset_id == "victoria-rim-boulders":
        rock_size_m = DEFAULT_CHALLENGE_CONFIG["rock_size_m"] if rock_size_m is None else rock_size_m
        rock_count = DEFAULT_CHALLENGE_CONFIG["rock_count"] if rock_count is None else rock_count
        friction = DEFAULT_CHALLENGE_CONFIG["friction"] if friction is None else friction
        start_offset_m = DEFAULT_CHALLENGE_CONFIG["start_offset_m"] if start_offset_m is None else start_offset_m
        if not ROCK_SIZE_RANGE_M[0] <= rock_size_m <= ROCK_SIZE_RANGE_M[1]:
            raise ValueError(f"Rock size must be from {ROCK_SIZE_RANGE_M[0]} to {ROCK_SIZE_RANGE_M[1]} m.")
        if isinstance(rock_count, bool) or not ROCK_COUNT_RANGE[0] <= rock_count <= ROCK_COUNT_RANGE[1]:
            raise ValueError(f"Rock-field density must be from {ROCK_COUNT_RANGE[0]} to {ROCK_COUNT_RANGE[1]} rocks.")
        if not LOOSE_SOIL_FRICTION_RANGE[0] <= friction <= LOOSE_SOIL_FRICTION_RANGE[1]:
            raise ValueError(
                f"Loose-soil friction must be from {LOOSE_SOIL_FRICTION_RANGE[0]} to "
                f"{LOOSE_SOIL_FRICTION_RANGE[1]}."
            )
        if not math.isfinite(start_offset_m) or abs(start_offset_m) > 6.0:
            raise ValueError("The start offset along the wall must be from -6 to 6 m.")
    if preset_id == "victoria-rim-boulders":
        design_constraints = design_constraints or DEFAULT_CHALLENGE_CONFIG["design_constraints"]
        design_constraints = (design_constraints if isinstance(design_constraints, RobotDesignConstraints)
                             else RobotDesignConstraints.model_validate(design_constraints))
    challenge = SceneSpec(
        domain="robot_terrain",
        title=preset["title"],
        environment=Environment(body=preset["body"], surface=preset["surface"]),
        terrain=Terrain(kind="real", preset=preset["terrain"], roughness=0.0),
        robot=Robot(),
        design_constraints=design_constraints,
        mission=Mission(start_place="lowest", goal_place="highest", goal_radius_m=2.0),
        obstacles=([Obstacle(kind="rock", height_m=rock_size_m, count=rock_count, placement="path",
                             path_fraction=0.5)] if preset_id == "victoria-rim-boulders" else []),
        duration_s=duration_s,
        time_limit_given=True,
    ).fill_defaults()
    if preset_id == "victoria-rim-boulders":
        challenge.environment.friction = friction
        challenge.assumptions.append(
            f"Loose regolith friction is set to {friction:.2f} (calibration range "
            f"{LOOSE_SOIL_FRICTION_RANGE[0]:.2f}–{LOOSE_SOIL_FRICTION_RANGE[1]:.2f}); "
            f"{rock_count} path rocks are {rock_size_m:.2f} m high (calibration range "
            f"{ROCK_SIZE_RANGE_M[0]:.2f}–{ROCK_SIZE_RANGE_M[1]:.2f} m)."
        )
        limits = challenge.design_constraints
        assert limits is not None
        max_size = (f"≤ {limits.max_size_m[0]:g} × {limits.max_size_m[1]:g} m"
                    if limits.max_size_m is not None else "no size cap")
        power_limit = (f", power proxy ≤ {limits.max_power_proxy:g}"
                       if limits.max_power_proxy is not None else "")
        challenge.assumptions.append(
            f"Must fit the lander: ≤ {limits.max_mass_kg:g} kg, {max_size}{power_limit}."
        )
    if preset["endpoints"] == "steepest":
        endpoints = (steepest_climb_points(
            challenge, span_m, start_height_percentile=30, goal_height_percentile=80,
        ) if preset_id == "victoria-rim-boulders" else steepest_climb_points(challenge, span_m))
        if preset_id == "victoria-rim-boulders" and start_offset_m:
            dx = endpoints["goal_xy"][0] - endpoints["start_xy"][0]
            dy = endpoints["goal_xy"][1] - endpoints["start_xy"][1]
            distance = math.hypot(dx, dy)
            offset = (dy / distance * start_offset_m, -dx / distance * start_offset_m)
            endpoints["start_xy"] = [round(endpoints["start_xy"][0] + offset[0], 2),
                                     round(endpoints["start_xy"][1] + offset[1], 2)]
            endpoints["goal_xy"] = [round(endpoints["goal_xy"][0] + offset[0], 2),
                                     round(endpoints["goal_xy"][1] + offset[1], 2)]
        challenge.mission = Mission(
            start_xy=endpoints["start_xy"], goal_xy=endpoints["goal_xy"], goal_radius_m=2.0,
        )
        for entry in challenge.robots:
            entry.start_xy, entry.start_place = endpoints["start_xy"], None
            entry.goal_xy, entry.goal_place = endpoints["goal_xy"], None
        if preset_id == "victoria-rim-boulders":
            distance = math.dist(endpoints["start_xy"], endpoints["goal_xy"])
            # 3 m/s is the slowest allowed rover top speed; twice its travel time is a generous lower bound.
            challenge.duration_s = max(duration_s, 2 * distance / DESIGN_SPACE["rover"]["drive_speed"][0])
            challenge.time_limit_given = True
    return challenge


def calibration_quality(results: dict | list[dict]) -> dict:
    """Apply the calibration target: no reaches, and the best design closes 20–80% of the route."""
    rows = _result_rows(results)
    best = max((float(row.get("progress_percent", 0)) for row in rows), default=0.0)
    reached = any(bool(row.get("reached_goal")) for row in rows)
    return {"qualifies": not reached and 20.0 <= best <= 80.0,
            "best_progress_percent": round(best, 1), "any_reached": reached}


def feasibility_reference_designs(constraints: RobotDesignConstraints | dict | None) -> list[dict]:
    """Return the legal rover with the biggest wheels and the legal walker with the longest legs.

    Calibration races these to prove that a design inside the lander limits can clear the course."""
    limits = (constraints.model_dump() if isinstance(constraints, RobotDesignConstraints)
              else dict(constraints or {}))
    rover_space, walker_space = DESIGN_SPACE["rover"], DESIGN_SPACE["walker"]
    mass = min(float(limits.get("max_mass_kg", 12.0)), 12.0)
    chassis_length, chassis_width = rover_space["chassis_length"][0], rover_space["chassis_width"][0]
    max_length, max_width = map(float, limits.get("max_size_m") or (2.0, 2.0))
    wheel_radius = min(rover_space["wheel_radius"][1], (max_length - chassis_length) / 2)
    wheel_width = min(0.09, (max_width - chassis_width - 0.02) / 2)
    designs = [
        {"id": "REF-R", "name": "Reference big-wheel rover", "color": COLORS[0], "robot": {
            "kind": "rover", "wheels": 6, "wheel_radius": round(wheel_radius, 3),
            "wheel_width": round(max(rover_space["wheel_width"][0], wheel_width), 3),
            "chassis_length": chassis_length, "chassis_width": chassis_width,
            "mass_kg": mass, "drive_speed": 4.0}},
        {"id": "REF-W", "name": "Reference long-leg walker", "color": COLORS[1], "robot": {
            "kind": "walker", "legs": 6, "leg_length": walker_space["leg_length"][1],
            "chassis_length": min(0.6, max_length), "chassis_width": min(0.35, max_width - 0.15),
            "mass_kg": mass, "gait_freq_hz": 1.5, "stride_deg": 15.0}},
    ]
    return [_constrain_design(design, limits) if limits else design for design in designs]


def _validate_race_designs(
    designs: list[dict], constraints: RobotDesignConstraints | dict | None = None,
) -> list[dict]:
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
        if violations := _constraint_violations(robot, constraints):
            raise ValueError(f"{design_id} violates challenge design constraints: {'; '.join(violations)}.")
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


def _nearest_rock(event: dict, run: dict) -> dict | None:
    if event.get("x") is None or event.get("y") is None:
        return None
    scene = run.get("race_scene") or {}
    geoms, poses = scene.get("geoms") or [], scene.get("poses") or []
    rocks = []
    for course_item in scene.get("course") or []:
        if course_item.get("kind") not in {"rock", "boulder"}:
            continue
        names = set(course_item.get("geoms") or [])
        for index, geom in enumerate(geoms):
            if geom.get("name") not in names or index >= len(poses):
                continue
            pose = poses[index]
            distance = math.hypot(float(pose[0]) - float(event["x"]), float(pose[1]) - float(event["y"]))
            rocks.append((distance, course_item, geom))
    if not rocks:
        return None
    distance, course_item, geom = min(rocks, key=lambda row: row[0])
    if distance > max(1.0, float(course_item.get("height_m", 0)) * 2.5):
        return None
    try:
        rock_index = int(geom["name"].rsplit("_", 1)[1]) + 1
    except (KeyError, ValueError):
        rock_index = 1
    return {
        "name": f"Rock {rock_index}", "size_m": course_item.get("height_m"),
        "distance_m": round(distance, 2),
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
    rock = _nearest_rock(event, run)
    if rock:
        event["rock"] = rock
    return {
        "id": design["id"], "name": design["name"], "color": design.get("color"),
        "kind": design["robot"]["kind"], "reached_goal": reached,
        "robot": design["robot"],
        "progress_percent": round(progress, 1),
        "time_s": round(float(event_time if event_time is not None else time_limit_s), 2),
        "outcome": outcome, "event": event, "reason": design.get("reason"),
        "run_id": run.get("id"),
        "samples": [
            {key: sample.get(key) for key in
             ("time_s", "x", "y", "distance_to_goal_m", "local_slope_deg", "step_m", "slip_ratio")}
            for sample in samples
        ],
    }


def _generation_replay(runs: dict[str, dict], results: list[dict], time_limit_s: float) -> dict:
    """Pack independent candidate ghost tracks into one shared-terrain 3D race replay."""
    first = next((runs.get(row["id"]) for row in results if runs.get(row["id"])), None)
    if first is None or not first.get("race_scene"):
        raise RuntimeError("The race worker did not return a 3D scene for replay.")
    first_scene = first["race_scene"]
    first_ghost = first.get("ghost")
    if not first_ghost:
        raise RuntimeError("The race worker did not record robot poses for replay.")
    first_robot_ids = set(first_ghost.get("geom_ids") or [])
    static_indices = [index for index in range(len(first_scene["geoms"])) if index not in first_robot_ids]
    static_scene = {key: value for key, value in first_scene.items()
                    if key not in {"geoms", "poses", "robots", "follow_geom", "robot_root_body"}}
    static_scene["geoms"] = []
    static_scene["poses"] = []
    for new_id, old_id in enumerate(static_indices):
        geom = dict(first_scene["geoms"][old_id])
        geom["id"] = new_id
        static_scene["geoms"].append(geom)
        static_scene["poses"].append(first_scene["poses"][old_id])

    robot_geoms, robots, tracks, failures = [], {}, [], []
    next_geom_id = len(static_scene["geoms"])
    for index, row in enumerate(results):
        run = runs.get(row["id"], {})
        scene, ghost = run.get("race_scene"), run.get("ghost")
        if not scene or not ghost:
            raise RuntimeError(f"The race worker did not record a 3D replay for {row['id']}.")
        old_ids = ghost.get("geom_ids") or []
        if not old_ids or any(geom_id >= len(scene.get("geoms", [])) for geom_id in old_ids):
            raise RuntimeError(f"The race replay for {row['id']} has invalid robot geometry.")
        initial = []
        new_ids = []
        for old_id in old_ids:
            geom = dict(scene["geoms"][old_id])
            geom["id"] = next_geom_id
            rgba = list(geom.get("rgba") or [0.4, 0.6, 0.8, 1.0])
            color = row.get("color") or COLORS[index % len(COLORS)]
            rgb = [int(color[offset:offset + 2], 16) / 255 for offset in (1, 3, 5)]
            geom["rgba"] = [*rgb, rgba[3] if len(rgba) > 3 else 1.0]
            robot_geoms.append(geom)
            initial.append(scene["poses"][old_id])
            new_ids.append(next_geom_id)
            next_geom_id += 1
        robot_id = f"evolution_{index}"
        old_follow = run.get("race_follow_geom", old_ids[0])
        follow = new_ids[old_ids.index(old_follow)] if old_follow in old_ids else new_ids[0]
        robots[robot_id] = {"label": row["name"], "follow_geom": follow, "color": color}
        raw_frames = json.loads(zlib.decompress(base64.b64decode(ghost["data"])))
        tracks.append((new_ids, raw_frames, initial))
        if not row.get("reached_goal") and row.get("event", {}).get("x") is not None:
            failures.append({
                "x": row["event"]["x"], "y": row["event"]["y"],
                "outcome": row["outcome"], "label": f"✕ {row['outcome']} · {row['name']}",
                "color": color,
            })

    if not tracks:
        raise RuntimeError("The generation race has no recorded robot tracks.")
    frame_step_s = 0.2
    frame_count = max(1, int(math.ceil(time_limit_s / frame_step_s)))
    frame_times = [round(index * frame_step_s, 3) for index in range(frame_count + 1)]
    packed_frames, indices = [], [0] * len(tracks)
    for time_s in frame_times:
        poses = list(static_scene["poses"])
        for track_index, (new_ids, raw_frames, initial) in enumerate(tracks):
            while indices[track_index] + 1 < len(raw_frames) and raw_frames[indices[track_index] + 1][0] <= time_s:
                indices[track_index] += 1
            current_poses = raw_frames[indices[track_index]][1] if raw_frames else initial
            poses.extend(current_poses)
        packed_frames.append([time_s, poses])
    compressed = zlib.compress(json.dumps(packed_frames, separators=(",", ":"), allow_nan=False).encode(), 9)
    return {
        "scene": static_scene,
        "track": {
            "encoding": "deflate-base64", "frame_count": len(packed_frames),
            "frame_step_s": frame_step_s, "robots": robots, "geoms": robot_geoms,
            "frames": base64.b64encode(compressed).decode("ascii"),
            "failure_markers": failures,
        },
    }


def race_generation(
    challenge: SceneSpec, designs: list[dict], seed: int, record_replay: bool = True,
) -> dict:
    """Run each candidate in a parallel worker on the same cloned challenge and terrain seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Race seed must be a nonnegative integer.")
    if not isinstance(challenge, SceneSpec) or challenge.domain != "robot_terrain":
        raise ValueError("A generation race requires a robot-terrain SceneSpec.")
    base = challenge.model_copy(deep=True).fill_defaults()
    if (base.terrain is None or base.mission is None or not base.mission.has_goal()
            or (base.mission.start_xy is None and base.mission.start_place is None)):
        raise ValueError("A generation race challenge must define both a start and a goal.")
    if len(base.robots) != 1:
        raise ValueError("A generation race challenge must start with one robot.")
    candidates = _validate_race_designs(designs, base.design_constraints)
    base.terrain.seed = seed
    time_limit_s = base.duration_s

    from . import batch

    variations = []
    for candidate in candidates:
        kind = candidate["robot"]["kind"]
        robot = {key: value for key, value in candidate["robot"].items()
                 if key not in {"kind", "robot_id", "controller_id"}}
        variations.append({
            "label": f"{candidate['id']} {candidate['name']}",
            "seed": seed,
            "robot_kind": kind,
            "robot": robot,
        })
    runs = batch.run_batch(base, variations, baseline=False, record=record_replay, race_telemetry=True)
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
    replay_runs = {
        candidate["id"]: by_label[f"{candidate['id']} {candidate['name']}"] for candidate in candidates
    }
    replay = _generation_replay(replay_runs, results, time_limit_s) if record_replay else None
    return {
        "challenge": base.title, "seed": seed, "time_limit_s": time_limit_s,
        "terrain": base.terrain.model_dump(mode="json"),
        "start": {"place": base.mission.start_place, "xy": base.mission.start_xy},
        "goal": {"place": base.mission.goal_place, "xy": base.mission.goal_xy,
                 "radius_m": base.mission.goal_radius_m},
        "results": results,
        **({"replay_scene": replay["scene"]} if replay else {}),
        **({"race_replay": replay["track"]} if replay else {}),
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
    if event.get("rock"):
        rock = event["rock"]
        where.append(f"{rock['name']} ({float(rock['size_m']):.2f} m rock)")
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
    constraints: RobotDesignConstraints | dict | None = None,
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
        if violations := _constraint_violations(Robot.model_validate(robot), constraints):
            raise ValueError(
                f"Child {index + 1} ({name}, parent {parent['id']}) violates challenge design constraints: "
                f"{'; '.join(violations)}. Parent stowed facts: {_stowed_facts(parent['robot'], constraints)}."
            )
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


def _constraint_fallback_change(
    parent: dict, outcome: str, attempt: int, constraints: RobotDesignConstraints | dict,
) -> tuple[dict, str]:
    robot = parent["robot"]
    limits = constraints.model_dump() if isinstance(constraints, RobotDesignConstraints) else constraints
    if robot["kind"] == "rover":
        if outcome == "stuck" and robot["wheels"] == 4 and attempt % 2 == 0:
            return {"wheels": 6}, "try six wheel contacts within the lander envelope"
        if outcome == "flipped":
            mass = min(float(limits["max_mass_kg"]), robot["mass_kg"] + 1.0 + attempt)
            if mass > robot["mass_kg"]:
                return {"mass_kg": mass}, "add permitted mass to resist the measured flip"
            return {"wheel_radius": max(0.10, robot["wheel_radius"] - 0.02 * (attempt + 1))}, \
                "lower the rover's center of mass after the measured flip"
        speed = robot["drive_speed"]
        if outcome == "too slow":
            value = min(12.0, speed + 0.5 * (attempt + 1))
            why = "increase drive speed while preserving the lander fit"
        else:
            value = max(3.0, speed - 0.5 * (attempt + 1))
            why = "retune drive speed because the size limit prevents larger running gear"
        return {"drive_speed": value}, why
    if outcome == "flipped":
        mass = min(float(limits["max_mass_kg"]), robot["mass_kg"] + 1.0 + attempt)
        if mass > robot["mass_kg"]:
            return {"mass_kg": mass}, "add permitted mass to resist the measured flip"
        return {"leg_length": max(0.12, robot["leg_length"] - 0.02 * (attempt + 1))}, \
            "shorten the legs to lower the body after the measured flip"
    gait = robot["gait_freq_hz"]
    if outcome == "too slow":
        return {"gait_freq_hz": min(3.0, gait + 0.2 * (attempt + 1))}, \
            "increase gait frequency within the stowed-size limit"
    return {"gait_freq_hz": max(1.0, gait - 0.2 * (attempt + 1))}, \
        "retune gait frequency while preserving the lander fit"


def _fallback_children(
    parents: list[dict], elite: dict, n_children: int, generation: int,
    raced_fingerprints: set[tuple], fallback_reason: str,
    constraints: RobotDesignConstraints | dict | None = None,
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
                alternatives = [(changes, mutation_reason)]
                if constraints is not None:
                    alternatives.extend(
                        _constraint_fallback_change(parent, outcome, variant, constraints)
                        for variant in range(8)
                    )
                for candidate_changes, candidate_reason in alternatives:
                    try:
                        robot, applied = _bounded_changes(parent, candidate_changes)
                    except ValueError:
                        continue
                    if _constraint_violations(Robot.model_validate(robot), constraints):
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
                        "reason": f"{parent['id']} was {outcome}{_failure_fact(parent)}; {candidate_reason} "
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
            if accepted:
                break
        if accepted is None:
            raise ValueError("No unique bounded fallback mutation remains for this generation.")
        children.append(accepted)
    if {elite["robot"]["kind"], *(row["robot"]["kind"] for row in children)} != {"rover", "walker"}:
        raise ValueError("The next generation must include at least one rover and one walker.")
    return children


def next_generation(
    results: dict | list[dict], n: int = 4,
    constraints: RobotDesignConstraints | dict | None = None,
) -> list[dict]:
    """Keep the ranked elite and request n-1 bounded, failure-guided children."""
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(COLORS):
        raise ValueError("A generation needs 2-8 designs.")
    rows = _result_rows(results)
    parents = _validate_race_designs(rows, constraints)
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
            **({"stowed": _stowed_facts(row["robot"], constraints)} if constraints is not None else {}),
        } for row in parents]
        messages = [
            {"role": "system", "content": MUTATION_SYSTEM + "\nDESIGN_SPACE: " + json.dumps(DESIGN_SPACE)},
            {"role": "user", "content": json.dumps({
                "elite_id": elite["id"], "n_children": n - 1,
                "generation": generation,
                "design_constraints": _constraint_facts(constraints),
                "FACTS": payload,
            })},
        ]
        try:
            children, attempts = llm.chat_json(
                messages, config.BUILDER_MODEL,
                lambda reply: _validate_children(reply.get("children"), parents, elite,
                                                 n - 1, generation, raced_fingerprints, constraints),
            )
        except OpenAIError as error:
            fallback_reason = f"Designer request failed ({type(error).__name__}); used bounded rule mutations."
            logger.warning("%s", fallback_reason)
        else:
            if children is None:
                logger.warning("Generation %s child proposals failed validation: %s", generation, attempts)
    if children is None:
        children = _fallback_children(parents, elite, n - 1, generation,
                                      raced_fingerprints, fallback_reason, constraints)
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


def _validate_winner_design(design: dict) -> dict:
    if not isinstance(design, dict) or not isinstance(design.get("robot"), dict):
        raise ValueError("Winner verification requires a design with a robot object.")
    raw_robot = design["robot"]
    kind = raw_robot.get("kind")
    if kind not in DESIGN_SPACE:
        raise ValueError("Only procedural rover and walker designs can be verified.")
    if any(isinstance(raw_robot.get(key), bool) for key in DESIGN_SPACE[kind]):
        raise ValueError("Winner design parameters must not be booleans.")
    robot = Robot.model_validate(raw_robot)
    if robot.kind not in DESIGN_SPACE:
        raise ValueError("Only procedural rover and walker designs can be verified.")
    for key, (low, high) in DESIGN_SPACE[robot.kind].items():
        value = getattr(robot, key)
        if not low <= value <= high:
            raise ValueError(f"{design.get('id', 'Design')} has {key} outside the {robot.kind} design space.")
    name = _text(design.get("name") or design.get("id"), "Design name", 40)
    return {**design, "name": name, "robot": robot.model_dump(mode="json")}


def verify_winner(challenge: SceneSpec, design: dict) -> dict:
    """Re-run a candidate on five new terrain/start seeds and report its Wilson interval."""
    from . import batch, cost, reliability

    candidate = _validate_winner_design(design)
    if not isinstance(challenge, SceneSpec) or challenge.domain != "robot_terrain":
        raise ValueError("Winner verification requires a robot-terrain SceneSpec.")
    base = challenge.model_copy(deep=True).fill_defaults()
    if base.terrain is None or len(base.robots) != 1:
        raise ValueError("Winner verification requires one robot and a terrain.")
    if violations := _constraint_violations(Robot.model_validate(candidate["robot"]), base.design_constraints):
        raise ValueError(f"{candidate['id']} violates challenge design constraints: {'; '.join(violations)}.")
    seed = base.terrain.seed
    robot = {key: value for key, value in candidate["robot"].items()
             if key not in {"kind", "robot_id", "controller_id"}}
    variations = [{
        "label": f"Verification seed {index}",
        "seed": seed + index,
        "perturb_seed": index,
        "robot_kind": candidate["robot"]["kind"],
        "robot": robot,
    } for index in range(1, 6)]
    with cost.track():
        runs = batch.run_batch(base, variations, baseline=False, record=False)
    if len(runs) != 5:
        raise RuntimeError(f"Winner verification expected 5 runs, received {len(runs)}.")
    scored, passed = [], 0
    for run in runs:
        if run.get("error"):
            raise RuntimeError(f"Winner verification run failed: {run['error']}")
        summary = run.get("summary") or {}
        reached = bool(summary.get("goal_reached"))
        passed += int(reached)
        start_distance = summary.get("start_distance_to_goal_m")
        closest = summary.get("closest_distance_m")
        progress = 100.0 if reached else (
            max(0.0, min(100.0, 100.0 * (start_distance - closest) / start_distance))
            if isinstance(start_distance, (int, float)) and start_distance > 0
            and isinstance(closest, (int, float)) else 0.0
        )
        scored.append({
            "run_id": run.get("id"), "seed": run.get("changes", {}).get("seed"),
            "reached_goal": reached, "outcome": summary.get("outcome"),
            "progress_percent": round(progress, 1),
            "time_s": summary.get("time_to_goal_s"),
            "final_distance_m": summary.get("final_distance_m"),
        })
    low, high = reliability.wilson(passed, len(runs))
    return {
        "passed": passed, "total": len(runs), "reliable": passed >= 4,
        "wilson_95": [round(low, 3), round(high, 3)],
        "text": reliability.ci_text(passed, len(runs)),
        "message": f"{candidate['name']} reaches the goal {passed} of 5 times ({reliability.ci_text(passed, 5)}).",
        "results": scored,
    }


def _margin_knob(challenge: SceneSpec, failures: list[dict]) -> str | None:
    from . import finder

    scene = challenge.model_copy(deep=True).fill_defaults()
    robot = scene.robots[0].robot
    fits, _ = finder.scene_knobs(scene, robot)
    scores = {"slope_deg": 0, "obstacle_scale": 0, "friction": 0}
    for row in failures:
        if row.get("reached_goal"):
            continue
        outcome = row.get("outcome")
        event = row.get("event") or {}
        if outcome in {"slipped"}:
            scores["friction"] += 3
        if event.get("local_slope_deg") is not None:
            scores["slope_deg"] += max(1, int(float(event["local_slope_deg"]) // 5))
        if event.get("step_m") is not None and float(event["step_m"]) > 0.03:
            scores["obstacle_scale"] += 2
    available = [knob for knob in scores if knob in fits]
    if not available or max(scores[knob] for knob in available) == 0:
        return None
    return max(available, key=lambda knob: (scores[knob], -("slope_deg", "obstacle_scale", "friction").index(knob)))


def _margin_text(limit: dict, obstacles: tuple[float, float] | None = None) -> str:
    if limit["never_broke"]:
        return f"It did not fail within Finder's tested {limit['name']} range."
    boundary = (float(limit["last_ok"]) + float(limit["first_fail"])) / 2
    uncertainty = abs(float(limit["first_fail"]) - float(limit["last_ok"])) / 2
    knob = limit["knob"]
    if knob == "slope_deg":
        delta = max(0.0, boundary - float(limit["baseline"]))
        return f"It would fail if the slope were about {delta:.1f}° steeper (±{uncertainty:.1f}°)."
    if knob == "friction":
        return f"It would fail if friction dropped below about {boundary:.2f} (±{uncertainty:.2f})."
    if knob == "obstacle_scale" and obstacles is not None:
        current, climb = obstacles
        rock_boundary = boundary * climb
        rock_uncertainty = uncertainty * climb
        return (f"It would fail if the rock were about {max(0.0, rock_boundary - current):.2f} m taller "
                f"(±{rock_uncertainty:.2f} m).")
    return f"It would fail near {boundary:.2f} for {limit['name']} (±{uncertainty:.2f})."


def margin_search(
    challenge: SceneSpec, design: dict, failures: list[dict] | None = None, budget: int = 6,
) -> dict:
    """Use Finder's one-condition binary search, including baseline, in at most six runs."""
    from . import finder

    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 6:
        raise ValueError("Margin search budget must be an integer from 1 to 6.")
    candidate = _validate_winner_design(design)
    scene = challenge.model_copy(deep=True).fill_defaults()
    if scene.terrain is None or len(scene.robots) != 1:
        return {"available": False, "reason": "Margin search needs one robot on a terrain.", "runs": 0}
    entry = scene.robots[0]
    entry.robot = Robot.model_validate(candidate["robot"])
    scene.robot = entry.robot.model_copy(deep=True)
    knob = _margin_knob(scene, failures or [])
    if knob is None:
        if scene.terrain.kind == "real" and any(
            (row.get("event") or {}).get("local_slope_deg") is not None
            for row in failures or []
        ):
            reason = ("The limiting feature was slope, but this real elevation map cannot be steepened "
                      "without replacing measured terrain.")
        else:
            reason = "No supported limiting feature is adjustable in this challenge."
        return {"available": False, "reason": reason, "runs": 0}
    if knob == "slope_deg" and scene.terrain.kind == "flat":
        scene = finder.prepare(scene, [knob])
    fits, _ = finder.scene_knobs(scene, entry.robot)
    if knob not in fits:
        return {"available": False, "reason": f"Finder cannot search {knob} for this challenge.", "runs": 0}
    obstacle_context = finder._obstacle_context(scene, entry.robot)
    context = {
        "base": fits, "robot_id": entry.id, "mass": entry.robot.mass_kg,
        "obstacles": obstacle_context,
    }
    scene = finder.prepare(scene, [knob])
    runner = finder.Runner(scene, entry.id, "task", budget=budget)
    baseline = runner.run([({"phase": 0, "knob": knob, "value": fits[knob]}, {})])[0]
    if baseline["broke"] is not False:
        return {
            "available": False, "reason": "The design did not pass the baseline margin run.",
            "knob": knob, "runs": len(runner.evidence), "evidence": runner.evidence,
        }
    limit = finder.search(runner, [knob], context, None)[0]
    result = {
        "available": True, "knob": knob, "name": limit["name"],
        "last_ok": limit["last_ok"], "first_fail": limit["first_fail"],
        "step": limit["step"], "never_broke": limit["never_broke"],
        "runs": len(runner.evidence), "evidence": runner.evidence,
    }
    result["text"] = _margin_text(limit, obstacle_context)
    return result


def _terrain_slope_p95(challenge: SceneSpec, height_scale: float) -> float:
    from .backends.mujoco_backend import make_heights

    scene = challenge.model_copy(deep=True).fill_defaults()
    scene.terrain.height_scale = height_scale
    heights, _ = make_heights(scene)
    spacing = float(scene.terrain.size_m) / (heights.shape[0] - 1)
    dy, dx = np.gradient(heights, spacing, spacing)
    return float(np.percentile(np.degrees(np.arctan(np.hypot(dx, dy))), 95))


def terrain_margin_search(challenge: SceneSpec, design: dict, budget: int = 6) -> dict:
    """Run separate Finder searches for boulder size and surface friction."""
    from . import finder

    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 6:
        raise ValueError("Each terrain margin search budget must be an integer from 1 to 6.")
    candidate = _validate_winner_design(design)
    scene = challenge.model_copy(deep=True).fill_defaults()
    if scene.terrain is None or scene.terrain.kind != "real" or len(scene.robots) != 1:
        unavailable = {"available": False, "reason": "Terrain margins need one robot on real elevation data.",
                       "runs": 0}
        return {"height_scale": unavailable, "friction": unavailable.copy(),
                "runs": 0, "text": unavailable["reason"]}
    entry = scene.robots[0]
    entry.robot = Robot.model_validate(candidate["robot"])
    scene.robot = entry.robot.model_copy(deep=True)
    fits, _ = finder.scene_knobs(scene, entry.robot)
    margins = {}
    rocks = [obstacle for obstacle in scene.obstacles if obstacle.kind in {"rock", "boulder"}]
    has_rock_field = bool(rocks)
    search_knobs = ("obstacle_scale", "friction") if has_rock_field else ("height_scale", "friction")
    for knob in search_knobs:
        result_key = "rock_size" if knob == "obstacle_scale" else knob
        if knob == "obstacle_scale" and has_rock_field:
            current_rock = max(float(obstacle.height_m or 0) for obstacle in rocks)
            search_base = {**fits, "obstacle_scale": 1.0}
            obstacle_context = (current_rock, current_rock)
        else:
            search_base = fits
            obstacle_context = finder._obstacle_context(scene, entry.robot)
        if knob not in search_base:
            margins[result_key] = {
                "available": False,
                "reason": "Friction is already at Finder's minimum." if knob == "friction"
                else "Finder cannot scale this terrain.",
                "runs": 0,
            }
            continue
        prepared = finder.prepare(scene, [])
        context = {
            "base": search_base, "robot_id": entry.id, "mass": entry.robot.mass_kg,
            "obstacles": obstacle_context,
        }
        runner = finder.Runner(prepared, entry.id, "task", budget=budget)
        baseline = runner.run([({"phase": 0, "knob": knob, "value": search_base[knob]}, {})])[0]
        if baseline["broke"] is not False:
            margins[result_key] = {
                "available": False, "reason": "The design did not pass the baseline margin run.",
                "runs": len(runner.evidence), "evidence": runner.evidence,
            }
            continue
        limit = finder.search(
            runner, [knob], context, None,
            harsh_values={"obstacle_scale": 2.0} if knob == "obstacle_scale" else None,
        )[0]
        margin = {
            "available": True, "knob": knob, "name": limit["name"],
            "baseline": limit["baseline"], "last_ok": limit["last_ok"],
            "first_fail": limit["first_fail"], "step": limit["step"],
            "never_broke": limit["never_broke"],
            "runs": len(runner.evidence), "evidence": runner.evidence,
        }
        if knob == "height_scale":
            base_p95 = _terrain_slope_p95(scene, float(fits[knob]))
            tested_scale = float(limit["last_ok"])
            delta = max(0.0, _terrain_slope_p95(scene, tested_scale) - base_p95)
            if limit["never_broke"]:
                margin["text"] = (
                    f"Still climbs out at terrain ×{tested_scale:.2f}; P95 slopes rise about {delta:.1f}° "
                    "and failure was not found in the tested range."
                )
            else:
                margin["text"] = (
                    f"Still climbs out with P95 slopes about {delta:.1f}° steeper (terrain ×{tested_scale:.2f}); "
                    f"fails above ×{float(limit['first_fail']):.2f}."
                )
        elif knob == "obstacle_scale":
            if limit["never_broke"]:
                margin["text"] = (
                    f"Still climbs with {current_rock * float(limit['last_ok']):.2f} m rocks; "
                    "failure was not found in the tested range."
                )
            else:
                last_ok = float(limit["last_ok"]) * current_rock
                first_fail = float(limit["first_fail"]) * current_rock
                margin["text"] = (
                    f"Still climbs with {last_ok:.2f} m rocks; fails above about {first_fail:.2f} m."
                )
        elif limit["never_broke"]:
            margin["text"] = (
                f"Still climbs out at friction {float(limit['last_ok']):.2f}; "
                "failure was not found in the tested range."
            )
        else:
            margin["text"] = (
                f"Still climbs out at friction {float(limit['last_ok']):.2f}; "
                f"fails below about {float(limit['first_fail']):.2f}."
            )
        margins[result_key] = margin
    result_keys = ["rock_size" if key == "obstacle_scale" else key for key in search_knobs]
    margin_text = "; ".join(
        margins[key].get("text") or margins[key].get("reason", "not available")
        for key in result_keys
    )
    return {
        **margins, "runs": sum(margin["runs"] for margin in margins.values()),
        "available": any(margin["available"] for margin in margins.values()),
        "text": margin_text,
    }


def _honesty_line(challenge: SceneSpec) -> str:
    scene = challenge.model_copy(deep=True).fill_defaults()
    if scene.terrain is not None and scene.terrain.kind == "real":
        from .terrain_data import load_preset, load_structure_patch

        _, metadata = (load_structure_patch(scene.terrain.preset, scene.terrain.structure_crop_size_m)
                       if scene.terrain.structure_crop_size_m is not None else load_preset(scene.terrain.preset))
        accuracy = metadata.get("accuracy_note") or (
            f"{metadata['meters_per_pixel']:g} m per source pixel"
            if metadata.get("meters_per_pixel") is not None else "not specified in terrain metadata"
        )
        accuracy = accuracy.rstrip(" .")
        return (f"Tested in simulation on real elevation data (accuracy: {accuracy}). "
                "Designs are simplified procedural robots.")
    terrain = scene.terrain.kind if scene.terrain is not None else "unspecified"
    return (f"Tested in simulation on {terrain} terrain (not real elevation data). "
            "Designs are simplified procedural robots.")


def _escalate_to_margin_failure(challenge: SceneSpec, margin: dict) -> tuple[SceneSpec, dict] | None:
    rocks = [obstacle for obstacle in challenge.obstacles if obstacle.kind in {"rock", "boulder"}]
    rock_limit = margin.get("rock_size") or {}
    if rocks and rock_limit.get("available") and rock_limit.get("first_fail") is not None:
        original_size = max(float(obstacle.height_m or 0) for obstacle in rocks)
        fail_size = original_size * float(rock_limit["first_fail"])
        if math.isfinite(fail_size) and fail_size > original_size:
            harder = challenge.model_copy(deep=True)
            for obstacle in harder.obstacles:
                if obstacle.kind in {"rock", "boulder"}:
                    obstacle.height_m = fail_size
            harder.title = f"{challenge.title} (Gen 0 edge: {fail_size:.2f} m rocks)"
            harder.assumptions.append(
                f"Escalated after a verified generation-zero design failed with approximately {fail_size:.2f} m rocks."
            )
            return harder, {
                "triggered": True, "feature": "rock size",
                "original_rock_size_m": original_size,
                "gen0_failed_at_rock_size_m": fail_size,
                "challenge_rock_size_m": fail_size,
                "margin": rock_limit,
            }
    friction_limit = margin.get("friction") or {}
    if (friction_limit.get("available") and friction_limit.get("first_fail") is not None
            and float(friction_limit["first_fail"]) < float(challenge.environment.friction)):
        fail_friction = float(friction_limit["first_fail"])
        harder = challenge.model_copy(deep=True)
        harder.environment.friction = fail_friction
        harder.title = f"{challenge.title} (Gen 0 edge: friction {fail_friction:.2f})"
        harder.assumptions.append(
            f"Escalated after a verified generation-zero design failed below friction {fail_friction:.2f}."
        )
        return harder, {
            "triggered": True, "feature": "friction",
            "original_friction": challenge.environment.friction,
            "gen0_failed_at_friction": fail_friction,
            "challenge_friction": fail_friction,
            "margin": friction_limit,
        }
    return None


def evolve(
    challenge: SceneSpec, max_generations: int = 5, n: int = 4, seed: int = 0,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Propose, race, and mutate generations until success or the bounded race budget is spent."""
    if (isinstance(max_generations, bool) or not isinstance(max_generations, int)
            or not 1 <= max_generations <= MAX_EVOLUTION_GENERATIONS):
        raise ValueError(f"max_generations must be an integer from 1 to {MAX_EVOLUTION_GENERATIONS}.")
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= len(COLORS):
        raise ValueError("A generation needs 2-8 designs.")
    if max_generations * n > MAX_EVOLUTION_RUNS:
        raise ValueError(f"Evolution is limited to {MAX_EVOLUTION_RUNS} candidate races.")
    from . import cost

    with cost.track() as spent:
        original_challenge = challenge.model_copy(deep=True).fill_defaults()
        active_challenge = original_challenge.model_copy(deep=True)
        if on_event:
            on_event({"type": "agent", "agent": "Designer", "status": "working",
                      "message": "Proposing diverse procedural robots from the place's terrain facts…"})
        designs = propose_generation0(active_challenge, n)
        if on_event:
            on_event({"type": "agent", "agent": "Designer", "status": "done",
                      "message": f"Proposed {len(designs)} bounded designs for generation 0."})
        generations, family_tree, all_raced = [], [], []
        success = False
        runs_used = 0
        races_used = 0
        replay_scene = None
        winner = None
        winner_verification = None
        winner_margin = None
        rejected_winners = []
        budget_stop = None
        escalation = None
        original_winner = None
        original_verification = None
        original_margin = None
        verified_on_active: dict[tuple, dict] = {}
        for generation in range(max_generations):
            if races_used + n > MAX_EVOLUTION_RUNS:
                budget_stop = (f"Evolution stopped after {races_used}/{MAX_EVOLUTION_RUNS} candidate races; "
                               f"the next generation would exceed the race budget.")
                break
            if on_event:
                on_event({"type": "agent", "agent": "Simulator", "status": "working",
                          "message": f"Racing generation {generation} side by side on the same challenge…"})
            race = race_generation(active_challenge, designs, seed)
            runs_used += len(race["results"])
            races_used += len(race["results"])
            generation_scene = race.pop("replay_scene", None)
            replay_scene = replay_scene or generation_scene
            race["replay_scene"] = generation_scene
            race["generation"] = generation
            race["challenge_conditions"] = {
                "friction": active_challenge.environment.friction,
                "rocks": [
                    {"size_m": obstacle.height_m, "count": obstacle.count}
                    for obstacle in active_challenge.obstacles if obstacle.kind in {"rock", "boulder"}
                ],
            }
            generations.append(race)
            if on_event and generation_scene is not None:
                on_event({"type": "race_scene", "scene": generation_scene, "generation": generation})
            if on_event:
                on_event({"type": "generation", "generation": race})
            current_by_id = {design["id"]: design for design in designs}
            for result in race["results"]:
                source = current_by_id[result["id"]]
                result.update(
                    robot=source["robot"], reason=source.get("reason"),
                    color=source.get("color"), source=source.get("source"),
                )
                all_raced.append({**result, "generation": generation})
            # Stop rule: verify the two fastest designs that reached the goal; a pass is >= 4/5.
            reached = sorted((row for row in race["results"] if row["reached_goal"]),
                             key=lambda row: (float(row.get("time_s") or float("inf")), row["id"]))
            contenders = reached[:VERIFIED_PER_GENERATION]
            for result in contenders:
                raced_copy = next(row for row in reversed(all_raced) if row["id"] == result["id"])
                key = _fingerprint(result["robot"])
                verification = verified_on_active.get(key)
                if verification is None:
                    if on_event:
                        on_event({"type": "agent", "agent": "Simulator", "status": "working",
                                  "message": f"Verifying apparent winner {result['id']} on five new seeds…"})
                    verification_challenge = active_challenge.model_copy(deep=True)
                    if verification_challenge.terrain is not None:
                        verification_challenge.terrain.seed = seed
                    verification = verify_winner(verification_challenge, result)
                    runs_used += verification["total"]
                    verified_on_active[key] = verification
                    if not verification["reliable"]:
                        rejected_winners.append({"id": result["id"], "verification": verification})
                result["verification"] = verification
                result["verified_winner"] = verification["reliable"]
                raced_copy["verification"] = verification
                raced_copy["verified_winner"] = verification["reliable"]
                if verification["reliable"]:
                    continue
                result["race_reached_goal"] = True
                result["reached_goal"] = False
                failed_verifications = [run for run in verification.get("results", []) if not run["reached_goal"]]
                failure_classes = [run.get("outcome") for run in failed_verifications]
                failure_classes = [
                    {"fell": "flipped", "left_terrain": "stuck", "out_of_time": "too slow"}.get(item, item)
                    for item in failure_classes
                ]
                result["outcome"] = max(
                    (item for item in failure_classes if item in {"flipped", "stuck", "slipped", "too slow"}),
                    key=failure_classes.count, default="too slow",
                )
                result["progress_percent"] = max(
                    (float(run.get("progress_percent", 0)) for run in failed_verifications), default=0.0,
                )
                raced_copy.update({
                    "race_reached_goal": True, "reached_goal": False,
                    "outcome": result["outcome"], "progress_percent": result["progress_percent"],
                })
            passing = [row for row in contenders if row["verification"]["reliable"]]
            escalated_now = False
            if passing:
                # More reliable wins; ties go to the faster race time.
                result = max(passing, key=lambda row: (
                    int(row["verification"].get("passed", 0)), -float(row.get("time_s") or float("inf")),
                ))
                verification = result["verification"]
                verification_challenge = active_challenge.model_copy(deep=True)
                if verification_challenge.terrain is not None:
                    verification_challenge.terrain.seed = seed
                margin = terrain_margin_search(verification_challenge, result, budget=6)
                runs_used += margin.get("runs", 0)
                escalation_result = (_escalate_to_margin_failure(active_challenge, margin)
                                     if generation == 0 and escalation is None else None)
                if escalation_result is not None:
                    active_challenge, escalation = escalation_result
                    verified_on_active = {}
                    escalated_now = True
                    escalation.update({
                        "message": "Gen 0 already climbs out. Raising the challenge to find its edge.",
                        "gen0_winner_id": result["id"],
                        "gen0_verification": verification["text"],
                        "gen0_margin": margin,
                    })
                    original_winner = result
                    original_verification = verification
                    original_margin = margin
                    result["verified_on_original_challenge"] = True
                    next(row for row in reversed(all_raced) if row["id"] == result["id"])[
                        "verified_on_original_challenge"] = True
                    if on_event:
                        on_event({"type": "agent", "agent": "Simulator", "status": "working",
                                  "message": escalation["message"]})
                else:
                    success = True
                    winner = result
                    winner_verification = verification
                    winner_margin = margin
                    if on_event:
                        on_event({"type": "agent", "agent": "Simulator", "status": "done",
                                  "message": f"Winner verified: {verification['text']}."})
            race["stop_check"] = {
                "reached": [row["id"] for row in reached],
                "verified": [{"id": row["id"], "passed": row["verification"].get("passed"),
                              "total": row["verification"].get("total"),
                              "reliable": row["verification"]["reliable"]} for row in contenders],
                "escalated": escalated_now, "stopped": success,
            }
            if on_event:
                checked = ", ".join(
                    f"{row['id']} {row['verification'].get('passed', '?')}/{row['verification'].get('total', '?')}"
                    for row in contenders
                ) or "none"
                on_event({"type": "agent", "agent": "Simulator", "status": "working",
                          "message": (f"Generation {generation} stop check: reached "
                                      f"{', '.join(race['stop_check']['reached']) or 'none'}; verified {checked}; "
                                      f"escalated {'yes' if escalated_now else 'no'}; "
                                      f"{'stop' if success else 'continue'}.")})
            if on_event:
                on_event({"type": "agent", "agent": "Simulator", "status": "done",
                          "message": f"Generation {generation} race complete."})
            if success:
                break
            if generation + 1 < max_generations and races_used + n <= MAX_EVOLUTION_RUNS:
                if on_event:
                    on_event({"type": "agent", "agent": "Analyst", "status": "working",
                              "message": "Explaining failure locations and outcomes from race telemetry…"})
                    on_event({"type": "agent", "agent": "Analyst", "status": "done",
                              "message": _failure_report(max(race["results"], key=lambda row: row["progress_percent"]))})
                    on_event({"type": "agent", "agent": "Designer", "status": "working",
                              "message": "Proposing children linked to the measured failures…"})
                generation_results = {
                    "generation": generation,
                    "results": race["results"],
                    "raced_designs": all_raced,
                }
                next_designs = (
                    next_generation(generation_results, n) if active_challenge.design_constraints is None
                    else next_generation(generation_results, n, active_challenge.design_constraints)
                )
                for child in next_designs[1:]:
                    family_tree.append({
                        "parent_id": child["parent_id"], "child_id": child["id"],
                        "generation": generation + 1, "changes": child["parameter_changes"],
                        "reason": child["reason"], "source": child.get("source"),
                        "result": None,
                    })
                designs = next_designs
                if on_event:
                    on_event({"type": "tree", "family_tree": family_tree, "designs": next_designs})
                    on_event({"type": "agent", "agent": "Designer", "status": "done",
                              "message": f"Generation {generation + 1} designs and family links are ready."})
            elif generation + 1 < max_generations:
                budget_stop = (f"Evolution stopped after {races_used}/{MAX_EVOLUTION_RUNS} candidate races; "
                               f"{n} more candidate races would exceed the race budget.")
        by_id = {row["id"]: row for row in all_raced}
        for edge in family_tree:
            edge["result"] = {
                key: by_id[edge["child_id"]].get(key)
                for key in ("outcome", "progress_percent", "reached_goal", "event")
            } if edge["child_id"] in by_id else None
        ranked_rows = (all_raced if escalation is None else
                       [row for row in all_raced if row["generation"] > 0] or all_raced)
        best = max(ranked_rows, key=lambda row: (
            bool(row["reached_goal"]), float(row.get("progress_percent", 0)),
            -float(row.get("time_s", float("inf"))) if row["reached_goal"] else 0,
        ))
        if success:
            margin_text = winner_margin["text"] if winner_margin.get("available") else winner_margin.get("reason")
            report = (f"Goal reached by {winner['id']} and verified "
                      f"({winner_verification['text']}). {margin_text}")
            if escalation is not None:
                if escalation["feature"] == "rock size":
                    evolved_limit = (winner_margin.get("rock_size") or {}).get("last_ok")
                    current_rock = max(
                        float(obstacle.height_m or 0) for obstacle in active_challenge.obstacles
                        if obstacle.kind in {"rock", "boulder"}
                    )
                    evolved_limit = (current_rock * float(evolved_limit)
                                     if evolved_limit is not None else current_rock)
                    report = (
                        f"Gen 0 climbed the original wall; evolved designs handle rocks up to "
                        f"{evolved_limit:.2f} m (gen 0 failed at "
                        f"{escalation['gen0_failed_at_rock_size_m']:.2f} m). {report}"
                    )
                else:
                    report = (
                        f"Gen 0 climbed the original wall; evolution continued at friction "
                        f"{escalation['gen0_failed_at_friction']:.2f}. {report}"
                    )
        else:
            report = _failure_report(best)
            if escalation is not None:
                if escalation["feature"] == "rock size":
                    report = (
                        f"{escalation['message']} Gen 0 climbed the original wall "
                        f"({escalation['original_rock_size_m']:.2f} m rocks); no evolved design verified on the "
                        f"raised {escalation['challenge_rock_size_m']:.2f} m rocks. {report}"
                    )
                else:
                    report = f"{escalation['message']} {report}"
            if rejected_winners:
                attempts: dict[str, list[str]] = {}
                for row in rejected_winners:
                    attempts.setdefault(row["id"], []).append(row["verification"]["text"])
                rejected = ", ".join(
                    f"{design_id} was unreliable ({texts[-1]}"
                    + (f"; rejected {len(texts)} times" if len(texts) > 1 else "") + ")"
                    for design_id, texts in attempts.items()
                )
                report += f" Rejected apparent winner(s): {rejected}."
            if budget_stop:
                report += f" {budget_stop}"
        if on_event:
            on_event({"type": "agent", "agent": "Analyst", "status": "done",
                      "message": report if not success else
                      f"{winner['id']} verified {winner_verification['text']}; {winner_margin.get('text') or winner_margin.get('reason')}"})
        return {
            "challenge": active_challenge.title, "success": success,
            "challenge_spec": active_challenge.model_dump(mode="json"),
            "challenge_scene": (generations[-1].get("replay_scene") if generations else None) or replay_scene,
            "challenge_conditions": {
                "friction": active_challenge.environment.friction,
                "duration_s": active_challenge.duration_s,
                "rocks": [
                    {"size_m": obstacle.height_m, "count": obstacle.count}
                    for obstacle in active_challenge.obstacles if obstacle.kind in {"rock", "boulder"}
                ],
                "design_constraints": (active_challenge.design_constraints.model_dump(mode="json")
                                       if active_challenge.design_constraints else None),
            },
            "original_challenge_conditions": {
                "friction": original_challenge.environment.friction,
                "rocks": [
                    {"size_m": obstacle.height_m, "count": obstacle.count}
                    for obstacle in original_challenge.obstacles if obstacle.kind in {"rock", "boulder"}
                ],
            },
            "race_scene": replay_scene,
            "generations": generations, "family_tree": family_tree,
            "best": best, "winner": winner, "verification": winner_verification,
            "margin": winner_margin, "honesty": _honesty_line(challenge),
            "rejected_winners": rejected_winners, "escalation": escalation,
            "original_winner": original_winner,
            "original_verification": original_verification,
            "original_margin": original_margin,
            "report": report,
            "race_count": races_used, "simulation_count": runs_used,
            "cost": spent.snapshot(),
        }
