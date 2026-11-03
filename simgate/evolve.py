"""Bounded generation-zero design proposals. No simulation or selection is performed."""

import json
import logging
import math
from typing import Any

import numpy as np

from . import config, llm
from .spec import Robot, SceneSpec

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
