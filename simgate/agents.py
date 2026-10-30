"""The agent team: Planner, Router, Builder, Checker, Explainer.

Each agent works with or without a model. Without one (MOCK mode), simple
rules stand in so the pipeline can be tested offline.
"""
import json
import re

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import config, llm
from .backends import REGISTRY, describe
from .backends.base import BuildResult, Simulator
from .batch import Variation, _validate_variation
from .spec import SceneSpec
from .terrain_data import list_presets

SCHEMA_HINT = json.dumps(SceneSpec.model_json_schema(), separators=(",", ":"))

PLANNER_SYSTEM = f"""You are the Planner in a physics sandbox for people with no simulation experience.
Turn the user's request into a JSON scene spec. Available simulators:
{describe()}

Rules:
- Choose the domain that fits the question.
- Use realistic values. If the user doesn't say something, pick a sensible default and add a short plain-English line to "assumptions".
- environment.body is one of earth, moon, mars, titan, europa, enceladus. Saturn and Jupiter have no solid surface: use titan or europa and note it in assumptions.
- Leave environment.gravity and environment.friction null unless the user gives numbers; the system fills real reference values.
- robot.kind is "rover" (wheels), "walker" (built-in gait), or "library" (a registered library robot).
  Use "walker" for built-in walkers: legs, dog, quadruped, insect, spider, hexapod. legs is 4 or 6.
- For G1, a humanoid, or a Unitree robot, select robot_id "g1" and policy_id "g1_walking".
  Only use robot and policy IDs from the library list below; never invent IDs.
- Keep the walker's gait_freq_hz and stride_deg at their defaults unless the user asks to walk faster/slower or take bigger steps; large changes can make it fall.
- Real terrain: when the user asks for a real place or "real terrain", set terrain.kind to "real" and terrain.preset to one of the available presets below. Never invent a preset name; if none fits, use generated terrain and say so in assumptions.
- Start and goal: when the user wants the robot to get somewhere ("reach", "get to", "go to", "from X to Y",
  "drive down to the bottom", "climb to the top"), fill "mission". Prefer named places in start_place / goal_place:
  start_area, center, lowest, highest, north, south, east, west. The system finds them on the actual terrain
  (crater floor, valley or bottom = lowest; rim, summit, peak or top = highest). Use start_xy / goal_xy only when the
  user gives coordinates: metres, (0, 0) at the centre, +x east, +y north, terrain spans ±size_m/2.
  Leave start_heading_deg null (the robot starts facing the goal) and leave duration_s at its default unless the
  user gives a time limit. Without a destination, leave mission null.
- Obstacles: when the user asks for an obstacle course or names obstacles (rocks, boulders, a step, stairs,
  a gap/trench/ditch/crevasse, a ramp, a wall, logs), fill "obstacles" in the order they should be met.
  Give kind, size (small/medium/large) and count; use height_m only for an exact number the user gives
  (for a gap it is the width). Leave placement "path" unless the user says scattered or names a place.
  Never give coordinates for obstacles; the system places them. A course gets a goal on its far side
  automatically, so mission can stay null. Use terrain kind "flat" for a course unless the user asks for
  other terrain. When editing, keep existing obstacles unless the user removes them.
- Reply with ONLY the JSON object.

Available real-terrain presets:
{{presets}}

Available robot library:
{{library}}

JSON schema:
{SCHEMA_HINT}"""

EDIT_RULES = """

You are now EDITING an existing spec. Apply the user's change, keep everything else the same,
and return the full updated JSON. If the change needs a different simulator, change "domain"."""

EXPLAIN_SYSTEM = """You are the Explainer in a physics sandbox for non-experts.
Given the scene and the run metrics, explain in 3–5 short plain sentences what happened and why,
using the physics (gravity, friction, slope, resonance, drift). Then suggest one concrete thing to try next,
phrased as something the user could type. No markdown headings, no bullet lists.

Evidence rules:
- Base the explanation on the measurements you're given. State measured facts plainly
  (e.g. "it tipped backward at 5.1 s on a 32° patch, facing uphill").
- Anything you infer rather than measure, mark with "likely". Never invent causes that weren't measured:
  no specific rocks, bumps, foot placements or wind unless the data shows them.
- Robot metrics: fall_direction is relative to the way the robot was facing; slope_along_heading_deg > 0 means
  uphill ahead, < 0 downhill ahead; cross_slope_deg is the sideways tilt of the ground; downhill_side says
  which side of the robot was downhill; local_slope_deg is measured where the fall began (or at the end if it
  didn't fall); tilt_trace shows tilt over time, so you can tell a sudden fall from a slow build-up.
- Slip: slip_ratio is the share of foot/wheel contact time the touching point slid faster than 0.3 m/s.
  For the walkers' simple fixed gait, 0.15–0.30 is normal scuffing on firm ground. Above about 0.4 is real
  slipping. Wheels rolling normally are near 0.
- Tested settings: the walkers use an open-loop rhythmic gait tuned at gait_freq_hz 2.0 and stride_deg 15.
  At those settings the 6-legged walker stayed upright on all 9 test terrains (flat, hills, rocky, craters,
  sand, ice, a 10° gravel slope, on Earth, Mars, Moon, Titan and Europa); the 4-legged walker fell on the
  10° gravel slope. If you suggest gait settings outside the tested values, say they are untested.
- left_terrain = true means the robot ended outside the simulated patch of ground (it slid or drove off the edge);
  mention it, because the 3D view will show it falling into empty space.
- Start and goal: when goal_reached is present, lead with whether the robot reached the goal and how long it took,
  or how close it got (closest_distance_m, final_distance_m). outcome says how the attempt ended: reached, fell,
  stuck (no real movement in its last 3 s), out_of_time (still moving when time ran out) or left_terrain.
  path_efficiency is progress toward the goal divided by the distance travelled (1.0 = a straight line).
  The robot steers itself straight at the goal; it does not plan a route around steep ground.
- Multi-robot runs: use leaderboard rank and per-robot measurements. Lead with who won or arrived, then explain
  why the others did not, using only their reported outcomes, distances, tilt, and contacts. Each robot's kind is
  given and its "robot" lists only the properties that robot has. Never compare a property one robot doesn't
  have (a walker has no wheels or wheel speed; a rover has no legs or gait).
- Obstacles: obstacles lists each one in course order with its height (gap: width_m is the opening, height_m its
  depth), whether the robot touched it (touched_at_s) and got past it (cleared). stopped_by is the first obstacle it
  didn't get past. robot_rough_limits are measured limits for this robot (climb_m: tallest step it usually
  climbs; gap_m: widest gap it usually crosses, 0 = it can't cross gaps). Compare the obstacle to those limits
  to explain why, e.g. "the 0.15 m step is taller than the 0.09 m it can usually climb".
- For buildings, explain using the period, drift and resonance numbers given."""

EXPLAIN_STRESS_SYSTEM = """You are the Stress-test Explainer in a physics sandbox for non-experts.
Explain the measured results in 3–5 short plain sentences. State how many runs passed, compare the
baseline with the variations, and identify the clearest pattern across the runs: which changed
condition aligns with failures and any measured threshold supported by the data. Treat association
as an inference and say "likely"; do not claim causation or invent thresholds. Use only reported
metrics, and finish with one concrete suggestion to type next. No markdown headings or bullet lists.
In a race each robot's kind is given and its "robot" lists only the properties it has: never compare a property
one robot doesn't have (a walker has no wheels or wheel speed; a rover has no legs or gait). Walker gaits were
tested only at 2.0 Hz and a 15° stride; call any other gait setting untested.
"""


class PlannedVariation(Variation):
    label: str = Field(min_length=1, max_length=80)


class StressPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variations: list[PlannedVariation] = Field(min_length=8, max_length=16)


STRESS_PLANNER_SYSTEM = f"""You are the Stress planner in a physical simulation sandbox.
Choose 8–16 useful condition variations for the user's robustness question. The unchanged scene is
added separately as the Baseline; do not include it in your variations.
Use only the supported knobs in the schema. Vary gravity without changing the environment body,
and use surface rather than friction when comparing named materials. Make variations relevant:
"will it work on Mars?" should emphasize Mars gravity and surfaces; "how robust?" should mix
gravity, surfaces, payload, and disturbances. Only use slope_deg for slope terrain and obstacle_scale
when the scene has obstacles. Return JSON only, matching this schema:
{json.dumps(StressPlan.model_json_schema(), separators=(",", ":"))}"""


def presets_text() -> str:
    found = list_presets()
    if not found:
        return "(none installed)"
    return "\n".join(f"- {name}: {m['label']} ({m.get('body', '?')}, {m['size_m']:.0f} m square, "
                     f"{m['relief_m']:.1f} m relief, typical slope {m.get('typical_slope_deg', '?')}°)" for name, m in found.items())


def planner_system() -> str:
    """Build per request so newly installed terrains and library cards are available."""
    from .library import LibraryRegistry

    registry = LibraryRegistry()
    library = []
    for robot in registry.robots():
        policies = ", ".join(
            f"{policy.id} ({policy.name})"
            for policy in registry.compatible_policies(robot.id)
        ) or "none"
        library.append(
            f"- robot_id={robot.id}: {robot.name} ({robot.kind}); compatible policies: {policies}"
        )
    return (
        PLANNER_SYSTEM.replace("{presets}", presets_text())
        .replace("{library}", "\n".join(library) if library else "(none installed)")
    )


AFFIRM = re.compile(r"^\s*(y|yes|yep|yeah|ok|okay|sure|go|run|run it|looks good|looks right|correct|perfect|do it)\b", re.I)


# ── Planner ────────────────────────────────────────────────────────────────

def _mock_plan(text: str, current: SceneSpec | None = None) -> SceneSpec:
    t = text.lower()
    spec = current.model_copy(deep=True) if current else None
    is_structure = any(w in t for w in ("building", "house", "tower", "frame", "story", "storey", "earthquake", "quake"))
    is_robot = any(w in t for w in ("robot", "rover", "walk", "leg", "dog", "quadruped", "hexapod", "spider", "insect", "humanoid", "unitree")) or bool(re.search(r"\bg1\b", t))
    if spec is None or (is_structure and spec.domain != "structure") or (is_robot and spec.domain != "robot_terrain"):
        spec = SceneSpec(domain="structure" if is_structure else "robot_terrain")
    spec.assumptions = []

    for body in ("earth", "moon", "mars", "titan", "europa", "enceladus"):
        if body in t:
            spec.environment.body, spec.environment.gravity = body, None
    if "saturn" in t:
        spec.environment.body, spec.environment.gravity = "titan", None
        spec.assumptions.append("Saturn has no solid surface, so I used its moon Titan.")
    surface_given = False
    for surface in ("ice", "sand", "mud", "gravel", "regolith", "rock"):
        if re.search(rf"\b{surface}\b(?! field)", t):   # "rocks" (obstacles) and "rocky" (terrain) are not the surface
            spec.environment.surface, spec.environment.friction, surface_given = surface, None, True
    if not surface_given and current is None:   # sensible ground for each world
        typical = {"moon": "regolith", "mars": "regolith", "titan": "sand", "europa": "ice", "enceladus": "ice"}
        spec.environment.surface = typical.get(spec.environment.body, spec.environment.surface)
        spec.environment.friction = None

    if spec.domain == "robot_terrain":
        from .spec import Robot, Terrain
        spec.terrain = spec.terrain or Terrain()
        spec.robot = spec.robot or Robot()
        for kind in ("slope", "hills", "rocky", "craters", "flat"):
            if kind in t or (kind == "slope" and ("hill" in t and "climb" in t)):
                spec.terrain.kind = kind
        real = list_presets()
        named = [n for n, meta in real.items() if n.lower() in t or (   # an uploaded terrain: only its whole name
            meta["label"].lower() if meta.get("uploaded") else meta["label"].split()[0].lower()) in t]
        wants_real = any(w in t for w in ("real terrain", "real mars", "real data", "nasa", "hirise", "real ground"))
        if named or (wants_real and real):
            pick = named[0] if named else next((n for n, meta in real.items() if meta.get("body") in t), next(iter(real)))
            spec.terrain.kind, spec.terrain.preset = "real", pick
        elif wants_real:
            spec.assumptions.append("No real-terrain data is installed yet, so I generated terrain instead.")
        if m := re.search(r"(\d+)\s*(?:°|deg|degree)", t):
            spec.terrain.kind, spec.terrain.slope_deg = "slope", float(m.group(1))
        if "steeper" in t:   # 10° steeper; hills and flat ground count as 0°
            base = spec.terrain.slope_deg if spec.terrain.kind == "slope" else 0.0
            spec.terrain.kind = "slope"
            spec.terrain.slope_deg = min(base + 10, 45)
        if "bigger rock" in t or "rougher" in t:
            spec.terrain.roughness = min(spec.terrain.roughness + 0.3, 1.0)
        rb = spec.robot
        if any(w in t for w in ("walk", "leg", "dog", "quadruped", "hexapod", "spider", "insect")):
            rb.kind = "walker"
        if "wheel" in t or "rover" in t:
            rb.kind = "rover"
        if re.search(r"\bg1\b", t) or "humanoid" in t or "unitree" in t:
            rb.kind, rb.robot_id, rb.policy_id = "library", "g1", "g1_walking"
        # A count applies only to the part it names: "six-legged" never sets wheels, "six wheels" never sets legs.
        for number, count in (("six|6", 6), ("four|4", 4)):
            if re.search(rf"\b(?:{number})[\s-]*leg", t):
                rb.legs = count
            if re.search(rf"\b(?:{number})[\s-]*wheel", t):
                rb.wheels = count
        if any(w in t for w in ("hexapod", "insect", "spider")):
            rb.legs = 6
        _mock_obstacles(spec, t, new_scene=current is None)
        _mock_mission(spec, t)
        if "bigger wheel" in t or "larger wheel" in t:
            rb.wheel_radius = round(rb.wheel_radius * 1.5, 3)
        if "longer leg" in t:
            rb.leg_length = round(min(rb.leg_length * 1.3, 0.4), 3)
        if "faster" in t:
            rb.drive_speed *= 1.5
            rb.gait_freq_hz = round(min(rb.gait_freq_hz * 1.25, 3.0), 2)
        if "slower" in t:
            rb.drive_speed /= 1.5
            rb.gait_freq_hz = round(max(rb.gait_freq_hz / 1.25, 0.8), 2)
        what = library_robot_name(rb) if rb.kind == "library" else (
            f"{rb.legs}-legged walker" if rb.kind == "walker" else f"{rb.wheels}-wheel rover"
        )
        if spec.terrain.kind == "real":
            spec.title = f"{what.capitalize()} on real terrain: {list_presets()[spec.terrain.preset]['label']}"
        else:
            spec.title = f"{what.capitalize()} on {spec.terrain.kind} terrain, {spec.environment.body.title()}"
    else:
        from .spec import Load, Structure
        spec.structure = spec.structure or Structure()
        spec.load = spec.load or Load()
        if m := re.search(r"(\d+)[\s-]*(?:story|storey|floor)", t):
            spec.structure.stories = int(m.group(1))
        if "taller" in t:
            spec.structure.stories += 2
        if "stronger" in t or "stiffer" in t:
            spec.structure.section = "heavy"
        if "harder" in t or "stronger quake" in t:
            spec.load.pga_g = round(spec.load.pga_g * 1.5, 2)
        if m := re.search(r"(\d+(?:\.\d+)?)\s*g\b", t):
            spec.load.pga_g = float(m.group(1))
        spec.title = f"{spec.structure.stories}-story frame in an earthquake"
    if spec.domain == "robot_terrain" and spec.robots and spec.robot is not None:
        first = spec.robots[0]
        first.robot = spec.robot.model_copy(deep=True)
        if spec.mission is not None:
            first.start_xy = spec.mission.start_xy
            first.start_place = spec.mission.start_place
            first.start_heading_deg = spec.mission.start_heading_deg
            if spec.mission.has_goal():
                for entry in spec.robots:
                    entry.goal_xy = spec.mission.goal_xy
                    entry.goal_place = spec.mission.goal_place
                    entry.goal_radius_m = spec.mission.goal_radius_m
    return spec.fill_defaults()


TYPICAL_SURFACE = {"moon": "regolith", "mars": "regolith", "titan": "sand", "europa": "ice", "enceladus": "ice"}
_COUNT_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def library_robot_name(robot) -> str:
    """A library robot's name for titles and text: "G1 humanoid", or an uploaded robot's own name."""
    from .library import LibraryRegistry

    try:
        name = LibraryRegistry().robot(robot.robot_id).name
    except ValueError:
        return "library robot"
    return name[:1] + name[1:].lower()


def library_robot_kind(robot) -> str:
    """wheeled, legged, humanoid or other, from the robot's card."""
    from .library import LibraryRegistry

    try:
        return LibraryRegistry().robot(robot.robot_id).kind
    except ValueError:
        return "other"


def scene_title(spec: SceneSpec) -> str:
    """The rules planner's title, rebuilt from a finished spec: robots, terrain (slope angle, unusual
    surface, real place), world and course. Used after every edit so the title matches the scene."""
    if spec.domain == "structure":
        return f"{spec.structure.stories}-story frame in an earthquake"

    def what(robot) -> str:
        if robot.kind == "library":
            return library_robot_name(robot)
        return f"{robot.legs}-legged walker" if robot.kind == "walker" else f"{robot.wheels}-wheel rover"

    robots = [entry.robot for entry in spec.robots] or ([spec.robot] if spec.robot else [])
    groups: dict[str, int] = {}
    for robot in robots:
        groups[what(robot)] = groups.get(what(robot), 0) + 1
    names = [name if count == 1 else f"{_COUNT_WORDS[count]} {name}s" for name, count in groups.items()] or ["Scene"]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    terrain, environment = spec.terrain, spec.environment
    if terrain is not None and terrain.kind == "real":
        where = f"real terrain: {list_presets()[terrain.preset]['label']}"
    else:
        surface = environment.surface if environment.surface != TYPICAL_SURFACE.get(environment.body, "rock") else ""
        if terrain is not None and terrain.kind == "slope":
            where = f"a {terrain.slope_deg:g}° {surface + ' ' if surface else ''}slope, {environment.body.title()}"
        else:
            kind = terrain.kind if terrain is not None else "hills"
            where = f"{surface + ' ' if surface else ''}{kind} terrain, {environment.body.title()}"
    course = " with an obstacle course" if spec.obstacles else ""
    return f"{listed[0].upper()}{listed[1:]} on {where}{course}"


PLACE_WORDS = [("lowest", ("lowest", "bottom", "crater floor", "valley", "low point")),
               ("highest", ("highest", "top", "summit", "peak", "rim", "high point")),
               ("east", ("far side",)),
               ("center", ("center", "centre", "middle")),
               ("north", ("north",)), ("south", ("south",)), ("east", ("east",)), ("west", ("west",))]


OBSTACLE_WORDS = [("boulder", r"boulders?"), ("rock", r"rocks\b|rock field|rock garden|stones?"),
                  ("stairs", r"stairs|staircase"), ("step", r"\bsteps?\b(?! up)|ledges?|curbs?|kerbs?"),
                  ("gap", r"gaps?|trench|ditch|crevasse|crack"), ("ramp", r"ramps?"), ("wall", r"walls?"), ("log", r"logs?")]
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "ten": 10,
                "some": 5, "few": 3, "a few": 3, "several": 6, "many": 12, "lots of": 15}


def _mock_obstacles(spec: SceneSpec, t: str, new_scene: bool = True) -> None:
    """Rule-based obstacles for MOCK mode: 'add 10 small rocks, a big step and a gap'.
    A new course starts on flat ground; adding obstacles to an existing scene keeps its terrain."""
    from .spec import Obstacle
    if re.search(r"\b(?:no|remove|clear|without) (?:the )?(?:obstacles|course)", t):
        spec.obstacles = []
        return
    found = []
    for kind, pattern in OBSTACLE_WORDS:
        for m in re.finditer(pattern, t):
            if kind == "step" and re.match(r"steps? (?:up|on|by|of the stair)", t[m.start():m.end() + 14]):
                continue
            before = t[max(0, m.start() - 30):m.start()]
            count = 1
            if n := re.search(r"(\d+)\s+(?:\w+\s+)?$", before):
                count = int(n.group(1))
            else:
                for word, value in NUMBER_WORDS.items():
                    if re.search(rf"\b{word}\s+(?:\w+\s+)?$", before):
                        count = value
            size = "large" if re.search(r"(?:big|large|tall|high|huge|wide|deep)\s+$", before) else \
                   "small" if re.search(r"(?:small|little|tiny|low|short|narrow)\s+$", before) else "medium"
            plural = {"rock": 8, "boulder": 4, "log": 2}
            if kind in plural and m.group(0).endswith("s") and count == 1:
                count = plural[kind]
            found.append((m.start(), Obstacle(kind=kind, size=size, count=min(count, 30),
                                              placement="scattered" if "scatter" in t else "path")))
            break    # one entry per kind per message
    if found:
        if new_scene and not spec.obstacles and spec.terrain.kind not in ("real",) \
                and not any(k in t for k in ("hill", "slope", "crater", "rocky")):
            spec.terrain.kind, spec.terrain.roughness = "flat", 0.0
        spec.obstacles = list(spec.obstacles) + [ob for _, ob in sorted(found, key=lambda f: f[0])]


def _mock_mission(spec: SceneSpec, t: str) -> None:
    """Rule-based start/goal for MOCK mode: 'from the west to the summit', 'reach the bottom', 'goal at 5, 3'."""
    from .spec import Mission

    def place_in(fragment: str) -> str | None:
        hits = [(fragment.find(w), name) for name, words in PLACE_WORDS for w in words if w in fragment]
        return min(hits)[1] if hits else None

    nums = r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?"
    goal_xy = re.search(r"(?:goal|go to|get to|reach)\s*(?:at|to)?\s*" + nums, t)
    places = "|".join(w for _, ws in PLACE_WORDS for w in ws)
    wants = goal_xy or any(w in t for w in ("reach", "get to", "go to", "goal", "destination", "make it to", "drive to", "walk to", "climb to", "far side")) \
        or re.search(r"\bfrom (?:the )?.+? to (?:the )?(?:" + places + ")", t) \
        or re.search(r"(?<!next )\bto (?:the )?(?:" + places + r")\b", t)   # "walking ... to the summit", "race ... to the bottom"
    if not wants:
        return
    ms = spec.mission or Mission()
    if goal_xy:
        ms.goal_xy, ms.goal_place = [float(goal_xy.group(1)), float(goal_xy.group(2))], None
    if m := re.search(r"from (?:the )?(.+?)\s+to\b(.*)", t):
        ms.start_place = place_in(m.group(1)) or ms.start_place
        if not goal_xy:
            ms.goal_place = place_in(m.group(2)) or ms.goal_place
    elif not goal_xy:
        ms.goal_place = place_in(t) or ms.goal_place
    if not ms.has_goal():
        ms.goal_place = "center"
        spec.assumptions.append("No place named for the goal, so I put it in the centre of the terrain.")
    spec.mission = ms


def _llm_spec(system: str, user: str) -> SceneSpec:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_error = ""
    for _ in range(2):
        if last_error:
            messages.append({"role": "user", "content": f"That JSON was invalid: {last_error}. Return the corrected JSON only."})
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        try:
            return SceneSpec.model_validate(llm.extract_json(reply)).fill_defaults()
        except (ValueError, ValidationError) as e:
            last_error = str(e)[:500]
    raise RuntimeError(f"Planner could not produce a valid spec: {last_error}")


def plan(text: str) -> SceneSpec:
    if config.MOCK_MODE:
        return _mock_plan(text)
    return _llm_spec(planner_system(), text)


def edit(spec: SceneSpec, instruction: str) -> SceneSpec:
    if config.MOCK_MODE:
        return _mock_plan(instruction, spec)
    clean = spec.model_copy(deep=True)
    clean.assumptions = []
    return _llm_spec(planner_system() + EDIT_RULES, f"Current spec:\n{clean.model_dump_json()}\n\nChange requested: {instruction}")


def plan_stress(spec: SceneSpec, text: str) -> list[dict]:
    """Choose validated condition overrides for a robot-scene stress test."""
    if spec.domain != "robot_terrain" or spec.robot is None:
        raise ValueError("Stress tests currently support robot scenes only.")
    fallback = [
        {"label": "Moon gravity", "gravity": 1.62},
        {"label": "Mars gravity", "gravity": 3.71},
        {"label": "Earth gravity", "gravity": 9.81},
        {"label": "Sand", "surface": "sand"},
        {"label": "Ice", "surface": "ice"},
        {"label": "30 N sideways push", "push_n": 30},
    ]
    if not any(entry.robot.kind == "library" for entry in spec.robots):   # a library robot's body is fixed
        fallback.insert(5, {"label": "+5 kg payload", "payload_kg": 5})
    if spec.obstacles:
        fallback.append({"label": "Obstacles ×1.3", "obstacle_scale": 1.3})
    if config.MOCK_MODE:
        return fallback

    messages = [
        {"role": "system", "content": STRESS_PLANNER_SYSTEM},
        {"role": "user", "content": (
            f"Scene:\n{spec.model_dump_json()}\n"
            f"Applicable-condition facts: terrain.kind={spec.terrain.kind if spec.terrain else 'none'}; "
            f"obstacles_present={bool(spec.obstacles)}.\n\nStress-test request: {text}"
        )},
    ]
    last_error = ""
    for _ in range(2):
        if last_error:
            messages.append({
                "role": "user",
                "content": (
                    f"The previous plan was invalid: {last_error}. "
                    "Return corrected JSON only. Do not use slope_deg unless terrain.kind is slope; "
                    "do not use obstacle_scale unless obstacles are present."
                ),
            })
        reply = llm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        try:
            plan = StressPlan.model_validate(llm.extract_json(reply))
            variations = [variation.model_dump(exclude_none=True) for variation in plan.variations]
            for variation in variations:
                _validate_variation(spec, variation)
            return variations
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:500]
    return [{**variation, "label": f"Fallback: {variation['label']}"} for variation in fallback]


STRESS_REQUEST = re.compile(
    r"\bstress[- ]tests?\b|\bhow robust\b|\btest (?:it )?under different conditions\b|\bwill it work on\b",
    re.I,
)


def is_stress_test(text: str) -> bool:
    return bool(STRESS_REQUEST.search(text))


def is_approval(text: str) -> bool:
    return bool(AFFIRM.match(text))


RERUN = re.compile(r"^\s*(run (it )?again|re-?run|again|repeat( it)?|one more time)\b", re.I)


def is_rerun(text: str) -> bool:
    """An explicit request to run the current scene again (checked before plain approval)."""
    return bool(RERUN.match(text))


RUN_ANYWAY = re.compile(r"^\s*run (it )?anyway\b", re.I)


def is_run_anyway(text: str) -> bool:
    """The explicit override for running a scene whose checks still fail."""
    return bool(RUN_ANYWAY.match(text))


# ── Router ─────────────────────────────────────────────────────────────────

def route(spec: SceneSpec) -> Simulator:
    return REGISTRY[spec.domain]()


# ── Builder ────────────────────────────────────────────────────────────────

BUILDER_SYSTEM = """You are the Builder agent. You write simulation code in the simulator's own language.
You receive a working template and a scene spec. Improve or extend the template to match the spec
(for example add obstacles or details the user asked for) while keeping every named body, joint and
actuator the template has. Reply with ONLY the complete code in one fenced block."""


def build(sim: Simulator, spec: SceneSpec, log, feedback: str | None = None) -> BuildResult:
    """Template build by default; in llm mode the model writes the code with a compile-and-fix loop."""
    if config.MOCK_MODE or config.BUILDER_MODE != "llm" or not hasattr(sim, "template"):
        return sim.build(spec)

    template_code = sim.template(spec)[0]
    lang = "xml" if sim.language.startswith("mjcf") else "python"
    request = f"Simulator: {sim.name} ({sim.language})\nSpec:\n{spec.model_dump_json()}\n\nTemplate:\n```{lang}\n{template_code}\n```"
    if feedback:
        request += f"\n\nChecker feedback to resolve:\n{feedback}\nPreserve the scene's named robot bodies, joints, and actuators."
    messages = [
        {"role": "system", "content": BUILDER_SYSTEM},
        {"role": "user", "content": request},
    ]
    for attempt in range(1, config.BUILDER_MAX_RETRIES + 1):
        reply = llm.chat(messages, model=config.BUILDER_MODEL)
        code = llm.extract_code(reply, lang)
        try:
            result = sim.build(spec, code=code)
            log(f"Code compiled on attempt {attempt}.")
            return result
        except Exception as e:  # noqa: BLE001 - simulator errors vary by engine
            err = str(e).splitlines()[0][:300]
            log(f"Attempt {attempt} failed to compile: {err}. Sending the error back to fix.")
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"The simulator rejected this code: {err}\nFix it and return the full code."}]
    log("Falling back to the verified template.")
    return sim.build(spec)


# ── Explainer ──────────────────────────────────────────────────────────────

def _mock_explain(spec: SceneSpec, m: dict) -> str:
    if m.get("simulator") == "MuJoCo":
        if m.get("robots"):
            leaderboard = m.get("leaderboard", [])
            if not leaderboard:
                return "The robots ran, but no leaderboard was recorded. Try running the race again."
            winner = leaderboard[0]
            if winner.get("goal_reached"):
                text = f"{winner['label']} placed first and reached the goal in {winner['time_to_goal_s']} s."
            elif m["robots"][winner["id"]].get("final_distance_m") is not None:
                text = (f"{winner['label']} made the most progress but missed the goal by "
                        f"{m['robots'][winner['id']].get('final_distance_m')} m.")
            else:
                text = (f"{winner['label']} placed first with "
                        f"{m['robots'][winner['id']].get('forward_progress_m')} m of forward progress.")
            for entry in leaderboard[1:]:
                robot = m["robots"][entry["id"]]
                if entry.get("goal_reached"):
                    detail = f"arrived in {entry['time_to_goal_s']} s"
                else:
                    detail = f"finished {robot.get('final_distance_m')} m from the goal ({robot.get('outcome')})"
                text += f" {entry['label']} {detail}, with a maximum tilt of {robot['max_tilt_deg']}°."
            text += f" The robots recorded {m.get('robot_contacts', 0)} cross-robot contact pairs."
            return text + " Try changing their starting lanes or widening the goal area."
        g, f = m["gravity"], m["friction"]
        walker = spec.robot.kind == "walker"
        library = library_robot_kind(spec.robot) if spec.robot.kind == "library" else None
        who, moved, grip = (
            ("The humanoid", "walked", "feet") if library == "humanoid" else
            (f"The {library_robot_name(spec.robot).lower()}", "drove", "wheels") if library == "wheeled" else
            (f"The {library_robot_name(spec.robot).lower()}", "walked" if library == "legged" else "moved", "feet")
            if library else
            ("The walker", "walked", "feet") if walker else
            ("The rover", "drove", "wheels")
        )
        slope, along, down = m.get("local_slope_deg"), m.get("slope_along_heading_deg"), m.get("downhill_side")
        slip, slip_late = m.get("slip_ratio"), m.get("slip_ratio_last_1s_before_fall")
        goal = ""
        if "goal_reached" in m:
            if m["goal_reached"]:
                goal = f"{who} reached the goal in {m['time_to_goal_s']} s, travelling {m['path_length_m']} m. "
            else:
                why = {"fell": "it fell", "stuck": "it got stuck", "out_of_time": "it ran out of time",
                       "left_terrain": "it went off the edge"}.get(m["outcome"], m["outcome"])
                goal = f"{who} missed the goal: {why}, {m['final_distance_m']} m short (closest {m['closest_distance_m']} m). "
        if m["flipped"]:
            text = f"{who} tipped {m.get('fall_direction') or 'over'} at {m['flipped_at_s']} s"
            if slope is not None:
                facing = "uphill" if (along or 0) > 3 else "downhill" if (along or 0) < -3 else "across the slope"
                text += f", on a {slope}° patch while facing {facing}"
                if down in ("left", "right"):
                    text += f" with the ground dropping away to its {down}"
            text += "."
            if slip_late is not None:
                text += f" In the last second before it fell, its {grip} were sliding {slip_late:.0%} of the time."
        elif m["forward_progress_m"] < 0.5:
            text = f"{who} barely made progress ({m['forward_progress_m']} m)."
            if slip is not None:
                text += f" Its {grip} were sliding {slip:.0%} of the time with friction {f}."
        else:
            text = f"{who} {moved} {m['forward_progress_m']} m forward and climbed {m['climbed_m']} m, tilting at most {m['max_tilt_deg']}°."
            if m.get("max_slope_on_path_deg") is not None:
                text += f" The steepest ground it crossed was {m['max_slope_on_path_deg']}°."
        if m.get("left_terrain"):
            text += " It ended up off the edge of the terrain."
        if m.get("obstacles_total"):
            lim, sb = m.get("robot_rough_limits") or {}, m.get("stopped_by")
            course = f" It got past {m['obstacles_cleared']} of {m['obstacles_total']} obstacles."
            if sb:
                if sb["kind"] == "gap":
                    what, limit = f"the {sb.get('width_m')} m wide gap", (f"wider than the about {lim.get('gap_m')} m it can usually cross"
                                                                          if lim.get("gap_m") else "and this robot can't cross gaps")
                else:
                    what, limit = f"the {sb['height_m']} m {sb['kind']}", f"compared with the about {lim.get('climb_m')} m it can usually climb"
                course += f" It was stopped by {what}, {limit}."
            text += course
        text = goal + text
        text += f" Gravity here is {g} m/s², so it weighs {g / 9.81:.2f}× what it would on Earth, which changes how much grip its {grip} get."
        return f"{text} {_mock_tip(spec, m)}."
    ratio = m["shaking_freq_hz"] / m["building_freq_hz"]
    resonance = " The shaking frequency is close to the building's own frequency, so it resonated and swayed much more." if 0.8 < ratio < 1.25 else ""
    return (f"The {m['stories']}-story frame swayed up to {m['max_roof_displacement_cm']} cm at the roof, "
            f"with a worst story drift of {m['max_story_drift_pct']}%, which suggests {m['damage_estimate']}.{resonance} "
            f"Its natural period is {m['period_s']} s (code estimate {m['code_period_estimate_s']} s). "
            f"Try: \"make it stiffer\" or \"make the quake harder\".")


ROBOT_FIELDS = {   # the design properties each kind of robot actually has
    "rover": ("kind", "wheels", "wheel_radius", "wheel_width", "drive_speed", "chassis_length", "chassis_width", "mass_kg"),
    "walker": ("kind", "legs", "leg_length", "gait_freq_hz", "stride_deg", "chassis_length", "chassis_width", "mass_kg"),
    "library": ("kind", "robot_id", "policy_id", "command_speed_m_s", "mass_kg"),
}


def robot_view(robot: dict) -> dict:
    """A robot's design with only the properties its kind has (no drive_speed for a walker, no gait for a rover)."""
    return {key: robot[key] for key in ROBOT_FIELDS.get(robot.get("kind"), tuple(robot)) if key in robot}


def explainer_metrics(metrics: dict) -> dict:
    """Run metrics for the Explainer: each robot's kind, and only the design properties that apply to it."""
    view = dict(metrics)
    if isinstance(view.get("robot"), dict):
        view["robot"] = robot_view(view["robot"])
    if isinstance(view.get("robots"), dict):
        view["robots"] = {
            robot_id: {**m, "kind": m["robot"].get("kind"), "robot": robot_view(m["robot"])} if isinstance(m.get("robot"), dict) else m
            for robot_id, m in view["robots"].items()
        }
    return view


TESTED_GAIT_HZ, TESTED_STRIDE_DEG = 2.0, 15.0
_GAIT_NUMBERS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*hz\b", re.I),
    re.compile(r"gait[ _]freq(?:uency)?(?:_hz)?\D{0,20}?(\d+(?:\.\d+)?)", re.I),
]
_STRIDE_NUMBERS = [
    re.compile(r"stride(?:_deg)?\D{0,30}?(\d+(?:\.\d+)?)\s*(?:°|deg)", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?)\s+(?:of\s+)?stride", re.I),
]


def mark_untested_gait(text: str, spec: SceneSpec) -> str:
    """Walker gaits were tested only at 2.0 Hz / 15°: if the text suggests other gait numbers without saying
    "untested", append "(untested settings)"."""
    if not any(entry.robot.kind == "walker" for entry in spec.robots) or "untested" in text.lower():
        return text
    hz = [float(value) for pattern in _GAIT_NUMBERS for value in pattern.findall(text)]
    strides = [float(value) for pattern in _STRIDE_NUMBERS for value in pattern.findall(text)]
    if any(abs(value - TESTED_GAIT_HZ) > 1e-6 for value in hz) or any(abs(value - TESTED_STRIDE_DEG) > 1e-6 for value in strides):
        return text.rstrip() + " (untested settings)"
    return text


def _explain_with_facts(system: str, user: str, spec: SceneSpec, metrics: dict, fallback) -> str:
    """Ask the Explainer with a FACTS block built in code; a reply that contradicts FACTS gets one retry with the
    exact problem, then the rules text is used. A tip naming a setting the robots don't have is replaced by the
    rules tip."""
    from .facts import FACTS_RULE, checked, drop_invalid_tips, facts, facts_block

    found = facts(spec, metrics)
    messages = [{"role": "system", "content": f"{system}\n- {FACTS_RULE}"},
                {"role": "user", "content": f"{facts_block(found)}\n\n{user}"}]

    def ask(feedback: str | None) -> str:
        if feedback is not None:
            messages.extend([{"role": "assistant", "content": reply[0]}, {"role": "user", "content": feedback}])
        reply[0] = llm.chat(messages, temperature=0.4).strip()
        return reply[0]

    reply = [""]
    text, _ = checked(ask, found, fallback)
    text, wrong_settings = drop_invalid_tips(text, found)
    if wrong_settings:
        text = f"{text} {_mock_tip(spec, metrics)}.".strip()
    return mark_untested_gait(text, spec)


def _mock_tip(spec: SceneSpec, m: dict) -> str:
    """A next step that fits what the scene has: a course suggests wheels or legs, a slope suggests less slope or
    more grip; never "make the slope steeper" on flat ground or a course."""
    robot, kind = spec.robot, spec.terrain.kind if spec.terrain else "flat"
    failed = bool(m.get("flipped")) or m.get("goal_reached") is False
    other = "Mars" if spec.environment.body == "moon" else "the Moon"
    if robot.kind == "walker":
        if robot.legs == 4:
            return 'Try: "use six legs"'
        if spec.obstacles:
            return 'Try: "make its legs longer"'
        if kind == "slope":
            return 'Try: "make the slope less steep" or "make the surface rock"' if failed else 'Try: "make the slope steeper"'
        return f'Try: "put it on {other}" or "make the terrain rocky"'
    if robot.kind == "library":
        return f'Try: "put it on {other}"'
    if spec.obstacles:
        if not failed:
            return 'Try: "make the obstacles bigger"'
        return 'Try: "give it bigger wheels" or "use six wheels"' if robot.wheels == 4 else 'Try: "give it bigger wheels"'
    if kind == "slope":
        return 'Try: "make the slope less steep" or "make the surface rock"' if failed else 'Try: "make the slope steeper"'
    if m.get("flipped"):
        return 'Try: "give it bigger wheels" or "use six wheels"'
    return f'Try: "put it on {other}" or "make it walk"'


def explain(spec: SceneSpec, metrics: dict) -> str:
    if config.MOCK_MODE:
        return _mock_explain(spec, metrics)
    user = f"Scene: {spec.title}\nAssumptions: {spec.assumptions}\nMetrics: {json.dumps(explainer_metrics(metrics))}"
    if spec.domain != "robot_terrain":
        return llm.chat([{"role": "system", "content": EXPLAIN_SYSTEM}, {"role": "user", "content": user}],
                        temperature=0.4).strip()
    return _explain_with_facts(EXPLAIN_SYSTEM, user, spec, metrics, lambda: _mock_explain(spec, metrics))


def _mock_explain_stress(results: list[dict]) -> str:
    measured = [result for result in results if not result.get("error")]
    passed = [result for result in measured if result.get("passed")]
    failed = [result for result in measured if not result.get("passed")]
    if not measured:
        return "No runs produced usable measurements. Try the stress test again with a shorter simulation duration."
    if all(isinstance(result.get("robots"), dict) for result in measured):
        return _mock_explain_race_stress(measured)

    text = f"{len(passed)} of {len(results)} runs passed."
    baseline = next((result for result in measured if result.get("label") == "Baseline"), None)
    if baseline:
        summary = baseline["summary"]
        text += (f" Baseline measured {summary.get('forward_progress_m', 0)} m forward progress "
                 f"and {summary.get('max_tilt_deg', 0)}° maximum tilt.")

    if not failed:
        text += " No failures were measured, so these conditions did not establish a failure threshold."
        return text + " Try a steeper slope or a larger payload."
    pattern, suggestion = _stress_pattern(passed, failed)
    return f"{text} {pattern} {suggestion}"


def _mock_explain_race_stress(results: list[dict]) -> str:
    """A race's stress test, robot by robot: each robot's pass count, its baseline result, and the changed
    condition its failures share (only for a robot that both passed and failed)."""
    ids = list(results[0]["robots"])
    label = {robot_id: results[0]["robots"][robot_id]["label"] for robot_id in ids}
    text = "; ".join(f"{label[robot_id]} passed {sum(r['robots'][robot_id]['passed'] for r in results)} of {len(results)} runs"
                     for robot_id in ids) + "."
    baseline = next((result for result in results if result.get("label") == "Baseline"), None)
    if baseline:
        text += " Baseline: " + "; ".join(f"{label[robot_id]} {baseline['robots'][robot_id]['reason']}" for robot_id in ids) + "."
    suggestion = "Try a steeper slope or a larger payload."
    for robot_id in ids:
        passed = [r for r in results if r["robots"][robot_id]["passed"]]
        failed = [r for r in results if not r["robots"][robot_id]["passed"]]
        if not failed:
            continue
        if not passed:
            text += f" {label[robot_id]} failed in every condition, including the baseline, so no changed condition explains it."
            suggestion = f"Try \"why did the {label[robot_id]} fail?\" to test causes one at a time."
            continue
        pattern, suggestion = _stress_pattern(passed, failed)
        text += f" {label[robot_id]}: {pattern}"
    return f"{text} {suggestion}"


def _stress_pattern(passed: list[dict], failed: list[dict]) -> tuple[str, str]:
    """The changed condition the failed runs share, and a suggestion to isolate it."""
    knobs = ("gravity", "surface", "friction", "payload_kg", "slope_deg", "obstacle_scale", "push_n", "seed")
    counts = {
        key: sum(key in result.get("changes", {}) for result in failed)
        for key in knobs
    }
    key = max(counts, key=counts.get)
    if not counts[key]:
        pattern = "The failures did not share a changed condition, so no single-knob pattern is clear."
        suggestion = "Try repeating one failed run while changing only one condition."
    else:
        failed_values = [result["changes"][key] for result in failed if key in result.get("changes", {})]
        passed_values = [result["changes"][key] for result in passed if key in result.get("changes", {})]
        numeric_failed = [value for value in failed_values if isinstance(value, (int, float))]
        numeric_passed = [value for value in passed_values if isinstance(value, (int, float))]
        if key == "surface":
            pattern = f"Failures were measured on {', '.join(map(str, dict.fromkeys(failed_values)))}."
        elif numeric_failed and numeric_passed and max(numeric_failed) < min(numeric_passed):
            pattern = (f"Runs failed at {key} values up to {max(numeric_failed):g}, while passing "
                       f"examples started at {min(numeric_passed):g}; that boundary likely matters.")
        elif numeric_failed and numeric_passed and min(numeric_failed) > max(numeric_passed):
            pattern = (f"Runs failed at {key} values from {min(numeric_failed):g}, while passing "
                       f"examples ended at {max(numeric_passed):g}; that boundary likely matters.")
        else:
            pattern = f"Failures clustered in runs that changed {key} to {', '.join(map(str, dict.fromkeys(failed_values)))}."
        sample = failed[0]
        suggestion = f"Try repeating {sample['label']} while varying only {key} to isolate its effect."
    return pattern, suggestion


CRASH_NOTE = "That's a problem running the simulation, not a physics result"


def explain_stress(spec: SceneSpec, results: list[dict]) -> str:
    """Explain the runs that ran. Crashed runs are reported as a system problem and never analysed as physics;
    when every run crashed there is nothing for the Explainer to read."""
    from .batch import crash_report, measured

    crashed, results = crash_report(results), measured(results)
    if not results:
        return f"{crashed}. {CRASH_NOTE}, so there is nothing to analyse." if crashed else "No runs to explain."
    if config.MOCK_MODE:
        text = _mock_explain_stress(results)
    else:
        evidence = [{**{key: value for key, value in result.items() if key != "ghost"},
                     "summary": explainer_metrics(result.get("summary") or {})} for result in results]
        baseline = next((result for result in results if result.get("label") == "Baseline"), results[0])
        text = _explain_with_facts(   # FACTS describe the scene and the baseline run
            EXPLAIN_STRESS_SYSTEM, f"Scene: {spec.title}\nAssumptions: {spec.assumptions}\n"
                                   f"Stress-test results: {json.dumps(evidence)}",
            spec, baseline.get("summary") or {}, lambda: _mock_explain_stress(results))
    return f"{crashed}. {CRASH_NOTE}, so those runs are left out. {text}" if crashed else text
