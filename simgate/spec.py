"""The neutral scene spec: the one format every agent and simulator shares.

The planner turns plain language into a SceneSpec, the router reads its
`domain` to pick a simulator, and each simulator backend builds from it.
Adding a new simulator means adding a domain and a backend, nothing else.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from . import presets


class Environment(BaseModel):
    body: str = "earth"                 # earth, moon, mars, titan, europa, enceladus, custom
    gravity: Optional[float] = None     # m/s^2; filled from presets if missing
    surface: str = "rock"               # rock, regolith, gravel, sand, mud, ice
    friction: Optional[float] = None    # filled from presets if missing


class Terrain(BaseModel):
    kind: Literal["flat", "hills", "rocky", "slope", "craters", "real"] = "hills"
    preset: Optional[str] = None        # for kind="real": a saved real-terrain preset, e.g. "jezero"
    slope_deg: float = 0.0
    roughness: float = Field(0.3, ge=0.0, le=1.0)
    size_m: float = 20.0
    seed: int = 0


class Robot(BaseModel):
    kind: Literal["rover", "walker", "library"] = "rover"
    robot_id: Optional[str] = None
    policy_id: Optional[str] = None
    # rover
    wheels: Literal[4, 6] = 4
    wheel_radius: float = 0.15
    wheel_width: float = 0.08
    drive_speed: float = 6.0            # wheel speed, rad/s
    # walker (legged robot with a rhythmic trot/tripod gait)
    legs: Literal[4, 6] = 4
    leg_length: float = 0.22            # length of each leg segment, m
    gait_freq_hz: float = 2.0           # gait cycles per second (tuned for stability)
    stride_deg: float = 15.0            # hip swing amplitude (tuned for stability)
    # shared
    chassis_length: float = 0.6
    chassis_width: float = 0.4
    mass_kg: float = 10.0


Place = Literal["start_area", "center", "lowest", "highest", "north", "south", "east", "west"]


class Mission(BaseModel):
    """Where the robot starts and where it should go. Coordinates are metres on the terrain:
    (0, 0) is the centre, +x is east, +y is north, and the terrain spans ±size_m/2.
    A named place (e.g. "lowest") is resolved on the actual terrain, so the model never guesses numbers."""
    start_xy: Optional[list[float]] = None
    start_place: Optional[Place] = None
    start_heading_deg: Optional[float] = None   # 0 = facing east (+x), 90 = north; default: facing the goal
    goal_xy: Optional[list[float]] = None
    goal_place: Optional[Place] = None
    goal_radius_m: float = Field(1.0, gt=0.1, le=5.0)

    def has_goal(self) -> bool:
        return self.goal_xy is not None or self.goal_place is not None


class RobotEntry(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    label: str = Field(min_length=1, max_length=80)
    robot: Robot
    start_xy: Optional[list[float]] = None
    start_place: Optional[Place] = None
    start_heading_deg: Optional[float] = None
    goal_xy: Optional[list[float]] = None
    goal_place: Optional[Place] = None
    goal_radius_m: float = Field(1.0, gt=0.1, le=5.0)

    def mission(self) -> Mission:
        return Mission(
            start_xy=self.start_xy,
            start_place=self.start_place,
            start_heading_deg=self.start_heading_deg,
            goal_xy=self.goal_xy,
            goal_place=self.goal_place,
            goal_radius_m=self.goal_radius_m,
        )


ObstacleKind = Literal["rock", "boulder", "step", "stairs", "gap", "ramp", "wall", "log"]


class Obstacle(BaseModel):
    """Something on the course. Placement is computed on the terrain, never guessed by the model:
    "path" = on the line from start to goal (in the order listed), "scattered" = anywhere clear,
    "place" = around a named place."""
    kind: ObstacleKind
    size: Literal["small", "medium", "large"] = "medium"
    count: int = Field(1, ge=1, le=30)      # rocks/boulders/logs: a field of this many; others: copies in a row
    placement: Literal["path", "scattered", "place"] = "path"
    place: Optional[Place] = None
    height_m: Optional[float] = Field(None, gt=0.0, le=2.0)   # exact height (gap: depth stays fixed, this is its width)


class Structure(BaseModel):
    stories: int = Field(2, ge=1, le=20)
    bays: int = Field(1, ge=1, le=6)
    story_height_m: float = 3.0
    bay_width_m: float = 5.0
    section: Literal["light", "medium", "heavy"] = "medium"
    floor_mass_t: float = 20.0          # tonnes per floor


class Load(BaseModel):
    kind: Literal["earthquake"] = "earthquake"
    pga_g: float = 0.3                  # peak ground acceleration in g
    freq_hz: float = 1.5                # dominant shaking frequency
    duration_s: float = 10.0


class Disturbance(BaseModel):
    push_n: float = Field(gt=0, le=1000)
    duration_s: float = Field(0.3, gt=0)


class SceneSpec(BaseModel):
    domain: Literal["robot_terrain", "structure"]
    title: str = "Untitled scene"
    environment: Environment = Environment()
    terrain: Optional[Terrain] = None
    robot: Optional[Robot] = None
    mission: Optional[Mission] = None   # start and goal for a robot (optional)
    robots: list[RobotEntry] = Field(default_factory=list, max_length=6)
    obstacles: list[Obstacle] = []      # an obstacle course (robot scenes only)
    structure: Optional[Structure] = None
    load: Optional[Load] = None
    disturbance: Optional[Disturbance] = None
    duration_s: float = 8.0
    assumptions: list[str] = []         # defaults the agent chose, shown to the user

    @model_validator(mode="after")
    def validate_robot_entries(self) -> "SceneSpec":
        ids = [entry.id for entry in self.robots]
        if len(set(ids)) != len(ids):
            raise ValueError("robot entry ids must be unique")
        return self

    def fill_defaults(self) -> "SceneSpec":
        """Fill physical values from reference presets and record what was assumed."""
        env = self.environment
        t = self.terrain
        if self.domain == "robot_terrain" and t is not None and t.kind == "real":
            from .terrain_data import load_preset   # local import: only needed for real terrain
            if not t.preset:
                raise ValueError("Real terrain needs a preset name.")
            _, meta = load_preset(t.preset)
            t.size_m = float(meta["size_m"])
            if env.body != meta.get("body", env.body):   # the data decides which world this is
                env.body, env.gravity, env.surface, env.friction = meta["body"], None, meta.get("surface", env.surface), None
            elif "surface" not in env.model_fields_set and env.friction is None:
                env.surface = meta.get("surface", env.surface)   # nobody chose a surface: use the site's own
            self.assumptions.append(
                f"Terrain: real elevation data, {meta['label']} ({meta['size_m']:.0f} m square, "
                f"{meta['relief_m']:.1f} m of relief, typical slope {meta.get('typical_slope_deg', '?')}°). Credit: {meta['credit']}.")
        if env.gravity is None:
            g = presets.gravity_for(env.body)
            env.gravity = g if g is not None else 9.81
            self.assumptions.append(f"Gravity set to {env.gravity} m/s² for {env.body}.")
        if env.friction is None:
            f = presets.friction_for(env.surface)
            env.friction = f if f is not None else 0.8
            if self.domain == "robot_terrain":
                self.assumptions.append(f"Friction set to {env.friction} for a {env.surface} surface.")
        if self.domain == "robot_terrain":
            self.terrain = self.terrain or Terrain()
            if not self.robots:
                self.robot = self.robot or Robot()
                mission = self.mission or Mission()
                self.robots = [
                    RobotEntry(
                        id="r1",
                        label=self.robot.kind.title(),
                        robot=self.robot.model_copy(deep=True),
                        start_xy=mission.start_xy,
                        start_place=mission.start_place,
                        start_heading_deg=mission.start_heading_deg,
                        goal_xy=mission.goal_xy,
                        goal_place=mission.goal_place,
                        goal_radius_m=mission.goal_radius_m,
                    )
                ]
            else:
                shared_goal = next(
                    (
                        entry for entry in self.robots
                        if entry.goal_xy is not None or entry.goal_place is not None
                    ),
                    None,
                )
                if shared_goal is not None:
                    for entry in self.robots:
                        if entry.goal_xy is None and entry.goal_place is None:
                            entry.goal_xy = shared_goal.goal_xy
                            entry.goal_place = shared_goal.goal_place
                            entry.goal_radius_m = shared_goal.goal_radius_m
                self.robot = self.robots[0].robot.model_copy(deep=True)
                self.mission = self.robots[0].mission()
            for entry in self.robots:
                if entry.robot.kind == "library":
                    if not entry.robot.robot_id or not entry.robot.policy_id:
                        raise ValueError(f"Library robot entry {entry.id!r} requires robot_id and policy_id.")
                    from .library import LibraryRegistry

                    registry = LibraryRegistry()
                    robot_card = registry.robot(entry.robot.robot_id)
                    compatible = {card.id for card in registry.compatible_policies(robot_card.id)}
                    if entry.robot.policy_id not in compatible:
                        raise ValueError(
                            f"Policy {entry.robot.policy_id!r} is not compatible with robot {robot_card.id!r}."
                        )
                    entry.robot.mass_kg = robot_card.mass_kg
            if self.robot.kind == "library":
                if not self.robot.robot_id or not self.robot.policy_id:
                    raise ValueError("Library robots require robot_id and policy_id.")
                from .library import LibraryRegistry

                registry = LibraryRegistry()
                robot_card = registry.robot(self.robot.robot_id)
                compatible = {card.id for card in registry.compatible_policies(robot_card.id)}
                if self.robot.policy_id not in compatible:
                    raise ValueError(
                        f"Policy {self.robot.policy_id!r} is not compatible with robot {robot_card.id!r}."
                    )
                self.robot.mass_kg = robot_card.mass_kg
            if self.obstacles and self.terrain.kind != "real":
                from .backends.obstacles import course_length_needed   # local import: avoids a cycle
                path = 0.73 * self.terrain.size_m                         # default start to the far-side goal
                need = course_length_needed(self.obstacles)
                if need > path and self.terrain.size_m < 60:
                    self.terrain.size_m = float(min(60, int(need / 0.73) + 2))
                    self.assumptions.append(f"Made the terrain {self.terrain.size_m:.0f} m wide so the course fits.")
            if self.obstacles and (self.mission is None or not self.mission.has_goal()):
                self.mission = self.mission or Mission()       # a course needs a finish line
                self.mission.goal_place = "east" if self.mission.start_place in (None, "start_area", "west") else "center"
                self.assumptions.append("Put the goal on the far side of the obstacle course.")
            if self.mission is not None and self.mission.has_goal() and self.duration_s == 8.0:
                # The 8 s default is too short to get anywhere. Runs end early once the goal is reached.
                self.duration_s = 45.0 if self.robot.kind == "walker" else 30.0
                self.assumptions.append(f"Gave the robot up to {self.duration_s:.0f} s to reach the goal.")
        else:
            self.structure = self.structure or Structure()
            self.load = self.load or Load()
            self.duration_s = self.load.duration_s
        self.assumptions = list(dict.fromkeys(self.assumptions))
        return self
