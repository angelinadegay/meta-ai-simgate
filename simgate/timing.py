"""The default time limit of a scene with a goal: the straight-line distance to it at the robot's nominal speed,
with room to spare. The slowest robot sets it in a race."""
import math

from .spec import Robot, SceneSpec

SLACK = 2.5                    # times the time the distance takes at the nominal speed
MIN_S, MAX_S = 20.0, 120.0
# Measured on flat Earth rock over 12 s, start-up included: the rover 0.83 m/s at its default 6 rad/s on 0.15 m
# wheels (0.9 m/s rolling), the 4-legged walker 0.37 m/s and the hexapod 0.77 m/s at the default 2 Hz gait, and the
# G1 0.55 m/s on a 0.6 m/s command.
ROLLING_SHARE = 0.92           # of wheel speed × radius
WALKER_M_S = {4: 0.37, 6: 0.77}   # at 2 Hz
LIBRARY_SHARE = 0.92           # of the commanded speed


def nominal_speed(robot: Robot) -> float:
    """m/s on flat ground, following the robot's own settings."""
    if robot.kind == "rover":
        return ROLLING_SHARE * robot.drive_speed * robot.wheel_radius
    if robot.kind == "walker":
        return WALKER_M_S[robot.legs] * robot.gait_freq_hz / 2.0
    return LIBRARY_SHARE * robot.command_speed_m_s


def _distances(spec: SceneSpec) -> list[tuple[str, Robot, float]]:
    """(label, robot, straight-line metres from its start to its goal) for each robot that has a goal."""
    from .backends.mujoco_backend import make_heights, resolve_mission

    heights, _ = make_heights(spec)
    # one robot: the scene's mission (a course puts its finish line there); a race: each robot's own
    entries = [(entry.label, entry.robot, entry.mission()) for entry in spec.robots] if len(spec.robots) > 1 else \
              [("the robot", spec.robot, spec.mission)]
    out = []
    for label, robot, mission in entries:
        if robot is None or mission is None or not mission.has_goal():
            continue
        one = spec.model_copy(deep=True)
        one.robot, one.mission = robot, mission
        resolved = resolve_mission(one, heights)
        start, goal = resolved["start"], resolved["goal"]
        out.append((label, robot, max(0.0, math.dist(start, goal) - resolved["radius"])))
    return out


def time_limit(spec: SceneSpec) -> tuple[float, str] | None:
    """The time limit and the sentence that explains it, or None without a goal."""
    rows = _distances(spec)
    if not rows:
        return None
    label, robot, metres = max(rows, key=lambda row: row[2] / nominal_speed(row[1]))
    speed = nominal_speed(robot)
    seconds = min(MAX_S, max(MIN_S, round(metres / speed * SLACK)))
    who = f" ({label} is the slowest to its goal)" if len(rows) > 1 else ""
    return seconds, (f"Gave {'the robots' if len(rows) > 1 else 'the robot'} up to {seconds:.0f} s to reach the goal: "
                     f"{metres:.1f} m at about {speed:.2f} m/s, times {SLACK:g}, kept within {MIN_S:.0f}–{MAX_S:.0f} s{who}.")
