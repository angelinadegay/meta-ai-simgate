import asyncio

import pytest

from simgate import agents, config
from simgate.backends.mujoco_backend import MuJoCoSimulator
from simgate.director import STANDING_ONLY
from simgate.orchestrator import Session
from simgate.spec import Environment, Robot, RobotEntry, SceneSpec, Terrain


async def handle(prompt: str):
    events = []

    async def emit(event):
        events.append(event)

    session = Session(emit)
    await session.handle(prompt)
    return session, events


def two_robot_checks(body: str, terrain: Terrain):
    """A standing G1 beside a rover: the scene a race used to build, checked without moving either robot."""
    spec = SceneSpec(domain="robot_terrain", terrain=terrain, environment=Environment(body=body),
                     robots=[RobotEntry(id="g1", label="G1", robot=Robot(kind="library", robot_id="g1", controller_id="g1_stand"),
                                        start_xy=[-2, -1], goal_xy=[4, -1]),
                             RobotEntry(id="rover", label="Rover", robot=Robot(kind="rover"), start_xy=[-2, 1], goal_xy=[4, 1])]
                     ).fill_defaults()
    simulator = MuJoCoSimulator()
    checks = simulator.validate(spec, agents.build(simulator, spec, lambda _: None))
    g1 = next(entry for entry in spec.robots if entry.robot.robot_id == "g1")
    return checks, next(check for check in checks if check.name == f"{g1.label} stands for 1 s at zero command")


@pytest.mark.parametrize("prompt", ["Race a Unitree G1 against a rover on flat ground",
                                    "Race a Unitree G1 against a rover on Mars hills"])
def test_a_g1_race_is_refused_because_it_only_stands(prompt, monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    session, events = asyncio.run(handle(prompt))
    assert session.spec is None and not any(event["type"] == "run_start" for event in events)
    assert any(STANDING_ONLY in str(event) for event in events)


@pytest.mark.parametrize(("body", "terrain"), [("earth", Terrain(kind="flat")),
                                               ("mars", Terrain(kind="slope", slope_deg=1))])
def test_g1_beside_a_rover_passes_its_stand_check_without_robot_motion(body, terrain):
    checks, stand = two_robot_checks(body, terrain)
    assert next(check for check in checks if check.name == "Robots rest on terrain").passed
    assert stand.passed, stand
    assert "root height change" in stand.detail


def test_g1_on_a_steep_slope_still_fails_its_stand_check():
    _, stand = two_robot_checks("earth", Terrain(kind="slope", slope_deg=30))
    assert not stand.passed, stand
    assert "root height change" in stand.detail
