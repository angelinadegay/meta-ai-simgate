import asyncio

import pytest

from simgate import config
from simgate.orchestrator import Session


async def build_race(prompt: str):
    events = []

    async def emit(event):
        events.append(event)

    session = Session(emit)
    await session.handle(prompt)
    return session, events


def run_race(prompt: str, monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    return asyncio.run(build_race(prompt))


def g1_stand_check(session, events):
    checks = next(event["checks"] for event in events if event["type"] == "checks")
    g1 = next(entry for entry in session.spec.robots if entry.robot.robot_id == "g1")
    return next(check for check in checks if check["name"] == f"{g1.label} stands for 1 s at zero command")


@pytest.mark.parametrize(
    ("prompt", "expected_body", "expected_terrain"),
    [
        ("Race a Unitree G1 against a rover on flat ground", "earth", "flat"),
        ("Race a Unitree G1 against a rover on Mars hills", "mars", "hills"),
    ],
)
def test_g1_race_stand_check_passes_without_robot_motion(
    prompt, expected_body, expected_terrain, monkeypatch
):
    session, events = run_race(prompt, monkeypatch)

    assert session.spec.environment.body == expected_body
    assert session.spec.terrain.kind == expected_terrain
    assert len(session.spec.robots) == 2
    checks = next(event["checks"] for event in events if event["type"] == "checks")
    assert next(check for check in checks if check["name"] == "Robots rest on terrain")["passed"]
    stand = g1_stand_check(session, events)
    assert stand["passed"], stand
    assert "root height change" in stand["detail"]


def test_unstable_g1_spawn_still_fails_its_stand_check(monkeypatch):
    session, events = run_race(
        "Race a Unitree G1 against a rover on a 30 degree slope", monkeypatch
    )

    stand = g1_stand_check(session, events)
    assert not stand["passed"], stand
    assert "root height change" in stand["detail"]
