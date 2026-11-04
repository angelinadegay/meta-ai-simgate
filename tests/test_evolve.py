import json

import numpy as np
import pytest

from simgate import config, evolve, llm
from simgate.spec import Environment, Mission, Robot, SceneSpec, Terrain


def challenge(kind="flat"):
    return SceneSpec(
        domain="robot_terrain", title="Reach the hilltop",
        environment=Environment(body="mars", surface="regolith", friction=0.7),
        terrain=Terrain(kind=kind, slope_deg=12 if kind == "slope" else 0, roughness=0, size_m=10),
        robot=Robot(), mission=Mission(start_place="west", goal_place="highest"),
    ).fill_defaults()


def row(kind, **knobs):
    return {"name": "New design", "reason": "A proposal to test on this place.",
            "robot": {"kind": kind, **knobs}}


def test_continuous_values_clamp_and_choices_remain_discrete():
    rows = [row("rover", wheels=6, wheel_radius=9, wheel_width=-1, chassis_length=4,
                chassis_width=0, mass_kg=100, drive_speed=0),
            row("walker", legs=6, leg_length=0, gait_freq_hz=9, stride_deg=-2,
                chassis_length=0, chassis_width=5, mass_kg=1)]
    designs = evolve._validate_designs({"designs": rows}, 2)
    rover, walker = (design["robot"] for design in designs)
    assert {key: rover[key] for key in evolve.DESIGN_SPACE["rover"]} == {
        "wheels": 6, "wheel_radius": .4, "wheel_width": .05, "chassis_length": 1.2,
        "chassis_width": .3, "mass_kg": 40, "drive_speed": 3,
    }
    assert walker["legs"] == 6 and walker["leg_length"] == .12
    assert walker["gait_freq_hz"] == 3 and walker["stride_deg"] == 8
    assert walker["chassis_length"] == .4 and walker["chassis_width"] == .9 and walker["mass_kg"] == 5
    assert designs[0]["adjustments"] and designs[1]["adjustments"]
    with pytest.raises(ValueError, match="wheels must be"):
        evolve._validate_designs({"designs": [row("rover", wheels=5), row("walker")]}, 2)


@pytest.mark.parametrize("robots", [
    [row("rover"), row("rover"), row("walker")],
    [row("rover", wheel_radius=4), row("rover", wheel_radius=8), row("walker")],
    [row("walker"), row("walker", leg_length=.22), row("rover")],
])
def test_duplicates_rejected_after_defaults_and_clamping(robots):
    with pytest.raises(ValueError, match="Duplicate"):
        evolve._validate_designs({"designs": robots}, len(robots))


def test_mix_of_procedural_kinds_required_and_library_forbidden():
    with pytest.raises(ValueError, match="at least one rover and one walker"):
        evolve._validate_designs({"designs": [row("rover"), row("rover", wheels=6)]}, 2)
    for kind in ("library", "g1"):
        with pytest.raises(ValueError, match="G1 is not designable"):
            evolve._validate_designs({"designs": [row(kind), row("walker")]}, 2)
    with pytest.raises(ValueError, match="Not design knobs"):
        evolve._validate_designs({"designs": [row("rover", robot_id="g1"), row("walker")]}, 2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "large", None])
def test_invalid_numeric_values_rejected(value):
    with pytest.raises(ValueError, match="finite number"):
        evolve._validate_designs({"designs": [row("rover", mass_kg=value), row("walker")]}, 2)


@pytest.mark.parametrize("n", [2, 4, 8])
def test_fixed_fallback_returns_valid_diverse_designs_without_model_or_runs(monkeypatch, n):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: pytest.fail("No model in fallback"))
    from simgate import batch
    monkeypatch.setattr(batch, "run_batch", lambda *args, **kwargs: pytest.fail("No racing in E1"))
    scene = challenge()
    before = scene.model_dump()
    designs = evolve.propose_generation0(scene, n)
    assert len(designs) == n
    assert {design["robot"]["kind"] for design in designs} == {"rover", "walker"}
    assert [design["id"] for design in designs] == [f"G0-{chr(65 + i)}" for i in range(n)]
    assert len({design["color"] for design in designs}) == n
    assert len({json.dumps(design["robot"], sort_keys=True) for design in designs}) == n
    for design in designs:
        assert design["source"] == "rules" and design["fallback_reason"] and not design["tested"]
        robot = Robot.model_validate(design["robot"])
        for key, (low, high) in evolve.DESIGN_SPACE[robot.kind].items():
            assert low <= getattr(robot, key) <= high
        assert design["name"] and "\n" not in design["reason"]
    assert scene.model_dump() == before
    assert evolve.propose_generation0(scene, n) == designs


def test_model_gets_challenge_facts_and_builder_model(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    captured = []

    def chat(messages, model):
        captured.append((messages, model))
        return json.dumps({"designs": [
            row("rover", wheel_radius=9), row("walker"),
            row("rover", wheels=6), row("walker", legs=6),
        ]})

    monkeypatch.setattr(llm, "chat", chat)
    designs = evolve.propose_generation0(challenge("slope"))
    assert len(captured) == 1 and captured[0][1] == config.BUILDER_MODEL
    context = json.loads(captured[0][0][1]["content"])
    assert context["challenge"]["title"] == "Reach the hilltop"
    assert context["challenge"]["mission"]["goal_place"] == "highest"
    found = context["FACTS"]
    assert found["gravity_m_s2"] == 3.71 and found["friction"] == .7
    assert found["p95_slope_deg"] == pytest.approx(12)
    assert found["relief_m"] == pytest.approx(10 * np.tan(np.radians(12)))
    assert found["rock_sizes_m"] is None and "not measured" in found["rock_sizes_source"]
    assert all(design["source"] == "model" and design["fallback_reason"] is None for design in designs)
    assert designs[0]["robot"]["wheel_radius"] == .4 and designs[0]["adjustments"]


def test_terrain_card_rock_sizes_are_not_invented(monkeypatch):
    from simgate import terrain_data

    scene = challenge()
    scene.terrain = Terrain(kind="real", preset="test-card", roughness=0)
    heights = np.tile(np.linspace(0, 1, 8), (8, 1))
    monkeypatch.setattr(terrain_data, "load_preset", lambda name: (heights, {
        "label": "Measured patch", "size_m": 10, "relief_m": 1, "body": "mars", "credit": "Test data",
        "rock_sizes_m": [.05, .2, .4],
    }))
    found = evolve.place_facts(scene)
    assert found["place"] == "Measured patch"
    assert found["rock_sizes_m"] == [.05, .2, .4] and found["rock_sizes_source"] == "terrain card"


@pytest.mark.parametrize("reply", [
    "not JSON", '{"designs":null}', '{"designs":[{"robot":{"kind":"g1"}}]}',
    json.dumps({"designs": [row("rover")] * 4}),
])
def test_malformed_model_reply_retries_then_falls_back_cleanly(monkeypatch, caplog, reply):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    calls = []
    monkeypatch.setattr(llm, "chat", lambda *args: calls.append(args) or reply)
    designs = evolve.propose_generation0(challenge())
    assert len(calls) == 2 and "failed validation" in calls[1][0][-1]["content"]
    assert len(designs) == 4 and all(design["source"] == "rules" for design in designs)
    assert all("failed validation" in design["fallback_reason"] for design in designs)
    assert "using fixed design rules" in caplog.text


def test_unavailable_endpoint_falls_back_with_explicit_reason(monkeypatch, caplog):
    import httpx
    from openai import APIConnectionError

    monkeypatch.setattr(config, "MOCK_MODE", False)

    def unavailable(*args):
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost"))

    monkeypatch.setattr(llm, "chat", unavailable)
    designs = evolve.propose_generation0(challenge())
    assert len(designs) == 4
    assert all("APIConnectionError" in design["fallback_reason"] for design in designs)
    assert "Designer request failed" in caplog.text


@pytest.mark.parametrize("n", [0, 1, 9, True, 4.5])
def test_invalid_generation_size_is_an_explicit_error(n):
    with pytest.raises(ValueError, match="2-8"):
        evolve.propose_generation0(challenge(), n)


def test_structure_challenge_is_rejected():
    with pytest.raises(ValueError, match="robot-terrain SceneSpec"):
        evolve.propose_generation0(SceneSpec(domain="structure"))


def test_g1_in_challenge_is_untouched_and_never_a_design(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    scene = SceneSpec(
        domain="robot_terrain", terrain=Terrain(kind="flat"),
        robot=Robot(kind="library", robot_id="g1", controller_id="g1_stand"),
    ).fill_defaults()
    before = scene.model_dump()
    designs = evolve.propose_generation0(scene)
    assert scene.model_dump() == before
    assert all(design["robot"]["kind"] != "library" and design["robot"]["robot_id"] is None
               for design in designs)


def test_rule_mutations_still_fill_a_generation_when_parents_sit_at_the_design_limits():
    """Mock-mode Victoria run (12 kg, 0.8 x 0.5 m stowed): every failure-guided change was raced or broke a limit."""
    from simgate.spec import RobotDesignConstraints

    walker = {"kind": "walker", "robot_id": None, "controller_id": None, "wheels": 4, "wheel_radius": 0.15,
              "wheel_width": 0.08, "drive_speed": 6.0, "legs": 6, "leg_length": 0.32, "gait_freq_hz": 1.5,
              "stride_deg": 12.0, "chassis_length": 0.6, "chassis_width": 0.35, "mass_kg": 10.0, "command_speed_m_s": 0.6}
    rover = {**walker, "kind": "rover", "wheels": 6, "wheel_radius": 0.1, "wheel_width": 0.05, "drive_speed": 3.5,
             "legs": 4, "leg_length": 0.22, "gait_freq_hz": 2.0, "stride_deg": 15.0, "chassis_width": 0.38}
    parents = [
        {"id": "G1-C", "name": "Steady Six", "outcome": "flipped", "reached_goal": False, "progress_percent": 45.8,
         "robot": walker, "color": "#ef829f"},
        {"id": "G2-C", "name": "Steady Six 3", "outcome": "flipped", "reached_goal": False, "progress_percent": 44.3,
         "robot": {**walker, "mass_kg": 11.0}, "color": "#718cff"},
        {"id": "G2-A", "name": "Trail Four", "outcome": "slipped", "reached_goal": False, "progress_percent": 40.9,
         "robot": rover, "color": "#40d8a0"},
        {"id": "G2-B", "name": "Wide Six", "outcome": "stuck", "reached_goal": False, "progress_percent": 40.9,
         "robot": {**rover, "drive_speed": 3.0, "mass_kg": 12.0}, "color": "#f2bc55"},
    ]
    raced = {evolve._fingerprint(row["robot"]) for row in parents} | {evolve._fingerprint(robot) for robot in (
        {**rover, "drive_speed": 4.0}, {**rover, "drive_speed": 5.0, "wheels": 4},
        {**rover, "drive_speed": 3.5, "mass_kg": 12.0}, {**rover, "drive_speed": 4.0, "mass_kg": 12.0},
        {**walker, "leg_length": 0.28},
        {**walker, "gait_freq_hz": 2.0, "leg_length": 0.34, "legs": 4, "mass_kg": 7.0, "stride_deg": 18.0})}
    constraints = RobotDesignConstraints(max_mass_kg=12.0, max_size_m=(0.8, 0.5))
    children = evolve._fallback_children(parents, {**parents[0], "tested": True}, 3, 3, raced, "rules", constraints)
    assert len({evolve._fingerprint(child["robot"]) for child in children} - raced) == 3
    assert {child["robot"]["kind"] for child in children} == {"rover", "walker"}
    assert not any(evolve._constraint_violations(Robot.model_validate(child["robot"]), constraints) for child in children)
