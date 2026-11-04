import asyncio
import json

import pytest

from simgate import config
from simgate.director import Director
from simgate.next_actions import next_actions
from simgate.orchestrator import Session
from simgate.plan import Plan


@pytest.mark.parametrize("cue", [
    "where can it go", "can it reach the bedroom", "is it safe to walk",
    "safe route", "plan a route", "map where it can drive",
])
def test_director_traversability_cues(cue, monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    assert Director().plan(cue).workflow == "traversability"


def test_reach_bedroom_sets_the_mock_goal(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    assert Director().plan("In the apartment, can it reach the bedroom?").goal.place == "bedroom"


@pytest.mark.parametrize("mock", [True, False])
def test_ordered_room_request_keeps_both_destinations(monkeypatch, mock):
    from simgate import llm

    monkeypatch.setattr(config, "MOCK_MODE", mock)
    monkeypatch.setattr(llm, "chat", lambda *args: json.dumps({
        "workflow": "build", "summary": "Apartment rover",
        "world": {"terrain_kind": "indoor", "preset": "apartment"},
        "robots": [{"kind": "rover", "start_place": "kitchen"}],
        "goal": {"place": "living_room"},
    }))
    plan = Director().plan("In the apartment, can the rover go from the kitchen to the living room, then to the bedroom?")
    assert plan.workflow == "traversability"
    assert plan.via == ["living_room"] and plan.goal.place == "bedroom"


def test_multi_leg_report_compares_each_leg_and_keeps_replays_unique(monkeypatch):
    from simgate import batch, traversability
    from simgate.finder import limits_world
    from simgate.spec import Mission, Robot, SceneSpec, Terrain

    scene = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="indoor"), robot=Robot(),
                      mission=Mission(start_place="kitchen", goal_place="bedroom")).fill_defaults()
    report = {"robot_id": scene.robots[0].id, "world": limits_world(scene, scene.robots[0].id), "limits": [
        {"knob": knob, "reliable_ok": ok, "first_fail": fail}
        for knob, ok, fail in [("slope_deg", 10, 20), ("roughness_m", 0.02, 0.1),
                               ("step", 0.03, 0.08), ("friction", 0.4, 0.2)]
    ]}
    calls = []

    def fake_batch(base, variations, **kwargs):
        calls.append((base, variations))
        return [{"id": f"run-{i}", "label": variation["label"], "summary": {
            "outcome": "stuck" if variation["label"].startswith("Spot") else "reached",
            "goal_reached": not variation["label"].startswith("Spot"),
            "time_to_goal_s": 5, "path_length_m": 4,
        }} for i, variation in enumerate(variations, 1)]

    monkeypatch.setattr(batch, "run_batch", fake_batch)
    result = traversability.map_report(scene, finder_report=report, via=["living_room"])
    legs = result["comparison"]["legs"]
    assert len(legs) == 2
    assert legs[0]["route"]["start_xy"] == [-4.5, -2.5]
    assert legs[0]["route"]["goal_xy"] == legs[1]["route"]["start_xy"] == [2.4, -2.5]
    assert legs[1]["route"]["goal_xy"] == [-4.5, 3.2]
    assert all(leg["route"]["found"] for leg in legs)
    assert result["comparison"]["route"]["length_m"] == sum(leg["route"]["length_m"] for leg in legs)
    runs = result["comparison"]["runs"]
    assert len(runs) == 4 and len({run["id"] for run in runs}) == 4
    assert all(run["id"] in result["replays"] for run in runs)
    assert all(variations[0]["seed"] == variations[1]["seed"] for _, variations in calls[1:])


def test_structure_start_requires_supported_mission_target():
    with pytest.raises(ValueError, match="structure start must be the lander"):
        Plan(workflow="traversability", summary="map", start={"structure": "habitat_airlock"})


def test_meta_plan_schema_and_guard(monkeypatch):
    from simgate import llm

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "chat", lambda *args: json.dumps({
        "workflow": "traversability", "summary": "Gusev walking map",
        "world": {"terrain_kind": "real", "preset": "gusev-columbia-hills"},
        "robots": [{"kind": "walker"}], "goal": {"place": "highest"},
    }))
    result = Director().plan("Where on Gusev can the walker go safely? Plan a safe route to the summit and walk it.")
    assert result.workflow == "traversability"
    assert result.goal.place == "highest"
    assert result.world.preset == "gusev-columbia-hills"


def test_a_g1_traversability_map_is_refused_because_it_only_stands(monkeypatch):
    from simgate import llm
    from simgate.director import STANDING_ONLY

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "chat", lambda *args: json.dumps({
        "workflow": "traversability", "summary": "Gusev walking map",
        "world": {"terrain_kind": "real", "preset": "gusev-columbia-hills"},
        "robots": [{"robot_id": "g1"}], "goal": {"place": "highest"},
    }))
    result = Director().plan("Where on Gusev can the G1 walk safely? Plan a safe route to the summit and walk it.")
    assert result.workflow == "answer" and STANDING_ONLY in result.reply


def test_finder_next_action_opens_map():
    actions = next_actions("finder", {"report": {}}, {}, None)["actions"]
    action = next(action for action in actions if action["label"] == "Map where it can go here")
    assert action["action"]["workflow"] == "traversability"


def test_workflow_emits_scene_before_map_and_retains_replay(monkeypatch):
    from simgate import traversability

    monkeypatch.setattr(config, "MOCK_MODE", True)
    events = []
    report = {"type": "traversability", "robot_id": "r1", "summary": "Measured map ready.",
              "replays": {}, "honesty": "Measured limits only"}
    monkeypatch.setattr(traversability, "map_report", lambda *args: report)

    async def run():
        async def emit(event):
            events.append(event)
        session = Session(emit)
        await session.make("map", False, Plan(
            workflow="traversability", summary="Apartment rover map",
            world={"terrain_kind": "indoor", "preset": "apartment"},
            robots=[{"kind": "rover", "start_place": "kitchen"}], goal={"place": "bedroom"},
        ))
        assert session.last_map_report == report
        assert session.spec.mission.start_place == "kitchen"
        assert session.spec.mission.goal_place == "bedroom"

    asyncio.run(run())
    types = [event["type"] for event in events]
    assert types.index("scene") < types.index("traversability")


@pytest.mark.parametrize("with_goal", [False, True])
def test_map_report_serializes_measurements_and_exact_replay_recipes(monkeypatch, with_goal):
    from simgate import batch, cost, traversability
    from simgate.finder import limits_world
    from simgate.spec import Mission, Robot, SceneSpec, Terrain

    scene = SceneSpec(
        domain="robot_terrain", terrain=Terrain(kind="flat", size_m=8), robot=Robot(),
        mission=Mission(start_xy=[-2, 0], goal_xy=[2, 0] if with_goal else None),
    ).fill_defaults()
    limits = {"robot_id": scene.robots[0].id, "world": limits_world(scene, scene.robots[0].id), "limits": [
        {"knob": knob, "reliable_ok": ok, "first_fail": fail}
        for knob, ok, fail in [("slope_deg", 20, 30), ("roughness_m", 0.1, 0.2),
                               ("step", 0.2, 0.3), ("friction", 0.4, 0.3)]
    ]}
    calls = []

    def fake_batch(base, variations, **kwargs):
        calls.append((base, variations))
        cost.add_runs(len(variations))
        return [{"id": f"run-{i}", "label": variation["label"], "summary": {
            "outcome": "reached", "goal_reached": True, "time_to_goal_s": 5, "path_length_m": 4,
        }} for i, variation in enumerate(variations, 1)]

    monkeypatch.setattr(batch, "run_batch", fake_batch)
    result = json.loads(json.dumps(traversability.map_report(scene, finder_report=limits), allow_nan=False))
    assert result["classes"] and result["slope_deg"]
    assert result["honesty"].startswith("Map from measured limits")
    assert sum(result["percentages"].values()) == 100
    assert result["cost"]["runs"] == sum(len(variations) for _, variations in calls)
    assert all(spot["id"].startswith("spot-run-") for spot in result["spot_checks"])
    if with_goal:
        assert result["comparison"]["route"]["found"]
        assert all(run["id"].startswith("route-run-") for run in result["comparison"]["runs"])
    else:
        assert len(calls) == 1
        assert result["comparison"]["route"] is None
        assert "no goal selected" in result["comparison"]["text"]
    for base, variations in calls:
        for variation in variations:
            recipe = next(recipe for recipe in result["replays"].values() if recipe["variation"] == variation)
            assert recipe["scene_spec"] == base.model_dump(mode="json")
    assert len(result["replays"]) == result["cost"]["runs"]


def test_map_replay_streams_real_frames_without_replacing_source_scene(monkeypatch):
    from simgate.spec import Mission, Robot, SceneSpec, Terrain

    monkeypatch.setattr(config, "MOCK_MODE", True)
    scene = SceneSpec(
        domain="robot_terrain", terrain=Terrain(kind="flat", size_m=8), robot=Robot(),
        mission=Mission(start_xy=[-1.5, 0], goal_xy=[1.5, 0]),
        duration_s=0.3, time_limit_given=True,
    ).fill_defaults()
    events = []

    async def run():
        async def emit(event):
            events.append(event)

        session = Session(emit)
        session.spec = scene
        session.last_map_report = {"replays": {"spot-run-1": {
            "scene_spec": scene.model_dump(mode="json"),
            "variation": {"label": "replay", "robot_id": scene.robots[0].id,
                          "start_xy": [-1, 0], "goal_xy": [1, 0]},
        }}}
        await session.replay_map_run("spot-run-1")
        assert session.spec is scene
        assert session.last_map_report["replays"]

    asyncio.run(run())
    assert not [event for event in events if event["type"] == "error"]
    assert any(event["type"] == "frames" and event["frames"] for event in events)
    assert any(event["type"] == "metrics" for event in events)
    assert events[-1]["type"] == "cost" and events[-1]["cost"]["runs"] == 1
