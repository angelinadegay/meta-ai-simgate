import asyncio
from pathlib import Path
from types import SimpleNamespace

from simgate.plan import Plan
from simgate.spec import Robot, RobotEntry, SceneSpec, Terrain


def race_scene() -> SceneSpec:
    return SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="slope", slope_deg=5.0, roughness=0),
        robots=[
            RobotEntry(id="r1", label="Rover", robot=Robot(kind="rover"), goal_place="east"),
            RobotEntry(id="r2", label="Walker", robot=Robot(kind="walker", legs=4), goal_place="east"),
        ],
    ).fill_defaults()


def evidence_report() -> dict:
    single_change = {"slope_deg": 22.0}
    combined_change = {"slope_deg": 25.0, "friction": 0.4}
    return {
        "type": "envelope",
        "robot_id": "r2",
        "robot_label": "Walker",
        "explanation": "Walker fails at the confirmed slope.",
        "honesty": "Confirmed with three seeds.",
        "knobs": ["slope_deg", "friction"],
        "failures": [
            {"kind": "single", "run": "r4", "values": single_change,
             "distance": 0.2, "confirm": {"label": "reliable fail"}},
            {"kind": "combined", "run": "r9", "values": combined_change,
             "distance": 0.01, "confirm": {"label": "reliable fail"}},
        ],
        "evidence": [
            {"id": "r4", "broke": True, "change": single_change},
            {"id": "r9", "broke": True, "change": combined_change},
            {"id": "c4-s2", "confirms": "r4", "seed": 2, "broke": True,
             "failure": "fell", "change": {**single_change, "seed": 9, "perturb_seed": 2}},
            {"id": "c4-s1", "confirms": "r4", "seed": 1, "broke": True,
             "failure": "fell", "change": {**single_change, "seed": 8, "perturb_seed": 1}},
            {"id": "c9-s1", "confirms": "r9", "seed": 1, "broke": True,
             "failure": "missed the goal", "change": {**combined_change, "seed": 8, "perturb_seed": 1}},
        ],
    }


def test_race_actions_keep_finder_diagnose_and_redesign_robot_ids():
    from simgate.facts import facts
    from simgate.next_actions import next_actions

    scene = race_scene()
    run = next_actions(
        "run",
        {"kind": "run_failed", "passed": False, "metrics": {"robots": {
            "r1": {"goal_reached": True}, "r2": {"goal_reached": False},
        }}},
        {"robots": [{"robot": "Rover", "reason": "reached the goal"},
                     {"robot": "Walker", "reason": "fell over"}]},
        scene,
    )
    assert [action["action"]["robot_id"] for action in run["actions"]] == ["r2", "r2", "r2"]

    report = evidence_report()
    finder_actions = next_actions("finder", {"report": report}, facts(scene, {}), scene)["actions"]
    assert finder_actions[0]["action"] == {"type": "replay_weakest_spot", "robot_id": "r2"}
    assert [action["action"]["robot_id"] for action in finder_actions[1:]] == ["r2", "r2"]

    diagnosis = next_actions("diagnosis", {"report": {
        "type": "diagnosis", "robot_id": "r2", "reproduced": True,
        "hypotheses": [{"cause": "steep ground", "verdict": "confirmed"}],
    }}, facts(scene, {}), scene)
    assert diagnosis["actions"][0]["action"]["robot_id"] == "r2"
    assert diagnosis["actions"][2]["action"]["robot_id"] == "r2"

    redesign = next_actions("redesign", {"report": {
        "type": "redesign", "robot_id": "r2",
        "winner": {"changes": {"legs": 6}, "description": "six legs"},
    }}, facts(scene, {}), scene)
    assert all(action["action"]["robot_id"] == "r2" for action in redesign["actions"])


def test_session_replays_simplest_confirmed_failure_seed_with_frames(monkeypatch):
    from simgate import agents, config, finder
    from simgate.orchestrator import Session

    monkeypatch.setattr(config, "MOCK_MODE", True)
    report = evidence_report()
    seen_finder_targets = []

    def fake_find_failures(spec, robot_id, failure, budget, knobs, on_step):
        seen_finder_targets.append(robot_id)
        return report

    monkeypatch.setattr(finder, "find_failures", fake_find_failures)
    scene_specs = []
    run_specs = []

    class FakeSimulator:
        name = "MuJoCo"

        def run(self, spec, build):
            run_specs.append(spec.model_copy(deep=True))
            yield [{"t": 0.0, "poses": {}}]

        def summary(self):
            return {"goal_reached": False, "robots": {"r1": {"goal_reached": True},
                                                          "r2": {"goal_reached": False}}}

    fake_sim = FakeSimulator()
    monkeypatch.setattr(agents, "route", lambda spec: fake_sim)

    def fake_build(sim, spec, log):
        scene_specs.append(spec.model_copy(deep=True))
        return SimpleNamespace(scene={"kind": "mujoco", "robots": {"r1": {}, "r2": {}}},
                               code="<mujoco/>", language="MuJoCo")

    monkeypatch.setattr(agents, "build", fake_build)

    async def flow():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        session.spec = race_scene()
        plan = Plan(workflow="find_failures", summary="Where it breaks", robot_id="r2")
        await session.make("find where it breaks", editing=True, plan=plan)
        next(event for event in events if event["type"] == "envelope")
        await session.handle_workflow_action("replay_weakest_spot", "r2")
        return session, events

    session, events = asyncio.run(flow())

    assert seen_finder_targets == ["r2"]
    assert session.last_finder_report is report
    replay_spec = scene_specs[-1]
    assert len(replay_spec.robots) == 2 and replay_spec.robots[1].id == "r2"
    assert replay_spec.terrain.slope_deg == 22.0
    assert replay_spec.terrain.seed == 8 and replay_spec.perturb_seed == 1
    assert run_specs[-1].terrain.seed == 8 and run_specs[-1].perturb_seed == 1
    replay_events = events[events.index(next(event for event in events if event["type"] == "envelope")) + 1:]
    types = [event["type"] for event in replay_events]
    assert types.index("run_start") < types.index("frames") < types.index("metrics")
    assert not any(event["type"] == "diagnosis" for event in replay_events)
    assert any("confirmed run r4" in event.get("message", "") for event in replay_events)
    assert session.spec.terrain.slope_deg == 5.0


def test_session_typed_analysis_actions_pass_selected_robot_id():
    from simgate.orchestrator import Session

    async def verify():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        session.spec = race_scene()
        plans = []

        async def capture_make(text, editing, plan=None):
            plans.append(plan)

        session.make = capture_make
        for workflow in ("diagnose", "redesign", "find_failures"):
            await session.handle_workflow_action(workflow, "r2")
        assert [plan.workflow for plan in plans] == ["diagnose", "redesign", "find_failures"]
        assert all(plan.robot_id == "r2" for plan in plans)
    asyncio.run(verify())


def test_server_and_browser_route_targeted_finder_actions():
    root = Path(__file__).resolve().parent.parent
    page = (root / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    server = (root / "server.py").read_text(encoding="utf-8")
    assert 'type: "workflow_action", workflow, robot_id: robotId' in page
    assert 'sendWorkflowAction("replay_weakest_spot", option.action.robot_id, option.label)' in page
    assert 'msg.get("type") == "workflow_action"' in server
    assert 'await session.handle_workflow_action(msg.get("workflow", ""), msg.get("robot_id"))' in server
    assert 'await session.show_design(msg.get("robot_id"))' in server
