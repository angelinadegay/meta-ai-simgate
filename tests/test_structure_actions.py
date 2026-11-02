import asyncio
import copy

import pytest

from simgate import config
from simgate.next_actions import next_actions
from simgate.orchestrator import Session
from simgate.spec import Environment, Mission, Obstacle, Robot, SceneSpec, Terrain, TerrainStructure


def action_scene(kind):
    return SceneSpec(
        domain="robot_terrain", environment=Environment(body="earth"),
        terrain=Terrain(kind="flat", roughness=0, size_m=50), robot=Robot(mass_kg=10),
        mission=Mission(start_xy=[-12, 0], goal_place="east"),
        structures=[TerrainStructure(kind=kind, size="small" if kind == "bridge" else "medium",
                                     placement="over_gap" if kind == "bridge" else "near_goal")],
        obstacles=[Obstacle(kind="gap", placement="path")] if kind == "bridge" else [],
    ).fill_defaults()


@pytest.mark.parametrize("kind,operation,field,expected", [
    ("bridge", "stronger_members", "section", "SHS60x60x4"),
    ("habitat", "stronger_members", "section", "SHS200x200x10"),
    ("lander", "stronger_members", "section", "SHS150x150x8"),
    ("bridge", "mid_support", "mid_support", True),
    ("habitat", "thinner_shielding", "shielding_thickness_m", 0.25),
])
def test_structure_action_changes_only_stated_parameter_and_rechecks(monkeypatch, kind, operation, field, expected):
    monkeypatch.setattr(config, "MOCK_MODE", True)

    async def exercise():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.build_and_check(action_scene(kind))
        before = session.spec.model_dump(mode="json")
        old_build = session.build
        events.clear()
        await session.handle_structure_action(operation, 0)
        assert not [event for event in events if event["type"] == "error"], events
        after = session.spec.model_dump(mode="json")
        before.pop("assumptions")
        after.pop("assumptions")
        before["structures"][0][field] = expected
        assert after == before
        assert session.build is not old_build
        assert any(event["type"] == "checks" for event in events)
        report = session.build.scene["structures"][0]["structural_check"]
        if operation == "stronger_members":
            section = report["section"] if kind == "bridge" else (report["frame"]["section"] if kind == "habitat"
                       else report["cases"]["static"]["frame"]["section"])
            assert section == expected
        if operation == "mid_support":
            assert report["mid_support"]
            assert "struct0_bridge_mid_support" in session.build.code
        if operation == "thinner_shielding":
            assert report["shielding_thickness_m"] == 0.25
            assert any("reduced radiation shielding" in event.get("message", "") for event in events)

    asyncio.run(exercise())


def test_structure_actions_are_specific_and_maximum_section_is_not_offered():
    base = {"kind": "bridge", "size": "small", "request_index": 2, "mid_support": False,
            "structural_check": {"passed": False, "utilisation_pct": 130, "summary": "130% at mid-span"}}
    guidance = next_actions("checks", {"kind": "check_failed", "structures": [base]}, {}, action_scene("bridge"))
    assert [item["label"] for item in guidance["actions"]] == ["Use stronger members", "Add a mid support"]
    assert all(item["action"]["structure_index"] == 2 for item in guidance["actions"])
    largest = {**base, "section": "SHS80x80x4", "mid_support": True}
    assert next_actions("checks", {"structures": [largest]}, {}, action_scene("bridge"))["actions"] == []
    passed = {**base, "structural_check": {"passed": True}}
    ready = next_actions("scene", {"kind": "ready", "structures": [passed]}, {}, action_scene("bridge"))
    assert [item["label"] for item in ready["actions"]] == ["Run the mission", "Find where it breaks", "What if a marsquake hits?"]


def test_malformed_structure_action_does_not_change_scene(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)

    async def exercise():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.build_and_check(action_scene("lander"))
        before = copy.deepcopy(session.spec.model_dump())
        for operation, index in (("mid_support", 0), ("stronger_members", 99), ("unknown", 0)):
            await session.handle_structure_action(operation, index)
            assert session.spec.model_dump() == before
        assert len([event for event in events if event["type"] == "error"]) == 3

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["habitat", "lander", "bridge"])
def test_synthetic_marsquake_rechecks_without_changing_baseline(monkeypatch, kind):
    monkeypatch.setattr(config, "MOCK_MODE", True)

    async def exercise():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        await session.build_and_check(action_scene(kind))
        before = copy.deepcopy(session.spec.model_dump(mode="json"))
        code = session.build.code
        scene = copy.deepcopy(session.build.scene)
        baseline_checks = copy.deepcopy([check.__dict__ for check in session.checks])
        events.clear()
        await session.handle_structure_action("marsquake")
        assert not [event for event in events if event["type"] == "error"]
        result = next(event["report"] for event in events if event["type"] == "structure_shaking")
        assert result["synthetic"] and "Synthetic" in result["label"]
        assert result["pga_m_s2"] == pytest.approx(0.2 * before["environment"]["gravity"])
        assert result["structures"][0]["kind"] == kind
        assert "not measured Mars seismic data" in result["honesty"]
        assert session.spec.model_dump(mode="json") == before
        assert session.build.code == code and session.build.scene == scene
        assert [check.__dict__ for check in session.checks] == baseline_checks

    asyncio.run(exercise())


def test_mid_support_is_a_real_reaction_and_reduces_sag():
    from simgate.structural.bridge import check_bridge
    from simgate.structural.truss import solve_truss

    simple = check_bridge(6, 5000)
    supported = check_bridge(6, 5000, mid_support=True)
    assert supported["checks"]["sag"]["utilisation_pct"] < simple["checks"]["sag"]["utilisation_pct"]
    result = solve_truss(span_m=6, panels=4, depth_m=0.75, point_loads={3: (0, -10_000)}, mid_support=True)
    assert result.displacements_m[3][1] == pytest.approx(0, abs=1e-12)
    assert result.reactions_n[3][1] > 10_000


def test_upgrading_one_structure_does_not_change_another(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)

    async def exercise():
        async def emit(event):
            pass

        session = Session(emit)
        spec = action_scene("bridge")
        spec.structures.append(TerrainStructure(kind="habitat", placement="near_goal"))
        await session.build_and_check(spec)
        before = copy.deepcopy(session.spec.model_dump(mode="json"))
        records = {record["request_index"]: record for record in session.build.scene["structures"]}
        assert records[0]["kind"] == "bridge" and records[1]["kind"] == "habitat"
        before_reports = {index: copy.deepcopy(record["structural_check"]) for index, record in records.items()}
        await session.handle_structure_action("stronger_members", 0)
        after = session.spec.model_dump(mode="json")
        before.pop("assumptions")
        after.pop("assumptions")
        before["structures"][0]["section"] = "SHS60x60x4"
        assert after == before
        records = {record["request_index"]: record for record in session.build.scene["structures"]}
        assert records[1]["structural_check"] == before_reports[1]

    asyncio.run(exercise())


def test_ready_mission_actions_use_existing_workflows_without_editing_scene(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)

    async def exercise():
        async def emit(event):
            pass

        session = Session(emit)
        await session.build_and_check(action_scene("bridge"))
        before = copy.deepcopy(session.spec.model_dump())
        calls = []

        async def run():
            calls.append("run")

        async def make(text, editing, plan):
            calls.append((plan.workflow, plan.robot_id, editing))

        session.run = run
        session.make = make
        guidance = next_actions("scene", {"kind": "ready", "structures": session.build.scene["structures"]}, {}, session.spec)
        for item in guidance["actions"][:2]:
            action = item["action"]
            await session.handle_workflow_action(action["workflow"], action["robot_id"])
        assert calls == ["run", ("find_failures", session.spec.robots[0].id, True)]
        assert session.spec.model_dump() == before

    asyncio.run(exercise())