"""Hackathon compliance: Meta models only, no third-party trained models, a standing-only G1."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from simgate import agents, config, evolve, llm
from simgate.director import STANDING_ONLY, Director
from simgate.library import LibraryRegistry
from simgate.library.uploads import AI_CONTROLLER_REJECTED, UploadError, save_upload
from simgate.spec import Mission, Robot, SceneSpec, Terrain

ROOT = Path(__file__).resolve().parents[1]
G1 = Robot(kind="library", robot_id="g1", controller_id="g1_stand")


class FakeClient:
    """The OpenAI-compatible client, scripted: records every model name it is asked for."""

    def __init__(self):
        self.models = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kwargs):
        self.models.append(model)
        system = messages[0]["content"]
        if "You are the Director" in system:
            reply = json.dumps({"workflow": "build", "summary": "Rover on Mars hills", "run_now": False,
                                "world": {"body": "mars", "terrain_kind": "hills"}, "robots": [{"kind": "rover"}]})
        elif "DESIGN_SPACE" in system:
            reply = "not a design"   # rejected twice, so the rules (no model) designs are used
        else:
            reply = "The rover moved 2.5 m on flat ground."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def test_plan_design_and_explanation_call_only_meta_models(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "_client", client)
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=Robot(kind="rover"),
                     mission=Mission(goal_place="east")).fill_defaults()

    plan = Director().plan("A rover on Mars hills")
    designs = evolve.propose_generation0(spec, 4)
    text = agents.explain(spec, {"simulator": "MuJoCo", "forward_progress_m": 2.5, "max_tilt_deg": 4.0,
                                 "flipped": False, "robot": spec.robot.model_dump()})

    assert plan.workflow == "build" and len(designs) == 4 and text
    assert len(client.models) >= 3, client.models
    assert all(config.is_meta_model(name) for name in client.models), client.models
    assert {config.PLANNER_MODEL, config.BUILDER_MODEL} <= set(client.models)


def test_defaults_are_meta_and_other_models_are_refused(monkeypatch):
    assert config.META_BASE_URL == "https://api.meta.ai/v1"
    assert config.is_meta_model("muse-spark-1.3") and config.is_meta_model("muse-wonder-1.0")
    monkeypatch.setattr(config, "DEV_ALLOW_OTHER_MODELS", False)
    monkeypatch.setattr(config, "PLANNER_MODEL", "gpt-4o-mini")
    with pytest.raises(RuntimeError, match="only runs Meta models"):
        config.check_models()
    monkeypatch.setattr(llm, "_client", FakeClient())
    with pytest.raises(ValueError, match="only calls Meta models"):
        llm.chat([{"role": "user", "content": "hi"}], "gpt-4o-mini")
    monkeypatch.setattr(config, "DEV_ALLOW_OTHER_MODELS", True)   # the dev-only override
    config.check_models()
    for path in (ROOT / ".env.example", ROOT / "simgate" / "config.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert "openai.com" not in text and "gpt-" not in text and "anthropic" not in text, path


@pytest.mark.parametrize("kind, files", [
    ("robot", [("robot.xml", b"<mujoco/>"), ("walk.onnx", b"onnx")]),
    ("robot", [("policy.pt", b"PK")]),
    ("terrain", [("motion.onnx", b"onnx")]),
    ("policy", [("motion.onnx", b"onnx")]),
])
def test_uploaded_ai_controllers_are_rejected(kind, files):
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(UploadError) as error:
            save_upload(kind, files, "Walk", "Me", "MIT", Path(folder))
        assert str(error.value) == AI_CONTROLLER_REJECTED == "SimGate doesn't run uploaded AI controllers in this version."
        assert not any(Path(folder).rglob("*.onnx")) and not any(Path(folder).rglob("*.pt"))


def test_onnx_upload_is_rejected_by_the_server():
    from fastapi.testclient import TestClient

    import server

    client = TestClient(server.app)   # no startup: this test doesn't need the key check
    for kind in ("robot", "policy"):
        answer = client.post("/upload", data={"type": kind, "name": "Walk", "credit": "Me", "license": "MIT"},
                             files=[("files", ("motion.onnx", b"onnx bytes"))])
        assert answer.status_code == 400 and answer.json()["error"] == AI_CONTROLLER_REJECTED
    assert "controllers" in client.get("/library").json() and "policies" not in client.get("/library").json()


def test_g1_stand_controller_has_no_model():
    registry = LibraryRegistry()
    robot, controllers = registry.robot("g1"), registry.compatible_controllers("g1")
    assert robot.controller == "stand" and [card.id for card in controllers] == ["g1_stand"]
    card = controllers[0].model_dump()
    assert not {"file", "format", "obs_layout", "recurrent_state_size"} & set(card)
    assert len(card["default_angles"]) == len(card["kps"]) == len(card["kds"]) == 12
    assert STANDING_ONLY in card["notes"]
    # nothing in the project is a trained model, and nothing loads one
    assert not [path for pattern in ("*.onnx", "*.pt", "*.pth", "*.ckpt")
                for path in (ROOT / "simgate").rglob(pattern)]
    assert not (ROOT / "third_party" / "unitree_rl_gym" / "deploy").exists()
    assert not (ROOT / "simgate" / "policy_runtime.py").exists()
    assert "onnxruntime" not in (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for path in (ROOT / "simgate").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import onnxruntime" not in text and "import torch" not in text, path


def test_g1_stands_ten_seconds_on_flat_ground():
    from simgate.backends.mujoco_backend import MuJoCoSimulator

    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=G1,
                     duration_s=10.0).fill_defaults()
    sim = MuJoCoSimulator()
    list(sim.run(spec, sim.build(spec, view=False), record=False))
    summary = sim.summary()
    assert not summary["flipped"] and summary["max_tilt_deg"] < 5 and summary["distance_m"] < 0.05, summary
    assert summary["stopped_early"] is None   # a standing robot is never "stuck": it stands to the time limit


@pytest.mark.parametrize("text", [
    "Make the G1 walk to the north edge",
    "Race a G1 and a rover to the east side",
    "Put a G1 humanoid on Mars hills and walk to the highest point",
    "Find a safe route to the summit",
    "Make it pass",
])
def test_director_refuses_walking_tasks_for_the_g1(monkeypatch, text):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    scene = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=G1).fill_defaults()
    plan = Director().plan(text, scene)
    assert plan.workflow == "answer", plan
    assert STANDING_ONLY in plan.reply and "rover or a walker" in plan.reply


def test_director_refuses_a_model_plan_that_walks_the_g1(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: json.dumps({
        "workflow": "build", "summary": "G1 walks north", "world": {"terrain_kind": "flat"},
        "robots": [{"robot_id": "g1"}], "goal": {"place": "north"}, "run_now": False}))
    plan = Director().plan("G1 humanoid walking north")
    assert plan.workflow == "answer" and STANDING_ONLY in plan.reply


def test_g1_standing_question_uses_the_standing_preset(monkeypatch):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    scene = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="flat", roughness=0), robot=G1).fill_defaults()
    plan = Director().plan("Can the humanoid stand safely on this Mars slope?", scene)
    assert (plan.workflow, plan.failure) == ("find_failures", "safety")
    built = Director().plan("Put a G1 humanoid on Mars hills")
    assert built.workflow == "build" and [(r.robot_id, r.controller_id) for r in built.robots] == [("g1", "g1_stand")]
    assert built.goal is None


def test_standing_preset_tilts_the_ground_and_pushes():
    import math

    from simgate import finder
    from simgate.backends.mujoco_backend import gravity_vector
    from simgate.spec import Environment

    gusev = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="real", preset="gusev-columbia-hills"),
                      environment=Environment(body="mars"), robot=G1)
    fits, _ = finder.scene_knobs(gusev, G1)
    assert {"tilt_deg", "push_n"} <= set(fits) and not {"slope_deg", "speed", "speed_slower"} & set(fits)
    assert finder.change("tilt_deg", 4.0, {"base": fits}) == {"ground_tilt_deg": 4.0}
    tilted = gravity_vector(Environment(gravity=3.71, ground_tilt_deg=10.0), 90.0)
    gx, gy, gz = (float(value) for value in tilted.split())
    assert math.isclose(math.hypot(gx, gy, gz), 3.71, rel_tol=1e-3) and gy < 0 and abs(gx) < 1e-3   # uphill ahead
