"""Running SimGate from a fresh clone: model mode with a valid Meta key, mock mode without one, and the saved demos."""
import asyncio
import gzip
import json
import re
from pathlib import Path

import pytest

from simgate import config, cost, demos, llm

ROOT = Path(__file__).resolve().parents[1]


class _Models:
    def __init__(self, ids=None, error=None):
        self.ids, self.error, self.calls = ids, error, 0

    def with_options(self, **_kwargs):
        return self

    @property
    def models(self):
        return self

    def list(self):
        self.calls += 1
        if self.error:
            raise self.error
        return [type("Model", (), {"id": name}) for name in self.ids]


class _Unauthorized(Exception):
    status_code = 401


@pytest.fixture
def mode(monkeypatch):
    """Set the key and flag as a fresh start would see them, and the API the key check talks to."""
    monkeypatch.setattr(config, "MODE_STATUS", "")

    def start(key="", flag=False, api=None):
        monkeypatch.setattr(config, "LLM_API_KEY", key)
        monkeypatch.setattr(config, "FORCE_MOCK", flag)
        monkeypatch.setattr(config, "MOCK_MODE", flag or not key)
        monkeypatch.setattr(llm, "_client", api)
        return llm.check_key()
    return start


def test_no_key_means_mock_mode_without_calling_anything(mode):
    api = _Models(error=AssertionError("no key: nothing to call"))
    assert mode(api=api) == "Key missing/invalid: LLM_API_KEY is not set. Running in mock mode."
    assert config.MOCK_MODE and api.calls == 0
    assert config.mode_label() == "Mock mode: rules, no model"


def test_the_flag_means_mock_mode_even_with_a_key(mode):
    api = _Models(ids=["muse-spark-1.3", "muse-wonder-1.0"])
    assert mode(key="sk-meta", flag=True, api=api) == "SIMGATE_MOCK_MODE=1. Running in mock mode."
    assert config.MOCK_MODE and api.calls == 0


def test_a_valid_key_is_checked_once_and_means_model_mode(mode):
    api = _Models(ids=["muse-spark-1.3", "muse-wonder-1.0", "muse-image-1.0"])
    assert mode(key="sk-meta", api=api) == "Connected to Meta Model API (muse-spark-1.3, muse-wonder-1.0)"
    assert not config.MOCK_MODE and api.calls == 1
    assert config.mode_label() == "Meta models: muse-spark-1.3 · muse-wonder-1.0"


@pytest.mark.parametrize(("api", "reason"), [
    (_Models(error=_Unauthorized("401 Unauthorized")), "the Meta Model API rejected the key (401 Unauthorized)"),
    (_Models(error=TimeoutError("timed out")), "couldn't reach the Meta Model API (TimeoutError)"),
    (_Models(ids=["muse-spark-1.3"]), "the key can't use muse-wonder-1.0"),
])
def test_an_invalid_key_says_why_and_means_mock_mode_without_crashing(mode, api, reason):
    assert mode(key="sk-wrong", api=api) == f"Key missing/invalid: {reason}. Running in mock mode."
    assert config.MOCK_MODE and config.mode_label() == "Mock mode: rules, no model"


def test_startup_and_the_ui_show_the_mode(monkeypatch, caplog):
    import server
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "MODE_STATUS", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-wrong")
    monkeypatch.setattr(config, "FORCE_MOCK", False)
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "_client", _Models(error=_Unauthorized("401 Unauthorized")))
    with caplog.at_level("WARNING"), TestClient(server.app) as client:   # startup runs the key check
        health = client.get("/health").json()
        with client.websocket_connect("/ws") as socket:
            hello = socket.receive_json()
    assert "Key missing/invalid" in caplog.text
    assert health["mock_mode"] and health["mode"] == "Mock mode: rules, no model"
    assert hello["mode"] == "Mock mode: rules, no model" and hello["mode_status"].startswith("Key missing/invalid")
    page = (ROOT / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    assert '$("mode").textContent = ev.mode' in page   # the header shows the server's mode line
    assert re.search(r"max-width: 860px\) \{ \.top \.mode \{ display: inline-block", page)   # on phones too


def test_mock_mode_makes_zero_api_calls(monkeypatch):
    from simgate.orchestrator import Session

    def no_calls(*_args, **_kwargs):
        raise AssertionError("mock mode must not call the API")

    monkeypatch.setattr(config, "MOCK_MODE", True)
    monkeypatch.setattr(llm, "client", no_calls)
    events = []

    async def emit(event):
        events.append(event)

    asyncio.run(Session(emit).handle("a rover on flat ground, go"))
    spent = [event["cost"] for event in events if event["type"] == "cost"][-1]
    assert spent["model_calls"] == 0 and spent["runs"] >= 1 and "0 model calls" in spent["text"]
    steps = [event for event in events if event["type"] == "step"]
    assert steps and all(step["model"] == "rules" for step in steps)   # every card says rules produced it
    with pytest.raises(llm.ModelUnavailable):
        llm.chat([{"role": "user", "content": "hi"}])


class _Down:
    """An API that times out on every chat call."""
    def __init__(self):
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **_kwargs):
        from openai import APITimeoutError

        self.calls += 1
        raise APITimeoutError(request=None)


def test_an_api_failure_mid_session_falls_back_to_rules_labels_it_and_continues(monkeypatch):
    from simgate.orchestrator import Session

    api = _Down()
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "_client", api)
    monkeypatch.setattr(llm, "_down_until", 0.0)
    events = []

    async def emit(event):
        events.append(event)

    asyncio.run(Session(emit).handle("a rover on flat ground, go"))
    assert not [event for event in events if event["type"] == "error"]
    assert any(event["type"] == "metrics" for event in events), "the run still happened"
    plan = next(event for event in events if event["type"] == "step" and event.get("tool") == "plan")
    assert plan["model"] == "rules (model unavailable)" and "didn't answer (APITimeoutError)" in plan["result"]
    spent = [event["cost"] for event in events if event["type"] == "cost"][-1]
    assert spent["model_failures"] >= 1 and "fell back to rules (model unavailable)" in spent["text"]
    assert api.calls == 1, "after one failure the next calls fail fast instead of each waiting out a timeout"
    page = (ROOT / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    assert "if (ev.cost.model_failures) addMsg(" in page


def test_every_model_reply_path_survives_an_unavailable_model(monkeypatch):
    from simgate.facts import checked

    def down(*_args, **_kwargs):
        raise llm.ModelUnavailable("APITimeoutError")

    monkeypatch.setattr(llm, "chat", down)
    assert llm.chat_json([], "muse-spark-1.3", dict)[0] is None
    assert checked(lambda _feedback: llm.chat([]), {}, lambda: "rules text")[0] == "rules text"


def test_env_example_has_exactly_the_variables_the_code_reads():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    settings = dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#"))
    assert settings == {"LLM_API_KEY": "", "LLM_BASE_URL": "https://api.meta.ai/v1", "PLANNER_MODEL": "muse-spark-1.3",
                        "BUILDER_MODEL": "muse-wonder-1.0", "SIMGATE_MOCK_MODE": "0"}
    code = (ROOT / "simgate" / "config.py").read_text(encoding="utf-8")
    assert all(f'os.getenv("{name}"' in code for name in settings)
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert all(f"{name}: \"${{{name}:-" in compose for name in settings)   # each with a default: no .env still starts


def test_missing_insight_model_uses_the_fallback_shape(monkeypatch, tmp_path, capsys):
    import server
    from simgate import structures
    from simgate.spec import TerrainStructure

    lander = TerrainStructure(kind="lander", placement="near_start")
    if structures.INSIGHT_GLB.is_file():
        assert structures._dimensions(lander, 0.5, "mars")["record"]["visual_model"] == "nasa_insight"
    monkeypatch.setattr(structures, "INSIGHT_GLB", tmp_path / "missing.glb")
    record = structures._dimensions(lander, 0.5, "mars")["record"]
    assert "visual_model" not in record and record["legs"] == 4   # the simple fallback shape

    monkeypatch.setattr(server, "INSIGHT_GLB", tmp_path / "missing.glb")
    from fastapi.testclient import TestClient

    assert TestClient(server.app).get("/assets/nasa/insight_lander.glb").status_code == 404


def test_fetch_assets_checks_the_hash_and_only_warns_when_it_fails(tmp_path, capsys):
    import io
    import sys

    sys.path.insert(0, str(ROOT))
    from scripts import fetch_assets

    content = b"glb bytes"
    import hashlib

    asset = {"name": "Test model", "path": "assets/nasa/x.glb", "url": "https://example.test/x.glb",
             "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    assert fetch_assets.fetch(asset, tmp_path, opener=lambda *_a, **_k: io.BytesIO(content))
    assert (tmp_path / "assets/nasa/x.glb").read_bytes() == content
    (tmp_path / "assets/nasa/x.glb").unlink()
    assert not fetch_assets.fetch(asset, tmp_path, opener=lambda *_a, **_k: io.BytesIO(b"something else"))
    assert not fetch_assets.fetch(asset, tmp_path, opener=lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")))
    assert not (tmp_path / "assets/nasa/x.glb").exists() and not list((tmp_path / "assets/nasa").glob("*.part"))
    assert "WARNING Test model not downloaded" in capsys.readouterr().err
    assert fetch_assets.main.__doc__ is None and "return 0" in Path(fetch_assets.__file__).read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "RUN python scripts/fetch_assets.py" in dockerfile
    assert "assets/" in (ROOT / ".dockerignore").read_text(encoding="utf-8").split()   # no raw data in the image


def test_every_terrain_preset_loads_without_raw_terrain_data(monkeypatch):
    from simgate import terrain_data

    def no_raw_data(*_args, **_kwargs):
        raise AssertionError("loading a preset must not read raw terrain data")

    monkeypatch.setattr(terrain_data, "load_structure_patch", no_raw_data)
    presets = terrain_data.list_presets()
    assert len(presets) >= 12
    for name in presets:
        heights, meta = terrain_data.load_preset(name)
        assert heights.ndim == 2 and heights.size and meta["label"], name


def test_a_larger_structure_patch_without_raw_data_keeps_the_shipped_patch(monkeypatch):
    from simgate import structures, terrain_data
    from simgate.spec import SceneSpec, Terrain, TerrainStructure

    def raw_data_missing(*_args, **_kwargs):
        raise FileNotFoundError("assets/nasa/terrain_raw/...IMG")

    monkeypatch.setattr(terrain_data, "load_structure_patch", raw_data_missing)
    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind="real", preset="gusev-columbia-hills"),
                     structures=[TerrainStructure(kind="habitat", placement="near_start"),
                                 TerrainStructure(kind="lander", placement="near_goal")]).fill_defaults()
    size = spec.terrain.size_m
    structures.prepare_structure_terrain(spec)
    assert spec.terrain.size_m == size and spec.terrain.structure_crop_size_m is None
    assert any("Kept the shipped" in note and "FileNotFoundError" in note for note in spec.assumptions)


def test_saved_demos_are_small_recorded_with_meta_models_and_listed():
    listed = demos.list_demos()
    assert {demo["id"] for demo in listed} == {"crater-evolution", "apartment-traversability", "gusev-safe-route"}
    for demo in listed:
        size = (demos.DEMOS_DIR / demo["file"]).stat().st_size
        assert size == demo["size_bytes"] < 5_000_000, demo["id"]
        assert demo["recorded"]["mock_mode"] is False and demo["recorded"]["planner_model"].startswith("muse-")
        assert demo["recorded"]["builder_model"].startswith("muse-")
    page = (ROOT / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    assert "Demo runs: replay instantly (no key needed)" in page


@pytest.mark.parametrize("mock", [True, False])
def test_a_demo_replays_with_zero_simulations_and_zero_model_calls(monkeypatch, mock):
    from simgate import agents
    from simgate.orchestrator import Session

    def nothing_runs(*_args, **_kwargs):
        raise AssertionError("a demo replay must not simulate or call a model")

    monkeypatch.setattr(config, "MOCK_MODE", mock)
    monkeypatch.setattr(llm, "chat", nothing_runs)
    monkeypatch.setattr(agents, "route", nothing_runs)
    events = []

    async def emit(event):
        events.append(event)

    async def replay():
        session = Session(emit)
        with cost.track() as spent:
            await session.play_demo("apartment-traversability")
            report = next(event["report"] for event in events if event["type"] == "traversability")
            safe = next(run for run in report["comparison"]["runs"] if run["label"] == "Safe route")
            await session.replay_map_run(safe["id"])   # "Watch safe route": its recorded frames
        return spent, report

    spent, report = asyncio.run(replay())
    assert spent.runs == 0 and spent.model_calls == 0
    assert not [event for event in events if event["type"] == "error"]
    assert report["headline"].startswith("The rover can reach the bedroom via the ramp")
    assert any(event["type"] == "frames" for event in events) and any(event["type"] == "metrics" for event in events)
    costs = [event["cost"] for event in events if event["type"] == "cost"]
    assert costs and all(cost_["runs"] == 0 and cost_["model_calls"] == 0 for cost_ in costs)
    assert costs[0]["text"].startswith("Replay · 0 runs · 0 model calls (recorded: ")


def test_the_evolution_demo_is_served_compressed_for_the_browser():
    from fastapi.testclient import TestClient

    import server

    answer = TestClient(server.app).get("/demos/crater-evolution")
    report = answer.json()["report"]
    assert answer.status_code == 200 and report["challenge_spec"] and report["generations"]
    assert all("race_replay" in generation for generation in report["generations"])
    assert TestClient(server.app).get("/demos/nope").status_code == 404
    raw = gzip.decompress((demos.DEMOS_DIR / "crater-evolution.json.gz").read_bytes())
    assert json.loads(raw)["report"]["demo"]["mock_mode"] is False
