import asyncio

from simgate.director import Director
from simgate.spec import Mission, Robot, SceneSpec, Terrain


def gale_scene() -> SceneSpec:
    return SceneSpec(
        domain="robot_terrain",
        title="G1 Humanoid · Gale Crater",
        terrain=Terrain(kind="real", preset="gale"),
        robot=Robot(kind="library", robot_id="g1", policy_id="g1_walking"),
        mission=Mission(goal_place="lowest"),
    ).fill_defaults()


def run_flow(action: str | None = None):
    events = []

    async def emit(event):
        events.append(event)

    async def flow():
        session = __import__("simgate.orchestrator", fromlist=["Session"]).Session(emit)
        session.spec = gale_scene()
        await session.handle("Can you change terrain to Earth?")
        if action:
            await session.handle(action)
        return session

    session = asyncio.run(flow())
    return session, events


def test_gale_earth_change_asks_before_replacing_real_terrain():
    plan = Director().plan("Can you change terrain to Earth?", gale_scene())
    assert plan.workflow == "answer"
    assert "Gale Crater is real Mars terrain" in plan.reply
    assert "Earth gravity on this same terrain" in plan.reply
    assert "Earth terrain instead" in plan.reply
    assert "Keep Mars" in plan.reply

    _, events = run_flow()
    question = next(event for event in events if event["type"] == "question")
    assert "Gale Crater is real Mars terrain" in question["message"]
    assert [action["label"] for action in question["actions"]] == [
        "Earth gravity on this same terrain (a what-if)", "Earth terrain instead", "Keep Mars"]


def test_earth_gravity_option_keeps_gale_elevation():
    session, events = run_flow("Earth gravity on this same terrain")

    assert session.spec.environment.body == "earth"
    assert session.spec.environment.gravity == 9.81
    assert session.spec.terrain.kind == "real" and session.spec.terrain.preset == "gale"
    assert any("what-if" in assumption.lower() for assumption in session.spec.assumptions)
    assert any(event["type"] == "scene" for event in events)


def test_earth_terrain_option_uses_generated_hills(monkeypatch, tmp_path):
    import numpy as np

    from simgate import terrain_data

    monkeypatch.setattr(terrain_data, "TERRAIN_DIR", tmp_path / "terrains")
    monkeypatch.setattr(terrain_data, "_uploaded_dir", lambda: tmp_path / "uploads" / "terrain")
    monkeypatch.setattr(terrain_data, "load_preset", lambda name: (
        np.zeros((32, 32)),
        {"label": "Gale Crater", "body": "mars", "surface": "rock", "size_m": 10.0,
         "relief_m": 0.0, "typical_slope_deg": 0.0, "credit": "Test fixture"},
    ))
    session, events = run_flow("Earth terrain instead")

    assert session.spec.environment.body == "earth"
    assert session.spec.environment.surface == "rock"
    assert session.spec.terrain.kind == "hills" and session.spec.terrain.preset is None
    assert any("generated Earth hills" in assumption for assumption in session.spec.assumptions)
    assert any(event["type"] == "scene" for event in events)


def test_earth_terrain_option_uses_installed_real_earth_preset(monkeypatch):
    import numpy as np

    from simgate import terrain_data
    from simgate.orchestrator import Session

    preset_meta = {
        "gale": {"label": "Gale Crater", "body": "mars", "surface": "rock",
                 "size_m": 100.0, "relief_m": 10.0, "typical_slope_deg": 5.0, "credit": "Test data"},
        "earth-test": {"label": "Earth Test Site", "body": "earth", "surface": "rock",
                        "size_m": 100.0, "relief_m": 10.0, "typical_slope_deg": 5.0, "credit": "Test data"},
    }
    monkeypatch.setattr(terrain_data, "list_presets", lambda: {
        "earth-test": {**preset_meta["earth-test"], "uploaded": False},
    })
    monkeypatch.setattr(terrain_data, "load_preset", lambda name: (np.zeros((2, 2)), preset_meta[name]))

    async def verify():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        session.spec = gale_scene()
        built = []

        async def capture_build(spec):
            session.spec = spec
            built.append(spec)

        session.build_and_check = capture_build
        await session.handle("Change terrain to Earth")
        await session.handle("Earth terrain instead")
        return session, built, events

    session, built, events = asyncio.run(verify())

    assert len(built) == 1
    assert session.spec.environment.body == "earth"
    assert session.spec.terrain.kind == "real" and session.spec.terrain.preset == "earth-test"
    assert any("installed Earth terrain Earth Test Site" in assumption for assumption in session.spec.assumptions)
    assert not any(event["type"] == "error" for event in events)


def test_keep_mars_option_leaves_real_site_unchanged():
    session, events = run_flow("Keep Mars")

    assert session.spec.environment.body == "mars"
    assert session.spec.terrain.kind == "real" and session.spec.terrain.preset == "gale"
    assert any("question" == event["type"] for event in events)
