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


def test_earth_terrain_option_uses_generated_hills():
    session, events = run_flow("Earth terrain instead")

    assert session.spec.environment.body == "earth"
    assert session.spec.environment.surface == "rock"
    assert session.spec.terrain.kind == "hills" and session.spec.terrain.preset is None
    assert any("generated Earth hills" in assumption for assumption in session.spec.assumptions)
    assert any(event["type"] == "scene" for event in events)


def test_keep_mars_option_leaves_real_site_unchanged():
    session, events = run_flow("Keep Mars")

    assert session.spec.environment.body == "mars"
    assert session.spec.terrain.kind == "real" and session.spec.terrain.preset == "gale"
    assert any("question" == event["type"] for event in events)
