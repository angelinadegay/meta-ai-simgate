import pytest
from fastapi.testclient import TestClient

from simgate.backends.mujoco_backend import MuJoCoSimulator
from simgate.spec import Environment, Mission, Obstacle, Robot, SceneSpec, Terrain, TerrainStructure


def test_optional_nasa_asset_route_is_local_and_handles_absence(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "__file__", str(tmp_path / "server.py"))
    client = TestClient(server.app)
    assert client.get("/assets/nasa/insight_lander.glb").status_code == 404
    model = tmp_path / "assets" / "nasa" / "insight_lander.glb"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"glTF-local-test-fixture")
    response = client.get("/assets/nasa/insight_lander.glb")
    assert response.status_code == 200 and response.content == model.read_bytes()
    assert response.headers["content-type"] == "model/gltf-binary"
    assert client.get("/assets/nasa/other.glb").status_code == 404


@pytest.mark.parametrize("kind,placement", [("habitat", "near_goal"), ("lander", "near_start"), ("bridge", "over_gap")])
def test_structure_view_data_does_not_change_physics_or_summary(kind, placement):
    spec = SceneSpec(
        domain="robot_terrain", environment=Environment(body="mars"),
        terrain=Terrain(kind="flat", roughness=0, size_m=40), robot=Robot(kind="rover"),
        mission=Mission(start_xy=[-12, 0], goal_xy=[12, 0]), duration_s=2,
        structures=[TerrainStructure(kind=kind, placement=placement)],
        obstacles=[Obstacle(kind="gap", size="small", placement="path")] if kind == "bridge" else [],
    ).fill_defaults()
    results = []
    for view in (True, False):
        current = spec.model_copy(deep=True)
        simulator = MuJoCoSimulator()
        build = simulator.build(current, view=view)
        for _ in simulator.run(current, build, record=False):
            pass
        results.append((build.code, simulator.summary()))
    assert results[0] == results[1]