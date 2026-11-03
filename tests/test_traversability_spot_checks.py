import numpy as np

from simgate.spec import Robot, SceneSpec, Terrain
from simgate.traversability import features, verify


def _scene(kind="flat"):
    return SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind=kind, size_m=8 if kind == "flat" else 14),
        robot=Robot(),
    ).fill_defaults()


def _classified_map(scene):
    result = features(scene, scene.robots[0])
    classes = np.empty(result["blocked"].shape, dtype="<U8")
    classes[:, :4] = "green"
    classes[:, 4:8] = "amber"
    classes[:, 8:] = "red"
    classes[result["blocked"]] = "grey"
    return {**result, "classes": classes}


def _fake_batch(outcomes):
    captured = {}

    def run_batch(scene, variations, **kwargs):
        captured["scene"] = scene
        captured["variations"] = variations
        captured["kwargs"] = kwargs
        return [
            {"id": f"run-{index}", "label": variation["label"],
             "summary": {"outcome": outcome, "flipped": outcome == "fell"}}
            for index, (variation, outcome) in enumerate(zip(variations, outcomes), start=1)
        ]

    return run_batch, captured


def test_spot_disagreements_correct_cells_and_report_run_ids():
    scene = _scene()
    fake_batch, captured = _fake_batch(["reached", "stuck", "fell", "reached", "stuck", "reached"])

    result = verify(scene, scene.robots[0], _classified_map(scene), run_batch_fn=fake_batch)

    assert len(captured["variations"]) == 6
    assert captured["kwargs"] == {"baseline": False, "record": False}
    assert result["agreement"] == 2
    assert result["agreement_total"] == 6
    assert result["unreliable"] is True
    assert "2 of 6 spot checks agreed with the map." in result["agreement_text"]
    assert "2 spots disagreed:" in result["agreement_text"]
    assert "the selected area was predicted safe but the robot got stuck; map corrected" in result["agreement_text"]
    assert "predicted safe but the robot got stuck; map corrected" in result["agreement_text"]
    assert (
        "predicted unsafe but the robot crossed without falling or getting stuck; map corrected"
        in result["agreement_text"]
    )

    disagreements = [spot for spot in result["spot_checks"] if spot["corrected"]]
    assert len(disagreements) == 2
    for spot in disagreements:
        row, col = spot["row"], spot["col"]
        expected = "green" if spot["passed"] else "red"
        assert result["classes"][row, col] == expected
        assert result["measured"][row, col]
        assert result["run_ids"][row, col] == spot["id"]


def test_four_judged_agreements_meet_reliability_threshold():
    scene = _scene()
    fake_batch, _ = _fake_batch(["reached", "reached", "fell", "reached", "stuck", "stuck"])

    result = verify(scene, scene.robots[0], _classified_map(scene), run_batch_fn=fake_batch)

    assert result["agreement"] == 4
    assert result["unreliable"] is False
    assert sum(not spot["judged"] for spot in result["spot_checks"]) == 2


def test_indoor_spot_batch_crosses_a_doorway():
    scene = _scene("indoor")
    map_result = features(scene, scene.robots[0])
    map_result["classes"] = np.full(map_result["blocked"].shape, "amber", dtype="<U8")
    map_result["classes"][map_result["blocked"]] = "grey"
    fake_batch, captured = _fake_batch(["reached", "reached"])

    result = verify(scene, scene.robots[0], map_result, run_batch_fn=fake_batch)

    doorway = next(variation for variation in captured["variations"] if "doorway" in variation["label"])
    assert doorway["start_xy"][0] == doorway["goal_xy"][0]
    assert doorway["start_xy"][1] < 0.6 < doorway["goal_xy"][1]
    assert doorway["start_heading_deg"] == 90
    assert result["agreement"] == 0
    assert len(result["spot_checks"]) == 2


def test_batch_variation_applies_the_spot_mission_to_its_target_robot():
    from simgate.batch import _validate_variation

    scene = _scene()
    _, varied, _ = _validate_variation(scene, {
        "label": "spot",
        "robot_id": scene.robots[0].id,
        "start_xy": [-1.5, 0],
        "goal_xy": [1.5, 0],
        "start_heading_deg": 0,
        "goal_radius_m": 0.25,
    })

    assert varied.robots[0].start_xy == [-1.5, 0]
    assert varied.robots[0].goal_xy == [1.5, 0]
    assert varied.mission.start_xy == [-1.5, 0]
    assert varied.mission.goal_xy == [1.5, 0]
