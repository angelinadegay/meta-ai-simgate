import pytest

from simgate import evolve
from simgate.spec import Environment, Mission, Robot, SceneSpec, Terrain


def _challenge():
    return SceneSpec(
        domain="robot_terrain",
        title="Reach the hilltop",
        environment=Environment(body="mars", surface="regolith", friction=0.7),
        terrain=Terrain(kind="flat", size_m=10),
        robot=Robot(),
        mission=Mission(start_place="west", goal_place="east"),
    ).fill_defaults()


def _samples(moving=False, slip=0.0):
    return [
        {"time_s": float(second), "x": second * 0.2 if moving else 1.0, "y": 2.0,
         "distance_to_goal_m": 10 - second * 0.2 if moving else 10.0,
         "local_slope_deg": 4.0, "step_m": 0.03, "slip_ratio": slip}
        for second in range(7)
    ]


def _summary(*, reached=False, progress=50, flipped=False, samples=None):
    return {
        "goal_reached": reached, "time_to_goal_s": 12 if reached else None,
        "start_distance_to_goal_m": 10.0, "closest_distance_m": 10 - progress / 10,
        "flipped": flipped, "flipped_at_s": 3.0 if flipped else None,
        "final_xy": [2.0, 3.0],
        "race_telemetry": {"samples": samples if samples is not None else _samples()},
    }


def test_score_detects_all_outcomes_from_telemetry():
    design = {"id": "G0-A", "name": "Test", "robot": {"kind": "rover"}}
    cases = [
        (_summary(reached=True), "reached"),
        (_summary(flipped=True), "flipped"),
        (_summary(samples=_samples()), "stuck"),
        (_summary(samples=_samples(slip=0.8)), "slipped"),
        (_summary(samples=_samples(moving=True)), "too slow"),
    ]
    for summary, expected in cases:
        scored = evolve._score_race_result(design, {"summary": summary}, 90)
        assert scored["outcome"] == expected
        assert scored["event"]["local_slope_deg"] == 4.0
        assert scored["event"]["step_m"] == 0.03
        assert scored["event"]["x"] is not None
        assert scored["time_s"] == (12 if expected == "reached" else 3 if expected == "flipped"
                                    else 5 if expected in ("stuck", "slipped") else 90)


def test_race_runs_designs_on_same_seed_and_ranks_from_simulation(monkeypatch):
    designs = evolve._rules(4)
    calls = {}

    def fake_batch(scene, variations, **kwargs):
        calls["scene"] = scene
        calls["variations"] = variations
        calls["kwargs"] = kwargs
        results = []
        summaries = [
            _summary(reached=True),
            {**_summary(reached=True), "time_to_goal_s": 25},
            _summary(progress=80, samples=_samples(moving=True)),
            _summary(progress=30, samples=_samples(moving=True)),
        ]
        for variation, summary in zip(variations, summaries):
            results.append({"label": variation["label"], "id": variation["label"], "summary": summary})
        return results

    monkeypatch.setattr("simgate.batch.run_batch", fake_batch)
    result = evolve.race_generation(_challenge(), designs, seed=19)

    assert calls["scene"].terrain.seed == 19
    assert [variation["seed"] for variation in calls["variations"]] == [19] * 4
    assert {variation["robot_kind"] for variation in calls["variations"]} == {"rover", "walker"}
    assert calls["kwargs"] == {"baseline": False, "record": False, "race_telemetry": True}
    assert result["start"] == {"place": "west", "xy": None}
    assert result["goal"] == {"place": "east", "xy": None, "radius_m": 1.0}
    assert [row["rank"] for row in result["results"]] == [1, 2, 3, 4]
    assert [row["id"] for row in result["results"]] == ["G0-A", "G0-B", "G0-C", "G0-D"]
    assert [row["outcome"] for row in result["results"]] == ["reached", "reached", "too slow", "too slow"]
    assert result["results"][0]["time_s"] < result["results"][1]["time_s"]
    assert result["results"][2]["progress_percent"] > result["results"][3]["progress_percent"]


def test_victoria_climb_out_resolves_measured_floor_and_rim():
    from simgate.backends.mujoco_backend import make_heights, place_xy, resolve_mission

    scene = evolve.challenge_preset("victoria-climb-out")
    heights, _ = make_heights(scene)
    mission = resolve_mission(scene, heights)
    assert scene.terrain.preset == "victoria-opportunity"
    assert scene.duration_s == 90
    assert scene.environment.gravity == pytest.approx(3.71)
    assert scene.environment.friction == pytest.approx(0.7)
    assert mission["radius"] == 2
    assert mission["start"] == pytest.approx(place_xy(heights, scene.terrain.size_m, "lowest"))
    assert mission["goal"] == pytest.approx(place_xy(heights, scene.terrain.size_m, "highest"))


def test_race_rejects_out_of_space_and_library_candidates():
    designs = evolve._rules(4)
    designs[0]["robot"]["wheel_radius"] = 2.0
    with pytest.raises(ValueError, match="outside"):
        evolve._validate_race_designs(designs)
    designs = evolve._rules(4)
    designs[0]["robot"]["kind"] = "library"
    with pytest.raises(ValueError, match="library robots"):
        evolve._validate_race_designs(designs)
    designs = evolve._rules(4)
    designs[0]["robot"]["unbounded_knob"] = 1
    with pytest.raises(ValueError, match="Unknown robot settings"):
        evolve._validate_race_designs(designs)
