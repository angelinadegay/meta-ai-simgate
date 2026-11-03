import json

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


def _generation_results(designs, *, progress=40, outcome="stuck", slope=12):
    return [{
        **design, "reached_goal": False, "progress_percent": progress,
        "time_s": 90, "outcome": outcome,
        "event": {"x": -2.0, "y": 3.0, "local_slope_deg": slope, "step_m": 0.22},
    } for design in designs]


def test_next_generation_keeps_elite_and_generates_unique_bounded_children(monkeypatch):
    monkeypatch.setattr(evolve.config, "MOCK_MODE", True)
    designs = evolve._rules(4)
    raced = _generation_results(designs)
    expected_elite = max(raced, key=lambda row: row["progress_percent"])
    next_designs = evolve.next_generation({"generation": 0, "results": raced}, n=4)

    assert next_designs[0]["id"] == expected_elite["id"]
    assert next_designs[0]["robot"] == expected_elite["robot"]
    assert len(next_designs) == 4
    assert len({row["color"] for row in next_designs}) == 4
    assert {row["robot"]["kind"] for row in next_designs} == {"rover", "walker"}
    for child in next_designs[1:]:
        assert child["parent_id"] in {row["id"] for row in raced}
        assert 1 <= len(child["changes"]) <= 2
        assert child["changed_parameters"] == list(child["changes"])
        assert child["reason"] and "stuck" in child["reason"]
        assert evolve._fingerprint(child["robot"]) not in {
            evolve._fingerprint(row["robot"]) for row in raced
        }


def test_child_validation_rejects_more_than_two_parameters_and_clamps():
    rows = _generation_results(evolve._rules(4))
    elite = rows[0]
    with pytest.raises(ValueError, match="one or two"):
        evolve._validate_children([{
            "parent_id": rows[1]["id"], "name": "Too many",
            "reason": "A reason linked to failure.",
            "changes": {"leg_length": .3, "stride_deg": 18, "gait_freq_hz": 2},
        }], rows, elite, 1, 1, set())
    children = evolve._validate_children([{
        "parent_id": rows[1]["id"], "name": "Clamped",
        "reason": "Longer legs address the stuck result.",
        "changes": {"leg_length": 9},
    }], rows, elite, 1, 1, set())
    assert children[0]["robot"]["leg_length"] == .4
    assert children[0]["changes"] == {"leg_length": .4}
    duplicate = dict(rows[1]["robot"])
    duplicate["leg_length"] = .3
    with pytest.raises(ValueError, match="already raced"):
        evolve._validate_children([{
            "parent_id": rows[1]["id"], "name": "Duplicate",
            "reason": "The original parameters did not address its failure.",
            "changes": {"leg_length": .3},
        }], rows, elite, 1, 1, {evolve._fingerprint(duplicate)})


def test_designer_receives_failure_facts_and_returns_three_children(monkeypatch):
    monkeypatch.setattr(evolve.config, "MOCK_MODE", False)
    rows = _generation_results(evolve._rules(4))
    captured = []

    def fake_chat_json(messages, model, parse):
        context = json.loads(messages[1]["content"])
        captured.append((context, model))
        children = [
            {"parent_id": "G0-B", "name": f"Child {index}",
             "reason": f"G0-B was stuck at the measured step; increase leg length {index}.",
             "changes": {"leg_length": .29 + .02 * index}}
            for index in range(3)
        ]
        return parse({"children": children}), [{"valid": True}]

    monkeypatch.setattr(evolve.llm, "chat_json", fake_chat_json)
    next_designs = evolve.next_generation({"generation": 0, "results": rows}, n=4)

    assert len(next_designs) == 4
    assert len(captured) == 1
    facts = captured[0][0]["FACTS"]
    assert {fact["id"] for fact in facts} == {row["id"] for row in rows}
    assert all(fact["outcome"] == "stuck" and fact["event"]["local_slope_deg"] == 12 for fact in facts)
    assert captured[0][1] == evolve.config.BUILDER_MODEL
    assert all(child["source"] == "model" for child in next_designs[1:])


@pytest.mark.parametrize(("kind", "outcome", "expected"), [
    ("rover", "stuck", {"wheel_radius", "wheel_width"}),
    ("walker", "stuck", {"leg_length", "stride_deg"}),
    ("rover", "flipped", {"chassis_width", "wheel_radius"}),
    ("walker", "flipped", {"chassis_width", "leg_length"}),
    ("rover", "slipped", {"wheels", "drive_speed"}),
    ("walker", "slipped", {"chassis_width", "gait_freq_hz"}),
    ("rover", "too slow", {"drive_speed", "wheel_radius"}),
    ("walker", "too slow", {"gait_freq_hz", "stride_deg"}),
])
def test_fallback_mutations_match_failure_class(kind, outcome, expected):
    design = next(row for row in evolve._rules(4) if row["robot"]["kind"] == kind)
    changes, _ = evolve._fallback_change(
        {**design, "outcome": outcome}, outcome, 0,
    )
    assert set(changes) == expected


def test_evolve_stops_immediately_when_any_design_reaches_goal(monkeypatch):
    from simgate import cost

    monkeypatch.setattr(evolve, "propose_generation0", lambda challenge, n: evolve._rules(n))
    def fake_verify(challenge, design):
        cost.add_runs(3)
        return {"reliable": True, "total": 3, "text": "passes 3/3 (95% CI 44–100%)"}

    budgets = []

    def fake_margin(challenge, design, failures, budget=6):
        budgets.append(budget)
        cost.add_runs(budget)
        return {
            "available": True, "text": "It would fail if the slope were about 4° steeper.", "runs": budget,
        }

    monkeypatch.setattr(evolve, "verify_winner", fake_verify)
    monkeypatch.setattr(evolve, "margin_search", fake_margin)
    calls = []

    def fake_race(challenge, designs, seed):
        cost.add_runs(len(designs))
        calls.append(designs)
        rows = _generation_results(designs)
        rows[0].update(reached_goal=True, outcome="reached", time_s=21, progress_percent=100)
        return {"results": rows}

    monkeypatch.setattr(evolve, "race_generation", fake_race)
    report = evolve.evolve(_challenge(), max_generations=4, n=4, seed=3)

    assert len(calls) == 1
    assert report["success"] is True
    assert len(report["generations"]) == 1
    assert report["family_tree"] == []
    assert report["report"].startswith("Goal reached by ")
    assert "verified" in report["report"]
    assert budgets == [6]
    assert report["cost"]["runs"] == 13
    assert report["cost"]["runs"] <= evolve.MAX_EVOLUTION_RUNS
    assert report["honesty"].endswith("Designs are simplified procedural robots.")


def test_verify_winner_rejects_one_of_three_and_reports_new_seeds(monkeypatch):
    from simgate import batch

    calls = {}

    def fake_batch(scene, variations, **kwargs):
        calls.update(scene=scene, variations=variations, kwargs=kwargs)
        return [
            {"id": f"run-{index}", "changes": variation,
             "summary": {
                 "goal_reached": index == 1, "outcome": "reached" if index == 1 else "stuck",
                 "start_distance_to_goal_m": 10, "closest_distance_m": 5,
             }}
            for index, variation in enumerate(variations, start=1)
        ]

    monkeypatch.setattr(batch, "run_batch", fake_batch)
    candidate = evolve._rules(4)[0]
    result = evolve.verify_winner(_challenge(), candidate)

    assert result["passed"] == 1
    assert result["total"] == 3
    assert result["reliable"] is False
    assert [row["progress_percent"] for row in result["results"]] == [100, 50, 50]
    assert result["wilson_95"][0] < result["wilson_95"][1]
    assert [row["seed"] for row in calls["variations"]] == [1, 2, 3]
    assert [row["perturb_seed"] for row in calls["variations"]] == [1, 2, 3]
    assert calls["kwargs"] == {"baseline": False, "record": False}


def test_margin_search_uses_at_most_six_finder_runs(monkeypatch):
    from simgate import finder

    class FakeRunner:
        def __init__(self, scene, robot_id, failure, budget):
            self.budget = budget
            self.evidence = []

        @property
        def left(self):
            return self.budget - len(self.evidence)

        def run(self, jobs):
            rows = []
            for setting, _ in jobs[:self.left]:
                row = {
                    **setting, "id": f"r{len(self.evidence) + 1}",
                    "broke": setting["value"] > 15, "failure": "fell",
                }
                self.evidence.append(row)
                rows.append(row)
            return rows

    monkeypatch.setattr(finder, "Runner", FakeRunner)
    candidate = evolve._rules(4)[0]
    failure = {
        "reached_goal": False, "outcome": "stuck",
        "event": {"local_slope_deg": 12, "step_m": 0.02},
    }

    result = evolve.margin_search(_challenge(), candidate, [failure])
    constrained = evolve.margin_search(_challenge(), candidate, [failure], budget=2)

    assert result["available"] is True
    assert result["knob"] == "slope_deg"
    assert result["runs"] == 6
    assert "steeper" in result["text"]
    assert len(result["evidence"]) == result["runs"]
    assert constrained["runs"] == 2


def test_evolve_continues_after_an_unreliable_apparent_winner(monkeypatch):
    monkeypatch.setattr(evolve, "propose_generation0", lambda challenge, n: evolve._rules(n))

    def fake_next_generation(results, n):
        children = evolve._rules(n)
        for child in children[1:]:
            child.update(
                parent_id="G0-A", parameter_changes={"mass_kg": {"from": 10, "to": 12}},
                reason="The verification runs stuck; add mass for stability.",
            )
        return children

    monkeypatch.setattr(evolve, "next_generation", fake_next_generation)
    verify_calls = []

    def fake_verify(challenge, design):
        verify_calls.append(design["id"])
        return {
            "reliable": False, "passed": 1, "total": 3, "text": "passes 1/3 (95% CI 6–79%)",
            "results": [
                {"reached_goal": False, "outcome": "stuck", "progress_percent": 45},
                {"reached_goal": True, "outcome": "reached", "progress_percent": 100},
                {"reached_goal": False, "outcome": "stuck", "progress_percent": 50},
            ],
        }

    monkeypatch.setattr(evolve, "verify_winner", fake_verify)
    race_calls = []

    def fake_race(challenge, designs, seed):
        race_calls.append(designs)
        rows = _generation_results(designs)
        if len(race_calls) == 1:
            rows[0].update(reached_goal=True, outcome="reached", progress_percent=100)
        return {"results": rows}

    monkeypatch.setattr(evolve, "race_generation", fake_race)
    report = evolve.evolve(_challenge(), max_generations=2, n=4, seed=3)

    assert len(race_calls) == 2
    assert verify_calls == ["G0-A"]
    assert report["success"] is False
    assert report["rejected_winners"][0]["id"] == "G0-A"
    assert "unreliable" in report["report"]


def test_honesty_line_uses_real_terrain_accuracy_metadata(monkeypatch):
    from simgate import terrain_data

    metadata = {
        "label": "Victoria Crater", "size_m": 80.0, "relief_m": 5.0,
        "typical_slope_deg": 9.5, "credit": "NASA", "body": "mars",
        "surface": "regolith", "accuracy_note": "native-scale elevation retained.",
    }
    monkeypatch.setattr(terrain_data, "load_preset", lambda preset: (None, metadata))
    challenge = _challenge().model_copy(update={
        "terrain": Terrain(kind="real", preset="victoria-opportunity", roughness=0),
    })

    line = evolve._honesty_line(challenge)

    assert line == (
        "Tested in simulation on real elevation data (accuracy: native-scale elevation retained). "
        "Designs are simplified procedural robots."
    )


def test_evolve_reports_closest_failure_and_links_family_tree(monkeypatch):
    monkeypatch.setattr(evolve.config, "MOCK_MODE", True)
    monkeypatch.setattr(evolve, "propose_generation0", lambda challenge, n: evolve._rules(n))
    generation = 0

    def fake_race(challenge, designs, seed):
        nonlocal generation
        rows = _generation_results(designs, progress=40, slope=12)
        if generation:
            for row in rows:
                row["progress_percent"] = 60
                row["event"]["local_slope_deg"] = 18
            child = next(row for row in rows if row["id"] == "G1-A")
            child["progress_percent"] = 78
            child["event"]["local_slope_deg"] = 24
        generation += 1
        from simgate import cost

        cost.add_runs(len(rows))
        return {"results": rows}

    monkeypatch.setattr(evolve, "race_generation", fake_race)
    report = evolve.evolve(_challenge(), max_generations=2, n=4, seed=3)

    assert report["success"] is False
    assert "closest: G1-A, 78% of the way, stuck" in report["report"]
    assert "24°" in report["report"]
    assert len(report["family_tree"]) == 3
    assert all(edge["parent_id"].startswith("G0-") for edge in report["family_tree"])
    assert all(edge["child_id"].startswith("G1-") and edge["result"] for edge in report["family_tree"])
    assert all(
        all(set(change) == {"from", "to"} for change in edge["changes"].values())
        for edge in report["family_tree"]
    )
    assert all(
        any(f"{key} " in edge["reason"] for key in edge["changes"])
        for edge in report["family_tree"]
    )
    assert report["cost"]["runs"] == 8
    assert report["cost"]["model_calls"] == 0


def test_evolution_enforces_the_sixteen_race_budget():
    with pytest.raises(ValueError, match="limited to 16"):
        evolve.evolve(_challenge(), max_generations=5, n=4, seed=0)
