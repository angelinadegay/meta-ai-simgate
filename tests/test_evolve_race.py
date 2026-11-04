import asyncio
import json

import pytest

from simgate import evolve
from simgate.spec import Environment, Mission, Obstacle, Robot, RobotDesignConstraints, SceneSpec, Terrain


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
        assert len(scored["samples"]) == 7


def test_director_routes_victoria_robot_design_request_to_evolve():
    from simgate.director import Director

    prompt = "Design a robot that can climb out of Victoria Crater."
    plan = Director().plan(prompt)

    assert plan.workflow == "evolve"
    assert plan.world.preset == "victoria-opportunity"
    assert plan.world.terrain_kind == "real"
    assert plan.world.roughness == 0
    challenge = evolve.challenge_preset("victoria-rim-boulders")
    assert plan.evolution_challenge == "victoria-rim-boulders"
    assert plan.start.xy == challenge.mission.start_xy
    assert plan.goal.xy == challenge.mission.goal_xy
    assert len(plan.obstacles) == 1 and plan.obstacles[0].kind == "rock"
    assert plan.duration_s == 90
    assert plan.run_now


def test_victoria_design_constraints_clamp_generation_zero_and_reject_race_violations():
    limits = RobotDesignConstraints(
        max_mass_kg=12, max_size_m=(0.6, 0.5), max_power_proxy=60,
    )
    designs = evolve._validate_designs({"designs": [
        {"name": "Large rover", "reason": "A test rover.", "robot": {
            "kind": "rover", "wheels": 6, "wheel_radius": 0.4, "wheel_width": 0.2,
            "chassis_length": 1.2, "chassis_width": 0.9, "mass_kg": 40, "drive_speed": 12,
        }},
        {"name": "Large walker", "reason": "A test walker.", "robot": {
            "kind": "walker", "legs": 6, "leg_length": 0.4, "chassis_length": 1.2,
            "chassis_width": 0.9, "mass_kg": 40,
        }},
    ]}, 2, limits)

    for design in designs:
        robot = Robot.model_validate(design["robot"])
        assert not evolve._constraint_violations(robot, limits)
        assert robot.mass_kg <= 12
    with pytest.raises(ValueError, match="design constraints"):
        evolve._validate_race_designs(evolve._rules(4), limits)


def test_director_extracts_lander_mass_constraint_from_prompt():
    from simgate.director import Director

    plan = Director().plan("Design a robot under 12 kg that can climb out of Victoria Crater.")

    assert plan.workflow == "evolve"
    assert plan.design_constraints.max_mass_kg == 12
    assert plan.design_constraints.max_size_m == evolve.DEFAULT_DESIGN_CONSTRAINTS["max_size_m"]


def test_workflow_streams_live_family_tree_updates():
    from simgate.workflows import _stream_evolution

    class FakeDirector:
        _root_id = "root"

        def __init__(self):
            self.events = []

        async def emit(self, event):
            self.events.append(event)

        async def _emit_agent_step(self, agent, model, tool, parent_id, thought, result, status):
            self.events.append({"type": "agent", "agent": agent, "status": status})

    director = FakeDirector()
    tree = [{"parent_id": "G0-A", "child_id": "G1-A"}]
    designs = [{"id": "G1-A", "name": "Wide Rover"}]

    def job(on_event):
        on_event({"type": "tree", "family_tree": tree, "designs": designs})
        return {"report": "finished"}

    report = asyncio.run(_stream_evolution(job, director))

    assert report == {"report": "finished"}
    assert {"type": "evolution_tree", "family_tree": tree, "designs": designs} in director.events


def test_director_forwards_evolution_progress_events():
    from simgate.director import Director

    events = []

    async def capture(event):
        events.append(event)

    director = Director(capture)
    event = {"type": "evolution_tree", "family_tree": [], "designs": []}

    asyncio.run(director.emit(event))

    assert events == [event]


def test_evolve_cues_do_not_intercept_unrelated_prompts():
    from simgate.director import evolve_requested

    assert evolve_requested("design a robot for the rocky trail")
    assert evolve_requested("what robot could handle this place?")
    assert evolve_requested("invent a robot")
    assert evolve_requested("evolve")
    assert evolve_requested("best robot for this place")
    assert not evolve_requested("where can the rover go?")


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
    result = evolve.race_generation(_challenge(), designs, seed=19, record_replay=False)

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


def test_designer_sees_stowed_size_rule_and_parent_headroom(monkeypatch):
    monkeypatch.setattr(evolve.config, "MOCK_MODE", False)
    limits = RobotDesignConstraints(max_mass_kg=12, max_size_m=(0.6, 0.5))
    rows = _generation_results(evolve._rules(4, limits))
    captured = []

    def fake_chat_json(messages, model, parse):
        captured.append(json.loads(messages[1]["content"]))
        rover = next(row for row in rows if row["robot"]["kind"] == "rover")
        with pytest.raises(ValueError, match="stowed size .*chassis_length \\+ 2\\*wheel_radius"):
            parse({"children": [
                {"parent_id": rover["id"], "name": f"Big wheels {index}",
                 "reason": "Stuck on a rock; bigger wheels.", "changes": {"wheel_radius": 0.3}}
                for index in range(3)
            ]})
        return None, [{"valid": False}]

    monkeypatch.setattr(evolve.llm, "chat_json", fake_chat_json)
    evolve.next_generation({"generation": 0, "results": rows}, n=4, constraints=limits)

    context = captured[0]
    assert "chassis_length + 2*wheel_radius" in context["design_constraints"]["stowed_size_rule"]
    for fact in context["FACTS"]:
        stowed = fact["stowed"]
        assert stowed["stowed_length_m"] <= 0.6 + 1e-9 and stowed["stowed_width_m"] <= 0.5 + 1e-9
        assert stowed["mass_headroom_kg"] >= 0 and "width_headroom_m" in stowed


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

    def fake_margin(challenge, design, budget=6):
        budgets.append(budget)
        cost.add_runs(budget * 2)
        return {
            "available": True, "text": "Still climbs with steeper slopes; fails below friction 0.4.",
            "runs": budget * 2, "height_scale": {"runs": budget}, "friction": {"runs": budget},
        }

    monkeypatch.setattr(evolve, "verify_winner", fake_verify)
    monkeypatch.setattr(evolve, "terrain_margin_search", fake_margin)
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
    assert report["cost"]["runs"] == 19
    assert report["race_count"] == 4
    assert report["simulation_count"] == 19
    assert report["honesty"].endswith("Designs are simplified procedural robots.")


@pytest.mark.parametrize(("reliable", "expected_escalation"), [(False, False), (True, True)])
def test_evolve_escalates_only_after_verified_generation_zero_winner(
    monkeypatch, reliable, expected_escalation,
):
    from simgate import cost

    challenge = _challenge()
    challenge.obstacles = [Obstacle(kind="rock", height_m=0.2, count=12)]
    challenge.design_constraints = RobotDesignConstraints(max_mass_kg=12, max_size_m=(0.6, 0.5))
    monkeypatch.setattr(evolve.config, "MOCK_MODE", True)
    monkeypatch.setattr(
        evolve, "propose_generation0",
        lambda scene, n: evolve._rules(n, scene.design_constraints),
    )
    verify_calls = []
    monkeypatch.setattr(evolve, "verify_winner", lambda scene, design: (
        verify_calls.append((scene.obstacles[0].height_m, design["id"])) or {
            "reliable": reliable, "total": 5, "text": "passes 5/5",
            "results": ([] if reliable else [
                {"reached_goal": False, "outcome": "stuck", "progress_percent": 30},
            ]),
        }
    ))
    margin_calls = []

    def fake_margin(scene, design, budget=6):
        margin_calls.append((scene.obstacles[0].height_m, design["id"]))
        return {
            "available": True, "runs": 2, "text": "rock edge measured",
            "rock_size": {"available": True, "first_fail": 1.5, "last_ok": 1.4, "runs": 2},
        }

    monkeypatch.setattr(evolve, "terrain_margin_search", fake_margin)
    race_scenes = []

    def fake_race(scene, designs, seed):
        race_scenes.append(scene.model_copy(deep=True))
        cost.add_runs(len(designs))
        rows = _generation_results(designs)
        if len(race_scenes) == 1:
            rows[0].update(reached_goal=True, outcome="reached", time_s=20, progress_percent=100)
        return {"results": rows}

    monkeypatch.setattr(evolve, "race_generation", fake_race)
    report = evolve.evolve(challenge, max_generations=2, n=4, seed=3)

    assert len(race_scenes) == 2
    assert report["success"] is False
    assert bool(report["escalation"]) is expected_escalation
    assert len(margin_calls) == int(reliable)
    assert len(verify_calls) >= 1
    if expected_escalation:
        assert race_scenes[1].obstacles[0].height_m == pytest.approx(0.3)
        assert report["escalation"]["gen0_failed_at_rock_size_m"] == pytest.approx(0.3)
        assert "Gen 0 already climbs out." in report["report"]
        assert "no evolved design verified on the raised 0.30 m rocks" in report["report"]
    else:
        assert race_scenes[1].obstacles[0].height_m == pytest.approx(0.2)
        assert report["report"].count("G0-A was unreliable") <= 1


@pytest.mark.parametrize(("passes", "reliable"), [(1, False), (3, False), (4, True)])
def test_verify_winner_requires_four_of_five_and_reports_new_seeds(monkeypatch, passes, reliable):
    from simgate import batch

    calls = {}

    def fake_batch(scene, variations, **kwargs):
        calls.update(scene=scene, variations=variations, kwargs=kwargs)
        return [
            {"id": f"run-{index}", "changes": variation,
             "summary": {
                 "goal_reached": index <= passes, "outcome": "reached" if index <= passes else "stuck",
                 "start_distance_to_goal_m": 10, "closest_distance_m": 5,
             }}
            for index, variation in enumerate(variations, start=1)
        ]

    monkeypatch.setattr(batch, "run_batch", fake_batch)
    candidate = evolve._rules(4)[0]
    result = evolve.verify_winner(_challenge(), candidate)

    assert result["passed"] == passes
    assert result["total"] == 5
    assert result["reliable"] is reliable
    assert [row["progress_percent"] for row in result["results"]] == [
        100 if index <= passes else 50 for index in range(1, 6)
    ]
    assert result["wilson_95"][0] < result["wilson_95"][1]
    assert [row["seed"] for row in calls["variations"]] == [1, 2, 3, 4, 5]
    assert [row["perturb_seed"] for row in calls["variations"]] == [1, 2, 3, 4, 5]
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
            "reliable": False, "passed": 1, "total": 5, "text": "passes 1/5 (95% CI 6–79%)",
            "results": [
                {"reached_goal": False, "outcome": "stuck", "progress_percent": 45},
                {"reached_goal": True, "outcome": "reached", "progress_percent": 100},
                {"reached_goal": False, "outcome": "stuck", "progress_percent": 50},
                {"reached_goal": False, "outcome": "stuck", "progress_percent": 50},
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


def test_evolve_stops_after_generation_one_when_three_designs_reach(monkeypatch):
    monkeypatch.setattr(evolve, "propose_generation0", lambda challenge, n: evolve._rules(n))

    def fake_next_generation(results, n):
        children = evolve._rules(n)
        for child in children:
            child["id"] = child["id"].replace("G0-", "G1-")
        for child in children[1:]:
            child.update(parent_id="G0-A", parameter_changes={"mass_kg": {"from": 10, "to": 12}},
                         reason="Add mass for traction.")
        return children

    monkeypatch.setattr(evolve, "next_generation", fake_next_generation)
    passes = {"G1-A": 4, "G1-B": 5, "G1-C": 5}
    verify_calls = []

    def fake_verify(challenge, design):
        verify_calls.append(design["id"])
        passed = passes[design["id"]]
        return {"reliable": passed >= 4, "passed": passed, "total": 5,
                "text": f"passes {passed}/5", "results": []}

    monkeypatch.setattr(evolve, "verify_winner", fake_verify)
    monkeypatch.setattr(evolve, "terrain_margin_search", lambda challenge, design, budget=6: {
        "available": True, "runs": 0, "text": "margin measured",
    })
    race_calls = []

    def fake_race(challenge, designs, seed):
        race_calls.append(designs)
        rows = _generation_results(designs)
        if len(race_calls) == 2:
            for row, time_s in zip(rows[:3], (18, 20, 25)):
                row.update(reached_goal=True, outcome="reached", time_s=time_s, progress_percent=100)
        return {"results": rows}

    monkeypatch.setattr(evolve, "race_generation", fake_race)
    report = evolve.evolve(_challenge(), max_generations=5, n=4, seed=3)

    assert len(race_calls) == 2
    assert report["success"] is True
    assert len(report["generations"]) == 2
    assert verify_calls == ["G1-A", "G1-B"]
    assert report["winner"]["id"] == "G1-B"
    assert report["escalation"] is None
    assert report["generations"][1]["stop_check"]["stopped"] is True


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


def test_evolution_enforces_twenty_candidate_races_and_five_generations():
    with pytest.raises(ValueError, match="limited to 20"):
        evolve.evolve(_challenge(), max_generations=5, n=5, seed=0)
    with pytest.raises(ValueError, match="from 1 to 5"):
        evolve.evolve(_challenge(), max_generations=6, n=4, seed=0)


def test_hadley_and_meteor_presets_use_measured_steepest_segments():
    for preset in ("hadley-climb-out", "meteor-crater-climb"):
        challenge = evolve.challenge_preset(preset)
        segment = evolve.steepest_climb_points(challenge)
        assert challenge.mission.start_xy == segment["start_xy"]
        assert challenge.mission.goal_xy == segment["goal_xy"]
        assert challenge.robots[0].start_xy == segment["start_xy"]
        assert challenge.robots[0].goal_xy == segment["goal_xy"]
        assert segment["rise_m"] > 0
        assert segment["slope_deg"] >= (15 if preset == "hadley-climb-out" else 10)


def test_calibration_uses_no_winner_and_twenty_to_eighty_percent_progress():
    assert evolve.calibration_quality([
        {"progress_percent": 20, "reached_goal": False},
        {"progress_percent": 80, "reached_goal": False},
    ])["qualifies"] is True
    assert evolve.calibration_quality([
        {"progress_percent": 80, "reached_goal": True},
    ])["qualifies"] is False
    assert evolve.calibration_quality([
        {"progress_percent": 19.9, "reached_goal": False},
    ])["qualifies"] is False
    assert evolve.calibration_quality([
        {"progress_percent": 80.1, "reached_goal": False},
    ])["qualifies"] is False


def test_feasibility_reference_designs_are_the_biggest_legal_designs():
    limits = {"max_mass_kg": 12.0, "max_size_m": (0.8, 0.5)}
    rover, walker = evolve.feasibility_reference_designs(limits)
    assert rover["robot"]["wheel_radius"] == pytest.approx(0.2)
    assert walker["robot"]["leg_length"] == evolve.DESIGN_SPACE["walker"]["leg_length"][1]
    for design in (rover, walker):
        assert evolve._constraint_violations(Robot.model_validate(design["robot"]), limits) == []
    tight, _ = evolve.feasibility_reference_designs({"max_mass_kg": 12.0, "max_size_m": (0.6, 0.5)})
    assert tight["robot"]["wheel_radius"] == pytest.approx(0.1)


def test_calibration_requires_a_legal_design_to_climb_out(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_challenge.py"
    spec = importlib.util.spec_from_file_location("calibrate_challenge", path)
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
    tunings = [{"rock_size_m": 0.25, "rock_count": 24, "friction": 0.55,
                "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.6, 0.5]}},
               {"rock_size_m": 0.15, "rock_count": 12, "friction": 0.65,
                "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.8, 0.5]}}]

    def fake_run(seed, tuning, index, mode, designs=None):
        return {"trial": index, "generation_zero": mode, **tuning, "duration_s": 90.0,
                **evolve.calibration_quality([{"progress_percent": 44.0, "reached_goal": False}]),
                "outcomes": [{}] * 4}

    feasible = {0.6: False, 0.8: True}
    monkeypatch.setattr(calibration, "_run", fake_run)
    monkeypatch.setattr(calibration, "feasibility", lambda seed, tuning: {
        "feasible": feasible[tuning["design_constraints"]["max_size_m"][0]], "reason": "fake",
        "simulations": 2})
    result = calibration.calibrate(tunings=tunings, include_model=False)
    first, second = result["rows"]
    assert first["gen0_qualifies"] and not first["qualifies"]
    assert second["qualifies"] and result["default_row"] is second and result["qualified"]
    assert result["simulations"] == 4 + 4 + 2 + 2


def test_real_terrain_height_scale_changes_measured_slopes_as_expected():
    import numpy as np

    from simgate.backends.mujoco_backend import make_heights

    challenge = evolve.challenge_preset("hadley-climb-out")
    heights, _ = make_heights(challenge)
    scaled = challenge.model_copy(deep=True)
    scaled.terrain.height_scale = 1.2
    scaled_heights, _ = make_heights(scaled)
    spacing = challenge.terrain.size_m / (heights.shape[0] - 1)
    base_dy, base_dx = np.gradient(heights, spacing, spacing)
    scaled_dy, scaled_dx = np.gradient(scaled_heights, spacing, spacing)
    base_p95 = np.percentile(np.degrees(np.arctan(np.hypot(base_dx, base_dy))), 95)
    scaled_p95 = np.percentile(np.degrees(np.arctan(np.hypot(scaled_dx, scaled_dy))), 95)
    expected = np.degrees(np.arctan(1.2 * np.tan(np.radians(base_p95))))
    assert scaled_p95 > base_p95
    assert scaled_p95 == pytest.approx(expected, abs=1e-4)


def test_terrain_margin_search_uses_six_runs_for_each_finder_search(monkeypatch):
    from simgate import finder

    calls = []

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
                knob, value = setting["knob"], setting["value"]
                calls.append((knob, value))
                broke = value > 1.5 if knob == "height_scale" else value < 0.4
                row = {**setting, "id": f"r{len(self.evidence) + 1}",
                       "broke": broke, "failure": "missed the goal"}
                self.evidence.append(row)
                rows.append(row)
            return rows

    monkeypatch.setattr(finder, "Runner", FakeRunner)
    challenge = evolve.challenge_preset("hadley-climb-out")
    result = evolve.terrain_margin_search(challenge, evolve._rules(4)[0])
    by_knob = {knob: [value for name, value in calls if name == knob]
               for knob in ("height_scale", "friction")}

    assert result["height_scale"]["available"] is True
    assert result["friction"]["available"] is True
    assert len(by_knob["height_scale"]) <= 6
    assert len(by_knob["friction"]) <= 6
    assert result["runs"] <= 12
    assert "P95 slopes" in result["height_scale"]["text"]
    assert "fails below" in result["friction"]["text"]


def test_generation_replay_combines_recorded_robot_poses_on_one_scene():
    import base64
    import zlib

    def packed_ghost(start_x):
        frames = [
            [0.0, [[start_x, 0, 0.2, 1, 0, 0, 0], [start_x, 0.4, 0.1, 1, 0, 0, 0]]],
            [1.0, [[start_x + 1, 0, 0.2, 1, 0, 0, 0], [start_x + 1, 0.4, 0.1, 1, 0, 0, 0]]],
        ]
        return {
            "geom_ids": [1, 2],
            "data": base64.b64encode(zlib.compress(json.dumps(frames).encode())).decode(),
        }

    scene = {
        "kind": "mujoco", "world": {"body": "mars", "surface": "regolith"},
        "geoms": [
            {"id": 0, "name": "ground", "type": "hfield", "hfield": {"data": [0.0]}},
            {"id": 1, "name": "chassis", "type": "box", "rgba": [0.5, 0.5, 0.5, 1]},
            {"id": 2, "name": "wheel", "type": "cylinder", "rgba": [0.2, 0.2, 0.2, 1]},
        ],
        "poses": [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0.2, 1, 0, 0, 0], [0, 0.4, 0.1, 1, 0, 0, 0]],
    }
    results = [
        {"id": "G0-A", "name": "Red Rover", "color": "#ef829f", "reached_goal": False,
         "outcome": "stuck", "event": {"x": 1, "y": 2}},
        {"id": "G0-B", "name": "Green Walker", "color": "#40d8a0", "reached_goal": True,
         "outcome": "reached", "event": {"x": 3, "y": 4}},
    ]
    runs = {
        "G0-A": {"race_scene": scene, "ghost": packed_ghost(0), "race_follow_geom": 1},
        "G0-B": {"race_scene": scene, "ghost": packed_ghost(5), "race_follow_geom": 1},
    }

    replay = evolve._generation_replay(runs, results, 1.0)
    track = replay["track"]
    frames = json.loads(zlib.decompress(base64.b64decode(track["frames"])))

    assert len(replay["scene"]["geoms"]) == 1
    assert len(track["geoms"]) == 4
    assert len(track["robots"]) == 2
    assert len(track["failure_markers"]) == 1
    assert track["failure_markers"][0]["label"].startswith("✕ stuck")
    assert track["frame_count"] == len(frames)
    assert len(frames[0][1]) == 5


def test_director_uses_hadley_steep_climb_for_explicit_request():
    from simgate.director import Director

    plan = Director().plan("Design a robot that can climb out of Hadley Rille.")
    challenge = evolve.challenge_preset("hadley-climb-out")

    assert plan.workflow == "evolve"
    assert plan.world.preset == "apollo15-hadley-rille"
    assert plan.start.xy == challenge.mission.start_xy
    assert plan.goal.xy == challenge.mission.goal_xy


def test_director_uses_calibrated_challenge_as_demo_default():
    from simgate.director import Director

    plan = Director().plan("Design a robot for the calibrated default challenge.")

    assert plan.workflow == "evolve"
    assert plan.world.preset == "victoria-opportunity"
    assert plan.evolution_challenge == "victoria-rim-boulders"
    challenge = evolve.challenge_preset("victoria-rim-boulders")
    assert plan.start.xy == challenge.mission.start_xy
    assert plan.goal.xy == challenge.mission.goal_xy
    assert plan.duration_s >= 2 * (
        __import__("math").dist(plan.start.xy, plan.goal.xy) / 3.0
    )


def test_victoria_rim_boulders_uses_steep_climb_rocks_friction_and_generous_time():
    challenge = evolve.challenge_preset("victoria-rim-boulders")
    segment = evolve.steepest_climb_points(
        challenge, start_height_percentile=30, goal_height_percentile=80,
    )

    assert challenge.terrain.preset == "victoria-opportunity"
    assert challenge.mission.start_xy == segment["start_xy"]
    assert challenge.mission.goal_xy == segment["goal_xy"]
    assert challenge.obstacles[0].kind == "rock"
    assert challenge.obstacles[0].placement == "path"
    assert 0.15 <= challenge.obstacles[0].height_m <= 0.35
    assert 12 <= challenge.obstacles[0].count <= 30
    assert 0.45 <= challenge.environment.friction <= 0.65
    assert challenge.duration_s >= 2 * (
        __import__("math").dist(challenge.mission.start_xy, challenge.mission.goal_xy) / 3.0
    )
    assert challenge.duration_s == pytest.approx(90.0)
    assert any("friction is set to" in assumption for assumption in challenge.assumptions)


def test_victoria_boulder_calibration_rejects_out_of_range_physical_settings():
    with pytest.raises(ValueError, match="Rock size"):
        evolve.challenge_preset("victoria-rim-boulders", rock_size_m=0.36)
    with pytest.raises(ValueError, match="density"):
        evolve.challenge_preset("victoria-rim-boulders", rock_count=31)
    with pytest.raises(ValueError, match="friction"):
        evolve.challenge_preset("victoria-rim-boulders", friction=0.7)
    with pytest.raises(ValueError, match="offset"):
        evolve.challenge_preset("victoria-rim-boulders", start_offset_m=6.1)


def test_failure_event_names_nearest_rock_and_reason_uses_its_size():
    design = evolve._rules(4)[0]
    run = {
        "summary": {
            "goal_reached": False, "start_distance_to_goal_m": 10, "closest_distance_m": 5,
            "race_telemetry": {"samples": [
                {"time_s": 0, "x": 1.0, "y": 2.0, "local_slope_deg": 17, "step_m": 0.2},
                {"time_s": 5, "x": 1.0, "y": 2.0, "local_slope_deg": 17, "step_m": 0.2},
            ]},
        },
        "race_scene": {
            "course": [{"name": "obs0_rock", "kind": "rock", "height_m": 0.3,
                        "geoms": ["obs0_rock_0"]}],
            "geoms": [{"name": "obs0_rock_0"}],
            "poses": [[1.1, 2.0, 0.1]],
        },
    }

    result = evolve._score_race_result(design, run, 90)

    assert result["outcome"] == "stuck"
    assert result["event"]["rock"] == {"name": "Rock 1", "size_m": 0.3, "distance_m": 0.1}
    assert "0.30 m rock" in evolve._failure_fact(result)


def test_rock_margin_search_uses_finder_for_larger_rocks_and_friction(monkeypatch):
    from simgate import finder

    calls = []

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
                knob, value = setting["knob"], setting["value"]
                calls.append((knob, value))
                broke = value > 1.5 if knob == "obstacle_scale" else value < 0.4
                row = {**setting, "id": f"r{len(self.evidence) + 1}",
                       "broke": broke, "failure": "stuck at rock"}
                self.evidence.append(row)
                rows.append(row)
            return rows

    monkeypatch.setattr(finder, "Runner", FakeRunner)
    challenge = evolve.challenge_preset("victoria-rim-boulders")
    result = evolve.terrain_margin_search(challenge, evolve._rules(4)[0])

    assert result["rock_size"]["available"]
    assert result["friction"]["available"]
    assert "rock_size" in result and "height_scale" not in result
    assert len([row for row in calls if row[0] == "obstacle_scale"]) <= 6
    assert len([row for row in calls if row[0] == "friction"]) <= 6
    assert "rocks" in result["rock_size"]["text"]
    assert "friction" in result["friction"]["text"]
