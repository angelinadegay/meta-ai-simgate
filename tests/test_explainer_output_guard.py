from simgate.spec import Mission, Robot, SceneSpec, Terrain


def scene_and_metrics():
    spec = SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="hills", roughness=0.3),
        robot=Robot(kind="rover"),
        mission=Mission(goal_place="east"),
    ).fill_defaults()
    metrics = {
        "simulator": "MuJoCo",
        "goal_reached": True,
        "time_to_goal_s": 4.0,
        "distance_m": 5.0,
        "max_tilt_deg": 10.3,
        "max_slope_on_path_deg": 10.3,
        "robot": spec.robot.model_dump(),
    }
    return spec, metrics


def test_explainer_facts_are_values_and_output_strips_labels_knobs_and_tips():
    from simgate.facts import (explainer_output_problems, explainer_values_block, facts,
                               strip_explainer_artifacts)

    spec, metrics = scene_and_metrics()
    prompt_facts = explainer_values_block(spec, metrics)
    assert '"terrain":"hills"' in prompt_facts
    assert '"maximum_path_angle_degrees":10.3' in prompt_facts
    assert "steepest slope crossed:" not in prompt_facts
    assert "gait_freq_hz" not in prompt_facts and "drive_speed" not in prompt_facts

    leaked = ("The rover reached the goal. The steepest slope crossed: 10.3°. "
              "Set gait_freq_hz to 1.5. Try typing 'make it rougher'.")
    problems = explainer_output_problems(leaked, facts(spec, metrics))
    assert any("FACTS label" in issue for issue in problems)
    assert any("field or setting name" in issue for issue in problems)
    assert any("suggests a next action" in issue for issue in problems)
    assert strip_explainer_artifacts(leaked, facts(spec, metrics)) == "The rover reached the goal."


def test_stress_explainer_conditions_hide_raw_robot_setting_names():
    from simgate.agents import _explainer_conditions

    conditions = _explainer_conditions({
        "robot": {"gait_freq_hz": 1.5, "stride_deg": 10},
        "robot_id": "r2",
        "slope_deg": 12.0,
    })

    assert conditions == [
        {"condition": "gait frequency", "value": 1.5},
        {"condition": "stride angle", "value": 10},
        {"condition": "slope", "value": 12.0},
    ]


def test_repeated_outcome_retries_once_then_uses_sanitized_rules_text(monkeypatch):
    from simgate import agents, config, llm

    spec, metrics = scene_and_metrics()
    calls = []
    responses = iter([
        "The rover reached the goal in 4 s. It reached the goal again.",
        "The rover reached the goal in 4 s. It reached the goal again.",
    ])

    def fake_chat(messages, *args, **kwargs):
        calls.append(messages)
        return next(responses)

    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(agents, "_mock_explain",
                        lambda _spec, _metrics: 'The rover reached the goal in 4 s. Try: "use six wheels".')

    result = agents.explain(spec, metrics)

    assert len(calls) == 2
    assert "repeats the same robot outcome" in calls[1][-1]["content"]
    assert result == "The rover reached the goal in 4 s."
    assert result.count("reached the goal") == 1
    prompt = "\n".join(message["content"] for call in calls for message in call)
    assert "gait_freq_hz" not in prompt and "drive_speed" not in prompt
    assert "Try typing" not in prompt
