from simgate.redesign import DesignSpace, _message
from simgate.spec import Robot


def _report(*, tried, diagnosis, changes, closeness=0.4, result="still missed the goal", decisions=None):
    return {
        "winner": None,
        "fragile": None,
        "tried_count": tried,
        "rounds": [{"candidates": [{"changes": changes, "closeness": closeness, "result": result}]}],
        "decisions": decisions or [],
        "diagnosis": {"conclusion": diagnosis},
        "message": "",
    }


def test_winner_message_includes_the_diagnosis_and_early_stop_reason():
    space = DesignSpace(Robot(kind="library", robot_id="g1", policy_id="g1_walking"), None)
    report = {
        "winner": {"description": "walking speed 0.4 m/s", "robustness": {"passed": 3, "total": 3}},
        "fragile": None,
        "tried_count": 4,
        "diagnosis": {"conclusion": "A steep route was confirmed as the cause."},
    }

    message = _message(report, space)

    assert "Diagnosis used: A steep route was confirmed as the cause." in message
    assert "only walking speed can change for this trained policy" in message
    assert "command_speed_m_s" not in message


def test_bound_message_names_only_a_limit_reached_by_a_tried_candidate():
    space = DesignSpace(Robot(kind="rover"), None)
    reached = _report(tried=2, diagnosis="The step was confirmed as the cause.",
                      changes={"wheel_radius": 0.35})

    message = _message(reached, space)

    assert "wheel radius tried up to 0.35 m; the maximum allowed is 0.35 m" in message
    assert "Diagnosis used: The step was confirmed as the cause." in message

    not_reached = _report(tried=2, diagnosis="The step was confirmed as the cause.",
                          changes={"wheel_radius": 0.34})
    message = _message(not_reached, space)

    assert "No tried design reached a permitted setting limit." in message
    assert "maximum allowed" not in message


def test_stopped_search_explains_why_it_used_less_than_the_candidate_budget():
    space = DesignSpace(Robot(kind="library", robot_id="g1", policy_id="g1_walking"), None)
    report = _report(tried=1, diagnosis="A steep route was confirmed as the cause.",
                     changes={"command_speed_m_s": 0.3},
                     decisions=[{"decision": "stop", "reason": "No further candidate was allowed."}])

    message = _message(report, space)

    assert "No design passed after 1 candidate designs." in message
    assert "only walking speed can change for this trained policy" in message
    assert "A steep route was confirmed as the cause." in message
    assert "command_speed_m_s" not in message
