from pathlib import Path


def test_hunt_round_batches_six_reasons_and_keeps_each_in_details():
    from simgate import finder

    combos = [
        {"values": {"slope_deg": 14.0 + index, "push_n": 50.0 + index},
         "reason": f"Measured slope {10 + index}° and push {40 + index} N may fail together."}
        for index in range(6)
    ]
    events = []

    class Runner:
        def run(self, jobs):
            return [{"id": f"hunt-{index}", "broke": False, "failure": None}
                    for index, _ in enumerate(jobs, 1)]

    context = {"robot_id": "r1", "base": {"slope_deg": 0.0, "push_n": 0.0}}
    results = finder._hunt_round(Runner(), combos, context, 1, events.append)
    proposal = next(event for event in events if event["agent"] == "Hunter" and event["tool"] == "propose")

    assert len([event for event in events if event["agent"] == "Hunter" and event["tool"] == "propose"]) == 1
    assert proposal["thought"] == "Hunter proposed 6 slope + push combinations"
    for index, combo in enumerate(combos, 1):
        assert f"Combination {index}:" in proposal["result"]
        assert combo["reason"] in proposal["result"]
    assert len(results) == 6


def test_browser_step_details_show_batched_hunter_reasons():
    page = (Path(__file__).resolve().parent.parent / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    assert 'entry.result ? `Result: ${entry.result}` : ""' in page
    assert 'detail.textContent = entries.map((entry, index) => [' in page