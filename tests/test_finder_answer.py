import asyncio
from pathlib import Path

from simgate.plan import Plan
from simgate.spec import Robot, SceneSpec, Terrain


def finder_report():
    slope = {
        "knob": "slope_deg", "name": "slope", "baseline": 0.0, "last_ok": 5.6,
        "first_fail": 7.5, "failure": "fell", "never_broke": False,
        "shown": {"baseline": "0.0°", "last_ok": "5.6°", "first_fail": "7.5°",
                   "harsh": "30.0°", "step": "±1.0°"},
        "confirm": {
            "ok_at": {"passed": 3, "total": 3},
            "fails_at": {"label": "reliable fail", "passed": 0, "total": 3},
        },
    }
    push = {
        "knob": "push_n", "name": "sideways push", "baseline": 0.0, "last_ok": 66.0,
        "first_fail": 70.0, "failure": "fell", "never_broke": False,
        "shown": {"baseline": "0 N", "last_ok": "66 N", "first_fail": "70 N",
                   "harsh": "80 N", "step": "±5 N"},
        "confirm": {
            "ok_at": {"passed": 1, "total": 3},
            "fails_at": {"label": "reliable fail", "passed": 0, "total": 3},
            "step_down": {"status": "tested", "value": 63.0, "shown": "63 N", "passed": 3, "total": 3},
        },
    }
    surprise = {
        "kind": "combined", "text": "7.0° slope and 63 N sideways push", "failure": "fell",
        "distance": 0.1, "values": {"slope_deg": 7.0, "push_n": 63.0},
        "surprising": True, "confirm": {"label": "reliable fail"},
    }
    return {
        "type": "envelope", "robot_id": "r1", "robot_label": "G1",
        "failure": "safety", "runs": 40, "limits": [slope, push],
        "failures": [surprise], "dropped": [], "unconfirmed": [], "map": None,
        "honesty": "40 runs, limits accurate to one search step; simulation only. Confirmation: 1 failure kept on 3 seeds.",
        "explanation": "Model interpretation must not replace the code-built answer.",
    }


def g1_scene():
    return SceneSpec(
        domain="robot_terrain",
        terrain=Terrain(kind="flat", roughness=0),
        robot=Robot(kind="library", robot_id="g1", policy_id="g1_walking"),
    ).fill_defaults()


def test_finder_answer_is_code_built_with_surprise_and_separate_honesty():
    from simgate.finder import answer_headline, describe, surprising_failure_line

    report = finder_report()
    headline = answer_headline(report)
    surprise = surprising_failure_line(report)

    assert headline == (
        "G1: reliably OK on slopes up to 5.6°, falls at 7.5°; "
        "sideways push OK up to 63 N (unreliable above), falls at 70 N."
    )
    assert surprise == (
        "Most surprising confirmed failure: G1 fell when 7.0° slope and 63 N sideways push were combined."
    )
    report["headline"] = headline
    report["surprising_failure_line"] = surprise
    report["honesty"] = "40 runs. Confirmation: 1 failure kept on 3 seeds."
    report["explanation"] = "Model interpretation must not replace the code-built answer."
    assert describe(report) == f"{headline} {surprise}"
    assert "honesty" not in describe(report).lower()
    assert "model interpretation" not in describe(report).lower()


def test_session_finder_question_uses_code_answer_and_small_honesty_note(monkeypatch):
    from simgate import config, finder
    from simgate.orchestrator import Session

    monkeypatch.setattr(config, "MOCK_MODE", True)
    report = finder_report()
    report["headline"] = finder.answer_headline(report)
    report["surprising_failure_line"] = finder.surprising_failure_line(report)
    monkeypatch.setattr(finder, "find_failures", lambda *_args, **_kwargs: report)

    async def flow():
        events = []

        async def emit(event):
            events.append(event)

        session = Session(emit)
        session.spec = g1_scene()
        plan = Plan(workflow="find_failures", summary="Where it breaks", robot_id="r1")
        await session.make("find where it breaks", editing=True, plan=plan)
        return events

    events = asyncio.run(flow())
    answer = next(event for event in events if event["type"] == "question")
    assert answer["message"] == f"{report['headline']} {report['surprising_failure_line']}"
    assert answer["note"] == report["honesty"]


def test_finder_insight_uses_limits_as_headline_and_honesty_as_small_text():
    root = Path(__file__).resolve().parent.parent
    page = (root / "static" / "vendor" / "index.html").read_text(encoding="utf-8")
    envelope_case = page.split('case "envelope": {', 1)[1].split('case "redesign"', 1)[0]
    assert 'addInsight(ev.report.headline || "Operating limits found"' in envelope_case
    assert 'ev.report.surprising_failure_line || ""' in envelope_case
    assert 'textEl("span", ev.report.honesty, "cost-note")' in envelope_case
    assert '$("envelopeHonesty").textContent = r.honesty || ""' in page
