import asyncio
from pathlib import Path
from types import SimpleNamespace


def browser_source() -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / "static" / "vendor" / "index.html").read_text(encoding="utf-8")


def test_director_fallback_reason_stays_in_debug_details(monkeypatch):
    from simgate import config, workflows
    from simgate.director import Director

    monkeypatch.setattr(config, "MOCK_MODE", True)
    director = Director()
    director.used_rule_planner = True
    director.plan = lambda *_args, **_kwargs: SimpleNamespace(
        workflow="build", summary="Build the requested scene.", then=None,
    )
    world = SimpleNamespace(spec=SimpleNamespace(assumptions=[]))

    async def execute(*_args, **_kwargs):
        return world, "The scene is ready."

    events = []

    async def capture_step(event):
        events.append(event)

    monkeypatch.setattr(workflows, "execute", execute)
    director.step_callback = capture_step
    message = asyncio.run(director.execute_plan("Build a rover"))[1]

    assert message == "The scene is ready."
    assert "planner" not in message.lower()
    assert any("used the rules planner" in event["result"] for event in events)


def test_stage_statuses_and_check_summaries_are_plain_language():
    page = browser_source()
    for status in (
        'setStatus("Analysing", "finding")',
        'setStatus("Running", "running")',
        'setStatus("Ready")',
        'setStatus("Needs your input", "input")',
        'setStatus("Failed checks", "failed")',
    ):
        assert status in page

    failure_summary = page.split("function checkFailureSummary", 1)[1].split("function updateSceneContext", 1)[0]
    checks_renderer = page.split("function renderChecks", 1)[1].split("async function decodeGhost", 1)[0]
    assert "first.detail" not in failure_summary
    assert "Open Show details for the measurements." in failure_summary
    assert 'textContent = c.detail' in checks_renderer


def test_repeated_steps_collapse_and_show_details_opens_every_occurrence():
    page = browser_source()
    assert 'state.status === "done" ? "done"' in page
    assert 'entries.length > 1 ? ` ×${entries.length}` : ""' in page
    assert 'entries.map((entry, index) => [' in page
    assert '`Occurrence ${index + 1}`' in page
    assert 'entry.args && Object.keys(entry.args).length' in page

    show_panel = page.split("function showPanel(", 1)[1].split("function hidePanel", 1)[0]
    assert "activeStepDetails.open = true" in show_panel
    assert 'activeStepList.querySelectorAll(".step-row")' in show_panel
    assert "step.open = true" in show_panel