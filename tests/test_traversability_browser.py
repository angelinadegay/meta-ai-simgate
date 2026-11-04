import json
from pathlib import Path

import pytest

from test_mars_base_browser import browser_page


def test_map_only_overlay_handles_absent_route(browser_page):
    result = browser_page.evaluate("""async () => {
        const {mapOverlay} = await import('/static/vendor/traversability.js');
        const report = {cell_size_m: .5, x_m: [0], y_m: [0], classes: [['amber']],
            comparison: {route: null}, spot_checks: []};
        const visual = mapOverlay(report, () => 0, () => {});
        return {meshes: visual.root.children.length, markers: visual.markers.length};
    }""")
    assert result == {"meshes": 1, "markers": 0}


@pytest.mark.parametrize("name", ["mars-rover", "apartment-rover", "apartment-walker", "apartment-rover-two-leg"])
def test_live_map_overlay_card_toggle_and_replay(browser_page, name):
    root = Path(__file__).resolve().parents[1]
    saved = root / "reports" / "traversability-live" / name / "replay.json"
    if not saved.exists():
        pytest.skip("Run scripts\\run_traversability_examples.py to produce live evidence first.")
    replay = json.loads(saved.read_text(encoding="utf-8"))
    context = browser_page.context.browser.new_context(viewport={"width": 1600, "height": 1050})
    page = context.new_page()
    errors, requests, sockets = [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def connect(socket):
        sockets.append(socket)
        socket.on_message(lambda message: requests.append(json.loads(message)))
        socket.send(json.dumps({"type": "spec", "spec": replay["spec"]}))
        socket.send(json.dumps({"type": "scene", "scene": replay["scene"], "code": replay["code"]}))
        socket.send(json.dumps({"type": "traversability", "report": replay["report"]}))
        socket.send(json.dumps({"type": "cost", "cost": replay["cost"]}))

    page.route_web_socket("**/ws", connect)
    page.goto(browser_page.url + "?visual-capture=1", wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    assert not errors, errors
    page.locator("#mapCard").wait_for(state="visible", timeout=10000)
    assert page.locator("#mapHeadline").inner_text() == (replay["report"].get("headline") or replay["report"]["comparison"]["text"])
    route_runs = page.locator("#mapRouteRuns button")   # the route runs are watchable without opening details
    assert route_runs.count() == len(replay["report"]["comparison"]["runs"]) and route_runs.first.is_visible()
    first_route = next(run for run in replay["report"]["comparison"]["runs"] if "straight" not in run["label"].lower())
    assert route_runs.first.inner_text().startswith("▶ Watch") and "safe route" in route_runs.first.inner_text()
    route_runs.first.click()
    page.wait_for_timeout(100)
    assert requests[-1] == {"type": "replay_map_run", "run_id": first_route["id"]}
    page.locator("#mapDetails > summary").click()   # everything else is under "Show details"
    assert "amber" in page.locator("#mapLegend").inner_text()
    assert "real places combine features" in page.locator("#mapHonesty").inner_text()
    assert replay["report"]["agreement_text"] == page.locator("#mapAgreement").inner_text()
    assert "model call" in page.locator("#mapCost").inner_text()
    screenshot = root / "tests" / "screenshots" / "traversability"
    screenshot.mkdir(exist_ok=True, parents=True)
    page.screenshot(path=str(screenshot / f"{name}-map-routes.png"))
    enabled = page.locator("#view canvas").screenshot()
    page.locator("#mapToggle").click()
    assert page.locator("#mapToggle").get_attribute("aria-pressed") == "false"
    disabled = page.locator("#view canvas").screenshot()
    assert enabled != disabled
    page.locator("#mapToggle").click()
    page.locator("#mapReplays button").first.click()
    page.wait_for_timeout(100)
    assert requests[-1] == {"type": "replay_map_run", "run_id": replay["report"]["spot_checks"][0]["id"]}
    assert not errors, errors
    context.close()
