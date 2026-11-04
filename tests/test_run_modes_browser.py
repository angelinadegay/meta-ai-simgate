"""The header's mode line and the home page's demo runs, in the real page."""
import json
from pathlib import Path

from test_mars_base_browser import browser_page  # noqa: F401  (the served-page fixture)

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "tests" / "screenshots" / "run-modes"


def _page(browser_page, hello):
    """The page with a server that greets it with ``hello`` (the real server for everything else)."""
    context = browser_page.context.browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()
    errors, sent = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def connect(socket):
        server = socket.connect_to_server()
        server.on_message(lambda message: socket.send(
            json.dumps({**json.loads(message), **hello}) if json.loads(message).get("type") == "hello" else message))
        socket.on_message(lambda message: sent.append(json.loads(message)) or server.send(message))

    page.route_web_socket("**/ws", connect)
    page.goto(browser_page.url, wait_until="domcontentloaded")
    page.locator("#demoRuns").wait_for(state="visible", timeout=15000)
    return context, page, errors, sent


def test_header_shows_each_mode_and_the_demo_runs_replay(browser_page):
    SHOTS.mkdir(parents=True, exist_ok=True)
    for name, hello in (("mock", {"mock_mode": True, "mode": "Mock mode: rules, no model",
                                  "mode_status": "Key missing/invalid: LLM_API_KEY is not set. Running in mock mode."}),
                        ("model", {"mock_mode": False, "mode": "Meta models: muse-spark-1.3 · muse-wonder-1.0",
                                   "mode_status": "Connected to Meta Model API (muse-spark-1.3, muse-wonder-1.0)"})):
        context, page, errors, sent = _page(browser_page, hello)
        assert page.locator("#mode").inner_text() == hello["mode"]
        assert page.locator("#mode").get_attribute("title") == hello["mode_status"]
        assert ("Key missing/invalid" in page.locator("#log").inner_text()) == (name == "mock")
        assert "Demo runs: replay instantly (no key needed)" in page.locator("#demoRuns").inner_text()
        assert page.locator("#demoRunList button").all_inner_texts() == [
            "Crater evolution", "Gusev safe route", "Apartment: kitchen to bedroom"]
        page.locator(".top").screenshot(path=str(SHOTS / f"header-{name}.png"))
        if name == "mock":   # a demo replays in mock mode: the map card from the recording, no simulation
            page.get_by_role("button", name="Apartment: kitchen to bedroom").click()
            page.locator("#mapCard").wait_for(state="visible", timeout=15000)
            assert page.locator("#mapHeadline").inner_text().startswith("The rover can reach the bedroom via the ramp")
            assert {"type": "play_demo", "demo_id": "apartment-traversability"} in sent
            page.wait_for_timeout(500)
            assert "Replay · 0 runs · 0 model calls" in page.locator("#log").inner_text()
            page.screenshot(path=str(SHOTS / "demo-apartment-mock.png"))
        with page.expect_response("**/demos/crater-evolution") as answer:   # the evolution replays in the browser
            page.locator("#homeBrand").click()
            page.get_by_role("button", name="Crater evolution").click()
        assert answer.value.ok
        page.locator("#evolutionCard").wait_for(state="visible", timeout=20000)
        assert "no simulations or model calls were used" in page.locator("#log").inner_text()
        assert not errors, errors
        context.close()


def test_the_mode_line_stays_on_show_on_a_phone(browser_page):
    context = browser_page.context.browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto(browser_page.url, wait_until="domcontentloaded")
    page.locator("#demoRuns").wait_for(state="visible", timeout=15000)
    assert page.locator("#mode").is_visible() and page.locator("#mode").inner_text().startswith(("Mock mode", "Meta models"))
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.locator(".top").screenshot(path=str(SHOTS / "header-phone.png"))
    context.close()
