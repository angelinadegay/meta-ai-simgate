import json
from pathlib import Path

from test_mars_base_browser import browser_page
from test_structure_actions import action_scene

from simgate.backends.mujoco_backend import MuJoCoSimulator


def test_structure_result_card_and_typed_edit_dispatch(browser_page):
    spec = action_scene("habitat")
    build = MuJoCoSimulator().build(spec)
    record = build.scene["structures"][0]
    assert not record["structural_check"]["passed"]
    messages = []
    routes = []
    context = browser_page.context.browser.new_context(viewport={"width": 1100, "height": 900})
    page = context.new_page()

    def connect(route):
        routes.append(route)
        route.on_message(lambda data: messages.append(json.loads(data)))
        route.send(json.dumps({"type": "spec", "spec": spec.model_dump(mode="json")}))
        route.send(json.dumps({"type": "scene", "scene": build.scene, "code": build.code}))

    page.route_web_socket("**/ws", connect)
    page.goto(browser_page.url)
    page.locator("#detailsToggle").click()
    card = page.locator("#structureCards article")
    card.wait_for(state="visible")
    text = card.inner_text()
    assert "Fail" in text and "% utilisation" in text and "Governing:" in text
    assert "Member section: SHS150x150x8" in text
    assert "visual model isn't analysed" in text
    assert "Visual model: Procedural habitat" in text
    assert "reduced radiation shielding" in card.get_by_role("button", name="Thinner shielding").get_attribute("title")
    card.get_by_role("button", name="Use stronger members").click()
    page.wait_for_function("document.querySelector('#log').textContent.includes('Use stronger members')")
    assert messages == [{"type": "structure_action", "operation": "stronger_members", "structure_index": 0}]
    folder = Path(__file__).parent / "screenshots" / "structure-actions"
    folder.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(folder / "habitat-failure-desktop.png"))
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator("#closeDetails").click()
    page.locator("#detailsToggle").click()
    card.get_by_role("button", name="Thinner shielding").click()
    routes[0].send(json.dumps({"type": "structure_shaking", "report": {
        "synthetic": True, "label": "Synthetic marsquake what-if", "honesty": "Synthetic shaking, not measured Mars seismic data.",
        "structures": [{"request_index": 0, "passed": False, "utilisation_pct": 125,
                        "governing": {"mode": "stress", "member": 1, "time_s": 1.2}}],
    }}))
    card.locator(".structure-shaking").wait_for(state="visible")
    assert "member 1" in card.locator(".structure-shaking").inner_text()
    assert "not measured Mars seismic data" in card.locator(".structure-shaking").inner_text()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    page.screenshot(path=str(folder / "habitat-failure-mobile.png"))
    context.close()