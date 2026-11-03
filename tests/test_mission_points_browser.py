import json
from pathlib import Path

import pytest

from test_mars_base_browser import browser_page
from test_mission_points import existing_world
from simgate.backends.mujoco_backend import MuJoCoSimulator


def test_flag_markers_are_small_and_located_at_selected_points(browser_page):
    result = browser_page.evaluate("""async () => {
      const THREE=await import('three');const {createMissionFlag}=await import('/static/vendor/mission-markers.js');
      return ['start','goal'].map(kind=>{const flag=createMissionFlag(kind,4,5,2);
        const bounds=new THREE.Box3().setFromObject(flag).getSize(new THREE.Vector3());
        return {name:flag.name,position:flag.position.toArray(),size:bounds.toArray(),parts:flag.children.length};});
    }""")
    assert [item["name"] for item in result] == ["mission_marker_start", "mission_marker_goal"]
    for marker in result:
        assert marker["position"] == [4, 5, 2]
        assert marker["parts"] == 3
        assert 1.1 < marker["size"][2] < 1.3


@pytest.mark.parametrize("count", [1, 2])
def test_map_clicks_patch_robot_and_markers_without_scene_reload(browser_page, count):
    simulator = MuJoCoSimulator()
    spec = existing_world(count)
    build = simulator.build(spec)
    before_code = build.code
    state = {"spec": spec, "build": build}
    received, patches, errors = [], [], []
    context = browser_page.context.browser.new_context(viewport={"width": 1200, "height": 900})
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))

    def connect(route):
        route.send(json.dumps({"type": "spec", "spec": spec.model_dump(mode="json")}))
        route.send(json.dumps({"type": "scene", "scene": build.scene, "code": build.code}))

        def clicked(message):
            payload = json.loads(message)
            received.append(payload)
            if payload["type"] not in {"set_start", "set_goal"}:
                return
            updated, changed, patch = simulator.update_mission_point(state["spec"], state["build"],
                payload["type"].removeprefix("set_"), payload["x"], payload["y"], payload.get("robot_id"))
            state.update({"spec": updated, "build": changed})
            patches.append(patch)
            route.send(json.dumps({"type": "spec", "spec": updated.model_dump(mode="json")}))
            route.send(json.dumps({"type": "mission_updated", **patch}))

        route.on_message(clicked)

    page.route_web_socket("**/ws", connect)
    page.goto(browser_page.url)
    page.locator("#view canvas").wait_for(state="visible")
    page.wait_for_function("document.querySelectorAll('#structureCards article').length === 1")
    requests = []
    page.on("request", lambda request: requests.append(request.url))
    canvas = page.locator("#view canvas")
    size = canvas.bounding_box()
    for kind in ("start", "goal"):
        page.get_by_role("button", name=f"Set {kind}", exact=True).click()
        fraction = 0.5 if kind == "start" else 0.65
        canvas.click(position={"x": size["width"] * fraction, "y": size["height"] * 0.75})
        page.wait_for_function(f"document.querySelector('#log').textContent.includes('Set {kind} at')")
        page.wait_for_function("document.getElementById('player').hidden")
    assert [message["type"] for message in received] == ["set_start", "set_goal"]
    assert len(patches) == 2 and patches[0]["moved_poses"] and not patches[1]["moved_poses"]
    assert state["build"].code == before_code
    assert not errors and not requests
    folder = Path(__file__).parent / "screenshots" / "mission-points"
    folder.mkdir(parents=True, exist_ok=True)
    page.locator("#view").screenshot(path=str(folder / f"flags-{count}-robots.png"))
    context.close()