import json
from pathlib import Path

from test_mars_base_browser import browser_page
from test_indoor import apartment
from simgate.backends.mujoco_backend import MuJoCoSimulator


def test_apartment_tile_floor_plan_and_indoor_rendering(browser_page):
    browser_page.get_by_role("heading", name="Apartment", exact=True).wait_for(state="visible")
    spec = apartment()
    simulator = MuJoCoSimulator()
    build = simulator.build(spec)
    initial_code = build.code
    snapshot = json.dumps(build.scene)
    context = browser_page.context.browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def connect(route):
        route.send(json.dumps({"type": "spec", "spec": spec.model_dump(mode="json")}))
        route.send(json.dumps({"type": "scene", "scene": build.scene, "code": build.code}))

    page.route_web_socket("**/ws", connect)
    page.goto(browser_page.url)
    page.locator("#goalPill").wait_for(state="visible")
    assert "living" in page.locator("#goalPill").inner_text()
    folder = Path(__file__).parent / "screenshots" / "indoor"
    folder.mkdir(parents=True, exist_ok=True)
    page.locator("#view").screenshot(path=str(folder / "apartment-desktop.png"))
    before = page.locator("#view canvas").screenshot()
    box = page.locator("#view canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2, steps=8)
    page.mouse.up()
    assert page.locator("#view canvas").screenshot() != before
    rendered = page.evaluate("""async data => {
      const THREE=await import('three');const {decorateIndoor,apartmentThumbnail}=await import('/static/vendor/indoor-visuals.js');
      const before=JSON.stringify(data),meshes=[];
      for(const geom of data.geoms)meshes[geom.id]=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshStandardMaterial());
      decorateIndoor(data,meshes);
      const tile=data.geoms.find(geom=>geom.name==='indoor_floor_kitchen_tile');
      const rug=data.geoms.find(geom=>geom.name==='indoor_floor_rug');
      const thumbnail=apartmentThumbnail(data.indoor);
      return {unchanged:before===JSON.stringify(data),tileMap:!!meshes[tile.id].material.map,
        rugMap:!!meshes[rug.id].material.map,thumbnail:thumbnail.startsWith('data:image/png'),
        ceiling:data.geoms.some(geom=>geom.name.includes('ceiling'))};
    }""", build.scene)
    assert rendered == {"unchanged": True, "tileMap": True, "rugMap": True, "thumbnail": True, "ceiling": False}
    page.set_viewport_size({"width": 390, "height": 844})
    page.reload()
    page.locator("#goalPill").wait_for(state="visible")
    page.locator("#view").screenshot(path=str(folder / "apartment-mobile.png"))
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    assert not errors
    assert build.code == initial_code and json.dumps(build.scene) == snapshot
    context.close()