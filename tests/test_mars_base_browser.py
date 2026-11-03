import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def browser_page():
    playwright = pytest.importorskip("playwright.sync_api")
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port),
         "--no-access-log"],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in process.stdout:
            if "Uvicorn running" in line:
                break
            if "ERROR" in line:
                raise RuntimeError(line)
        with playwright.sync_playwright() as driver:
            browser = driver.chromium.launch(channel="msedge", headless=True,
                                             args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.goto(f"http://127.0.0.1:{port}/")
            yield page
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=15)


def test_horizon_joins_edge_and_preserves_physics_data(browser_page):
    result = browser_page.evaluate("""async () => {
      const THREE = await import('three');
      const {buildTerrainHorizon, continuedHeight, terrainSample} = await import('/static/vendor/terrain-horizon.js');
      const heightfield = {nrow:7, ncol:7, size:[3,3,1], data:[]};
      for (let row=0; row<7; row++) for (let col=0; col<7; col++) heightfield.data.push(col*.1+row*.2);
      const before = JSON.stringify(heightfield);
      const geometry = new THREE.PlaneGeometry(6,6,6,6);
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Array(49*3).fill(.6),3));
      const source = new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:0xffffff,roughness:.8,vertexColors:true}));
      const horizon = buildTerrainHorizon(heightfield,source);
      const points = horizon.geometry.attributes.position;
      let seamError=0, inside=0;
      for(let index=0;index<points.count;index++) {
        const x=points.getX(index),y=points.getY(index);
        if(Math.max(Math.abs(x),Math.abs(y))<3-1e-6)inside++;
        if(index<24)seamError=Math.max(seamError,Math.abs(points.getZ(index)-terrainSample(heightfield,x,y)));
      }
      return {seamError,inside,unchanged:before===JSON.stringify(heightfield),
        slope:(continuedHeight(heightfield,3.00001,0)-continuedHeight(heightfield,3,0))/.00001,
        visualOnly:horizon.userData.visualOnly,vertices:points.count,extent:horizon.geometry.attributes.position.getX(24*10+6)};
    }""")
    assert result["seamError"] < 1e-6
    assert result["slope"] == pytest.approx(0.1, rel=0.001)
    assert result["inside"] == 0 and result["unchanged"] and result["visualOnly"]
    assert result["vertices"] == 264
    assert result["extent"] > 2000


def test_true_visual_sizes_are_measured_from_geometry(browser_page):
        result = browser_page.evaluate("""async () => {
            const {buildStructureVisuals,measureStructureVisuals}=await import('/static/vendor/structures.js');
            const records=[
                {kind:'habitat',dimensions:{module_length_m:6,module_diameter_m:4,airlock_rise_m:1,ramp:{run_m:4.7,width_m:1.5}}},
                {kind:'bridge',dimensions:{span_m:3.4,deck_width_m:1.2}},
                {kind:'lander',dimensions:{footprint_m:6,height_m:2.25,deck_span_m:1.5,deck_height_m:1,body_height_m:.35,leg_footprint_m:2.5,legs:3,visual_model:'nasa_insight'}}
            ].map(record=>({...record,position:{x:0,y:0,z:0},orientation:{yaw_deg:0}}));
            const scene={world:{body:'mars'},structures:records};const before=JSON.stringify(scene);
            const visual=buildStructureVisuals(scene,{terrainHeight:()=>0});await visual.ready;
            return {measurements:measureStructureVisuals(visual.root),unchanged:before===JSON.stringify(scene),
                credit:visual.root.children[2].userData.credit||null};
        }""")
        habitat, bridge, lander = result["measurements"]
        assert habitat["body_size_m"] == pytest.approx([6, 4, 4], abs=1e-5)
        assert bridge["body_size_m"] == pytest.approx([3.4, 1.2, 0.19], abs=1e-5)
        assert max(lander["world_size_m"][:2]) == pytest.approx(6, abs=0.1)
        assert result["unchanged"]
        artifacts = Path(__file__).parent / "screenshots"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "measured-structure-sizes.json").write_text(json.dumps(result, indent=2))


@pytest.mark.parametrize("aspect", [1440 / 900, 390 / 844])
def test_base_camera_contains_structures_start_goal_and_eases_to_follow(browser_page, aspect):
        result = browser_page.evaluate("""async aspect => {
            const THREE=await import('three');const {frameBase,easeIntoFollow}=await import('/static/vendor/base-camera.js');
            const root=new THREE.Group(),parent=new THREE.Group();parent.rotation.x=-Math.PI/2;parent.add(root);
            for(const x of [-12,0,14]){const group=new THREE.Group(),skin=new THREE.Group();
                skin.add(new THREE.Mesh(new THREE.BoxGeometry(6,4,4),new THREE.MeshBasicMaterial()));
                group.position.set(x,0,2);group.add(skin);group.userData.skin=skin;root.add(group);}
            const camera=new THREE.PerspectiveCamera(50,aspect,.05,5000),controls={target:new THREE.Vector3()};
            const scene={mission:{start:[-20,-5],goal:[19,5],goal_z:4}};
            const bounds=frameBase(camera,controls,root,scene,()=>0);const projected=[];
            for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z]){
                const point=new THREE.Vector3(x,y,z).project(camera);projected.push([point.x,point.y]);}
            const initial=camera.position.clone(),robot=new THREE.Vector3(-20,1,5),desired=robot.clone().add(new THREE.Vector3(-3.2,2.2,2.6));
            easeIntoFollow(camera,controls,robot,1/60);const first=camera.position.distanceTo(initial);
            const complete=initial.distanceTo(desired);for(let i=0;i<240;i++)easeIntoFollow(camera,controls,robot,1/60);
            return {projected,first,complete,remaining:camera.position.distanceTo(desired)};
        }""", aspect)
        assert all(abs(axis) < 1 for point in result["projected"] for axis in point)
        assert 0 < result["first"] < result["complete"] * 0.1
        assert result["remaining"] < result["complete"] * 0.001


def test_habitat_has_visible_hardware_dust_and_optional_shielding(browser_page):
        result = browser_page.evaluate("""async () => {
            const {buildStructureVisuals}=await import('/static/vendor/structures.js');
            const record={kind:'habitat',position:{x:0,y:0,z:0},orientation:{yaw_deg:0},
                dimensions:{module_length_m:6,module_diameter_m:4,airlock_rise_m:1,ramp:{run_m:4.7,width_m:1.5}},
                structural_check:{shielding_thickness_m:.5}};
            delete record.structural_check;
            const bare=buildStructureVisuals({world:{body:'mars'},structures:[record]},{terrainHeight:()=>0});
            const skin=bare.root.children[0].userData.skin,counts={};skin.traverse(object=>{if(object.isMesh)counts[object.name]=(counts[object.name]||0)+1;});
            const segment=skin.getObjectByName('habitat_segment');const canvas=segment.material.map.image;
            const data=canvas.getContext('2d').getImageData(0,128,256,1).data;
            const lower=data[128*4]+data[128*4+1]+data[128*4+2],upper=data[0]+data[1]+data[2];
            return {counts,lower,upper,physicsUnchanged:!Object.hasOwn(record,'structural_check')};
        }""")
        assert result["counts"]["habitat_segment"] >= 5
        assert result["counts"]["habitat_rib_band"] >= 6
        assert result["counts"]["habitat_window"] == 6
        assert result["counts"]["airlock_frame"] == 1
        assert result["counts"]["ramp_side_rail"] == 2
        assert result["counts"]["ramp_rail_post"] == 10
        assert "regolith_shielding" not in result["counts"]
        assert result["lower"] < result["upper"] and result["physicsUnchanged"]


def test_follow_transition_survives_a_long_browser_frame(browser_page):
        result = browser_page.evaluate("""async () => {
            const THREE=await import('three');const {advanceFollowTransition}=await import('/static/vendor/base-camera.js');
            const camera=new THREE.PerspectiveCamera();camera.position.set(50,20,40);
            const controls={target:new THREE.Vector3()},robot=new THREE.Vector3();
            const before=camera.position.clone();const remaining=advanceFollowTransition(camera,controls,robot,10,2.2);
            return {remaining,moved:camera.position.distanceTo(before),distance:before.length()};
        }""")
        assert result["remaining"] == pytest.approx(2.1)
        assert 0 < result["moved"] < result["distance"] * 0.2


def test_jezero_base_desktop_mobile_screenshots_and_interaction(browser_page, monkeypatch):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from simgate import config
    from simgate.director import Director
    from simgate.workflows import _prepare_robot_world
    from simgate.backends.mujoco_backend import MuJoCoSimulator

    monkeypatch.setattr(config, "MOCK_MODE", True)
    director = Director()
    prompt = ("Build a Mars base on Jezero: a habitat, a lander, and a bridge over a trench. "
              "Can the rover drive from the lander to the habitat?")
    plan = director.plan(prompt)
    with ThreadPoolExecutor(max_workers=1) as executor:
        world = executor.submit(asyncio.run, _prepare_robot_world(plan, None, director)).result()
    build = MuJoCoSimulator().build(world.spec)
    messages = [json.dumps({"type": "spec", "spec": world.spec.model_dump(mode="json")}),
                json.dumps({"type": "scene", "scene": build.scene, "code": build.code})]
    sockets = []

    def connect(route):
        sockets.append(route)
        for message in messages:
            route.send(message)

    context = browser_page.context.browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    page.set_default_timeout(60_000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route_web_socket("**/ws", connect)
    page.goto(browser_page.url)
    page.wait_for_function("document.querySelectorAll('#structureCards article').length === 3")
    page.locator("#goalPill").filter(has_text="Goal: habitat airlock").wait_for(state="visible")
    folder = Path(__file__).parent / "screenshots" / "mars-base"
    folder.mkdir(parents=True, exist_ok=True)
    nasa = (Path(__file__).resolve().parents[1] / "assets/nasa/insight_lander.glb").is_file()
    if nasa:
        page.wait_for_function("Array.from(document.querySelectorAll('#structureCards a')).some(a=>a.textContent.includes('NASA InSight'))")
    page.locator("#view").screenshot(path=str(folder / "base-desktop.png"))
    for kind in ("habitat", "lander", "bridge"):
        page.locator("#detailsToggle").click()
        page.get_by_title(f"Focus on {kind}").click()
        page.locator("#detailsToggle").click()
        page.locator("#view").screenshot(path=str(folder / f"{kind}-desktop.png"))
    canvas = page.locator("#view canvas")
    before = canvas.screenshot()
    bounds = canvas.bounding_box()
    page.mouse.move(bounds["x"] + bounds["width"] * 0.45, bounds["y"] + bounds["height"] * 0.6)
    page.mouse.down()
    page.mouse.move(bounds["x"] + bounds["width"] * 0.65, bounds["y"] + bounds["height"] * 0.6, steps=12)
    page.mouse.up()
    assert before != canvas.screenshot()
    page.locator("#detailsToggle").click()
    page.get_by_title("Focus on lander").click()
    page.screenshot(path=str(folder / "nasa-details-desktop.png"))
    page.locator("#detailsToggle").click()
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator("#detailsToggle").click()
    page.get_by_title("Focus on lander").click()
    page.locator("#closeDetails").click()
    page.locator("#view").screenshot(path=str(folder / "lander-mobile.png"))
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    pixel_check = page.evaluate("""() => new Promise(resolve=>requestAnimationFrame(()=>{
          const canvas=document.querySelector('#view canvas'),gl=canvas.getContext('webgl2')||canvas.getContext('webgl');
          const pixels=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
          gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
          const colours=new Set();for(let i=0;i<pixels.length;i+=160)colours.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
          resolve(colours.size);
        }))""")
    assert pixel_check > 30
    page.locator("#detailsToggle").click()
    page.screenshot(path=str(folder / "nasa-details-mobile.png"))
    page.locator("#closeDetails").click()
    sockets[0].send(json.dumps({"type": "run_start"}))
    page.wait_for_function("!document.getElementById('player').hidden")
    page.evaluate("() => new Promise(resolve=>{let frames=0;function step(){if(++frames===90)resolve();else requestAnimationFrame(step);}requestAnimationFrame(step);})")
    page.locator("#view").screenshot(path=str(folder / "follow-mobile.png"))
    (folder / "verification.json").write_text(json.dumps({"browser_errors": errors, "canvas_colours": pixel_check,
        "goal": page.locator("#goalPill").inner_text(), "physics_terrain_m": world.spec.terrain.size_m,
        "preset": world.spec.terrain.preset, "nasa_loaded": nasa}, indent=2))
    assert not errors
    page.close()
    context.close()