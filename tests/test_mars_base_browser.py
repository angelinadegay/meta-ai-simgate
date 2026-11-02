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
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port)],
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