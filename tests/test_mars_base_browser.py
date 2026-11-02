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