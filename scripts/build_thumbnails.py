#!/usr/bin/env python3
"""Regenerate the home page's world-card thumbnails (static/thumbnails/<card key>.webp, 800×500).

python scripts/build_thumbnails.py [card key ...]

Each shipped world card is opened in the app itself (a local server in mock mode, headless Edge or Chromium), built
exactly as a visitor's scene, and captured with the app's own renderer, materials and sky: the world alone (no
robot or markers) in a three-quarter view. If a 3D capture fails, the card gets a smooth shaded-relief image
instead (simgate/thumbnails.py), never the raw heightmap. Uploaded worlds aren't shipped, so they're skipped;
the server draws their shaded relief on the fly. Needs `pip install -r requirements-dev.txt` and
`python -m playwright install chromium` (or a local Edge).
"""
from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simgate.thumbnails import SIZE, THUMBNAILS_DIR, shaded_relief, thumbnail_file, webp_bytes  # noqa: E402

CAPTURE_CSS = """
#view { position: fixed !important; inset: 0 !important; width: 100vw !important; height: 100vh !important;
        z-index: 2147483000 !important; }
#view > :not(canvas) { display: none !important; }
"""


def _server():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        port = connection.getsockname()[1]
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port)],
                               cwd=ROOT, env={**os.environ, "SIMGATE_MOCK_MODE": "1"},
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        if "Uvicorn running" in line:
            return process, port
        if "Traceback" in line or "ERROR" in line:
            break
    process.terminate()
    raise RuntimeError("the local server didn't start")


def _browser(driver):
    for options in ({"channel": "msedge"}, {}):
        try:
            return driver.chromium.launch(headless=True, args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                                               "--ignore-gpu-blocklist"], **options)
        except Exception:   # noqa: BLE001 - no Edge here: Playwright's own Chromium
            continue
    raise RuntimeError("no Chromium-based browser for the capture")


def capture(page, card: dict) -> bytes:
    """The card's world in the app's renderer, as WebP; raises when the capture fails or comes out blank."""
    from PIL import Image, ImageStat

    page.evaluate("window.__thumbnailPrevious = window.simgateThumbnailScene()")
    page.evaluate("key => window.simgateThumbnailLaunch(key)", card["key"])
    page.wait_for_function("window.simgateThumbnailScene() && window.simgateThumbnailScene() !== window.__thumbnailPrevious",
                           timeout=180_000)
    page.evaluate("() => window.simgateThumbnail(window.simgateThumbnailScene())")
    page.wait_for_timeout(1200)   # terrain textures and the structure skins settle
    image = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
    if image.size != SIZE:
        image = image.resize(SIZE, Image.LANCZOS)
    if max(ImageStat.Stat(image).stddev) < 6:
        raise RuntimeError("the capture is blank")
    return webp_bytes(image)


def relief(card: dict) -> bytes:
    """The fallback: a smooth shaded relief of the card's own ground."""
    from simgate.spec import Environment, SceneSpec, Terrain

    if card["terrain"] == "real":
        from simgate.terrain_data import load_preset

        heights, meta = load_preset(card["preset"])
        return webp_bytes(shaded_relief(heights, float(meta["size_m"]), meta.get("body", card["body"])))
    if card["terrain"] == "indoor":
        from simgate.indoor import get_layout, height_grid

        preset = card["preset"] or "apartment"
        return webp_bytes(shaded_relief(height_grid(preset, 160), max(get_layout(preset)["size_m"]), "indoor"))
    from simgate.backends.mujoco_backend import make_heights

    spec = SceneSpec(domain="robot_terrain", terrain=Terrain(kind=card["terrain"]),
                     environment=Environment(body=card["body"], surface=card["surface"])).fill_defaults()
    return webp_bytes(shaded_relief(make_heights(spec)[0], spec.terrain.size_m, card["body"]))


def main(keys: set[str]) -> int:
    from playwright.sync_api import sync_playwright

    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    server, port = _server()
    report = []
    try:
        with sync_playwright() as driver:
            browser = _browser(driver)
            page = browser.new_page(viewport={"width": SIZE[0], "height": SIZE[1]})
            page.goto(f"http://127.0.0.1:{port}/?thumbnail", wait_until="domcontentloaded")
            page.wait_for_function("window.simgateThumbnailCards && window.simgateThumbnailCards().length > 0",
                                   timeout=60_000)
            page.add_style_tag(content=CAPTURE_CSS)
            for card in page.evaluate("window.simgateThumbnailCards()"):
                if card["source"] == "upload" or (keys and card["key"] not in keys):
                    continue
                try:
                    data, how = capture(page, card), "3D"
                except Exception as error:   # noqa: BLE001 - any capture problem: the shaded relief instead
                    data, how = relief(card), f"shaded relief ({type(error).__name__}: {str(error)[:80]})"
                thumbnail_file(card["key"]).write_bytes(data)
                report.append((card["key"], how, len(data)))
                print(f"{card['key']}: {how}, {len(data) / 1000:.0f} KB", flush=True)
            browser.close()
    finally:
        server.terminate()
    total = sum(path.stat().st_size for path in THUMBNAILS_DIR.glob("*.webp"))
    print(f"{len(report)} thumbnails written; static/thumbnails/ holds {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:])))
