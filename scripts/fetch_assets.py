#!/usr/bin/env python3
"""Download the optional runtime assets that aren't in git (the Docker build runs this).

Today that is one file: NASA's InSight lander model (assets/nasa/insight_lander.glb), the visual for Mars landers.
The download is checked against its size and SHA-256. A failed download is a warning, never an error: without the
file, Mars landers use the simple fallback shape. Physics never uses the mesh.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = [
    {
        "name": "NASA InSight lander model",
        "path": "assets/nasa/insight_lander.glb",
        # https://science.nasa.gov/resource/insight-lander-3d-model/ (NASA/JPL-Caltech)
        "url": "https://assets.science.nasa.gov/content/dam/science/psd/mars/resources/gltf_files/24880_InSight_deployed.glb",
        "size": 22_357_936,
        "sha256": "1e944bedb23af07962ba9f2b88d195b561665f255573b29ba7d028293a179387",
    },
]
TIMEOUT_S = 120


def _matches(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]


def fetch(asset: dict, root: Path = ROOT, opener=urllib.request.urlopen) -> bool:
    """Download one asset if it's missing or wrong. True when the checked file is in place."""
    path = root / asset["path"]
    if _matches(path, asset):
        print(f"ok    {asset['name']}: {asset['path']} (already there)")
        return True
    partial = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with opener(asset["url"], timeout=TIMEOUT_S) as response:
            partial.write_bytes(response.read())
        if not _matches(partial, asset):
            raise ValueError(f"size or SHA-256 doesn't match ({partial.stat().st_size} bytes)")
        partial.replace(path)
    except Exception as exc:   # no network, a moved URL, a changed file: the app still runs
        partial.unlink(missing_ok=True)
        print(f"WARNING {asset['name']} not downloaded ({type(exc).__name__}: {exc}); "
              "Mars landers will use the simple fallback shape.", file=sys.stderr)
        return False
    print(f"ok    {asset['name']}: downloaded {asset['path']} ({asset['size']} bytes, SHA-256 checked)")
    return True


def main() -> int:
    for asset in ASSETS:
        fetch(asset)
    return 0   # always 0: a missing optional asset must never fail a build


if __name__ == "__main__":
    sys.exit(main())
