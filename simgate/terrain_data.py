"""Real terrain presets (e.g. NASA HiRISE elevation data), stored in simgate/terrains/.

Each preset is two files written by scripts/hirise_to_simgate.py:
  <name>.npz   heights_m: float32 [rows, cols], in metres, lowest point = 0.
               Row 0 is the SOUTH edge (-y) and column 0 the WEST edge (-x),
               matching how SimGate's generated terrain is stored.
  <name>.json  metadata: label, size, resolution, relief, slopes, source, credit.
Uploaded terrains keep theirs in their upload folder (uploads/terrain/<id>/preset.npz and preset.json); the
preset's name is the upload id.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# SIMGATE_TERRAIN_DIR lets tests point at a temporary folder.
TERRAIN_DIR = Path(os.getenv("SIMGATE_TERRAIN_DIR", Path(__file__).resolve().parent / "terrains"))


def _uploaded_dir() -> Path:
    from .library import uploads   # read at call time, so tests can point it elsewhere

    return uploads.UPLOADS_DIR / "terrain"


def _preset_files(name: str) -> tuple[Path, Path]:
    """(json, npz) for a preset: shipped presets first, then an uploaded terrain with that id."""
    shipped = (TERRAIN_DIR / f"{name}.json", TERRAIN_DIR / f"{name}.npz")
    if shipped[0].exists() or "/" in name or "\\" in name or name.startswith("."):
        return shipped
    folder = _uploaded_dir() / name
    return folder / "preset.json", folder / "preset.npz"


def list_presets() -> dict[str, dict]:
    out = {}
    if TERRAIN_DIR.exists():
        for meta_path in sorted(TERRAIN_DIR.glob("*.json")):
            if meta_path.with_suffix(".npz").exists():
                out[meta_path.stem] = json.loads(meta_path.read_text())
    uploaded = _uploaded_dir()
    if uploaded.is_dir():
        for folder in sorted(uploaded.iterdir()):
            if folder.name not in out and (folder / "preset.json").is_file() and (folder / "preset.npz").is_file():
                out[folder.name] = json.loads((folder / "preset.json").read_text())
    return out


@lru_cache(maxsize=16)
def load_preset(name: str) -> tuple[np.ndarray, dict]:
    meta_path, data_path = _preset_files(name)
    if not (meta_path.exists() and data_path.exists()):
        available = ", ".join(list_presets()) or "none yet (run scripts/hirise_to_simgate.py)"
        raise FileNotFoundError(f"No terrain preset named '{name}'. Available: {available}")
    heights = np.load(data_path)["heights_m"].astype(np.float64)
    return heights, json.loads(meta_path.read_text())


def resizable(meta: dict) -> bool:
    """An uploaded heightmap image: its width and height range were the uploader's choice, so a scene may change
    them. Real elevation data (HiRISE, GeoTIFF) keeps the scale it was measured at."""
    return bool(meta.get("uploaded")) and str(meta.get("source", "")).lower().endswith((".png", ".jpg"))


def scene_heights(terrain) -> np.ndarray:
    """A real terrain's heights for one scene: the preset's, scaled to the scene's height range when it has one."""
    heights = load_preset(terrain.preset)[0]
    top = float(heights.max())
    if terrain.preset_relief_m is not None and top > 0:
        heights = heights * (terrain.preset_relief_m / top)
    return heights


def resample(heights: np.ndarray, n: int) -> np.ndarray:
    """Bilinear resample to an n x n grid covering the same area."""
    rows, cols = heights.shape
    r = np.linspace(0, rows - 1, n)
    c = np.linspace(0, cols - 1, n)
    tmp = np.array([np.interp(c, np.arange(cols), heights[i]) for i in range(rows)])        # rows x n
    return np.array([np.interp(r, np.arange(rows), tmp[:, j]) for j in range(n)]).T       # n x n
