"""Real terrain presets (e.g. NASA HiRISE elevation data), stored in simgate/terrains/.

Each preset is two files written by scripts/hirise_to_simgate.py:
  <name>.npz   heights_m: float32 [rows, cols], in metres, lowest point = 0.
               Row 0 is the SOUTH edge (-y) and column 0 the WEST edge (-x),
               matching how SimGate's generated terrain is stored.
  <name>.json  metadata: label, size, resolution, relief, slopes, source, credit.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

# SIMGATE_TERRAIN_DIR lets tests point at a temporary folder.
TERRAIN_DIR = Path(os.getenv("SIMGATE_TERRAIN_DIR", Path(__file__).resolve().parent / "terrains"))


def list_presets() -> dict[str, dict]:
    if not TERRAIN_DIR.exists():
        return {}
    out = {}
    for meta_path in sorted(TERRAIN_DIR.glob("*.json")):
        if meta_path.with_suffix(".npz").exists():
            out[meta_path.stem] = json.loads(meta_path.read_text())
    return out


@lru_cache(maxsize=16)
def load_preset(name: str) -> tuple[np.ndarray, dict]:
    meta_path = TERRAIN_DIR / f"{name}.json"
    data_path = TERRAIN_DIR / f"{name}.npz"
    if not (meta_path.exists() and data_path.exists()):
        available = ", ".join(list_presets()) or "none yet (run scripts/hirise_to_simgate.py)"
        raise FileNotFoundError(f"No terrain preset named '{name}'. Available: {available}")
    heights = np.load(data_path)["heights_m"].astype(np.float64)
    return heights, json.loads(meta_path.read_text())


def resample(heights: np.ndarray, n: int) -> np.ndarray:
    """Bilinear resample to an n x n grid covering the same area."""
    rows, cols = heights.shape
    r = np.linspace(0, rows - 1, n)
    c = np.linspace(0, cols - 1, n)
    tmp = np.array([np.interp(c, np.arange(cols), heights[i]) for i in range(rows)])        # rows x n
    return np.array([np.interp(r, np.arange(rows), tmp[:, j]) for j in range(n)]).T       # n x n
