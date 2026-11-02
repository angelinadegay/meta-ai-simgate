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
import math
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


def load_ortho_image(name: str) -> tuple[bytes, dict] | None:
    """The optional RGB terrain image for the viewer; paths are constrained to the preset folder."""
    meta_path, _ = _preset_files(name)
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    filename = meta.get("ortho_file")
    if not filename:
        return None
    folder = meta_path.parent.resolve()
    image_path = (folder / filename).resolve()
    try:
        image_path.relative_to(folder)
    except ValueError:
        return None
    if not image_path.is_file():
        return None
    return image_path.read_bytes(), meta


def resizable(meta: dict) -> bool:
    """An uploaded heightmap image: its width and height range were the uploader's choice, so a scene may change
    them. Real elevation data (HiRISE, GeoTIFF) keeps the scale it was measured at."""
    return bool(meta.get("uploaded")) and str(meta.get("source", "")).lower().endswith((".png", ".jpg"))


def scene_heights(terrain) -> np.ndarray:
    """A real terrain's heights for one scene: the preset's, scaled to the scene's height range when it has one."""
    heights = (load_structure_patch(terrain.preset, terrain.structure_crop_size_m)[0]
               if terrain.structure_crop_size_m is not None else load_preset(terrain.preset)[0])
    top = float(heights.max())
    if terrain.preset_relief_m is not None and top > 0:
        heights = heights * (terrain.preset_relief_m / top)
    return heights


@lru_cache(maxsize=16)
def load_structure_patch(name: str, size_m: float) -> tuple[np.ndarray, dict]:
    """Select a larger measured source patch; never stretch or overwrite the preset."""
    import rasterio
    from scipy.ndimage import maximum_filter, uniform_filter

    from .terrain_import import geotiff_heights, inspect

    original, meta = load_preset(name)
    source = meta.get("source")
    if not source or not str(source).lower().endswith((".tif", ".tiff")):
        raise ValueError(f"{meta['label']} has no source GeoTIFF for a larger measured structure patch.")
    mpp = float(meta["meters_per_pixel"])
    requested_pixels = int(round(size_m / mpp))
    survey_size = min(requested_pixels * 3, *meta["source_shape"])
    row, col = meta["window_top_left_px"]
    row = max(0, min(meta["source_shape"][0] - survey_size, row + original.shape[0] // 2 - survey_size // 2))
    col = max(0, min(meta["source_shape"][1] - survey_size, col + original.shape[1] // 2 - survey_size // 2))
    with rasterio.Env(GDAL_HTTP_TIMEOUT=20, GDAL_HTTP_MAX_RETRY=1):
        survey, mpp, details = geotiff_heights(source, survey_size * mpp, pixel=(row, col))
    dy, dx = np.gradient(survey, mpp, mpp)
    slopes = np.degrees(np.arctan(np.hypot(dx, dy)))
    pad = maximum_filter(slopes, size=max(3, int(math.ceil(12 / mpp))))
    scores = uniform_filter((pad <= 8).astype(float), size=requested_pixels, mode="constant")
    border = requested_pixels // 2 + 1
    scores[:border] = scores[-border:] = -1
    scores[:, :border] = scores[:, -border:] = -1
    centre_row, centre_col = np.unravel_index(int(np.argmax(scores)), scores.shape)
    bottom = int(centre_row - requested_pixels // 2)
    left = int(centre_col - requested_pixels // 2)
    heights = survey[bottom:bottom + requested_pixels, left:left + requested_pixels].copy()
    offset = float(heights.min())
    heights -= offset
    report = inspect(heights, mpp)
    patch = {**meta, **report, "window_top_left_px": [row + survey.shape[0] - bottom - requested_pixels, col + left],
             "base_elevation_m": details["base_elevation_m"] + offset, "structure_patch": True}
    return heights, patch


def resample(heights: np.ndarray, n: int) -> np.ndarray:
    """Bilinear resample to an n x n grid covering the same area."""
    rows, cols = heights.shape
    r = np.linspace(0, rows - 1, n)
    c = np.linspace(0, cols - 1, n)
    tmp = np.array([np.interp(c, np.arange(cols), heights[i]) for i in range(rows)])        # rows x n
    return np.array([np.interp(r, np.arange(rows), tmp[:, j]) for j in range(n)]).T       # n x n
