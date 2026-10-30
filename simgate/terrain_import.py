"""Turn elevation data into SimGate terrain presets: GeoTIFF DTMs (e.g. NASA HiRISE) and greyscale heightmap
images. Used by scripts/hirise_to_simgate.py and by terrain uploads.

A preset's heights are metres above its lowest point, row 0 = SOUTH (-y), column 0 = WEST (-x), as
simgate/terrain_data.py reads them.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

MAX_TYPICAL_SLOPE_DEG = 22.0  # relief picking skips windows whose 95th-percentile slope is steeper (robots can't cross)
MAX_CANDIDATES = 40           # full-resolution reads to try before giving up
MAX_HEIGHTMAP_PX = 512        # larger heightmap images are resampled to this many pixels a side
# What the Inspector compares a terrain with: the robots' rough slope limits (as in the Checker) and a usable size.
STEEP_FOR = {"walkers": 20.0, "rovers": 25.0}
MIN_SIZE_M = 10.0


class TerrainError(ValueError):
    """Elevation data that can't become a preset; the message says why."""


def valid_mask(a: np.ndarray, nodata) -> np.ndarray:
    m = np.isfinite(a) & (a > -1e10) & (a < 1e10)       # HiRISE uses about -3.4e38 for "no data"
    if nodata is not None and np.isfinite(nodata):
        m &= a != nodata
    return m


def slope_deg(h: np.ndarray, mpp: float) -> np.ndarray:
    gy, gx = np.gradient(h, mpp)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def typical_slope(h: np.ndarray, mpp: float) -> float:
    """95th-percentile slope: robust to single noisy pixels, unlike the maximum."""
    return float(np.percentile(slope_deg(h, mpp), 95))


def candidates(ds, win_px: int, mode: str, center_frac: tuple[float, float]) -> list[tuple[int, int]]:
    """Rank windows using a low-resolution preview (cheap, even for a remote file)."""
    from rasterio.enums import Resampling

    step = max(1, win_px // 4)                       # preview pixel size, in source pixels
    ph, pw = max(1, ds.height // step), max(1, ds.width // step)
    prev = ds.read(1, out_shape=(ph, pw), resampling=Resampling.nearest).astype(np.float64)
    ok = valid_mask(prev, ds.nodata)
    k = max(1, win_px // step)                       # window size in preview pixels
    cy, cx = center_frac[1] * (ph - k), center_frac[0] * (pw - k)
    mpp_prev = abs(ds.transform.a) * step
    scored = []
    for r in range(0, ph - k + 1):
        for c in range(0, pw - k + 1):
            if not ok[r : r + k, c : c + k].all():
                continue
            w = prev[r : r + k, c : c + k]
            if mode == "relief":
                # The coarse preview understates steepness, so cap it at 2/3 of the real limit before ranking.
                if k > 1 and typical_slope(w, mpp_prev) > MAX_TYPICAL_SLOPE_DEG * 2 / 3:
                    continue
                score = float(w.max() - w.min())
            else:
                score = -((r - cy) ** 2 + (c - cx) ** 2)
            scored.append((score, min(r * step, ds.height - win_px), min(c * step, ds.width - win_px)))
    scored.sort(reverse=True)
    return [(r, c) for _, r, c in scored]


def read_window(ds, row: int, col: int, win_px: int) -> tuple[np.ndarray, np.ndarray]:
    from rasterio.windows import Window

    raw = ds.read(1, window=Window(col, row, win_px, win_px)).astype(np.float64)
    return raw, valid_mask(raw, ds.nodata)


def pick_window(ds, win_px: int, mode: str, center_frac: tuple[float, float]):
    """First candidate that is >= 98% valid at full resolution (and, for relief, crossable)."""
    mpp = abs(ds.transform.a)
    tried = 0
    for row, col in candidates(ds, win_px, mode, center_frac):
        if tried >= MAX_CANDIDATES:
            break
        tried += 1
        raw, ok = read_window(ds, row, col, win_px)
        if ok.mean() < 0.98:
            continue
        if mode == "relief" and typical_slope(np.where(ok, raw, np.median(raw[ok])), mpp) > MAX_TYPICAL_SLOPE_DEG:
            continue
        return row, col, raw, ok
    raise TerrainError(f"No suitable window in the first {tried} candidates. Try picking the center, a smaller size, "
                       "or another file.")


def geotiff_heights(source, size_m: float, pick: str = "relief", center: tuple[float, float] = (0.5, 0.5),
                    pixel: tuple[int, int] | None = None, require_metric: bool = False) -> tuple[np.ndarray, float, dict]:
    """Crop a size_m square from a DTM GeoTIFF (local path or URL) at full resolution, fill rare holes with the
    nearest valid height, and flip it so row 0 is south. Returns (heights, metres per pixel, source details)."""
    import rasterio

    with rasterio.open(source) as ds:
        if require_metric and (ds.crs is None or not ds.crs.is_projected
                               or (ds.crs.linear_units or "").lower() not in ("metre", "meter", "m", "metres", "meters")):
            raise TerrainError("This GeoTIFF has no projected coordinate system in metres (its pixel size isn't in "
                               "metres), so its scale can't be trusted. Reproject it to a metric CRS first.")
        mpp = abs(ds.transform.a)
        if abs(abs(ds.transform.e) - mpp) / mpp > 0.01:
            raise TerrainError(f"Non-square pixels ({ds.transform.a} x {ds.transform.e}); not supported yet.")
        win_px = int(round(size_m / mpp))
        if win_px < 8 or win_px > min(ds.height, ds.width):
            raise TerrainError(f"A {size_m:g} m square is {win_px} px; the file is {ds.height} x {ds.width} px at "
                               f"{mpp:.3f} m/px.")
        if pixel:
            row, col = pixel
            raw, ok = read_window(ds, row, col, win_px)
        else:
            row, col, raw, ok = pick_window(ds, win_px, pick, center)
        x0, y0 = ds.transform * (col, row)
        details = {"crs": ds.crs.to_string() if ds.crs else None, "source_shape": [ds.height, ds.width],
                   "window_top_left_px": [int(row), int(col)], "window_origin_projected_m": [round(x0, 2), round(y0, 2)]}

    valid_frac = float(ok.mean())
    if valid_frac < 0.98:
        raise TerrainError(f"The window is only {valid_frac:.0%} valid data. Move it or choose a smaller size.")
    if valid_frac < 1.0:                             # fill rare holes with the nearest valid height
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~ok, return_distances=False, return_indices=True)
        raw = raw[tuple(idx)]
    base = float(raw.min())
    details["base_elevation_m"] = round(base, 2)
    return np.flipud(raw - base), mpp, details     # GeoTIFF row 0 is NORTH; SimGate row 0 is SOUTH


def heightmap_heights(image: bytes, size_m: float, height_range_m: float) -> tuple[np.ndarray, float]:
    """A greyscale PNG/JPG heightmap: black = lowest, white = height_range_m higher; the image's top edge is
    north. Colour images use their luminance. Returns (heights, metres per pixel)."""
    from PIL import Image

    from .terrain_data import resample

    try:
        picture = Image.open(io.BytesIO(image))
        picture.load()
    except Exception as exc:  # noqa: BLE001 - any decoding problem is the image's
        raise TerrainError(f"The heightmap image can't be read: {exc}") from exc
    if picture.mode not in ("I;16", "I;16B", "I;16L", "I", "F", "L"):
        picture = picture.convert("L")
    grey = np.asarray(picture, dtype=np.float64)
    if grey.ndim != 2 or min(grey.shape) < 8:
        raise TerrainError("The heightmap must be a 2D image at least 8 x 8 pixels.")
    if grey.shape[0] != grey.shape[1]:
        raise TerrainError(f"The heightmap is {grey.shape[1]} x {grey.shape[0]} px; it must be square.")
    if max(grey.shape) > MAX_HEIGHTMAP_PX:
        grey = resample(grey, MAX_HEIGHTMAP_PX)
    low, high = float(grey.min()), float(grey.max())
    scaled = (grey - low) / (high - low) if high > low else np.zeros_like(grey)
    return np.flipud(scaled * height_range_m), size_m / grey.shape[0]


def inspect(heights: np.ndarray, mpp: float) -> dict:
    """The Inspector's report: size, relief, typical (95th percentile) and max slope, and plain warnings."""
    slopes = slope_deg(heights, mpp)
    size_m = heights.shape[0] * mpp
    report = {"size_m": round(size_m, 3), "meters_per_pixel": round(mpp, 4),
              "relief_m": round(float(heights.max() - heights.min()), 3),
              "typical_slope_deg": round(float(np.percentile(slopes, 95)), 1),
              "max_slope_deg": round(float(slopes.max()), 1), "mean_slope_deg": round(float(slopes.mean()), 1),
              "warnings": []}
    too_steep = [robots for robots, limit in STEEP_FOR.items() if report["typical_slope_deg"] > limit]
    if too_steep:
        report["warnings"].append(f"Typical slope {report['typical_slope_deg']}° is steeper than the "
                                  f"{' and '.join(too_steep)} can usually climb "
                                  f"({', '.join(f'{STEEP_FOR[robots]:g}°' for robots in too_steep)}).")
    if size_m < MIN_SIZE_M:
        report["warnings"].append(f"At {size_m:.1f} m across it is too small for a run with a goal "
                                  f"(at least {MIN_SIZE_M:g} m).")
    if heights.shape[0] < 16:
        report["warnings"].append(f"Only {heights.shape[0]} x {heights.shape[1]} height samples: detail will be coarse.")
    return report


def write_preset(out_dir: Path, name: str, heights: np.ndarray, meta: dict) -> dict:
    """Write <name>.npz (heights_m) and <name>.json (meta, with grid size and the Inspector's numbers)."""
    heights = heights.astype(np.float32)
    meta = {**meta, "grid": [int(heights.shape[0]), int(heights.shape[1])]}
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", heights_m=heights)
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta
