#!/usr/bin/env python3
"""Turn a NASA HiRISE elevation model (DTM GeoTIFF) into a SimGate terrain preset.

It crops a small square at FULL resolution (about 1 m per pixel), so the
terrain is robot-sized, unlike resampling the whole multi-km strip.

The input can be a local file or a URL. HiRISE DTMs on AWS are Cloud-Optimized
GeoTIFFs, so with a URL only the needed window is downloaded.

Examples
  # Jezero Crater, straight from NASA/USGS public data on AWS (no account needed):
  python scripts/hirise_to_simgate.py \\
      --input https://astrogeo-ard.s3.us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/ESP_045994_1985_ESP_046060_1985/DTEEC_045994_1985_046060_1985_U01.tif \\
      --name jezero --label "Jezero Crater, Mars" --size-m 40 --pick relief

  # A file you downloaded:
  python scripts/hirise_to_simgate.py --input DTEEC_045994_1985_046060_1985_U01.tif --name jezero

Data credit: NASA/JPL/University of Arizona HiRISE; DTMs released via USGS on AWS
Open Data under CC0 1.0 (public domain).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "simgate" / "terrains"
MAX_TYPICAL_SLOPE_DEG = 22.0  # --pick relief skips windows whose 95th-percentile slope is steeper (robots can't cross)
MAX_CANDIDATES = 40           # full-resolution reads to try before giving up


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
    raise SystemExit(f"No suitable window in the first {tried} candidates. Try --pick center, a smaller --size-m, or another file.")


def main() -> None:
    import rasterio

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="HiRISE DTM GeoTIFF: local path or https URL")
    p.add_argument("--name", required=True, help="preset name, e.g. jezero")
    p.add_argument("--label", help='display name, e.g. "Jezero Crater, Mars"')
    p.add_argument("--size-m", type=float, default=40.0, help="side of the square crop in metres (default 40)")
    p.add_argument("--pick", choices=["relief", "center"], default="relief",
                   help="relief = most interesting valid window (default); center = nearest valid to --center")
    p.add_argument("--center", type=float, nargs=2, default=(0.5, 0.5), metavar=("X_FRAC", "Y_FRAC"),
                   help="for --pick center: position as fractions of width and height (0..1)")
    p.add_argument("--pixel", type=int, nargs=2, metavar=("ROW", "COL"), help="exact top-left pixel; skips --pick")
    p.add_argument("--body", default="mars", help="which world the data is from (sets gravity), default mars")
    p.add_argument("--surface", default="regolith", help="surface type for friction, default regolith")
    p.add_argument("--credit", default="NASA/JPL/University of Arizona HiRISE; DTM via USGS on AWS Open Data (CC0 1.0)",
                   help="data credit shown to users; change it for non-HiRISE or test data")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = p.parse_args()

    with rasterio.open(a.input) as ds:
        mpp = abs(ds.transform.a)
        if abs(abs(ds.transform.e) - mpp) / mpp > 0.01:
            raise SystemExit(f"Non-square pixels ({ds.transform.a} x {ds.transform.e}); not supported yet.")
        win_px = int(round(a.size_m / mpp))
        if win_px < 8 or win_px > min(ds.height, ds.width):
            raise SystemExit(f"--size-m {a.size_m} is {win_px} px; the file is {ds.height} x {ds.width} px at {mpp:.3f} m/px.")
        if a.pixel:
            row, col = a.pixel
            raw, ok = read_window(ds, row, col, win_px)
        else:
            row, col, raw, ok = pick_window(ds, win_px, a.pick, tuple(a.center))
        x0, y0 = ds.transform * (col, row)
        crs = ds.crs.to_string() if ds.crs else None
        source_shape = [ds.height, ds.width]

    valid_frac = float(ok.mean())
    if valid_frac < 0.98:
        raise SystemExit(f"Window is only {valid_frac:.0%} valid data. Move it (--pick center / --pixel) or shrink --size-m.")
    if valid_frac < 1.0:                             # fill rare holes with the nearest valid height
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~ok, return_distances=False, return_indices=True)
        raw = raw[tuple(idx)]

    base = float(raw.min())
    heights = np.flipud(raw - base).astype(np.float32)   # GeoTIFF row 0 is NORTH; SimGate row 0 is SOUTH
    slopes = slope_deg(heights.astype(np.float64), mpp)
    size_m = win_px * mpp
    meta = {
        "name": a.name,
        "label": a.label or a.name,
        "kind": "real",
        "body": a.body,
        "surface": a.surface,
        "size_m": round(size_m, 3),
        "meters_per_pixel": round(mpp, 4),
        "grid": [int(heights.shape[0]), int(heights.shape[1])],
        "relief_m": round(float(heights.max()), 3),
        "base_elevation_m": round(base, 2),
        "max_slope_deg": round(float(slopes.max()), 1),
        "typical_slope_deg": round(float(np.percentile(slopes, 95)), 1),
        "mean_slope_deg": round(float(slopes.mean()), 1),
        "source": str(a.input),
        "source_shape": source_shape,
        "window_top_left_px": [int(row), int(col)],
        "window_origin_projected_m": [round(x0, 2), round(y0, 2)],
        "crs": crs,
        "credit": a.credit,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out_dir / f"{a.name}.npz", heights_m=heights)
    (a.out_dir / f"{a.name}.json").write_text(json.dumps(meta, indent=2))
    try:                                             # 16-bit preview image (north up), for galleries and docs
        from PIL import Image
        img = np.flipud(heights) / max(float(heights.max()), 1e-6) * 65535
        Image.fromarray(img.astype(np.uint16)).save(a.out_dir / f"{a.name}.png")
    except Exception:  # noqa: BLE001 - preview is optional
        pass
    print(f"Saved preset '{a.name}': {size_m:.1f} m square, {heights.shape[0]}x{heights.shape[1]} px at {mpp:.2f} m/px, "
          f"relief {meta['relief_m']} m, typical slope {meta['typical_slope_deg']}° (max {meta['max_slope_deg']}°) (window at row {row}, col {col}).")


if __name__ == "__main__":
    main()
