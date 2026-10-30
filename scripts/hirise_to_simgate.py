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
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simgate.terrain_import import TerrainError, geotiff_heights, inspect, write_preset  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "simgate" / "terrains"


def main() -> None:
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

    try:
        heights, mpp, details = geotiff_heights(a.input, a.size_m, a.pick, tuple(a.center), tuple(a.pixel) if a.pixel else None)
    except TerrainError as exc:
        raise SystemExit(str(exc)) from None
    report = inspect(heights, mpp)
    meta = {
        "name": a.name,
        "label": a.label or a.name,
        "kind": "real",
        "body": a.body,
        "surface": a.surface,
        "size_m": report["size_m"],
        "meters_per_pixel": report["meters_per_pixel"],
        "relief_m": report["relief_m"],
        "base_elevation_m": details["base_elevation_m"],
        "max_slope_deg": report["max_slope_deg"],
        "typical_slope_deg": report["typical_slope_deg"],
        "mean_slope_deg": report["mean_slope_deg"],
        "source": str(a.input),
        "source_shape": details["source_shape"],
        "window_top_left_px": details["window_top_left_px"],
        "window_origin_projected_m": details["window_origin_projected_m"],
        "crs": details["crs"],
        "credit": a.credit,
    }
    write_preset(a.out_dir, a.name, heights, meta)
    try:                                             # 16-bit preview image (north up), for galleries and docs
        from PIL import Image
        img = np.flipud(heights) / max(float(heights.max()), 1e-6) * 65535
        Image.fromarray(img.astype(np.uint16)).save(a.out_dir / f"{a.name}.png")
    except Exception:  # noqa: BLE001 - preview is optional
        pass
    row, col = details["window_top_left_px"]
    print(f"Saved preset '{a.name}': {report['size_m']:.1f} m square, {heights.shape[0]}x{heights.shape[1]} px at {mpp:.2f} m/px, "
          f"relief {meta['relief_m']} m, typical slope {meta['typical_slope_deg']}° (max {meta['max_slope_deg']}°) (window at row {row}, col {col}).")


if __name__ == "__main__":
    main()
