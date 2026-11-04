"""World-card thumbnails for the home page (visual only: nothing here touches physics).

The shipped worlds get 3D captures from the app's own renderer (scripts/build_thumbnails.py), saved in
static/thumbnails/<card key>.webp. When a capture fails, and for uploaded terrains (which aren't shipped), a smooth
shaded-relief image stands in: a hillshade blended with the world's colour ramp, upscaled bilinearly so no pixel
of the heightmap shows.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

THUMBNAILS_DIR = Path(__file__).resolve().parent.parent / "static" / "thumbnails"
SIZE = (800, 500)
# The home cards' plain titles; the full source label and accuracy notes stay in the card details.
CARD_TITLES = {
    "gusev-columbia-hills": "Gusev Crater: Columbia Hills (Spirit)",
    "victoria-opportunity": "Victoria Crater (Opportunity)",
    "apollo15-hadley-rille": "Hadley Rille (Apollo 15)",
    "apollo17-taurus-littrow": "Taurus-Littrow Valley (Apollo 17)",
    "meteor-crater-rim": "Meteor Crater rim (Arizona)",
    "meteor-crater-floor": "Meteor Crater floor (Arizona)",
    "ius-chasma-floor": "Ius Chasma",
    "arabia-northeast-mesas": "Arabia Terra mesas",
    "juventae-kieserite": "Juventae",
    "gale": "Gale Crater",
    "jezero": "Jezero Crater",
    "ares3": "Acidalia Planitia (Ares 3)",
}
# low → high ground, per world (the same families as the app's terrain materials)
RAMPS = {
    "mars": ["#5b2f22", "#8f4a31", "#b9734c", "#d9a57a", "#efd2b0"],
    "moon": ["#2d2f36", "#55585f", "#7f8188", "#a9aab0", "#d4d4d8"],
    "earth": ["#3e4f3a", "#6b7650", "#9c8f66", "#c3ad85", "#e6dcc6"],
    "titan": ["#3f2c17", "#6e4d26", "#9a7038", "#c29a5a", "#e0c690"],
    "europa": ["#5a6470", "#808a95", "#a8b1ba", "#cdd3d9", "#eef1f4"],
    "enceladus": ["#6d7a86", "#93a0ab", "#b7c2cb", "#d6dde3", "#f2f5f7"],
    "indoor": ["#8a6a4a", "#a27f5a", "#b9946b", "#cfae86", "#e4cfb1"],
}


def thumbnail_file(key: str) -> Path:
    return THUMBNAILS_DIR / f"{key}.webp"


def thumbnail_url(key: str) -> str | None:
    """The shipped thumbnail for a home card key ("real-gale", "mars-hills", "apartment"), if there is one."""
    return f"/static/thumbnails/{key}.webp" if thumbnail_file(key).is_file() else None


def _ramp(values: np.ndarray, colours: list[str]) -> np.ndarray:
    stops = np.array([[int(colour[i:i + 2], 16) for i in (1, 3, 5)] for colour in colours], dtype=float) / 255
    position = np.clip(values, 0, 1) * (len(stops) - 1)
    low = np.floor(position).astype(int).clip(0, len(stops) - 2)
    t = (position - low)[..., None]
    return stops[low] * (1 - t) + stops[low + 1] * t


def shaded_relief(heights: np.ndarray, size_m: float, body: str = "mars", size: tuple[int, int] = SIZE,
                  azimuth_deg: float = 315.0, altitude_deg: float = 40.0):
    """A smooth shaded-relief PIL image: the grid is upscaled bilinearly first, then lit and coloured, so the
    heightmap's cells never show as blocks."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter, zoom

    grid = np.asarray(heights, dtype=float)
    width, height = size
    # crop the square grid to the card's aspect, then upsample to the card's size (bilinear: order=1)
    rows = max(2, int(round(grid.shape[0] * height / width)))
    top = (grid.shape[0] - rows) // 2
    grid = grid[top:top + rows, :]
    fine = zoom(grid, (height / grid.shape[0], width / grid.shape[1]), order=1)
    fine = gaussian_filter(fine, sigma=max(1.0, width / grid.shape[1] / 4))   # soften cell edges left by the upscale
    cell = size_m / width
    exaggeration = 2.0 if np.ptp(fine) < 0.05 * size_m else 1.0   # flat floors still show their shape
    dy, dx = np.gradient(fine * exaggeration, cell, cell)
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    azimuth, altitude = np.radians(azimuth_deg), np.radians(altitude_deg)
    shade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    low, high = np.percentile(shade, [2, 98])
    shade = np.clip((shade - low) / ((high - low) or 1.0), 0, 1)   # each terrain's own light range
    span = np.ptp(fine) or 1.0
    colour = _ramp((fine - fine.min()) / span, RAMPS.get(body, RAMPS["mars"]))
    rgb = colour * (0.35 + 0.95 * shade[..., None])
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[::-1], "RGB")   # north up
    return image


def webp_bytes(image, quality: int = 78) -> bytes:
    out = io.BytesIO()
    image.save(out, "WEBP", quality=quality, method=6)
    return out.getvalue()
