"""Import the supplied HiRISE, LROC and USGS terrain sources at native metre scale."""
import argparse
import json
import math
import struct
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import transform
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter, uniform_filter

from simgate.terrain_import import geotiff_heights, inspect, slope_deg, valid_mask, write_preset

MAX_TYPICAL_DEG = 20.0
MAX_SLOPE_DEG = 30.0
MARS_SITES = [
    ("DTEEC_001414_1780_001612_1780_U01.IMG", "victoria-opportunity", "Victoria Crater (Opportunity)", "PSP_001414_1780"),
    ("DTEEC_001513_1655_001777_1650_U01.IMG", "gusev-columbia-hills", "Gusev / Columbia Hills area (Spirit)", "PSP_001513_1655"),
    ("DTEEC_007060_1760_006915_1760_A01.IMG", "juventae-kieserite", "Juventae Chasma Layered Kieserite Deposit", "PSP_007060_1760"),
    ("DTEED_026399_2085_026254_2085_A01.IMG", "arabia-northeast-mesas", "Craters and Mesas in Northeast Arabia Terra", "ESP_026399_2085"),
    ("DTEED_053753_1715_053819_1715_A01.IMG", "ius-chasma-floor", "Light-Toned Deposits along Ius Chasma Floor", "ESP_053753_1715"),
]


def pds_lunar_elevation_metadata(source):
    import pvl

    lines, consumed = [], 0
    with source.open("rb") as stream:
        while consumed < 1024 * 1024:
            line = stream.readline(4096)
            if not line:
                raise ValueError(f"{source.name} has no complete attached PDS label.")
            lines.append(line)
            consumed += len(line)
            if line.strip() == b"END":
                break
        else:
            raise ValueError(f"{source.name} has an unexpectedly large attached label.")
    label = pvl.loads(b"".join(lines).decode("ascii"))
    image = label["IMAGE"]
    if label["PDS_VERSION_ID"] != "PDS3" or label["TARGET_NAME"] != "MOON":
        raise ValueError("Expected a PDS3 lunar terrain product.")
    if not str(label["PRODUCT_ID"]).startswith("NAC_DTM_"):
        raise ValueError("Expected an LROC NAC DTM product, not arbitrary floating-point image data.")
    if image["SAMPLE_BITS"] != 32 or image["SAMPLE_TYPE"] not in {"PC_REAL", "IEEE_REAL"}:
        raise ValueError("Lunar elevation IMG must contain 32-bit floating-point samples.")
    sample_unit = image.get("UNIT", image.get("SAMPLE_UNIT"))
    if sample_unit is not None and str(sample_unit).upper() not in {"M", "METER", "METERS", "METRE", "METRES"}:
        raise ValueError(f"Unsupported lunar elevation unit {sample_unit!r}.")
    null_bits = int(image["CORE_NULL"])
    null = struct.unpack(">f", struct.pack(">I", null_bits))[0]
    scale = float(image.get("SCALING_FACTOR", 1))
    offset = float(image.get("OFFSET", 0))
    if scale != 1 or offset != 0:
        raise ValueError("This lunar product requires non-identity scale/offset conversion.")
    with rasterio.open(source) as dataset:
        if dataset.dtypes != ("float32",) or dataset.scales[0] != scale or dataset.offsets[0] != offset:
            raise ValueError("GDAL interpretation differs from the attached elevation label.")
        if not math.isclose(dataset.nodata, null, rel_tol=1e-7):
            raise ValueError("GDAL no-data differs from the attached CORE_NULL value.")
    product = str(label["PRODUCT_ID"])
    site = product.rsplit("_", 1)[0].removeprefix("NAC_DTM_")
    readme = (f"https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/"
              f"DATA/SDP/NAC_DTM/{site}/NAC_DTM_{site}_README.TXT")
    return {"pds_product_id": product, "sample_type": str(image["SAMPLE_TYPE"]),
            "sample_bits": 32, "sample_unit_in_label": None if sample_unit is None else str(sample_unit),
            "elevation_units": "metres", "elevation_units_source": "LROC NAC DTM product convention",
            "scaling_factor": scale, "offset": offset, "scale_offset_explicit": "SCALING_FACTOR" in image or "OFFSET" in image,
            "core_null_bits": f"0x{null_bits:08X}", "source_nodata": null,
            "reference_radius_m": 1737400.0, "product_readme_url": readme,
            "vertical_reference": "LOLA-tied lunar elevations",
            "special_pixel_bits": {key: f"0x{int(value):08X}" for key, value in image.items()
                                   if key.startswith("CORE_") and isinstance(value, int)}}


def projected_site(dataset, longitude, latitude, body):
    geographic = CRS.from_epsg(4326) if body == "earth" else CRS.from_dict({"proj": "longlat", "R": 1737400, "no_defs": True})
    east, north = transform(geographic, dataset.crs, [longitude], [latitude])
    return east[0], north[0]


def rank_candidates(dataset, size_m, preference=None):
    scale = max(1, max(dataset.width, dataset.height) / 1024)
    rows, cols = max(8, round(dataset.height / scale)), max(8, round(dataset.width / scale))
    preview = dataset.read(1, out_shape=(rows, cols), resampling=Resampling.nearest).astype(float)
    valid = valid_mask(preview, dataset.nodata)
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = preview[tuple(nearest)]
    mpp = abs(dataset.transform.a)
    row_step, col_step = dataset.height / rows, dataset.width / cols
    pixels = round(size_m / mpp)
    kernel = max(2, math.ceil(pixels / min(row_step, col_step)))
    relief = maximum_filter(filled, size=kernel) - minimum_filter(filled, size=kernel)
    coverage = uniform_filter(valid.astype(float), size=kernel + 2, mode="constant")
    gradient_y, gradient_x = np.gradient(filled, mpp * row_step, mpp * col_step)
    slopes = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
    mask = (coverage >= 0.995) & (maximum_filter(slopes, size=kernel) <= MAX_TYPICAL_DEG * 0.8)
    border = math.ceil(pixels / min(row_step, col_step) / 2) + 2
    mask[:border] = mask[-border:] = False
    mask[:, :border] = mask[:, -border:] = False
    row_grid, col_grid = np.indices(preview.shape)
    centre_rows, centre_cols = (row_grid + 0.5) * row_step, (col_grid + 0.5) * col_step
    score = relief.copy()
    if preference:
        point, inner, outer, mode = preference
        source_row, source_col = dataset.index(*point)
        distance = np.hypot((centre_rows - source_row) * mpp, (centre_cols - source_col) * mpp)
        mask &= (distance >= inner) & (distance <= outer)
        if mode == "floor":
            candidates = filled[mask]
            if candidates.size:
                mask &= filled <= np.percentile(candidates, 40)
            score = relief - 0.02 * distance / mpp
        elif mode == "rille":
            score = relief + 0.5 * slopes
        elif mode == "landing":
            score = -distance - slopes * 10
    ranked = np.argsort(np.where(mask, score, -np.inf).ravel())[::-1]
    used = set()
    locations = []
    for flat_index in ranked:
        if not mask.ravel()[flat_index]:
            break
        row_index, col_index = np.unravel_index(flat_index, mask.shape)
        row = round(centre_rows[row_index, col_index] - pixels / 2)
        col = round(centre_cols[row_index, col_index] - pixels / 2)
        key = (row // max(1, pixels // 2), col // max(1, pixels // 2))
        if key in used:
            continue
        used.add(key)
        if 0 <= row <= dataset.height - pixels and 0 <= col <= dataset.width - pixels:
            locations.append((row, col))
        if len(locations) == 800:
            break
    return locations, preview, valid


def choose_patch(path, size_m, preference=None):
    with rasterio.open(path) as dataset:
        if dataset.count != 1 or np.dtype(dataset.dtypes[0]).itemsize == 1:
            raise ValueError(f"{path.name} is an uncalibrated 8-bit/browse raster, not a numeric elevation DTM; "
                             "relief and slopes cannot be interpreted in metres.")
        if dataset.scales[0] != 1 or dataset.offsets[0] != 0:
            raise ValueError(f"{path.name} needs explicit vertical scale/offset handling before elevation import.")
        if not dataset.crs or not dataset.crs.is_projected or dataset.crs.linear_units.lower() not in {"metre", "meter", "m", "metres", "meters"}:
            raise ValueError(f"{path.name} is not a projected metre raster.")
        if abs(dataset.transform.b) > 1e-9 or abs(dataset.transform.d) > 1e-9 or dataset.transform.e >= 0:
            raise ValueError(f"{path.name} requires orientation/reprojection handling before import.")
        locations, preview, valid = rank_candidates(dataset, size_m, preference)
    for row, col in locations:
        try:
            heights, mpp, details = geotiff_heights(str(path), size_m, pixel=(row, col), require_metric=True)
        except ValueError:
            continue
        slopes = slope_deg(heights, mpp)
        if np.percentile(slopes, 95) <= MAX_TYPICAL_DEG and slopes.max() <= MAX_SLOPE_DEG:
            return heights, mpp, details, preview, valid
    raise ValueError(f"No native-resolution crossable patch found for {path.name} in the selected area.")


def shaded_image(heights, valid, body):
    nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = heights[tuple(nearest)]
    low, high = np.percentile(filled[valid], [1, 99])
    normalized = np.clip((filled - low) / max(high - low, 1e-6), 0, 1)
    palettes = {"mars": ([87, 70, 61], [215, 170, 126]), "moon": ([67, 76, 84], [225, 231, 235]),
                "earth": ([37, 91, 94], [227, 224, 168])}
    dark, light = (np.asarray(color) for color in palettes[body])
    gradient_y, gradient_x = np.gradient(filled)
    illumination = np.clip(0.85 + 0.25 * (-gradient_x + gradient_y) / np.sqrt(1 + gradient_x**2 + gradient_y**2), 0.5, 1.15)
    rgb = (dark + normalized[..., None] * (light - dark)) * illumination[..., None]
    rgb[~valid] = [241, 242, 244]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def write_preview(path, label, body, heights, preview, valid, details, report, source_shape, note):
    canvas = Image.new("RGB", (1040, 640), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("segoeui.ttf", 18)
        small = ImageFont.truetype("segoeui.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    draw.text((20, 16), label, fill="#202830", font=font)
    context = shaded_image(preview, valid, body)
    context.thumbnail((480, 470))
    offset_x, offset_y = 20 + (480 - context.width) // 2, 58 + (470 - context.height) // 2
    canvas.paste(context, (offset_x, offset_y))
    row, col = details["window_top_left_px"]
    pixels = details["window_size_px"]
    left = offset_x + col / source_shape[1] * context.width
    top = offset_y + row / source_shape[0] * context.height
    right, bottom = left + pixels / source_shape[1] * context.width, top + pixels / source_shape[0] * context.height
    draw.rectangle((left, top, right, bottom), outline="#ff3d38", width=2)
    draw.ellipse((left - 4, top - 4, left + 4, top + 4), outline="#ff3d38", width=1)
    patch = shaded_image(np.flipud(heights), np.ones(heights.shape, dtype=bool), body).resize((480, 470), Image.Resampling.NEAREST)
    canvas.paste(patch, (540, 58))
    draw.text((20, 535), "Source context (north up); red mark = selected patch", fill="#34434b", font=small)
    draw.text((540, 535), "Native-grid patch (north up); no new detail added", fill="#34434b", font=small)
    draw.text((20, 565), f"{report['meters_per_pixel']:.4f} m/px | {report['size_m']:.3f} m square | relief {report['relief_m']:.3f} m | slope P95 {report['typical_slope_deg']:.1f} deg", fill="#202830", font=font)
    draw.text((20, 595), note[:135], fill="#59646d", font=small)
    canvas.save(path)


def import_site(source, name, site, body, size_m, output, previews, preference=None, note="", site_url=None):
    vertical = pds_lunar_elevation_metadata(source) if body == "moon" and source.suffix.upper() == ".IMG" else {}
    heights, mpp, details, context, valid = choose_patch(source, size_m, preference)
    report = inspect(heights, mpp)
    uncontrolled = body == "mars" and "_U01" in source.stem
    label = f"{site}, {body.title()}" + (" (U01 uncontrolled)" if uncontrolled else "")
    credit = {"mars": "NASA/JPL-Caltech/University of Arizona HiRISE DTM", "moon": "NASA/GSFC/Arizona State University LROC NAC DTM",
              "earth": "USGS 3DEP 1 m DEM, AZ NorthEast D23"}[body]
    accuracy = "U01 uncontrolled: absolute position less precise; native-scale heights retained." if uncontrolled else ""
    metadata = {**report, **details, **vertical, "name": name, "label": label, "site": site, "kind": "real", "body": body,
                "surface": "regolith" if body in {"mars", "moon"} else "rock", "source": str(source.resolve()),
                "source_format": source.suffix.upper(), "source_page": site_url, "credit": credit,
                "accuracy_note": accuracy, "selection_note": note,
                "crossability_screen": {"typical_slope_limit_deg": MAX_TYPICAL_DEG, "max_slope_limit_deg": MAX_SLOPE_DEG,
                                         "verified_simulation": False}}
    if preference is not None:
        with rasterio.open(source) as dataset:
            row, col = details["window_top_left_px"]
            east, north = dataset.xy(row + (details["window_size_px"] - 1) / 2,
                                     col + (details["window_size_px"] - 1) / 2)
        metadata["selection_distance_from_anchor_m"] = round(math.dist((east, north), preference[0]), 2)
        metadata["selection_anchor_projected_m"] = list(preference[0])
    if body == "moon" and source.with_suffix(".TIF").is_file():
        metadata["visual_reference_source"] = str(source.with_suffix(".TIF").resolve())
        metadata["visual_reference_role"] = "Browse/shaded reference only; never sampled as elevation."
        metadata["source_page"] = f"https://data.lroc.im-ldi.com/lroc/view_rdr/{vertical['pds_product_id'].rsplit('_', 1)[0]}"
    image = previews / f"{name}.png"
    write_preview(image, label, body, heights, context, valid, details, report, details["source_shape"], accuracy or note)
    metadata["preview"] = str(image)
    metadata["preview_file"] = f"{name}.preview.png"
    shaded_image(np.flipud(heights), np.ones(heights.shape, dtype=bool), body).resize(
        (480, 480), Image.Resampling.NEAREST).save(output / metadata["preview_file"])
    write_preset(output, name, heights, metadata)
    print(f"{name}: {mpp:.4f} m/px, {report['size_m']:.3f} m, relief {report['relief_m']:.3f} m, P95 {report['typical_slope_deg']:.1f} deg", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("assets/nasa/terrain_raw"))
    parser.add_argument("--output", type=Path, default=Path("simgate/terrains"))
    parser.add_argument("--report", type=Path, default=Path("reports/local-terrain-sites"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    results, notes, blocked = [], [], []
    for filename, name, site, image_id in MARS_SITES:
        results.append(import_site(args.root / "mars" / filename, name, site, "mars", 80, args.output, args.report,
                                   site_url=f"https://www.uahirise.org/dtm/dtm.php?ID={image_id}"))
    lunar = [("NAC_DTM_APOLLO15_2_E260N0036.IMG", "apollo15-hadley-rille", "Apollo 15 / Hadley Rille edge", 3.583, 26.13222, "rille", 2000),
             ("NAC_DTM_APOLLO17_E203N0308.IMG", "apollo17-taurus-littrow", "Apollo 17 / Taurus-Littrow valley floor", 30.77168, 20.1908, "landing", 1500)]
    for filename, name, site, lon, lat, mode, radius in lunar:
        source = args.root / "moon" / filename
        with rasterio.open(source) as dataset:
            location = projected_site(dataset, lon, lat, "moon")
            if dataset.count != 1 or np.dtype(dataset.dtypes[0]).itemsize == 1:
                reason = "8-bit grayscale browse/shaded-relief image with no calibrated elevation scale; numeric DTM required."
                preview = dataset.read(1, out_shape=(max(8, round(dataset.height / max(dataset.shape) * 720)),
                                                     max(8, round(dataset.width / max(dataset.shape) * 720))))
                image = Image.fromarray(preview.astype(np.uint8)).convert("RGB")
                canvas = Image.new("RGB", (image.width, image.height + 80), "white")
                canvas.paste(image, (0, 80))
                draw = ImageDraw.Draw(canvas)
                draw.text((8, 8), site, fill="black")
                draw.text((8, 28), "BROWSE IMAGE ONLY - not metre heights; no relief/slope claim", fill="black")
                canvas.save(args.report / f"{name}.png")
                blocked.append({"name": name, "site": site, "meters_per_pixel": abs(dataset.transform.a),
                                "requested_patch_m": 160, "reason": reason, "relief_m": None,
                                "typical_slope_deg": None, "preview": f"{name}.png"})
                metadata_path = args.output / f"{name}.json"
                if metadata_path.is_file() and json.loads(metadata_path.read_text()).get("source") == str(source.resolve()):
                    for suffix in (".npz", ".json", ".preview.png"):
                        generated = args.output / f"{name}{suffix}"
                        if generated.is_file():
                            generated.unlink()
                notes.append(f"{site}: blocked - {reason}")
                continue
        try:
            result = import_site(source, name, site, "moon", 160, args.output, args.report,
                                 (location, 0, radius, mode), note=f"Near requested {mode} area; 5 m cells do not resolve rover-scale hazards.")
        except ValueError as error:
            fallback = f"Preferred lunar area unavailable: {error}; selected most interesting crossable patch in this DTM."
            notes.append(fallback)
            result = import_site(source, name, site + " (nearby fallback)", "moon", 160, args.output, args.report, note=fallback)
        results.append(result)
    earth = args.root / "earth" / "USGS_1M_12_x49y388_AZ_NorthEast_D23.tif"
    with rasterio.open(earth) as dataset:
        centre = projected_site(dataset, -111.022, 35.027, "earth")
        contains = dataset.bounds.left <= centre[0] <= dataset.bounds.right and dataset.bounds.bottom <= centre[1] <= dataset.bounds.top
        notes.append(f"Meteor Crater centre in EPSG:26912: E {centre[0]:.3f}, N {centre[1]:.3f}; inside tile: {contains}.")
    if contains:
        for part, inner, outer, mode in (("rim", 500, 850, "rim"), ("floor", 0, 350, "floor")):
            try:
                results.append(import_site(earth, f"meteor-crater-{part}", f"Meteor Crater {part}", "earth", 80,
                    args.output, args.report, (centre, inner, outer, mode), note=f"Slope-screened {part} patch; not a certified rover route."))
            except ValueError as error:
                notes.append(str(error))
    else:
        results.append(import_site(earth, "az-northeast-crossable", "Arizona NorthEast tile (Meteor Crater outside tile)", "earth", 80,
                                   args.output, args.report, note="Meteor Crater is not covered; selected most interesting crossable tile patch."))
    (args.report / "report.json").write_text(json.dumps({"presets": results, "blocked": blocked, "notes": notes}, indent=2), encoding="utf-8")
    lines = ["# Local terrain presets", "", *notes, "", "Slope screening: P95 <=20 deg, maximum <=30 deg; no route simulation.", "",
             "| Preset / site | m/px | Patch m | Relief m | P95 slope deg | Preview |", "|---|---:|---:|---:|---:|---|"]
    for result in results:
        lines.append(f"| {result['name']}: {result['site']} | {result['meters_per_pixel']:.4f} | {result['size_m']:.3f} | {result['relief_m']:.3f} | {result['typical_slope_deg']:.1f} | ![Preview]({result['name']}.png) |")
    lines.extend(["", "## Lunar source validation", "",
        "Apollo 15/17 elevation comes only from the attached-label PDS3 IMG DTMs: PC_REAL, 32-bit float, 5 m/pixel.",
        "The IMAGE labels omit an explicit vertical UNIT and scale/offset. Metres follow the LROC NAC DTM/LOLA product convention;",
        "the identity scale 1 and offset 0 are defaults verified against GDAL, not invented label entries.",
        "CORE_NULL is 0xFF7FFFFB (~-3.402822655e38); other ISIS special pixels are excluded as no-data.",
        "The supplied byte TIFFs remain local visual references and are never read as elevations.",
        "Both 160 m lunar patches contain 32 x 32 native samples; orbital data does not resolve sub-5 m hazards."])
    if blocked:
        lines.extend(["", "## Sources requiring numeric elevation data", ""])
        for source in blocked:
            lines.extend([f"### {source['site']}", f"{source['meters_per_pixel']:g} m/px map scale; requested patch {source['requested_patch_m']:g} m. "
                          f"Not imported: {source['reason']} Relief and slope are unavailable.",
                          f"![Browse reference only]({source['preview']})", ""])
    (args.report / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()