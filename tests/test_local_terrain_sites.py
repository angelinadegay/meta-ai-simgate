import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from rasterio.windows import Window
from PIL import Image

from scripts.import_local_terrain_sites import choose_patch, import_site, projected_site, pds_lunar_elevation_metadata


def write_raster(path, dtype="float32"):
    rows, cols = np.indices((128, 128))
    heights = (100 + rows * 0.04 + cols * 0.1).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=1, dtype=dtype,
                       crs="EPSG:26912", transform=from_origin(490000, 3880000, 1, 1)) as dataset:
        dataset.write(heights, 1)


def test_native_patch_preserves_source_samples_and_metre_scale(tmp_path):
    source = tmp_path / "source.tif"
    write_raster(source)
    heights, mpp, details, _, _ = choose_patch(source, 40)
    row, col = details["window_top_left_px"]
    with rasterio.open(source) as dataset:
        raw = dataset.read(1, window=Window(col, row, 40, 40))
    assert mpp == 1 and heights.shape == (40, 40)
    assert np.allclose(heights, np.flipud(raw - raw.min()), atol=1e-6)


def test_byte_browse_raster_is_not_accepted_as_elevation(tmp_path):
    source = tmp_path / "browse.tif"
    write_raster(source, "uint8")
    with pytest.raises(ValueError, match="browse raster"):
        choose_patch(source, 40)


def test_meteor_crater_coordinate_transform_matches_tile_location(tmp_path):
    source = tmp_path / "source.tif"
    write_raster(source)
    with rasterio.open(source) as dataset:
        east, north = projected_site(dataset, -111.022, 35.027, "earth")
    assert east == pytest.approx(497993, abs=10)
    assert north == pytest.approx(3876037, abs=10)


def test_imported_preview_and_u01_accuracy_note_are_on_cards(tmp_path, monkeypatch):
    from simgate import terrain_data
    from simgate.library import LibraryRegistry
    import server

    source = tmp_path / "DTEEC_TEST_U01.tif"
    write_raster(source)
    output, previews = tmp_path / "presets", tmp_path / "previews"
    output.mkdir()
    previews.mkdir()
    result = import_site(source, "test-native", "Native test site", "mars", 40, output, previews)
    metadata = json.loads((output / "test-native.json").read_text())
    assert metadata["preview_file"] == "test-native.preview.png"
    assert (output / metadata["preview_file"]).is_file()
    assert "absolute position less precise" in result["accuracy_note"]
    monkeypatch.setattr(terrain_data, "TERRAIN_DIR", output)
    card = next(card for card in LibraryRegistry().worlds() if card.terrain_preset == "test-native")
    assert "U01 uncontrolled" in card.name and "absolute position less precise" in card.credit
    view = server.library_view()
    world = next(world for world in view["worlds"] if world["terrain_preset"] == "test-native")
    assert world["preview_url"] == "/terrain-preview/test-native"
    response = TestClient(server.app).get(world["preview_url"])
    assert response.status_code == 200 and response.content.startswith(b"\x89PNG")
    assert terrain_data.load_ortho_image("test-native") is None
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    metadata["preview_file"] = "../outside.png"
    (output / "test-native.json").write_text(json.dumps(metadata))
    assert terrain_data.load_preview_image("test-native") is None


def write_pds_elevation(path, extra="", bits=32):
    label = f"""PDS_VERSION_ID = PDS3
RECORD_TYPE = FIXED_LENGTH
RECORD_BYTES = 128
FILE_RECORDS = 40
LABEL_RECORDS = 8
^IMAGE = 9
PRODUCT_ID = NAC_DTM_TEST_E0N0
TARGET_NAME = MOON
OBJECT = IMAGE_MAP_PROJECTION
MAP_PROJECTION_TYPE = EQUIRECTANGULAR
CENTER_LATITUDE = 0
CENTER_LONGITUDE = 180
A_AXIS_RADIUS = 1737.4 <KM>
B_AXIS_RADIUS = 1737.4 <KM>
C_AXIS_RADIUS = 1737.4 <KM>
MAP_SCALE = 5 <METERS/PIXEL>
LINE_PROJECTION_OFFSET = 0.5
SAMPLE_PROJECTION_OFFSET = 0.5
END_OBJECT = IMAGE_MAP_PROJECTION
OBJECT = IMAGE
LINES = 32
LINE_SAMPLES = 32
SAMPLE_TYPE = PC_REAL
SAMPLE_BITS = {bits}
BANDS = 1
BAND_STORAGE_TYPE = BAND_SEQUENTIAL
CORE_NULL = 16#FF7FFFFB#
{extra}
END_OBJECT = IMAGE
END
"""
    rows, cols = np.indices((32, 32))
    data = (100 + rows * 0.1 + cols * 0.2).astype("<f4")
    data[0, 0] = np.float32(-3.4028226550889045e38)
    encoded = label.encode("ascii")
    assert len(encoded) <= 1024
    path.write_bytes(encoded.ljust(1024, b" ") + data.tobytes())


def test_pds_lunar_label_records_defaults_and_null_pixels(tmp_path):
    source = tmp_path / "test.IMG"
    write_pds_elevation(source)
    metadata = pds_lunar_elevation_metadata(source)
    assert metadata["sample_type"] == "PC_REAL" and metadata["sample_bits"] == 32
    assert metadata["sample_unit_in_label"] is None
    assert metadata["scaling_factor"] == 1 and metadata["offset"] == 0
    assert not metadata["scale_offset_explicit"]
    assert metadata["core_null_bits"] == "0xFF7FFFFB"
    with rasterio.open(source) as dataset:
        assert dataset.dtypes == ("float32",)
        from simgate.terrain_import import valid_mask
        assert not valid_mask(dataset.read(1), dataset.nodata)[0, 0]


@pytest.mark.parametrize("extra,bits,message", [
    ("", 16, "32-bit"), ("UNIT = KM", 32, "elevation unit"),
    ("SCALING_FACTOR = 2", 32, "scale/offset"), ("OFFSET = 100", 32, "scale/offset"),
])
def test_pds_lunar_label_rejects_incompatible_elevation_metadata(tmp_path, extra, bits, message):
    source = tmp_path / "test.IMG"
    write_pds_elevation(source, extra, bits)
    with pytest.raises(ValueError, match=message):
        pds_lunar_elevation_metadata(source)


@pytest.mark.parametrize("name", ["apollo15-hadley-rille", "apollo17-taurus-littrow"])
def test_imported_lunar_preset_matches_actual_native_img(name):
    root = Path(__file__).resolve().parents[1]
    metadata_path = root / "simgate" / "terrains" / f"{name}.json"
    if not metadata_path.is_file():
        pytest.skip("Imported lunar preset is not present.")
    metadata = json.loads(metadata_path.read_text())
    source = Path(metadata["source"])
    if not source.is_file():
        pytest.skip("Raw source remains local and is not committed.")
    heights = np.load(metadata_path.with_suffix(".npz"))["heights_m"]
    with rasterio.open(source) as dataset:
        row, col = metadata["window_top_left_px"]
        raw = dataset.read(1, window=Window(col, row, metadata["window_size_px"], metadata["window_size_px"]))
        assert dataset.dtypes == ("float32",)
        assert abs(dataset.transform.a) == pytest.approx(5)
        assert heights.shape == raw.shape == (32, 32)
        assert np.array_equal(heights, np.flipud(raw - raw.min()))
    from simgate.terrain_import import slope_deg
    slope = slope_deg(heights, 5)
    assert np.percentile(slope, 95) <= 20 and slope.max() <= 30
    assert metadata["source_format"] == ".IMG"
    assert metadata["visual_reference_source"].endswith(".TIF")
    assert "ortho_file" not in metadata
    assert metadata["selection_distance_from_anchor_m"] <= (2000 if name.startswith("apollo15") else 1500)
    assert (metadata_path.parent / metadata["preview_file"]).is_file()