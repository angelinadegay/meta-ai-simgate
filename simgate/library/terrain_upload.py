"""Turn an uploaded terrain into a preset: a GeoTIFF goes through the HiRISE converter path (patch picking, hole
filling, orientation); a PNG/JPG heightmap uses the size and height range the user gives. The world and
surface are the user's; anything missing is asked for instead of guessed."""

from pathlib import Path, PurePosixPath

from .. import presets
from ..terrain_import import (TerrainError, geotiff_heights, geotiff_rgb_crop, heightmap_heights, inspect,
                              write_ortho_image, write_preset)
from .uploads import UploadError, UploadRecord

GEOTIFF, IMAGE = (".tif", ".tiff"), (".png", ".jpg")
DEFAULT_GEOTIFF_SIZE_M = 40.0
SIZE_RANGE_M, HEIGHT_RANGE_M = (1.0, 1000.0), (0.0, 500.0)
IMAGE_ORIENTATION = "The image's top edge is north (+y) and its left edge west (-x); black is the lowest point."


class NeedsAnswers(UploadError):
    """The upload can't become a terrain until the user answers these questions."""

    def __init__(self, questions: list[dict]) -> None:
        self.questions = questions
        super().__init__(" ".join(question["question"] for question in questions))


def prepare_terrain(record: UploadRecord, body: str | None, surface: str | None, size_m: float | None = None,
                    height_range_m: float | None = None) -> dict:
    """Write preset.npz/preset.json into the upload's folder and return the preset's metadata, including the
    Inspector's report. Raises NeedsAnswers for missing choices and UploadError for unusable data."""
    folder = Path(record.folder)
    source = _main_file(record)
    is_image = PurePosixPath(source).suffix.lower() in IMAGE
    questions = []
    if body not in presets.GRAVITY:
        questions.append({"field": "body", "question": "Which world is this terrain from? It sets the gravity.",
                          "choices": list(presets.GRAVITY)})
    if surface not in presets.SURFACE_FRICTION:
        questions.append({"field": "surface", "question": "What is the ground like? It sets the friction.",
                          "choices": list(presets.SURFACE_FRICTION)})
    if is_image and not _within(size_m, SIZE_RANGE_M):
        questions.append({"field": "size_m", "question": "How wide is the heightmap, in metres "
                          f"({SIZE_RANGE_M[0]:g} to {SIZE_RANGE_M[1]:g})?"})
    if is_image and not _within(height_range_m, HEIGHT_RANGE_M):
        questions.append({"field": "height_range_m", "question": "How many metres higher is white than black "
                          f"(the height range, {HEIGHT_RANGE_M[0]:g} to {HEIGHT_RANGE_M[1]:g})?"})
    if questions:
        raise NeedsAnswers(questions)

    try:
        if is_image:
            heights, mpp = heightmap_heights((folder / source).read_bytes(), size_m, height_range_m)
            details = {"orientation": IMAGE_ORIENTATION}
        else:
            heights, mpp, details = geotiff_heights(str(folder / source), size_m or DEFAULT_GEOTIFF_SIZE_M,
                                                    require_metric=True)
    except TerrainError as exc:
        raise UploadError(str(exc)) from exc
    report = inspect(heights, mpp)
    meta = {"name": record.id, "label": record.name, "kind": "real", "body": body, "surface": surface,
            **{key: value for key, value in report.items() if key != "warnings"},
            **details, "source": f"upload: {source}", "credit": record.credit, "license": record.license,
            "uploaded": True, "inspection": {"by": "Inspector", "warnings": report["warnings"]}}
    if not is_image:
        for ortho in _ortho_candidates(record, source):
            rgb = geotiff_rgb_crop(str(folder / source), str(folder / ortho),
                                   *details["window_top_left_px"], details["window_size_px"], details["source_shape"])
            if rgb is None:
                continue
            meta["ortho_file"] = write_ortho_image(folder, "preset", rgb)
            meta["ortho_source"] = f"upload: {ortho}"
            meta["ortho_credit"] = record.credit
            break
    return write_preset(folder, "preset", heights, meta)


def _main_file(record: UploadRecord) -> str:
    for suffixes in (GEOTIFF, IMAGE):
        found = [path for path in record.files if PurePosixPath(path).suffix.lower() in suffixes]
        if found:
            return found[0]
    raise UploadError("A terrain upload needs a GeoTIFF (.tif) elevation model or a PNG/JPG heightmap.")


def _ortho_candidates(record: UploadRecord, source: str) -> list[str]:
    """Only choose an explicitly named companion or an unambiguous single image in the upload."""
    paths = [path for path in record.files if path != source and PurePosixPath(path).suffix.lower() in (".tif", ".tiff", ".png", ".jpg")]
    stem = PurePosixPath(source).stem.lower()
    exact = [path for path in paths if PurePosixPath(path).stem.lower() in
             {f"{stem}_ortho", f"{stem}_rgb", "ortho", "rgb"}]
    named = [path for path in paths if any(word in PurePosixPath(path).stem.lower() for word in ("ortho", "rgb", "color"))]
    return list(dict.fromkeys(exact + named + (paths if len(paths) == 1 else [])))


def _within(value: float | None, bounds: tuple[float, float]) -> bool:
    return value is not None and bounds[0] <= value <= bounds[1]
