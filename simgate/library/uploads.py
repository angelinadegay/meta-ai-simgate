"""Safe storage for user uploads: terrains, robots (MJCF/URDF bodies) and policies (ONNX brains).

Every check happens in memory before anything is written. Nothing uploaded is ever imported or executed:
YAML is read with yaml.safe_load, XML is parsed without DOCTYPE/ENTITY declarations, MJCF plugins are refused,
and policies must be ONNX (a TorchScript or pickle file can run arbitrary code when loaded).
"""

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml

UploadKind = Literal["terrain", "robot", "policy"]
KINDS: tuple[str, ...] = ("terrain", "robot", "policy")
UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024        # what the user sends, zipped or not
MAX_EXTRACTED_BYTES = 200 * 1024 * 1024    # what a zip may expand to
MAX_ZIP_FILES = 500
ALLOWED_EXTENSIONS = {".xml", ".urdf", ".mjcf", ".stl", ".obj", ".msh", ".png", ".jpg", ".tif", ".tiff", ".onnx",
                      ".yaml", ".yml", ".json", ".md", ".txt"}
MAIN_FILES = {   # an upload of each kind must contain one of these
    "terrain": {".tif", ".tiff", ".png", ".jpg"},
    "robot": {".xml", ".urdf", ".mjcf"},
    "policy": {".onnx"},
}
TEXT_LIMIT = 200   # name, credit and license are free text shown on the card

ONNX_EXPORT_SNIPPET = """import torch
policy = torch.jit.load("policy.pt", map_location="cpu").eval()   # run this on your own machine
observation = torch.zeros(1, NUM_OBSERVATIONS)                      # the size your policy expects
torch.onnx.export(policy, observation, "policy.onnx", input_names=["observation"],
                  output_names=["action"], opset_version=17)"""
_PICKLE_EXTENSIONS = {".pt", ".pth", ".ts", ".pkl", ".pickle", ".ckpt", ".bin", ".jit"}
_PATH_ATTRIBUTES = {"file", "filename", "meshdir", "texturedir", "assetdir"}   # MJCF and URDF file references
_PACKAGE_URI = re.compile(r"^package://[^/]+/")


class UploadError(ValueError):
    """A rejected upload; the message is shown to the user as-is."""


@dataclass
class UploadRecord:
    id: str
    kind: str
    name: str
    credit: str
    license: str
    files: list[str]
    size_bytes: int
    sha256: str
    created: float
    source: str = "upload"
    folder: str = ""   # where it is on disk; not stored in upload.json

    def card(self) -> dict:
        data = asdict(self)
        data.pop("folder")
        return data


def save_upload(kind: str, files: list[tuple[str, bytes]], name: str, credit: str, license: str,
                uploads_dir: Path | None = None) -> UploadRecord:
    """Check an upload completely, then store it under uploads/<kind>/<id>/ with an upload.json record.
    ``files`` are (filename, content) pairs as sent; a single .zip is unpacked safely. The same content and name
    uploaded twice give the same id, and the stored copy is returned."""
    uploads_dir = uploads_dir or UPLOADS_DIR
    if kind not in KINDS:
        raise UploadError(f"Upload type must be one of: {', '.join(KINDS)}.")
    name, credit, license = (_text(label, value) for label, value in
                             (("Name", name), ("Credit", credit), ("License", license)))
    if not files:
        raise UploadError("No files were uploaded.")
    if sum(len(content) for _, content in files) > MAX_UPLOAD_BYTES:
        raise UploadError(f"The upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if len(files) == 1 and files[0][0].lower().endswith(".zip"):
        entries = _unzip(files[0][1])
    else:
        entries = {_safe_relative(filename): content for filename, content in files}
        if len(entries) != len(files):
            raise UploadError("Two uploaded files have the same name.")
    for path, content in entries.items():
        _check_file(kind, path, content)
    if not any(PurePosixPath(path).suffix.lower() in MAIN_FILES[kind] for path in entries):
        raise UploadError(f"A {kind} upload needs one of: {', '.join(sorted(MAIN_FILES[kind]))}.")

    digest = hashlib.sha256()
    for path in sorted(entries):
        digest.update(path.encode("utf-8") + b"\0" + entries[path] + b"\0")
    sha = digest.hexdigest()
    upload_id = f"{_slug(name)}-{sha[:8]}"
    folder = uploads_dir / kind / upload_id
    if (folder / "upload.json").is_file():
        return _read_record(folder)

    record = UploadRecord(id=upload_id, kind=kind, name=name, credit=credit, license=license,
                          files=sorted(entries), size_bytes=sum(len(c) for c in entries.values()),
                          sha256=sha, created=round(time.time(), 3))
    (uploads_dir / kind).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{upload_id}-", dir=uploads_dir / kind))
    try:   # write everything to a staging folder, then move it into place in one step
        for path, content in entries.items():
            target = staging / PurePosixPath(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (staging / "upload.json").write_text(json.dumps(record.card(), indent=2), encoding="utf-8")
        os.replace(staging, folder)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    record.folder = str(folder)
    return record


def discard_upload(record: UploadRecord) -> None:
    """Remove a stored upload that couldn't be used (e.g. a terrain that failed to convert)."""
    if record.folder:
        shutil.rmtree(record.folder, ignore_errors=True)


def list_uploads(kind: str | None = None, uploads_dir: Path | None = None) -> list[UploadRecord]:
    """Stored uploads, newest first; a folder without a readable upload.json is skipped."""
    uploads_dir = uploads_dir or UPLOADS_DIR
    records = []
    for each in (kind,) if kind else KINDS:
        root = uploads_dir / each
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            if folder.is_dir() and not folder.name.startswith(".") and (folder / "upload.json").is_file():
                try:
                    records.append(_read_record(folder))
                except (OSError, ValueError, TypeError):
                    continue
    return sorted(records, key=lambda record: record.created, reverse=True)


def _read_record(folder: Path) -> UploadRecord:
    record = UploadRecord(**json.loads((folder / "upload.json").read_text(encoding="utf-8")))
    record.folder = str(folder)
    return record


def _text(label: str, value: str) -> str:
    value = " ".join((value or "").split())
    if not value:
        raise UploadError(f"{label} is required.")
    if len(value) > TEXT_LIMIT:
        raise UploadError(f"{label} must be at most {TEXT_LIMIT} characters.")
    return value


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24].strip("-") or "upload"


def _safe_relative(filename: str) -> str:
    """A stored path: relative, forward slashes, no '..', no drive letter, no hidden or empty parts."""
    raw = (filename or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or raw.startswith("/") or re.match(r"^[A-Za-z]:", raw) or ".." in path.parts:
        raise UploadError(f"Unsafe file path in upload: {filename!r}.")
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts or any(part.startswith(".") for part in parts):
        raise UploadError(f"Unsafe file path in upload: {filename!r}.")
    return "/".join(parts)


def _unzip(data: bytes) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadError("The .zip file can't be read.") from exc
    infos = [info for info in archive.infolist() if not info.is_dir()]
    if len(infos) > MAX_ZIP_FILES:
        raise UploadError(f"The .zip has {len(infos)} files; the limit is {MAX_ZIP_FILES}.")
    if sum(info.file_size for info in infos) > MAX_EXTRACTED_BYTES:
        raise UploadError(f"The .zip expands to more than {MAX_EXTRACTED_BYTES // (1024 * 1024)} MB.")
    entries: dict[str, bytes] = {}
    total = 0
    for info in infos:
        if stat.S_ISLNK(info.external_attr >> 16):
            raise UploadError(f"The .zip contains a symbolic link: {info.filename!r}.")
        if info.filename.startswith("__MACOSX/") or PurePosixPath(info.filename).name == ".DS_Store":
            continue   # macOS archive leftovers
        path = _safe_relative(info.filename)
        with archive.open(info) as member:   # declared sizes can lie: count what actually comes out
            content = member.read(MAX_EXTRACTED_BYTES - total + 1)
        total += len(content)
        if total > MAX_EXTRACTED_BYTES:
            raise UploadError(f"The .zip expands to more than {MAX_EXTRACTED_BYTES // (1024 * 1024)} MB.")
        if path in entries:
            raise UploadError(f"The .zip contains {path!r} twice.")
        entries[path] = content
    if not entries:
        raise UploadError("The .zip is empty.")
    return entries


def _check_file(kind: str, path: str, content: bytes) -> None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _PICKLE_EXTENSIONS:
        raise UploadError(f"{path!r} is a PyTorch/pickle file, which can run code when it is loaded, so it isn't "
                          "accepted. Export the policy to ONNX on your own machine and upload the .onnx file:\n"
                          + ONNX_EXPORT_SNIPPET)
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadError(f"{path!r}: {suffix or 'files without an extension'} can't be uploaded. Allowed: "
                          f"{', '.join(sorted(ALLOWED_EXTENSIONS))}.")
    if suffix in (".yaml", ".yml"):
        try:
            yaml.safe_load(content.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise UploadError(f"{path!r} isn't valid YAML: {exc}") from exc
    elif suffix == ".json":
        try:
            json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise UploadError(f"{path!r} isn't valid JSON: {exc}") from exc
    elif suffix in (".xml", ".urdf", ".mjcf"):
        _check_model_xml(path, content)
    elif suffix == ".onnx" and kind != "policy":
        raise UploadError(f"{path!r}: ONNX files belong in a policy upload.")


def _check_model_xml(path: str, content: bytes) -> None:
    """MJCF/URDF: no DOCTYPE or entities, no plugins, and every file reference stays inside the upload folder."""
    head = content[:4096].lower()
    if b"<!doctype" in head or b"<!entity" in content.lower():
        raise UploadError(f"{path!r} declares a DOCTYPE or entities, which aren't allowed.")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise UploadError(f"{path!r} isn't valid XML: {exc}") from exc
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""
        if tag == "plugin":
            raise UploadError(f"{path!r} uses a MuJoCo <plugin>, which loads native code; plugins aren't allowed.")
        for attribute, value in node.attrib.items():
            if attribute.lower() in _PATH_ATTRIBUTES:
                _check_reference(path, attribute, value)


def _check_reference(path: str, attribute: str, value: str) -> None:
    """A file or folder named inside a model must be relative and stay in the upload (URDF package:// paths
    count from the upload's root)."""
    reference = _PACKAGE_URI.sub("", value.strip().replace("\\", "/"))
    if (reference.startswith("/") or re.match(r"^[A-Za-z]:", reference) or "://" in reference
            or ".." in PurePosixPath(reference).parts):
        raise UploadError(f"{path!r}: {attribute}={value!r} points outside the upload folder.")
