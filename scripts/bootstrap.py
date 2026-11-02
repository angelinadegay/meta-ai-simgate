#!/usr/bin/env python3
"""Fetch missing G1 source assets and import the shipped Mars terrain presets."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNITREE_REPOSITORY = "https://github.com/unitreerobotics/unitree_rl_gym.git"
UNITREE_SOURCE_ASSETS = (
    "resources/robots/g1_description/g1_12dof.xml",
    "resources/robots/g1_description/meshes/pelvis.STL",
    "deploy/deploy_mujoco/configs/g1.yaml",
    "LICENSE",
)
SIMGATE_POLICY_ASSET = "deploy/pre_train/g1/motion.onnx"
UNITREE_COPY_PATHS = (
    "resources/robots/g1_description",
    "deploy/pre_train/g1",
    "deploy/deploy_mujoco/configs/g1.yaml",
    "deploy/deploy_mujoco/deploy_mujoco.py",
    "LICENSE",
)
MARS_TERRAIN_SOURCES = {
    "gale": {
        "label": "Gale Crater, Mars",
        "source": "https://astrogeo-ard.s3.us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/PSP_008437_1750_PSP_008938_1750/DTEEC_008437_1750_008938_1750_U01.tif",
    },
    "jezero": {
        "label": "Jezero Crater, Mars",
        "source": "https://astrogeo-ard.s3.us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/ESP_045994_1985_ESP_046060_1985/DTEEC_045994_1985_046060_1985_U01.tif",
    },
    "ares3": {
        "label": "Acidalia Planitia (Ares 3 site), Mars",
        "source": "https://astrogeo-ard.s3.us-west-2.amazonaws.com/mars/mro/hirise/controlled/dtm/ESP_041277_2115_ESP_040776_2115/DTEEC_041277_2115_040776_2115_A01.tif",
    },
}


def _copy_missing(source: Path, destination: Path) -> None:
    if source.is_file():
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return
    if not source.is_dir():
        return
    for item in source.rglob("*"):
        target = destination / item.relative_to(source)
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def ensure_unitree_assets(root: Path = ROOT) -> None:
    destination = root / "third_party" / "unitree_rl_gym"
    missing = [path for path in UNITREE_SOURCE_ASSETS if not (destination / path).is_file()]
    if missing:
        print("Cloning Unitree assets required by the G1 robot...")
        try:
            with tempfile.TemporaryDirectory(prefix="simgate-unitree-") as temporary:
                checkout = Path(temporary) / "unitree_rl_gym"
                subprocess.run(
                    ["git", "clone", "--depth", "1", UNITREE_REPOSITORY, str(checkout)],
                    check=True,
                )
                for relative_path in UNITREE_COPY_PATHS:
                    _copy_missing(checkout / relative_path, destination / relative_path)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"Could not clone Unitree G1 assets: {error}") from error

    missing = [path for path in UNITREE_SOURCE_ASSETS if not (destination / path).is_file()]
    if missing:
        raise RuntimeError("The Unitree checkout is missing required G1 source files: " + ", ".join(missing))
    if not (destination / SIMGATE_POLICY_ASSET).is_file():
        raise RuntimeError(
            f"Missing SimGate-converted G1 policy: {destination / SIMGATE_POLICY_ASSET}. "
            "The upstream Unitree repository provides motion.pt, not this ONNX runtime file."
        )
    print("Unitree G1 source assets are ready.")


def ensure_terrain(root: Path, name: str, refresh: bool = False) -> None:
    terrain_dir = root / "simgate" / "terrains"
    metadata_path = terrain_dir / f"{name}.json"
    heights_path = terrain_dir / f"{name}.npz"
    if metadata_path.is_file() and heights_path.is_file() and not refresh:
        print(f"Mars terrain '{name}' is already installed.")
        return

    defaults = MARS_TERRAIN_SOURCES[name]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    source = metadata.get("source") or defaults["source"]
    label = metadata.get("label") or defaults["label"]
    importer = root / "scripts" / "hirise_to_simgate.py"
    if not importer.is_file():
        raise FileNotFoundError(f"Terrain importer not found: {importer}")

    print(f"Importing {label} from NASA/USGS HiRISE data...")
    subprocess.run(
        [
            sys.executable, str(importer), "--input", source, "--name", name,
            "--label", label, "--size-m", "40", "--body", "mars",
            "--surface", "regolith", "--pick", "relief", "--out-dir", str(terrain_dir),
        ],
        check=True,
        cwd=root,
    )
    if not metadata_path.is_file() or not heights_path.is_file():
        raise RuntimeError(f"Terrain import did not produce both {name}.json and {name}.npz.")


def ensure_terrains(root: Path = ROOT, names: tuple[str, ...] = tuple(MARS_TERRAIN_SOURCES),
                    refresh: bool = False) -> None:
    for name in names:
        ensure_terrain(root, name, refresh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--terrain", choices=tuple(MARS_TERRAIN_SOURCES), action="append",
                        help="ensure only this Mars terrain (may be repeated; default: all three)")
    parser.add_argument("--refresh-terrains", action="store_true",
                        help="re-import selected terrains from NASA/USGS instead of keeping installed files")
    parser.add_argument("--skip-unitree", action="store_true", help="do not clone missing G1 source assets")
    args = parser.parse_args(argv)

    if not args.skip_unitree:
        ensure_unitree_assets(args.root)
    ensure_terrains(args.root, tuple(args.terrain or MARS_TERRAIN_SOURCES), args.refresh_terrains)
    print("SimGate data bootstrap complete.")


if __name__ == "__main__":
    main()