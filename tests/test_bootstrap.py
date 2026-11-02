import json
from pathlib import Path

from scripts import bootstrap


def test_bootstrap_skips_clones_and_imports_when_assets_are_shipped(tmp_path, monkeypatch):
    unitree = tmp_path / "third_party" / "unitree_rl_gym"
    for relative_path in bootstrap.UNITREE_SOURCE_ASSETS + (bootstrap.SIMGATE_POLICY_ASSET,):
        asset = unitree / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.touch()
    terrain_dir = tmp_path / "simgate" / "terrains"
    terrain_dir.mkdir(parents=True)
    for name in bootstrap.MARS_TERRAIN_SOURCES:
        (terrain_dir / f"{name}.json").write_text("{}", encoding="utf-8")
        (terrain_dir / f"{name}.npz").touch()

    def unexpected_network_call(*_args, **_kwargs):
        raise AssertionError("Existing bundled assets should not trigger network calls.")

    monkeypatch.setattr(bootstrap.subprocess, "run", unexpected_network_call)
    bootstrap.ensure_unitree_assets(tmp_path)
    bootstrap.ensure_terrains(tmp_path)


def test_bootstrap_clones_missing_unitree_runtime_assets(tmp_path, monkeypatch):
    calls = []

    def fake_clone(command, check):
        calls.append(command)
        checkout = Path(command[-1])
        for relative_path in bootstrap.UNITREE_SOURCE_ASSETS:
            asset = checkout / relative_path
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.touch()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_clone)
    policy = tmp_path / "third_party" / "unitree_rl_gym" / bootstrap.SIMGATE_POLICY_ASSET
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.touch()
    bootstrap.ensure_unitree_assets(tmp_path)

    assert len(calls) == 1
    assert calls[0][:4] == ["git", "clone", "--depth", "1"]
    for relative_path in bootstrap.UNITREE_SOURCE_ASSETS:
        assert (tmp_path / "third_party" / "unitree_rl_gym" / relative_path).is_file()


def test_bootstrap_explains_that_upstream_does_not_supply_onnx(tmp_path, monkeypatch):
    unitree = tmp_path / "third_party" / "unitree_rl_gym"
    for relative_path in bootstrap.UNITREE_SOURCE_ASSETS:
        asset = unitree / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.touch()

    def unexpected_clone(*_args, **_kwargs):
        raise AssertionError("Complete Unitree source assets should not be cloned again.")

    monkeypatch.setattr(bootstrap.subprocess, "run", unexpected_clone)
    try:
        bootstrap.ensure_unitree_assets(tmp_path)
    except RuntimeError as error:
        assert "upstream Unitree repository provides motion.pt, not this ONNX runtime file" in str(error)
    else:
        raise AssertionError("A missing converted policy should be reported.")


def test_bootstrap_imports_missing_mars_terrain_from_metadata(tmp_path, monkeypatch):
    importer = tmp_path / "scripts" / "hirise_to_simgate.py"
    importer.parent.mkdir(parents=True)
    importer.touch()
    terrain_dir = tmp_path / "simgate" / "terrains"
    terrain_dir.mkdir(parents=True)
    metadata_path = terrain_dir / "gale.json"
    metadata_path.write_text(json.dumps({
        "label": "Gale Crater, Mars",
        "source": "https://example.test/gale.tif",
    }), encoding="utf-8")
    calls = []

    def fake_import(command, check, cwd):
        calls.append((command, cwd))
        (terrain_dir / "gale.npz").touch()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_import)
    bootstrap.ensure_terrain(tmp_path, "gale")

    command, cwd = calls[0]
    assert command[command.index("--input") + 1] == "https://example.test/gale.tif"
    assert command[command.index("--name") + 1] == "gale"
    assert command[command.index("--out-dir") + 1] == str(terrain_dir)
    assert cwd == tmp_path
    assert (terrain_dir / "gale.npz").is_file()