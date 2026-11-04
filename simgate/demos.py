"""Saved demo runs: recorded once with the Meta models, replayed instantly in either mode (no key needed).

A demo is one gzipped JSON file in demos/ (built from reports/ by scripts/build_demos.py), listed in
demos/index.json. Replaying one sends its recorded events: zero simulations and zero model calls.
"""
import gzip
import json
from functools import lru_cache
from pathlib import Path

DEMOS_DIR = Path(__file__).resolve().parent.parent / "demos"
REPLAY_COST = {"runs": 0, "wall_s": 0.0, "model_calls": 0}


def list_demos(root: Path = DEMOS_DIR) -> list[dict]:
    """The home page's demo list: id, title, prompt, kind and what it was recorded with."""
    index = root / "index.json"
    if not index.is_file():
        return []
    return [demo for demo in json.loads(index.read_text(encoding="utf-8")) if (root / demo["file"]).is_file()]


@lru_cache(maxsize=4)
def load_demo(demo_id: str, root: Path = DEMOS_DIR) -> dict:
    entry = next((demo for demo in list_demos(root) if demo["id"] == demo_id), None)
    if entry is None:
        raise ValueError(f"There is no saved demo called {demo_id!r}.")
    return {**entry, **json.loads(gzip.decompress((root / entry["file"]).read_bytes()))}


def replay_cost(recorded: dict | None) -> dict:
    """The cost line of a replay: nothing ran now; what the recording cost is said too."""
    text = "Replay · 0 runs · 0 model calls"
    if recorded:
        text += f" (recorded: {recorded['text']})"
    return {**REPLAY_COST, "text": text}
