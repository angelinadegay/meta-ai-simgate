"""Calibrate the Victoria Crater boulder challenge without changing its time limit."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simgate import evolve


def _run(seed: int, tuning: dict, index: int) -> dict:
    challenge = evolve.challenge_preset(
        "victoria-rim-boulders",
        duration_s=evolve.DEFAULT_CHALLENGE_DURATION_S,
        rock_size_m=tuning["rock_size_m"],
        rock_count=tuning["rock_count"],
        friction=tuning["friction"],
        start_offset_m=tuning["start_offset_m"],
    )
    race = evolve.race_generation(challenge, evolve._rules(4), seed, record_replay=True)
    quality = evolve.calibration_quality(race["results"])
    return {
        "challenge": "victoria-rim-boulders",
        "trial": index,
        "rock_size_m": tuning["rock_size_m"],
        "rock_count": tuning["rock_count"],
        "friction": tuning["friction"],
        "start_offset_m": tuning["start_offset_m"],
        "duration_s": challenge.duration_s,
        "start_xy": challenge.mission.start_xy,
        "goal_xy": challenge.mission.goal_xy,
        **quality,
        "outcomes": [
            {
                "id": row["id"], "name": row["name"], "outcome": row["outcome"],
                "progress_percent": row["progress_percent"], "event": row["event"],
            }
            for row in race["results"]
        ],
        "adjustment": tuning.get("adjustment", "default settings"),
    }


def calibrate(seed: int = 0, tunings: list[dict] | None = None) -> dict:
    """Try physical challenge variations only; route length and duration remain fixed."""
    candidates = tunings or [
        {
            "rock_size_m": evolve.DEFAULT_ROCK_SIZE_M,
            "rock_count": evolve.DEFAULT_ROCK_COUNT,
            "friction": evolve.DEFAULT_LOOSE_SOIL_FRICTION,
            "start_offset_m": 0.0,
            "adjustment": "default settings",
        },
        {"rock_size_m": 0.30, "rock_count": 24, "friction": 0.50,
         "start_offset_m": 0.0, "adjustment": "larger rocks and lower loose-soil friction"},
        {"rock_size_m": 0.20, "rock_count": 30, "friction": 0.45,
         "start_offset_m": 0.0, "adjustment": "denser rock field and lower loose-soil friction"},
        {"rock_size_m": 0.35, "rock_count": 12, "friction": 0.45,
         "start_offset_m": 0.0, "adjustment": "maximum rock size and lower loose-soil friction"},
        {"rock_size_m": 0.25, "rock_count": 30, "friction": 0.50,
         "start_offset_m": 3.0, "adjustment": "denser field, lower friction and shifted start along wall"},
        {"rock_size_m": 0.30, "rock_count": 24, "friction": 0.45,
         "start_offset_m": -3.0, "adjustment": "lower friction and shifted start along wall"},
    ]
    rows = []
    for index, tuning in enumerate(candidates, 1):
        row = _run(seed, tuning, index)
        rows.append(row)
        if row["qualifies"]:
            break
    qualifying = [row for row in rows if row["qualifies"]]
    chosen = qualifying[0] if qualifying else min(
        rows,
        key=lambda row: abs(row["best_progress_percent"] - 50) +
        (100 if row["any_reached"] else 0),
    )
    return {
        "rows": rows, "default": "victoria-rim-boulders", "default_row": chosen,
        "qualified": bool(qualifying), "seed": seed,
        "simulations": sum(len(row["outcomes"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calibrate(args.seed)
    print("| Trial | Rock size | Density | Friction | Start offset | Time limit | Reached | Best progress | Qualifies | Adjustment |")
    print("|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|---|")
    for row in result["rows"]:
        print(
            f"| {row['trial']} | {row['rock_size_m']:.2f} m | {row['rock_count']} rocks "
            f"| {row['friction']:.2f} | {row['start_offset_m']:+.1f} m | {row['duration_s']:.1f} s "
            f"| {'yes' if row['any_reached'] else 'no'} | {row['best_progress_percent']:.1f}% "
            f"| {'yes' if row['qualifies'] else 'no'} | {row['adjustment']} |"
        )
    if result["qualified"]:
        row = result["default_row"]
        print(f"Qualified demo challenge: {row['rock_size_m']:.2f} m rocks, {row['rock_count']} rocks, "
              f"friction {row['friction']:.2f}, wall offset {row['start_offset_m']:+.1f} m; "
              f"time remains {row['duration_s']:.1f} s.")
    else:
        print("No physical-only setting met the 20–80% no-reach calibration target.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
