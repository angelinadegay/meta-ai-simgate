"""Calibrate Victoria's physical course and lander-fit design constraints."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simgate import evolve
from simgate import config


DEFAULT_TUNINGS = [
    {"rock_size_m": 0.25, "rock_count": 24, "friction": 0.55,
     "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.6, 0.5]}},
    {"rock_size_m": 0.15, "rock_count": 12, "friction": 0.65,
     "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.6, 0.5]},
     "adjustment": "smallest, sparsest rocks on the firmest allowed soil; lander limits unchanged"},
    {"rock_size_m": 0.20, "rock_count": 12, "friction": 0.65,
     "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.8, 0.5]},
     "adjustment": "stowed length 0.6 → 0.8 m so 0.20 m wheels fit; sparse 0.20 m rocks, firm soil"},
    {"rock_size_m": 0.15, "rock_count": 12, "friction": 0.65,
     "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.8, 0.5]},
     "adjustment": "stowed length 0.6 → 0.8 m so 0.20 m wheels fit; smallest sparse rocks, firm soil"},
    {"rock_size_m": 0.15, "rock_count": 12, "friction": 0.60,
     "design_constraints": {"max_mass_kg": 12, "max_size_m": [0.8, 0.5]},
     "adjustment": "stowed length 0.6 → 0.8 m; smallest sparse rocks, friction 0.60"},
]


def _challenge(tuning: dict):
    return evolve.challenge_preset(
        "victoria-rim-boulders",
        duration_s=evolve.DEFAULT_CHALLENGE_DURATION_S,
        rock_size_m=tuning["rock_size_m"],
        rock_count=tuning["rock_count"],
        friction=tuning["friction"],
        design_constraints=tuning["design_constraints"],
    )


def feasibility(seed: int, tuning: dict) -> dict:
    """Race the best legal reference designs; the course is feasible only if one climbs out."""
    challenge = _challenge(tuning)
    designs = evolve.feasibility_reference_designs(challenge.design_constraints)
    race = evolve.race_generation(challenge, designs, seed, record_replay=False)
    return feasibility_summary(race["results"], designs)


def feasibility_summary(results: list[dict], designs: list[dict]) -> dict:
    robots = {design["id"]: design["robot"] for design in designs}
    reached = [row for row in results if row.get("reached_goal")]
    best = max(results, key=lambda row: float(row.get("progress_percent", 0)))
    return {
        "feasible": bool(reached),
        "reference": [
            {"id": row["id"], "name": row["name"], "robot": robots.get(row["id"]),
             "outcome": row["outcome"], "progress_percent": row["progress_percent"]}
            for row in results
        ],
        "reason": (f"{reached[0]['name']} (legal under the lander limits) reaches the rim"
                   if reached else
                   f"no legal reference design reaches the rim; best is {best['name']} at "
                   f"{float(best['progress_percent']):.1f}% ({best['outcome']})"),
        "simulations": len(results),
    }


def _run(seed: int, tuning: dict, index: int, mode: str, designs: list[dict] | None = None) -> dict:
    constraints = tuning["design_constraints"]
    challenge = _challenge(tuning)
    if designs is None:
        candidates = evolve._rules(4, challenge.design_constraints)
    else:
        candidates = [
            evolve._constrain_design(design, challenge.design_constraints)
            for design in designs
        ]
    race = evolve.race_generation(challenge, candidates, seed, record_replay=False)
    quality = evolve.calibration_quality(race["results"])
    return {
        "challenge": "victoria-rim-boulders",
        "trial": index,
        "generation_zero": mode,
        "rock_size_m": tuning["rock_size_m"],
        "rock_count": tuning["rock_count"],
        "friction": tuning["friction"],
        "design_constraints": constraints,
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


def calibrate(
    seed: int = 0, tunings: list[dict] | None = None, include_model: bool | None = None,
) -> dict:
    """Tune constraints and the E7 rock field; never change the course or time limit.

    A setting qualifies only if generation zero fails with 20–80% best progress AND a legal
    reference design (biggest wheels / longest legs inside the lander limits) climbs out."""
    candidates = tunings or DEFAULT_TUNINGS
    rows = []
    checked: dict[int, dict] = {}

    def judge(row: dict, tuning: dict) -> dict:
        row["gen0_qualifies"] = row["qualifies"]
        if row["qualifies"]:
            key = candidates.index(tuning)
            if key not in checked:
                checked[key] = feasibility(seed, tuning)
            row["feasibility"] = checked[key]
            row["qualifies"] = checked[key]["feasible"]
        else:
            row["feasibility"] = None
        return row

    rule_row = None
    rule_tuning = candidates[-1]
    for tuning in candidates:
        row = judge(_run(seed, tuning, len(rows) + 1, "rule-based"), tuning)
        rows.append(row)
        if row["qualifies"] and rule_row is None:
            rule_row, rule_tuning = row, tuning
            break
    run_model = not config.MOCK_MODE if include_model is None else include_model
    model_rows = []
    model_designs = None
    model_status = "skipped: no live Designer configured"
    if run_model:
        initial = evolve.challenge_preset(
            "victoria-rim-boulders", rock_size_m=rule_tuning["rock_size_m"],
            rock_count=rule_tuning["rock_count"], friction=rule_tuning["friction"],
            design_constraints=rule_tuning["design_constraints"],
        )
        model_designs = evolve.propose_generation0(initial, 4)
        model_status = "one model-designed generation-zero set tested against candidate settings"
        start_index = candidates.index(rule_tuning)
        model_candidates = candidates[start_index:]
        for tuning in model_candidates:
            row = judge(_run(seed, tuning, len(rows) + 1, "model-designed", model_designs), tuning)
            rows.append(row)
            model_rows.append(row)
            if row["qualifies"]:
                break
    qualifying_model = next((row for row in model_rows if row["qualifies"]), None)
    chosen = qualifying_model or rule_row or min(
        rows, key=lambda row: abs(row["best_progress_percent"] - 50) +
        (100 if row["any_reached"] else 0),
    )
    qualified = bool(rule_row and (not run_model or qualifying_model))
    return {
        "rows": rows, "default": "victoria-rim-boulders", "default_row": chosen,
        "rule_based_row": rule_row, "model_designed_row": qualifying_model,
        "model_designed_status": model_status,
        "qualified": qualified, "seed": seed,
        "feasibility_rule": ("a legal reference design (biggest wheels / longest legs inside the "
                             "lander limits) must reach the rim on the same course and time limit"),
        "simulations": (sum(len(row["outcomes"]) for row in rows)
                        + sum(check["simulations"] for check in checked.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = calibrate(args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("| Trial | Gen 0 | Rock size | Density | Friction | Mass cap | Stowed size | Time limit | Reached | Best progress | Legal design climbs out? | Qualifies |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|---|:---:|")
    for row in result["rows"]:
        limits = row["design_constraints"]
        check = row.get("feasibility")
        feasible = ("not checked (gen 0 outside target)" if check is None else
                    ("yes: " if check["feasible"] else "no: ") + check["reason"])
        print(
            f"| {row['trial']} | {row['generation_zero']} | {row['rock_size_m']:.2f} m "
            f"| {row['rock_count']} rocks | {row['friction']:.2f} | ≤{limits['max_mass_kg']:g} kg "
            f"| {limits['max_size_m'][0]:g} × {limits['max_size_m'][1]:g} m | {row['duration_s']:.1f} s "
            f"| {'yes' if row['any_reached'] else 'no'} | {row['best_progress_percent']:.1f}% "
            f"| {feasible} | {'yes' if row['qualifies'] else 'no'} |"
        )
    if result["qualified"]:
        row = result["default_row"]
        print(f"Qualified demo challenge: {row['rock_size_m']:.2f} m rocks, {row['rock_count']} rocks, "
              f"friction {row['friction']:.2f}, mass ≤{row['design_constraints']['max_mass_kg']:g} kg, "
              f"stowed size ≤{row['design_constraints']['max_size_m'][0]:g} × "
              f"{row['design_constraints']['max_size_m'][1]:g} m; time remains {row['duration_s']:.1f} s.")
    elif result["rule_based_row"]:
        row = result["rule_based_row"]
        print(f"Rule-based generation zero qualifies and a legal design climbs out at {row['rock_size_m']:.2f} m "
              f"rocks, {row['rock_count']} rocks, friction {row['friction']:.2f}, mass ≤"
              f"{row['design_constraints']['max_mass_kg']:g} kg, stowed size ≤"
              f"{row['design_constraints']['max_size_m'][0]:g} × {row['design_constraints']['max_size_m'][1]:g} m. "
              "The model-designed generation zero already climbs out there, so the live run relies on "
              "escalation (raise the rocks to the best design's edge) rather than a harder hand-tuned course.")
    else:
        print("No setting made generation zero meet the 20–80% no-reach target while a legal design climbs out.")


if __name__ == "__main__":
    main()
