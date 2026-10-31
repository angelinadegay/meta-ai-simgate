"""Honest uncertainty: run the same condition on several seeds and say how sure the result is.

Seed 0 is the condition exactly as it is (the run you saw). Seed k > 0 changes the terrain and obstacle randomness
(terrain seed + k) and nudges where each robot starts by up to ±1 cm and its heading by up to ±1° (see
SceneSpec.perturb_seed). Results are counted with a Wilson 95% interval, which stays sensible for the small counts
used here (3 of 3 passing is 44–100%, not "100%").
"""
import math
import statistics
from typing import Callable

from .batch import MAX_BATCH_RUNS, crash_report, run_batch
from .spec import SceneSpec

Z_95 = 1.96


def wilson(passed: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for passed/total (0–1 each side); (0, 1) when nothing ran."""
    if total <= 0:
        return 0.0, 1.0
    rate = passed / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def label(passed: int, total: int) -> str:
    """"reliable pass" (every seed passed), "reliable fail" (none did) or "borderline" (mixed)."""
    if total <= 0:
        return "not tested"
    return "reliable pass" if passed == total else "reliable fail" if passed == 0 else "borderline"


def ci_text(passed: int, total: int) -> str:
    """"passes 3/3 (95% CI 44–100%)"."""
    low, high = wilson(passed, total)
    return f"passes {passed}/{total} (95% CI {low * 100:.0f}–{high * 100:.0f}%)"


def seeded(spec: SceneSpec, variation: dict | None, k: int) -> dict:
    """The variation on seed k: unchanged for k = 0, otherwise the next terrain seed plus a start nudge."""
    variation = dict(variation or {})
    if k == 0:
        return variation
    base = variation.get("seed", spec.terrain.seed if spec.terrain is not None else 0)
    return {**variation, "seed": base + k, "perturb_seed": k}


def key_metric(summary: dict, robot_id: str | None = None) -> tuple[str, float | None]:
    """The number that says how close a run came: distance left to the goal, or forward progress without one."""
    from .diagnose import robot_metrics

    metrics = robot_metrics(summary or {}, robot_id)
    if metrics.get("has_goal"):
        return "distance_left_m", metrics.get("final_distance_m")
    return "progress_m", metrics.get("forward_progress_m")


def summarize(results: list[dict], robot_id: str | None = None) -> dict:
    """Pass count, rate, Wilson 95% interval, label and mean ± spread of the key metric over the runs that ran.
    A crashed run is a system problem, not a result: it is left out and reported."""
    from .diagnose import result_metrics

    ran = [result for result in results if not result.get("error")]
    passed = sum(bool(result_metrics(result, robot_id)["passed"] if robot_id else result.get("passed")) for result in ran)
    low, high = wilson(passed, len(ran))
    values = [value for name, value in (key_metric(result.get("summary") or {}, robot_id) for result in ran)
              if value is not None]
    name = key_metric((ran[0].get("summary") or {}) if ran else {}, robot_id)[0]
    out = {"passed": passed, "total": len(ran), "rate": round(passed / len(ran), 3) if ran else None,
           "ci95": [round(low, 3), round(high, 3)], "label": label(passed, len(ran)),
           "text": ci_text(passed, len(ran)),
           "metric": {"name": name, "mean": round(statistics.fmean(values), 2) if values else None,
                      "spread": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0 if values else None}}
    if (crashed := crash_report(results)):
        out["crashed"] = crashed
    return out


def run_seeded(spec: SceneSpec, variations: list[dict | None], seeds: int,
               on_result: Callable[[dict], None] | None = None, first_seed: int = 0) -> list[list[dict]]:
    """Every variation (None = the unchanged scene) on seeds first_seed..seeds-1, in parallel through the batch
    runner and in as few batches as its run limit allows. Returns one list of results per variation, in seed order."""
    own_seed = spec.terrain.seed if spec.terrain is not None else 0
    # the unchanged scene on seed 0 is written as its own seed, a change that changes nothing, so it runs in a batch
    jobs = [(index, k, {"label": f"{index}:{k}", **(seeded(spec, variation, k) or {"seed": own_seed})})
            for index, variation in enumerate(variations) for k in range(first_seed, seeds)]
    grouped: list[list[dict]] = [[] for _ in variations]
    chunk = MAX_BATCH_RUNS - 1
    for start in range(0, len(jobs), chunk):
        part = jobs[start:start + chunk]
        for (index, _, _), result in zip(part, run_batch(spec, [job[2] for job in part], on_result, baseline=False, record=False)):
            grouped[index].append(result)
    return grouped


def repeat(spec: SceneSpec, variation: dict | None = None, seeds: int = 3, robot_id: str | None = None) -> dict:
    """The same condition on ``seeds`` seeds, in parallel: pass count, rate, Wilson 95% interval, label and the
    key metric's mean ± spread."""
    results = run_seeded(spec, [variation], seeds)[0]
    return {"changes": dict(variation or {}), "seeds": seeds, **summarize(results, robot_id), "results": results}
