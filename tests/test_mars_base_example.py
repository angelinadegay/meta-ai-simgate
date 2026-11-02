import asyncio
import copy

import pytest

from scripts.run_mars_base_example import run_example, screenshot_frames
from simgate import config


def recorded_example():
    return {"spec": {"robot": {"kind": "rover", "wheel_radius": 0.15}},
            "scene": {"follow_geom": 0, "mission": {"goal_z": 1}, "structures": [
                {"kind": "bridge", "position": {"x": 2, "y": 0, "z": 0}, "orientation": {"yaw_deg": 0},
                 "dimensions": {"span_m": 2, "deck_width_m": 1}}]},
            "frames": [{"t": 0, "poses": [[0, 0, 0.15]]},
                       {"t": 1, "poses": [[2, 0, 0.18]]},
                       {"t": 2, "poses": [[4, 0, 1.15]]}],
            "metrics": {"goal_reached": True, "time_to_goal_s": 2}}


def test_live_capture_uses_recorded_bridge_and_height_correct_arrival():
    replay = recorded_example()
    before = copy.deepcopy(replay)
    assert screenshot_frames(replay) == {"overview": 0, "rover-on-bridge": 1, "arriving-at-airlock": 2}
    assert replay == before


def test_live_capture_refuses_a_false_planar_arrival():
    replay = recorded_example()
    replay["frames"][2]["poses"][0][2] = 0.15
    with pytest.raises(RuntimeError, match="raised goal"):
        screenshot_frames(replay)
    replay["metrics"]["goal_reached"] = False
    with pytest.raises(RuntimeError, match="No recorded airlock arrival"):
        screenshot_frames(replay)


def test_live_runner_refuses_mock_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    with pytest.raises(RuntimeError, match="refuses Mock mode"):
        asyncio.run(run_example(tmp_path))