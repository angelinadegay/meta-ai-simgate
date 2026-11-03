import numpy as np
import pytest

from simgate.library.cards import RobotCard
from simgate.spec import Obstacle, Robot, SceneSpec, Terrain
from simgate.traversability import features


def _scene(terrain, **kwargs):
    return SceneSpec(domain="robot_terrain", terrain=terrain, robot=Robot(), **kwargs)


def test_flat_ground_has_zero_slope_and_roughness():
    grid = features(_scene(Terrain(kind="flat", size_m=8)))

    assert grid["cell_size_m"] == 0.5
    assert np.max(grid["slope_deg"]) < 1e-6
    assert np.max(grid["roughness_m"]) < 1e-6
    assert grid["blocked"].dtype == bool


def test_known_slope_is_measured():
    grid = features(_scene(Terrain(kind="slope", slope_deg=12, roughness=0, size_m=8)))

    assert np.allclose(grid["slope_deg"][4:-4, 4:-4], 12, atol=0.1)


@pytest.mark.parametrize("height_m", [0.08, 0.15])
def test_course_step_is_reported(height_m):
    grid = features(_scene(Terrain(kind="flat", size_m=12), obstacles=[Obstacle(kind="step", height_m=height_m)]))

    assert np.max(grid["max_step_m"]) >= height_m


def test_robot_card_footprint_controls_blocked_cells():
    scene = _scene(Terrain(kind="indoor"))
    small_robot = RobotCard(id="small", name="Small", license="test", credit="test", kind="wheeled",
                            source="builtin", root_body="base", actuator_type="torque",
                            standing_height_m=0.3, mass_kg=2, footprint_length_m=0.3,
                            footprint_width_m=0.4)
    grid = features(scene, small_robot)
    doorway_col = int(np.argmin(np.abs(grid["x_m"] - 5.0)))
    doorway_row = int(np.argmin(np.abs(grid["y_m"] - 0.625)))

    assert not grid["blocked"][doorway_row, doorway_col]


def test_indoor_doorway_clearance_and_wet_surface_friction():
    scene = _scene(Terrain(kind="indoor"))
    grid = features(scene)
    doorway_col = int(np.argmin(np.abs(grid["x_m"] - 5.0)))
    doorway_row = int(np.argmin(np.abs(grid["y_m"] - 0.625)))
    wet_col = int(np.argmin(np.abs(grid["x_m"] + 5.625)))
    wet_row = int(np.argmin(np.abs(grid["y_m"] + 4.125)))

    assert grid["cell_size_m"] == 0.25
    assert np.isclose(grid["clearance_width_m"][doorway_row, doorway_col], 0.65)
    assert grid["friction"][wet_row, wet_col] == 0.15


def test_gusev_grid_matches_measured_terrain_size():
    grid = features(_scene(Terrain(kind="real", preset="gusev-columbia-hills")))

    assert grid["slope_deg"].shape == (160, 160)
    assert grid["x_m"].size == grid["y_m"].size == 160
    assert grid["cell_size_m"] == 0.5
