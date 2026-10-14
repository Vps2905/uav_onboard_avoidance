import numpy as np
from utils.reroute_planner import generate_reroute
from utils.trajectory_predictor import predict_future_position
import pytest
def test_velocity_units_are_pixels_per_second():
    assert np.allclose(predict_future_position([100,100],[10,-2],2),[120,96])
def test_reroute_is_deterministic_and_increases_clearance():
    p=np.array([100.,100.]);obstacle=np.array([110.,100.]);a=generate_reroute(p,obstacle)
    assert a==generate_reroute(p,obstacle)
    assert np.linalg.norm(np.asarray(a)-obstacle)>np.linalg.norm(p-obstacle)
def test_coincident_obstacle_has_finite_escape_direction():
    assert np.all(np.isfinite(generate_reroute([0,0],[0,0])))
def test_invalid_motion_rejected():
    with pytest.raises(ValueError):generate_reroute([float('nan'),0])
