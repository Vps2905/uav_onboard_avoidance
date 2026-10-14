import numpy as np
from src.optical_flow import compute_optical_flow,draw_flow
from src.kalman_tracker import KalmanObstacleTracker
import pytest
def test_identical_frames_have_zero_flow_and_valid_overlay():
    frame=np.zeros((64,64,3),np.uint8);flow=compute_optical_flow(frame,frame)
    assert flow.shape==(64,64,2) and np.max(np.abs(flow))<1e-6
    assert draw_flow(frame,flow).shape==frame.shape
def test_tracker_initializes_at_measurement_and_estimates_velocity():
    t=KalmanObstacleTracker();assert np.allclose(t.update([100,50],.5),[100,50])
    for i in range(1,12):p=t.update([100+10*i,50],.5)
    assert np.linalg.norm(p-[210,50])<3
    assert abs(t.velocity[0]-20)<3
    with pytest.raises(ValueError):t.update([0,0],dt=0)
