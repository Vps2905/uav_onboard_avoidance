"""Deterministic image-plane avoidance illustration; never a flight command."""
import math
import numpy as np
def generate_reroute(current_pos,obstacle=None,clearance=100.0):
    p=np.asarray(current_pos,dtype=float)
    if p.shape!=(2,) or not np.all(np.isfinite(p)) or not math.isfinite(clearance) or clearance<=0:raise ValueError('Finite 2D position and positive clearance required')
    if obstacle is None:return tuple(p+np.array([0.,clearance]))
    delta=p-np.asarray(obstacle,dtype=float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)):raise ValueError('Finite 2D obstacle required')
    distance=float(np.linalg.norm(delta));direction=delta/distance if distance>1e-9 else np.array([0.,1.])
    return tuple(p+direction*clearance)
