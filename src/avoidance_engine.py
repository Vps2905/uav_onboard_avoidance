from src.kalman_tracker import KalmanObstacleTracker
from utils.trajectory_predictor import predict_future_position
from utils.reroute_planner import generate_reroute
from utils.helpers import euclidean_distance

def run_avoidance_logic(current_position, detected_object):
    tracker = KalmanObstacleTracker()
    predicted_pos = tracker.update(detected_object)
    future = predict_future_position(predicted_pos, [1, 1])  # sample velocity

    if euclidean_distance(current_position, future) < 5:
        return generate_reroute(current_position)
    return current_position
