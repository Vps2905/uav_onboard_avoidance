import numpy as np

def predict_future_position(current_position, velocity_vector, seconds_ahead=1.0):
    future_pos = np.array(current_position) + np.array(velocity_vector) * 50 * seconds_ahead
    return future_pos
