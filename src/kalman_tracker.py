import cv2
import numpy as np

class KalmanObstacleTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def update(self, coord):
        measurement = np.array([[np.float32(coord[0])],
                                [np.float32(coord[1])]])
        self.kalman.correct(measurement)
        predicted = self.kalman.predict()
        return predicted[:2].flatten()
