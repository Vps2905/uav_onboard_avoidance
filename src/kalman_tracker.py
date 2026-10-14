import cv2
import numpy as np
class KalmanObstacleTracker:
    def __init__(self):
        self.kalman=cv2.KalmanFilter(4,2);self.initialized=False
        self.kalman.measurementMatrix=np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.processNoiseCov=np.eye(4,dtype=np.float32)*.03
        self.kalman.measurementNoiseCov=np.eye(2,dtype=np.float32)*.05
        self.kalman.errorCovPost=np.eye(4,dtype=np.float32)
    @property
    def velocity(self):return self.kalman.statePost[2:,0].copy()
    def update(self,coord,dt=1.0):
        point=np.asarray(coord,dtype=float)
        if point.shape!=(2,) or not np.all(np.isfinite(point)) or not np.isfinite(dt) or dt<=0:raise ValueError('Finite 2D observation and positive dt required')
        self.kalman.transitionMatrix=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)
        if not self.initialized:
            self.kalman.statePost=np.array([[point[0]],[point[1]],[0],[0]],np.float32);self.initialized=True
            return point.astype(np.float32)
        self.kalman.predict()
        corrected=self.kalman.correct(point.astype(np.float32).reshape(2,1))
        return corrected[:2,0].copy()
