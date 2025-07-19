import sys, os
import cv2
import numpy as np

sys.path.append("src")
sys.path.append("utils")

from optical_flow import compute_optical_flow, draw_flow
from kalman_tracker import KalmanObstacleTracker
from trajectory_predictor import predict_future_position
from reroute_planner import generate_reroute
from helpers import euclidean_distance

video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

trackers = [KalmanObstacleTracker(), KalmanObstacleTracker()]
current_position = (320, 240)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

ret, prev_frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    flow = compute_optical_flow(prev_frame, frame)
    vis = draw_flow(frame, flow)
    prev_frame = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 3000:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                detected.append((cx, cy))

    for i, obj in enumerate(detected[:2]):
        pred = trackers[i].update(obj)
        future = predict_future_position(pred, [1, 0], 1.5)

        cv2.circle(vis, obj, 8, (0, 0, 255), 2)
        cv2.circle(vis, tuple(pred.astype(int)), 8, (0, 255, 255), 2)

        distance = euclidean_distance(current_position, future)
        if distance < 100:
            reroute = generate_reroute(current_position)
            cv2.putText(vis, f"Rerouting#{i+1} to: {reroute}", (10, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(vis, tuple(map(int, reroute)), 10, (255, 0, 0), -1)

    cv2.circle(vis, current_position, 6, (0, 255, 0), -1)
    cv2.imshow("Multi-Drone Avoidance", vis)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
