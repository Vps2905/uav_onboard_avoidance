import cv2
import numpy as np

video_path = "data/test_video.mp4"
frame_width, frame_height = 640, 480
fps = 10
duration_sec = 10
total_frames = fps * duration_sec

video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

for i in range(total_frames):
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    x1 = min(50 + i * 5, frame_width - 50)
    y1 = 160
    x2 = max(frame_width - 50 - i * 5, 50)
    y2 = 320

    def draw_drone(frame, x, y):
        cv2.circle(frame, (x, y), 20, (255, 255, 255), -1)
        for dx, dy in [(-30, -30), (30, -30), (-30, 30), (30, 30)]:
            cv2.circle(frame, (x + dx, y + dy), 8, (255, 255, 255), 2)
            cv2.line(frame, (x, y), (x + dx, y + dy), (255, 255, 255), 1)

    draw_drone(frame, x1, y1)
    draw_drone(frame, x2, y2)

    cv2.putText(frame, f"Frame {i+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    video_writer.write(frame)

video_writer.release()
print("✅ Multi-drone simulation video saved to:", video_path)
