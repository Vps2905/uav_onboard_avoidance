import cv2

def compute_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def draw_flow(frame, flow, step=16):
    h, w = frame.shape[:2]
    vis = frame.copy()
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            end = (int(x + fx), int(y + fy))
            cv2.line(vis, (x, y), end, (0, 255, 0), 1)
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
    return vis
