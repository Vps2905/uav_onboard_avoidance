# UAV onboard obstacle-avoidance prototype

Vijay P S · [Portfolio](https://vijaypsportfolio.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/vijay-ps-362a66298/)

An existing offline computer-vision prototype using Farneback optical flow, bright-region detection, Kalman tracking and a simple image-plane escape heuristic. It is **not a flight controller**: it sends no MAVLink/PX4 commands and does not demonstrate autonomous drone flight.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --video /absolute/path/to/a/permitted/video.mp4
pip install pytest
PYTHONPATH=. python3 -m pytest test -q
```

The existing sample video remains in `data/test_video.mp4`; the path can be overridden. A desktop OpenCV build is needed for the video window. ESC closes it.

The public repair on 14 October 2026 removed unresolved README merge markers, initialized the tracker from the first observation, used video frame time to express velocity in pixels/second, corrected the predictor's unexplained 50× scale factor and made escape points deterministic. Existing source and history were retained.

Source map: `src/optical_flow.py` → motion visualization; `src/kalman_tracker.py` → position/velocity estimate; `utils/trajectory_predictor.py` → constant-velocity extrapolation; `utils/reroute_planner.py` → illustrative 2D escape; `main.py` → offline video loop.

Tests exercise optical flow, initialization, constant-velocity estimation, predictor units, deterministic clearance, degenerate positions and invalid inputs. Contour-order association remains a prototype limitation; it is not robust multi-object data association. Pixel clearance is not physical distance, and no depth calibration or real flight validation is claimed.
