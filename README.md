# Drone Detection - Unity Simulation

A YOLO-based drone detection pipeline, tested end-to-end against a Unity-simulated camera feed instead of real drone hardware.

## Why simulate

Testing computer vision models against real drones means real flight time, real hardware, and real risk. This project swaps that for a Unity simulation that streams synthetic camera frames over WebSocket, so the detection pipeline can be developed and validated without ever taking anything off the ground.

## What it does

- `camera_ws.py` — WebSocket server that receives simulated camera frames from Unity.
- `detection/` — runs YOLO object detection on incoming frames.
- `drone_ws.py` — WebSocket server for streaming drone position/telemetry data.
- `run_servers.py` — launches both WebSocket servers together.
- `test_camera_ws_stream.html` / `test_drone_ws_stream.html` — standalone browser clients for manually exercising each WebSocket stream without needing Unity running.

## Stack

- Python, YOLO
- Runs on an `yolo-mps` virtualenv — GPU-accelerated inference via Apple Metal Performance Shaders (Apple Silicon)
- WebSockets for all Unity <-> Python communication
- Unity (simulation side, not included in this repo)

## Running locally

```bash
python -m venv ~/venvs/yolo-mps
source ~/venvs/yolo-mps/bin/activate
pip install -r requirements.txt
./start.sh
```
