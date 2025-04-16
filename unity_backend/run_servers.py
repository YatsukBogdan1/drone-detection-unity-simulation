import subprocess
import signal
import sys
import os

# Commands to run
drone_cmd = [
    sys.executable, '-m', 'uvicorn', 'drone_ws:app', '--host', '0.0.0.0', '--port', '8000'
]
camera_cmd = [
    sys.executable, '-m', 'uvicorn', 'camera_ws:app', '--host', '0.0.0.0', '--port', '8001'
]

processes = []

def start_process(cmd):
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, cwd=os.path.dirname(__file__))

try:
    print('Starting drone_ws on port 8000...')
    p1 = start_process(drone_cmd)
    processes.append(p1)
    print('Starting camera_ws on port 8001...')
    p2 = start_process(camera_cmd)
    processes.append(p2)
    print('Both servers are running. Press Ctrl+C to stop.')
    # Wait for both processes
    while True:
        for p in processes:
            if p.poll() is not None:
                raise RuntimeError(f"Process {p.args} exited with code {p.returncode}")
        signal.pause()
except KeyboardInterrupt:
    print('\nShutting down servers...')
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print('Servers stopped.')
except Exception as e:
    print(f"Error: {e}")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    sys.exit(1)
