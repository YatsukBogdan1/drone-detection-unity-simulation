# --- WebSocket Camera Rotation Stream Endpoint ---
import asyncio
from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
from detection.detection import DroneDetector

class CameraWebSocket(WebSocketEndpoint):
    encoding = "json"

    def __init__(self, scope, receive, send, detector=None):
        super().__init__(scope, receive, send)
        self.detector = detector
        self._broadcast_task = None

    async def on_connect(self, websocket):
        await websocket.accept()
        self._broadcast_task = asyncio.create_task(self.broadcast_camera_rotation(websocket))

    async def broadcast_camera_rotation(self, websocket):
        ticks_per_second = 30
        while True:
            # Get latest angles from detector
            if self.detector and hasattr(self.detector, "current_angles"):
                rotX, rotY, rotZ = self.detector.current_angles
            else:
                rotX = rotY = rotZ = 0.0
            msg = {
                "type": "camera_rotation",
                "rotX": rotX,
                "rotY": rotY,
                "rotZ": rotZ
            }
            await websocket.send_json(msg)
            await asyncio.sleep(1.0 / ticks_per_second)

    def set_camera_rotation(self, rotX: float, rotY: float, rotZ: float):
        """
        Updates the camera rotation.
        Args:
            rotX (float): Rotation around X axis (pitch)
            rotY (float): Rotation around Y axis (yaw)
            rotZ (float): Rotation around Z axis (roll)
        """
        self.camera_rotation = {
            'rotX': rotX,
            'rotY': rotY,
            'rotZ': rotZ
        }

    async def generate_natural_camera_rotation(self):
        """
        Generates natural, smooth camera rotation values using sine/cosine waves.
        Updates self.camera_rotation several times per second.
        """
        import math, time
        t0 = time.time()
        ticks_per_second = 30
        while True:
            t = time.time() - t0
            # Oscillate each axis with different amplitude and frequency
            rotX = 10.0 * math.sin(0.4 * t) + 2.0 * math.sin(1.7 * t)
            rotY = 20.0 * math.sin(0.2 * t + 1.0) + 3.0 * math.cos(0.9 * t)
            rotZ = 5.0 * math.sin(0.6 * t + 2.0)
            self.camera_rotation = {
                'rotX': rotX,
                'rotY': rotY,
                'rotZ': rotZ
            }
            await asyncio.sleep(1.0 / ticks_per_second)

detector = DroneDetector()
detector.start()

def camera_ws_factory(scope, receive, send):
    return CameraWebSocket(scope, receive, send, detector=detector)

app = Starlette(debug=True, routes=[
    WebSocketRoute("/ws", camera_ws_factory),
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# To run: uvicorn index_ws:app --host 0.0.0.0 --port 8000