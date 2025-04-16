# --- WebSocket Camera Rotation Stream Endpoint ---
import asyncio
import logging
from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
import sys
import os
from detection.detection import DroneDetector

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class CameraWebSocket(WebSocketEndpoint):
    encoding = "json"

    def __init__(self, scope, receive, send):
        super().__init__(scope, receive, send)
        self._broadcast_task = None

    async def on_connect(self, websocket):
        await websocket.accept()
        logging.info("WebSocket client connected")
        self._broadcast_task = asyncio.create_task(self.broadcast_camera_rotation(websocket))

    async def broadcast_camera_rotation(self, websocket):
        ticks_per_second = 30
        try:
            while True:
                # Get latest angles from detector
                if detector and hasattr(detector, "current_angles"):
                    rotX, rotY, rotZ = detector.current_angles
                    logging.debug(f"Current angles: X={rotX:.1f}, Y={rotY:.1f}, Z={rotZ:.1f}")
                else:
                    rotX = rotY = rotZ = 0.0
                    logging.warning("No detector found or current_angles not available")
                
                msg = {
                    "type": "camera_rotation",
                    "rotX": rotX,
                    "rotY": rotY,
                    "rotZ": rotZ
                }
                await websocket.send_json(msg)
                logging.info(f"Camera rotation message sent: {msg}")
                await asyncio.sleep(1.0 / ticks_per_second)
        except asyncio.CancelledError:
            # Handle task cancellation
            logging.info("Camera rotation broadcast task cancelled")
        except Exception as e:
            logging.error(f"Error in broadcast_camera_rotation: {e}")

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

# Initialize the detector
detector = DroneDetector()
# Start the detector in a separate thread
detector.start()
logging.info("Drone detector initialized and started")

async def on_shutdown():
    """Shutdown handler to clean up resources"""
    logging.info("Server shutting down, stopping detector")
    if detector:
        detector.stop()

app = Starlette(
    debug=True, 
    routes=[
        WebSocketRoute("/ws", CameraWebSocket),
    ],
    on_shutdown=[on_shutdown]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# To run: uvicorn camera_ws:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting WebSocket server")
    uvicorn.run(app, host="0.0.0.0", port=8000)