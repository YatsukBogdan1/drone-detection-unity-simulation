import math
import asyncio
import logging
from starlette.applications import Starlette
from starlette.endpoints import WebSocketEndpoint
from starlette.routing import WebSocketRoute
from starlette.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Drone Path Generation ---
def generate_drone_path():
    total_distance = 20000  # 20 km in meters
    point_spacing = 0.1     # 10 cm in meters
    num_points = int(total_distance / point_spacing)
    radius = 20            # meters, arbitrary radius for a loop
    altitude = 2           # meters above ground
    path = []
    for i in range(num_points):
        theta = (2 * math.pi * i) / num_points
        x = radius * math.cos(theta) - 50
        y = altitude
        z = radius * math.sin(theta) + 170
        path.append({
            'x': x,
            'y': y,
            'z': z,
            'rotX': 0,
            'rotY': 0,
            'rotZ': 0
        })
    return path

# --- WebSocket Drone Stream Endpoint ---
class DroneWebSocket(WebSocketEndpoint):
    encoding = "json"

    async def on_connect(self, websocket):
        await websocket.accept()
        logging.info("Drone WebSocket client connected")
        self._broadcast_task = asyncio.create_task(self.broadcast_positions(websocket))

    async def on_disconnect(self, websocket, close_code):
        if hasattr(self, '_broadcast_task'):
            self._broadcast_task.cancel()

    async def broadcast_positions(self, websocket):
        path = generate_drone_path()
        ticks_per_second = 30
        path_spacing = 0.1
        current_speed = 140.0  # m/s
        max_speed = 160.0
        min_speed = 140.0
        acceleration = 2.0
        accelerating = True
        float_index = 0.0
        while True:
            index = int(float_index) % len(path)
            position = path[index].copy()
            position['speed'] = current_speed
            await websocket.send_json(position)
            logging.info(f"Drone position sent: {position}")
            # Update speed
            if accelerating:
                current_speed += acceleration / ticks_per_second
                if current_speed >= max_speed:
                    current_speed = max_speed
                    accelerating = False
            else:
                current_speed -= acceleration / ticks_per_second
                if current_speed <= min_speed:
                    current_speed = min_speed
                    accelerating = True
            meters_per_tick = current_speed / ticks_per_second
            points_per_tick = meters_per_tick / path_spacing
            float_index = (float_index + points_per_tick) % len(path)
            await asyncio.sleep(1.0 / ticks_per_second)


# --- Starlette App Setup ---
app = Starlette(debug=True, routes=[
    WebSocketRoute("/ws", DroneWebSocket),
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# To run: uvicorn index_ws:app --host 0.0.0.0 --port 8000