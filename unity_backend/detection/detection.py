from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import time
import math
import threading
from detection.sort import Sort
from detection.unity_stream_capture import ScreenCapture

class DroneDetector:
    def __init__(self, model_path=None, video_path=None, confidence=0.4, 
                 horizontal_fov=62.2, vertical_fov=48.8, headless=False):
        if model_path is None:
            # Always resolve relative to this file's directory
            model_path = os.path.join(os.path.dirname(__file__), "weights", "weightsM50Epoch.pt")
        # if video_path is None:
        #     # Always resolve relative to this file's directory
        #     video_path = os.path.join(os.path.dirname(__file__), "assets", "videos", "drone_video.mp4")
        #     # video_path = S
        self.window_title = "Drone Detection (MPS)"
        self.headless = headless
        self.video_path = video_path
        
        # Field of view parameters (for angle calculation)
        self.horizontal_fov = horizontal_fov  # degrees
        self.vertical_fov = vertical_fov      # degrees
        
        # Load YOLO model with MPS device for Apple Silicon
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Set confidence threshold
        self.model.conf = confidence
        
        # For FPS calculation
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = 0
        self.fps_values = []
        self.fps_buffer_size = 30  # Store the last 30 FPS values for averaging
        
        # Initialize SORT tracker
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        # Track color map (for consistent colors per ID)
        self.track_colors = {}
        
        # Store current angles for access by other threads
        self.current_angles = (0.0, 0.0, 0.0)  # (rotX, rotY, rotZ)
        
        # Thread control
        self._running = False
        self._thread = None
        
    def start(self):
        """Start the detector in a separate thread"""
        if self._thread and self._thread.is_alive():
            print("Detector already running")
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._process_video)
        self._thread.daemon = True  # Thread will exit when main program exits
        self._thread.start()
        print("Drone detector started in background thread")
    
    def _process_video(self):
        """Process video frames in a separate thread"""
        try:
            # Open video file
            self.video_capture = ScreenCapture(2)
            if not self.video_capture.isOpened():
                print(f"Error: Unable to open video file {self.video_path}")
                return
            
            # Get video properties
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        except Exception as e:
            print(f"Error initializing video capture: {e}")
            return
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        try:
            # Only create windows if not in headless mode
            if not self.headless:
                try:
                    cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_title, frame_width, frame_height)
                except cv2.error as e:
                    print(f"Warning: Could not create OpenCV window: {e}")
                    self.headless = True  # Switch to headless mode if window creation fails
            
            self.fps_start_time = time.time()
            
            while True:
                # Read frame
                start_time = time.time()
                ret_val, frame = self.video_capture.read()
                if not ret_val:
                    print("End of video or failed to read frame")
                    # Optionally loop the video
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Perform YOLO inference on the frame
                results = self.model(frame)
                
                # Process results
                if len(results) > 0:
                    # Get the frame for custom drawing
                    plotted_frame = frame.copy()
                    result = results[0]
                    
                    # Get detection information
                    detection_array = np.empty((0, 5))  # Format: [x1, y1, x2, y2, confidence]
                    
                    if len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
                        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                        
                        # Use class names from the model
                        class_names = result.names
                        
                        # Filter detections with confidence threshold
                        min_confidence = 0.4
                        keep_idxs = confidences >= min_confidence
                        boxes = boxes[keep_idxs]
                        confidences = confidences[keep_idxs]
                        class_ids = class_ids[keep_idxs]
                        
                        # Combine boxes and confidence scores into detection_array for SORT
                        if len(boxes) > 0:
                            detection_array = np.hstack((boxes, confidences.reshape(-1, 1)))
                    
                    # Update SORT tracker with filtered detections
                    tracked_objects = self.tracker.update(detection_array)
                    
                    # Draw the tracked bounding boxes and IDs on the frame
                    for tracked in tracked_objects:
                        x1, y1, x2, y2, track_id = tracked
                        track_id = int(track_id)
                        
                        # Convert to integers for drawing
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Calculate box center for angle calculation
                        box_center_x = (x1 + x2) // 2
                        box_center_y = (y1 + y2) // 2
                        
                        # Calculate angles relative to camera center
                        h_angle, v_angle = self.calculate_angles(box_center_x, box_center_y, 
                                                                frame_width, frame_height)
                        
                        # Update current_angles property with the latest values (use horizontal as Y rotation)
                        # X rotation is vertical angle (pitch), Y rotation is horizontal angle (yaw), Z rotation stays at 0
                        self.current_angles = (v_angle, h_angle, 0.0)
                        
                        # Assign a consistent color for this track ID
                        if track_id not in self.track_colors:
                            self.track_colors[track_id] = (np.random.randint(0, 255),
                                                          np.random.randint(0, 255),
                                                          np.random.randint(0, 255))
                        color = self.track_colors[track_id]
                        
                        # Draw bounding box, ID, and angles
                        cv2.rectangle(plotted_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(plotted_frame, (box_center_x, box_center_y), 5, (0, 0, 255), -1)  # center point
                        cv2.putText(plotted_frame, 
                                   f"ID: {track_id} ({h_angle:.1f}°, {v_angle:.1f}°)", 
                                   (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    plotted_frame = frame
                
                # --- FPS Calculation ---
                self.frame_count += 1
                elapsed_time = time.time() - self.fps_start_time
                if elapsed_time > 1:  # Update FPS every second
                    self.fps = self.frame_count / elapsed_time
                    self.fps_values.append(self.fps)
                    # Keep only the last N FPS values
                    if len(self.fps_values) > self.fps_buffer_size:
                        self.fps_values.pop(0)
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                
                # Calculate statistics
                avg_fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
                max_fps = max(self.fps_values) if self.fps_values else 0
                
                # Add performance metrics overlay
                frame_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Setup text parameters
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (0, 255, 0)  # Green
                font_thickness = 2
                line_type = cv2.LINE_AA
                
                # Draw metrics
                cv2.putText(plotted_frame, f"FPS: {self.fps:.1f} (Avg: {avg_fps:.1f}, Max: {max_fps:.1f})", 
                           (10, 30), font, font_scale, font_color, font_thickness, line_type)
                cv2.putText(plotted_frame, f"Frame time: {frame_time:.1f} ms", 
                           (10, 60), font, font_scale, font_color, font_thickness, line_type)
                cv2.putText(plotted_frame, f"Resolution: {frame_width}x{frame_height}", 
                           (10, 90), font, font_scale, font_color, font_thickness, line_type)
                cv2.putText(plotted_frame, f"Device: {self.device}", 
                           (10, 120), font, font_scale, font_color, font_thickness, line_type)
                
                # Display the frame only if not in headless mode
                if not self.headless:
                    try:
                        cv2.imshow(self.window_title, plotted_frame)
                    except cv2.error:
                        # If display fails, switch to headless mode
                        self.headless = True
                
                # Check for exit key and thread status
                if not self.headless:
                    keyCode = cv2.waitKey(1) & 0xFF
                    if keyCode == 27 or keyCode == ord('q') or not self._running:  # Esc or q key or thread stop signal
                        break
                elif not self._running:  # Just check running status in headless mode
                    break
                    
                # Add small sleep in headless mode to prevent CPU hogging
                if self.headless:
                    time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in video processing: {e}")
        finally:
            # Clean up resources but don't call self.stop() which would attempt to join the current thread
            if hasattr(self, 'video_capture') and self.video_capture is not None:
                self.video_capture.release()
            if not self.headless:
                cv2.destroyAllWindows()
            self._running = False
            print("Video processing thread finished.")
    
    def calculate_angles(
            self,
            pixel_x: float,
            pixel_y: float,
            frame_width: int,
            frame_height: int
        ) -> tuple[float, float]:
        """
        Return the yaw (horizontal) and pitch (vertical) offsets, in degrees, from the
        camera’s forward axis that would send a ray through the pixel (pixel_x, pixel_y).

        Args
        ----
        pixel_x, pixel_y : coordinates of the point in the image.  The origin (0, 0)
                        is the **TOP‑LEFT** corner, as in OpenCV / most image APIs.
        frame_width, frame_height : resolution of the frame in pixels.

        Returns
        -------
        (horizontal_angle, vertical_angle) : tuple[float, float]
            • horizontal_angle  < 0 → pixel lies to the **left** of centre  
            • vertical_angle    < 0 → pixel lies **below** centre (positive is up)
        """

        # 1. Normalise pixel to [-1, +1] range centred at the optical axis
        centre_x = frame_width  * 0.5
        centre_y = frame_height * 0.5
        nx = (pixel_x - centre_x) / centre_x      # +right,  -left
        ny = (centre_y - pixel_y) / centre_y      # +up,     -down   (invert y‑axis)

        # 2. Pre‑compute tangents of half‑FOV (degrees → radians first)
        tan_h = math.tan(math.radians(self.horizontal_fov) * 0.5)
        tan_v = math.tan(math.radians(self.vertical_fov)   * 0.5)

        # 3. Exact angles via arctangent
        yaw_rad   = math.atan(nx * tan_h)   # left/right
        pitch_rad = math.atan(ny * tan_v)   # up/down

        return math.degrees(yaw_rad), -math.degrees(pitch_rad)
        
    def stop(self):
        """Stop the detector thread"""
        self._running = False
        # Only join the thread if we're not called from within the thread itself
        if self._thread and self._thread.is_alive() and self._thread != threading.current_thread():
            self._thread.join(timeout=2.0)  # Wait for thread to finish with timeout
        if hasattr(self, 'video_capture') and self.video_capture is not None:
            self.video_capture.release()
        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignore OpenCV errors during cleanup
        print("Drone detection stopped.")


if __name__ == "__main__":
    detector = DroneDetector()
    detector.start()
