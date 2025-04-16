from ultralytics import YOLO
import cv2
import numpy as np
from time import time
import torch
import os
from detection.sort import Sort

class DroneDetector:
    def __init__(self, model_path=None, video_path=None, confidence=0.4, 
                 horizontal_fov=62.2, vertical_fov=48.8):
        if model_path is None:
            # Always resolve relative to this file's directory
            model_path = os.path.join(os.path.dirname(__file__), "weights", "weightsM50Epoch.pt")
        if video_path is None:
            # Always resolve relative to this file's directory
            video_path = os.path.join(os.path.dirname(__file__), "assets", "videos", "drone_video.mp4")
        self.window_title = "Drone Detection (MPS)"
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
        
    def start(self):
        # Open video file
        self.video_capture = cv2.VideoCapture(self.video_path)
        if not self.video_capture.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return
        
        # Get video properties
        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        try:
            cv2.namedWindow(self.window_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_title, frame_width, frame_height)
            
            self.fps_start_time = time()
            
            while True:
                # Read frame
                start_time = time()
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
                elapsed_time = time() - self.fps_start_time
                if elapsed_time > 1:  # Update FPS every second
                    self.fps = self.frame_count / elapsed_time
                    self.fps_values.append(self.fps)
                    # Keep only the last N FPS values
                    if len(self.fps_values) > self.fps_buffer_size:
                        self.fps_values.pop(0)
                    self.frame_count = 0
                    self.fps_start_time = time()
                
                # Calculate statistics
                avg_fps = sum(self.fps_values) / len(self.fps_values) if self.fps_values else 0
                max_fps = max(self.fps_values) if self.fps_values else 0
                
                # Add performance metrics overlay
                frame_time = (time() - start_time) * 1000  # Convert to milliseconds
                
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
                
                # Display the frame
                cv2.imshow(self.window_title, plotted_frame)
                
                # Check for exit key
                keyCode = cv2.waitKey(1) & 0xFF
                if keyCode == 27 or keyCode == ord('q'):  # Esc or q key
                    break
                
        finally:
            self.stop()
    
    def calculate_angles(self, pixel_x, pixel_y, frame_width, frame_height):
        """Calculate horizontal and vertical angles relative to camera center.
        
        Args:
            pixel_x: x-coordinate of the point in the image
            pixel_y: y-coordinate of the point in the image
            frame_width: width of the frame
            frame_height: height of the frame
            
        Returns:
            horizontal_angle, vertical_angle: Angles in degrees relative to camera center
        """
        # Calculate center of the frame
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        # Calculate normalized position from -1 to 1 (where 0 is center)
        norm_x = (pixel_x - center_x) / (frame_width / 2)
        norm_y = (pixel_y - center_y) / (frame_height / 2)
        
        # Calculate angles using the field of view
        horizontal_angle = norm_x * (self.horizontal_fov / 2)
        vertical_angle = norm_y * (self.vertical_fov / 2)  # Positive is down from center
        
        return horizontal_angle, -vertical_angle  # Invert vertical angle so positive is up
        
    def stop(self):
        if hasattr(self, 'video_capture') and self.video_capture is not None:
            self.video_capture.release()
        cv2.destroyAllWindows()
        print("Drone detection stopped.")


if __name__ == "__main__":
    detector = DroneDetector()
    detector.start()
