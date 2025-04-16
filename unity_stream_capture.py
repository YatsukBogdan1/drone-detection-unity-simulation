import numpy as np
import cv2
from mss import mss
import time

class ScreenCapture:
    """A class that mimics cv2.VideoCapture but uses mss for screen capture"""
    
    def __init__(self, source=0):
        """Initialize the screen capture.
        
        Args:
            source: If integer, treated as monitor number.
                   If dict, treated as region (e.g., {'top': 100, 'left': 0, 'width': 1920, 'height': 1080})
        """
        self.sct = mss()
        self.source = source
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps = 0
        self.is_opened = True
        
        # Determine capture region
        if isinstance(source, dict):
            self.region = source
        elif isinstance(source, int):
            # Use the specified monitor
            self.region = self.sct.monitors[source] if source < len(self.sct.monitors) else self.sct.monitors[0]
        else:
            # Default to monitor 0
            self.region = self.sct.monitors[0]
        
        # Get the resolution
        self.width = self.region['width']
        self.height = self.region['height']
    
    def read(self):
        """Read a new frame, similar to cv2.VideoCapture.read()
        
        Returns:
            tuple: (ret, frame) where ret is True if frame is read correctly
        """
        if not self.is_opened:
            return False, None
        
        try:
            # Capture screen
            sct_img = self.sct.grab(self.region)
            
            # Convert to numpy array (BGRA)
            frame_bgra = np.array(sct_img)
            
            # Convert BGRA to BGR (OpenCV format)
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            
            # Ensure it's contiguous and the correct type
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            
            # Update frame count and calculate FPS
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:
                self.last_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.start_time = time.time()
            
            return True, frame
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return False, None
    
    def isOpened(self):
        """Check if screen capture is open"""
        return self.is_opened
    
    def get(self, propId):
        """Gets a property value, similar to cv2.VideoCapture.get()"""
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        elif propId == cv2.CAP_PROP_FPS:
            return self.last_fps
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            # Screen capture doesn't have frame count, return -1
            return -1
        elif propId == cv2.CAP_PROP_POS_FRAMES:
            # Screen capture doesn't have position, return -1
            return -1
        else:
            return 0
    
    def set(self, propId, value):
        """Sets a property, similar to cv2.VideoCapture.set()"""
        # Most properties can't be set for screen capture
        return False
    
    def release(self):
        """Release resources"""
        self.is_opened = False
        # Close mss
        self.sct.close()


# Example usage
if __name__ == "__main__":
    # Create a screen capture for the primary monitor
    cap = ScreenCapture(1)
    
    # Or specify a custom region
    # cap = ScreenCapture({'top': 0, 'left': 0, 'width': 1920, 'height': 1080})
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow('Screen Capture', frame)
        
        # Display FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            print(f"FPS: {fps:.2f}")
        
        # Press 'q' to exit
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()