import cv2
import numpy as np
from ultralytics import YOLO
# To use a tracker, you might need to install `filterpy`
# pip install filterpy
from collections import defaultdict

# A simple bounding box tracker (like a simplified SORT)
# This helps maintain a consistent motion analysis for a car across frames
class SimpleTracker:
    def __init__(self):
        self.tracks = defaultdict(lambda: {"last_pos": None, "age": 0})
        self.next_id = 0

    def update(self, boxes):
        # This is a very basic tracker for demonstration. 
        # For a real system, consider using a library like `norfair` or implementing SORT.
        # For this example, we'll just track the largest box.
        if not boxes:
            return None, None
        
        # Find the box with the largest area
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
        largest_box = boxes[np.argmax(areas)]
        
        # We'll just assign ID 0 to the largest car for simplicity
        return 0, largest_box

# --- Configuration ---
# Use a raw string or forward slashes for Windows paths
MODEL_PATH = r"E:\RCN\model\yolo11l_08-02-2025_best.pt" 
VIDEO_PATH = r"E:\VS Code Folders\i3d\videos\vid_772.mp4"
MOVEMENT_THRESHOLD = 1.5  # Tunable: Average pixel movement to be considered "moving"

# --- Initialization ---
# Load YOLOv8 model
model = YOLO(MODEL_PATH, task='detect')
cap = cv2.VideoCapture(VIDEO_PATH)
tracker = SimpleTracker()

# Optical Flow parameters
farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

prev_gray = None
car_id = None
car_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. Detect and Track the Car ---
    results = model(frame, verbose=False, classes=[0])[0] # Assuming class 0 is the car
    
    # Get bounding boxes for cars
    car_boxes = [box.xyxy[0].cpu().numpy().astype(int) for box in results.boxes]
    
    # Update tracker
    if car_boxes:
        car_id, car_box = tracker.update(car_boxes)
    else:
        car_id, car_box = None, None # No car detected

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    status = "Unknown"
    
    # --- 2. Calculate Optical Flow (if we have a previous frame) ---
    if prev_gray is not None and car_box is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **farneback_params)
        
        # --- 3. Separate Car and Background Regions using Masks ---
        height, width = frame.shape[:2]
        background_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Black out the car region from the background mask
        x1, y1, x2, y2 = car_box
        cv2.rectangle(background_mask, (x1, y1), (x2, y2), 0, -1)
        
        car_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)

        # --- 4. Analyze Flow Vectors ---
        # Get average motion of the background (approximates camera motion)
        bg_flow = flow[background_mask == 255]
        # Avoid division by zero if background is empty
        avg_bg_flow = np.mean(bg_flow, axis=0) if bg_flow.size > 0 else np.array([0.0, 0.0])

        # Get average motion of the car
        car_flow = flow[car_mask == 255]
        avg_car_flow = np.mean(car_flow, axis=0) if car_flow.size > 0 else np.array([0.0, 0.0])

        # --- 5. Calculate Relative Motion and Make Decision ---
        relative_flow = avg_car_flow - avg_bg_flow
        relative_motion_magnitude = np.linalg.norm(relative_flow)

        if relative_motion_magnitude > MOVEMENT_THRESHOLD:
            status = "Moving"
            color = (0, 255, 0)
        else:
            status = "Stationary"
            color = (0, 0, 255)
            
        cv2.putText(frame, f"Car is: {status} ({relative_motion_magnitude:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # --- Display Info ---
    if car_box is not None:
        x1, y1, x2, y2 = car_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Car not detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update previous frame
    prev_gray = current_gray

    cv2.imshow("Movement Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()