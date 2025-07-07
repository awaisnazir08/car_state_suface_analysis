import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# A simple bounding box tracker (like a simplified SORT)
# This helps maintain a consistent motion analysis for a car across frames
class SimpleTracker:
    def __init__(self):
        # We don't need a complex tracker if we just follow the largest car
        pass

    def update(self, boxes):
        # This is a very basic tracker for demonstration.
        # For this example, we'll just track the largest box.
        if not boxes:
            return None, None
        
        # Find the box with the largest area
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        largest_box = boxes[np.argmax(areas)]
        
        # We'll just assign ID 0 to the largest car for simplicity
        return 0, largest_box

# --- Configuration ---
# Use a raw string or forward slashes for Windows paths
MODEL_PATH = r"E:\RCN\model\yolo11l_08-02-2025_best.pt"
VIDEO_PATH = r"Ralph1.-8.091613814046736,41.52594138988471.1741337616.r.mp4"
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
    winsize=21, # Increased for potentially more stable flow on fast objects
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# --- EMA Initialization ---
# This variable will store the smoothed motion magnitude over time.
ema_magnitude = 0.0
# Lower alpha = smoother output but slower to react to changes. 0.1-0.3 is a good range.
ALPHA = 0.2 

# Frame processing variables
prev_gray = None
car_id = None
car_box = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 1. Detect and Track the Car ---
    # Assuming class 0 is 'car'
    results = model(frame, verbose=False, classes=[0])[0]
    
    # Get bounding boxes for cars
    car_boxes = [box.xyxy[0].cpu().numpy().astype(int) for box in results.boxes]
    
    # Update tracker to find the main car
    if car_boxes:
        car_id, car_box = tracker.update(car_boxes)
    else:
        car_id, car_box = None, None # No car detected

    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Default status if no calculation is done
    status = "Unknown"
    color = (255, 0, 255) # Purple for Unknown
    
    # --- 2. Calculate Optical Flow (if we have a previous frame and a detected car) ---
    if prev_gray is not None and car_box is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, **farneback_params)
        
        # --- 3. Create a SINGLE, efficient mask ---
        height, width = frame.shape[:2]
        # Mask is True for background, False for car
        is_background = np.ones((height, width), dtype=bool)
        x1, y1, x2, y2 = car_box
        is_background[y1:y2, x1:x2] = False

        # --- 4. Analyze Flow Vectors ---
        # Get average motion of the background
        bg_flow = flow[is_background]
        avg_bg_flow = np.mean(bg_flow, axis=0) if bg_flow.size > 0 else np.array([0.0, 0.0])

        # Get average motion of the car
        car_flow = flow[~is_background] # Use `~` to invert the boolean mask
        avg_car_flow = np.mean(car_flow, axis=0) if car_flow.size > 0 else np.array([0.0, 0.0])

        # --- 5. Calculate Relative Motion ---
        relative_flow = avg_car_flow - avg_bg_flow
        relative_motion_magnitude = np.linalg.norm(relative_flow)
        
        # --- 6. EMA Update and Final Decision ---
        # Update the smoothed value using the current frame's magnitude
        ema_magnitude = (ALPHA * relative_motion_magnitude) + ((1 - ALPHA) * ema_magnitude)
        
        # Make the decision based on the SMOOTHED value
        if ema_magnitude > MOVEMENT_THRESHOLD:
            status = "Moving"
            color = (0, 255, 0) # Green
        else:
            status = "Stationary"
            color = (0, 0, 255) # Red
            
        # Display the smoothed EMA value for easier debugging and tuning
        cv2.putText(frame, f"Car is: {status} (Motion EMA: {ema_magnitude:.2f})", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # --- Display Info ---
    if car_box is not None:
        x1, y1, x2, y2 = car_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    else:
        # If no car is detected, reset the EMA so it doesn't hold an old value
        ema_magnitude = 0.0
        cv2.putText(frame, "Car not detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update previous frame for the next iteration
    prev_gray = current_gray

    cv2.imshow("Movement Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()