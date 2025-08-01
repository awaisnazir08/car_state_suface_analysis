import torch
import clip
from PIL import Image
import os
import cv2
import numpy as np
import pickle
from collections import deque
from pathlib import Path

# --- Configuration ---
CLIP_MODEL_NAME = "ViT-L/14@336px"
CLASSIFIER_PATH = "logistic_crash_model.pkl"
VISUALIZE = True # Set to True to see the video processed in real-time
VIDEO_PATH = Path(r"E:\VS Code Folders\i3d\videos\vid_746.mp4")

# --- Smoothing and Prediction Hyper-Parameters ---
FRAME_SKIP = 1  # Analyze every N-th frame; 1 means every frame
SMOOTHING_WINDOW_SIZE = 15  # Number of frames to average predictions over
CONFIDENCE_THRESHOLD = 0.5  # Threshold of "crash" votes to classify as a crash

class VideoClassifier:
    def __init__(self, clip_model_name, classifier_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model_name = clip_model_name
        self.classifier_path = classifier_path
        self.model, self.preprocess = self._load_clip_model()
        self.classifier = self._load_classifier()
    
    def _load_clip_model(self):
        print("Loading OpenAI CLIP model...")
        model, preprocess = clip.load(self.clip_model_name, device=self.device)
        model.eval()
        print(f"CLIP model loaded on {self.device}.")
        return model, preprocess
    
    def _load_classifier(self):
        print(f"Loading classifier from: {self.classifier_path}")
        with open(self.classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Classifier loaded successfully.")
        return classifier
    
    def process_frame(self, frame):
        """
        Processes a single frame to get a prediction.
        Args:
            frame (numpy.ndarray): The input video frame in BGR format.
        Returns:
            tuple: The predicted index (0 for 'no crash', 1 for 'crash') and the raw probability.
        """
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_feat = self.model.encode_image(img_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
        
        embedding_np = img_feat.cpu().numpy()
        pred_index = self.classifier.predict(embedding_np)[0]
        prob = self.classifier.predict_proba(embedding_np)[0].max()
        
        return pred_index, prob

def main():
    """
    Main function to run video inference with smoothing and visualization.
    """
    output_dir = Path("crashes")
    output_dir.mkdir(exist_ok=True)
    output_video_path = output_dir / f"{VIDEO_PATH.stem}_annotated.mp4"

    classifier = VideoClassifier(CLIP_MODEL_NAME, CLASSIFIER_PATH)
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # --- State Variables ---
    frame_count = 0
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    current_smoothed_state = 0 # starting assumption: no crash
    crash_ratio = 0.0

    print("Starting video processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- Prediction (on skipped frames) ---
        if frame_count % FRAME_SKIP == 0:
            prediction_label, _ = classifier.process_frame(frame)
            prediction_buffer.append(prediction_label)

            # --- Smoothing ---
            if len(prediction_buffer) == SMOOTHING_WINDOW_SIZE:
                crash_votes = sum(prediction_buffer)
                crash_ratio = crash_votes / SMOOTHING_WINDOW_SIZE
                current_smoothed_state = 1 if crash_ratio >= CONFIDENCE_THRESHOLD else 0
        
        label_text = "Crash" if current_smoothed_state == 1 else "No Crash"
        
        # Calculate the confidence score for the CURRENT displayed state
        confidence_score = crash_ratio if current_smoothed_state == 1 else 1 - crash_ratio

        display_text = f"{label_text} (State confidence: {confidence_score:.2f})"
        color = (0, 0, 255) if current_smoothed_state == 1 else (0, 255, 0) # Red for crash, Green for no crash
        
        # Add a black background for better text visibility
        (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(frame, (10, 10), (20 + text_width, 20 + text_height + 10), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (15, 15 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)

        if VISUALIZE:
            cv2.imshow('Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Annotated video saved to: {output_video_path}")

if __name__ == "__main__":
    main()