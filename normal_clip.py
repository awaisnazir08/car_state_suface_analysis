#!/usr/bin/env python
# onroad_offroad_clip_annotated.py
# Detects on‑road vs off‑road, then annotates and saves the video.

import cv2
import torch
import clip
from PIL import Image
from collections import deque
import numpy as np
import datetime
import csv
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1.  Input video (EDIT ONLY THIS)
# --------------------------------------------------------------------------- #
VIDEO_PATH = Path(r"E:\VS Code Folders\i3d\videos\vid_772.mp4")

# --------------------------------------------------------------------------- #
# 2.  Auto‑generate output paths
# --------------------------------------------------------------------------- #
# (i) Make a sub‑folder called “road” next to the video file
out_dir = Path("road")
out_dir.mkdir(exist_ok=True)

# (ii) Build names: <orig‑stem>_annotated.mp4  and  <orig‑stem>_log.csv
OUTPUT_VIDEO_PATH = out_dir / f"{VIDEO_PATH.stem}_annotated.mp4"
OUTPUT_CSV_PATH   = out_dir / f"{VIDEO_PATH.stem}_log.csv"

print("Output video  →", OUTPUT_VIDEO_PATH)
print("Output  log   →", OUTPUT_CSV_PATH)   # Path for the detailed log file

FRAME_SKIP = 1                                 # Analyze every N‑th frame for speed
SMOOTHING_WINDOW_SIZE = 15                     # Temporal consensus window
CONFIDENCE_THRESHOLD = 0.5                     # ≥50 % off‑road votes → state change

# --------------------------------------------------------------------------- #
# 2. Load CLIP (OpenAI repo, not Hugging Face)
# --------------------------------------------------------------------------- #
def load_clip_model():
    print("Loading OpenAI CLIP...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    model.eval()
    print(f"Model loaded on {device}.")
    return model, preprocess, device

# --------------------------------------------------------------------------- #
# 3. Text prompts
# --------------------------------------------------------------------------- #
ON_ROAD_PROMPTS = [
    "A rally car racing on a designated rally stage",
    "A rally car at high speed on a tarmac race track",
    "A rally car drifting on a gravel track through a forest",
    "A rally car navigating a snowy mountain race course",
    "A photo of a rally car competing on the intended road",
    "A rally car under control on a wet and muddy path",
    'A photo of a rally car driving on a country side road',
    'A photo of a rally car driving on a road between green fields',
    'A photo of a rally car driving very fast on road',
    'A photo of a rally car driving very quickly on road with motion blur in the background',
    'A photo of a rally car on the road',
    'A rally car driving on a road',
]

OFF_ROAD_PROMPTS = [
    "A rally car that has crashed or spun off the road",
    "A rally car that has gone into a field or a ditch",
    "A rally car hitting a tree off the side of the track",
    "A photo of a rally car accident during a race",
    "A car that has lost control and left the designated race course",
    "A rally car stuck in the mud after leaving the track",
    'A photo of a car on the side of the road',
    'A photo of a car in the field',
    'A photo of a car in the ditch',
    'A photo of a car in greenery'

]

# --------------------------------------------------------------------------- #
# 4. Main video‑processing loop
# --------------------------------------------------------------------------- #
def analyze_and_annotate_video(model, preprocess, device):
    all_prompts = ON_ROAD_PROMPTS + OFF_ROAD_PROMPTS
    on_road_count = len(ON_ROAD_PROMPTS)

    # Encode every prompt **once** for efficiency
    with torch.no_grad():
        text_tokens = clip.tokenize(all_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # --- Video Input and Output Setup ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: cannot open video file {VIDEO_PATH}")
        return

    # Get video properties for the output writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    # 'mp4v' is a good choice for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    print(f"Output video will be saved to {OUTPUT_VIDEO_PATH}")

    # --- State and Logging Variables ---
    frame_num = 0
    current_state = "on_road"
    prediction_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    analysis_log = []

    print("\n--- Starting video analysis and annotation ---")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        # --- Analysis is done intermittently ---
        if frame_num % FRAME_SKIP == 0:
            # BGR → RGB → tensor for CLIP
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                logits = 100.0 * img_feat @ text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            prob_on = probs[:on_road_count].sum()
            prob_off = probs[on_road_count:].sum()
            frame_pred = "on_road" if prob_on > prob_off else "off_road"
            prediction_buffer.append(frame_pred)
            
            # Update the smoothed state only after the buffer is full
            if len(prediction_buffer) == SMOOTHING_WINDOW_SIZE:
                off_votes = prediction_buffer.count("off_road")
                ratio = off_votes / SMOOTHING_WINDOW_SIZE
                new_state = "off_road" if ratio >= CONFIDENCE_THRESHOLD else "on_road"
                current_state = new_state # Update the state for drawing

            # Log the analysis results
            timestamp_sec = frame_num / fps
            analysis_log.append({
                "frame": frame_num,
                "timestamp_sec": f"{timestamp_sec:.2f}",
                "raw_prediction": frame_pred,
                "smoothed_state": current_state
            })

        # --- Annotation is done for EVERY frame ---
        # This ensures the output video is smooth
        text_to_display = f"STATE: {current_state.upper()}"
        color = (0, 255, 0) if current_state == "on_road" else (0, 0, 255) # Green for on-road, Red for off-road (BGR)
        
        # Add a black background rectangle for text readability
        (text_width, text_height), _ = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(frame, (10, 10), (20 + text_width, 30 + text_height), (0,0,0), -1)

        # Put the state text on the frame
        cv2.putText(frame, text_to_display, (20, 20 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Write the annotated frame to the output video
        video_writer.write(frame)
        
        frame_num += 1
        if frame_num % (int(fps)*5) == 0: # Print progress every 5 seconds of video
             print(f"  Processed {str(datetime.timedelta(seconds=int(frame_num/fps)))} of video...")


    # --- Cleanup and Finalizing ---
    cap.release()
    video_writer.release()
    print("\n--- Video analysis finished ---")

    # Save the CSV log file
    if analysis_log:
        print(f"Writing analysis log to {OUTPUT_CSV_PATH}...")
        with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=analysis_log[0].keys())
            writer.writeheader()
            writer.writerows(analysis_log)
        print("Log saved successfully.")
    
    print(f"Annotated video saved successfully to {OUTPUT_VIDEO_PATH}")


# --------------------------------------------------------------------------- #
# 5. Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model, preprocess, device = load_clip_model()
    if model:
        analyze_and_annotate_video(model, preprocess, device)