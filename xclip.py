#!/usr/bin/env python
# Detects activities (crash, drift, jump) in rally videos using X-CLIP.

import cv2
import torch
from transformers import XCLIPProcessor, XCLIPModel
from PIL import Image
from collections import deque
import numpy as np
import csv
import datetime
from pathlib import Path

VIDEO_PATH = Path(r"E:\VS Code Folders\i3d\videos\vid_857.mp4")

out_dir = Path("events_detection")
out_dir.mkdir(exist_ok=True)

OUTPUT_CSV_PATH   = out_dir / f"{VIDEO_PATH.stem}_log.csv"

print("Output  log   →", OUTPUT_CSV_PATH)

# --- Model & Processing ---
# How long each video clip we analyze should be (in seconds)
CLIP_DURATION_SEC = 1.0

# How far to slide the window forward for the next analysis (in seconds).
# A smaller stride means more overlap and better temporal resolution, but more computation.
SLIDE_STRIDE_SEC = 0.5

# X-CLIP models are trained on a fixed number of frames. Do not change this unless you use a different model.
NUM_SAMPLED_FRAMES = 8

# The minimum confidence score required to log an event.
DETECTION_THRESHOLD = 0.5 # confidence


# We group multiple descriptive prompts for each activity to improve robustness.
ACTIVITY_PROMPTS = {
    "crash": [
        "a video of a rally car crashing",
        "a video of a rally car spinning out of control and hitting something",
        "a video of a rally car rolling over",
        'a video of a rally car crashing side of the road',
        'a video of a rally car hitting a side wall',
        'a video of a rally car going into a ditch',
        'a video of a rally car hitting a tree',
        'a video of a rally car hitting a person',
    ],
    "drift": [
        "a video of a rally car drifting around a corner",
        "a video of a rally car sliding sideways on the road",
        "a video of a powerslide",
        'a video of a rally car drifting on a muddy road',
        'a video of a rally car drifting on gravel',
        'a video of a rally car drifting on a race track'
    ],
    "jump": [
        "a video of a rally car jumping over a crest",
        "a video of a rally car going airborne",
        "a video of a rally car landing after a jump",
        'a video of a rally car in air',
        'a video of a rally car with all wheels in the air',
        'a video of a rally car performing a jump',
        'a video of a rally car going in the air from ground'
    ],
    "normal_driving": [
        "a video of a rally car driving normally on a race track",
        "a video of a rally car driving straight at high speed",
        "a video of a rally car navigating a turn without incident",
        'a video of a rally car driving on a muddy road',
        'a video of a rally car driving on a countryside road',
        'a video of a rally car driving on a road',
    ]
}

def analyze_rally_video():
    """
    Analyzes a rally video using a sliding window approach with X-CLIP.
    """
    print("Loading X-CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/xclip-base-patch16"
    
    try:
        processor = XCLIPProcessor.from_pretrained(model_name)
        model = XCLIPModel.from_pretrained(model_name).to(device)
        model.eval()
        print(f"Model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Prepare prompts and video reader ---
    all_prompts = [prompt for sublist in ACTIVITY_PROMPTS.values() for prompt in sublist]
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    clip_frames = int(CLIP_DURATION_SEC * fps)
    stride_frames = int(SLIDE_STRIDE_SEC * fps)
    
    frame_buffer = deque(maxlen=clip_frames)
    event_log = []
    frame_num = 0

    print("\n--- Starting Video Analysis ---")
    print(f"Analyzing in {CLIP_DURATION_SEC}s clips, sliding by {SLIDE_STRIDE_SEC}s...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR (OpenCV) to RGB (PIL) and add to buffer
        frame_buffer.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        # Once the buffer is full and we are at a stride interval, process the clip
        if len(frame_buffer) == clip_frames and frame_num % stride_frames == 0:
            
            # --- Sample frames evenly from the buffer ---
            indices = np.linspace(0, len(frame_buffer) - 1, NUM_SAMPLED_FRAMES, dtype=int)
            sampled_frames = [list(frame_buffer)[i] for i in indices]

            # --- Process with X-CLIP ---
            inputs = processor(
                text=all_prompts,
                videos=sampled_frames,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Get probabilities
            logits_per_video = outputs.logits_per_video
            probs = logits_per_video.softmax(dim=1).cpu().numpy()[0]

            # --- Aggregate and interpret results ---
            class_probs = {}
            current_pos = 0
            for activity, prompts in ACTIVITY_PROMPTS.items():
                # Sum the probabilities of all prompts for this activity
                class_probs[activity] = np.sum(probs[current_pos : current_pos + len(prompts)])
                current_pos += len(prompts)
            
            # Identify all activities that exceed the detection threshold
            for activity, prob in class_probs.items():
                if activity != "normal_driving" and prob >= DETECTION_THRESHOLD:
                    start_time = (frame_num - clip_frames) / fps
                    end_time = frame_num / fps
                    
                    log_entry = {
                        "start_time_sec": f"{start_time:.2f}",
                        "end_time_sec": f"{end_time:.2f}",
                        "detected_activity": activity,
                        "confidence": f"{prob:.2%}"
                    }
                    event_log.append(log_entry)
                    
                    start_ts = str(datetime.timedelta(seconds=int(start_time)))
                    print(f"[{start_ts}] DETECTED: {activity.upper()} (Confidence: {prob:.2%})")

        frame_num += 1

    cap.release()
    print("\n--- Video Analysis Finished ---")

    # --- Save results to CSV ---
    if event_log:
        print(f"Writing {len(event_log)} detected events to {OUTPUT_CSV_PATH}...")
        with open(OUTPUT_CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=event_log[0].keys())
            writer.writeheader()
            writer.writerows(event_log)
        print("Log saved successfully.")
    else:
        print("No specific activities detected above the threshold.")


if __name__ == "__main__":
    analyze_rally_video()