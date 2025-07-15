# Rally Car Video Analysis Toolkit

This repository contains a suite of tools for analyzing rally car videos, including activity detection (crash, drift, jump), on-road/off-road classification, car movement tracking, and annotation processing for machine learning datasets.

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Script Usage](#script-usage)
  - [1. XCLIP Activity Detection (`xclip.py`)](#1-xclip-activity-detection-xclippy)
  - [2. On-Road/Off-Road Detection (`normal_clip.py`)](#2-on-roadoff-road-detection-normal_clippy)
  - [3. Car Movement Tracking (`track_with_ema.py`, `track.py`)](#3-car-movement-tracking-track_with_emapy-trackpy)
  - [4. ActivityNet Annotation Preprocessing (`activitynet_preprocess.ipynb`)](#4-activitynet-annotation-preprocessing-activitynet_preprocessipynb)
- [Requirements](#requirements)
- [Notes](#notes)

## Features
- **Activity Detection**: Detects rally car activities (crash, drift, jump, normal driving) in videos using X-CLIP (HuggingFace).
- **On-Road/Off-Road Classification**: Classifies and annotates video frames as on-road or off-road using OpenAI CLIP.
- **Car Movement Tracking**: Tracks car movement and determines if the car is moving or stationary using YOLOv8 and optical flow.
- **Annotation Processing**: Converts Labelbox NDJSON exports into ActivityNet-style JSON for use with action recognition models.

## Repository Structure
```
.
├── xclip.py                  # Rally activity detection using X-CLIP (HuggingFace)
├── normal_clip.py            # On-road/off-road detection and annotation using OpenAI CLIP
├── track_with_ema.py         # Car movement tracking with YOLOv8 and EMA smoothing
├── track.py                  # Car movement tracking with YOLOv8 (basic)
├── activitynet_preprocess.ipynb # Jupyter notebook for processing Labelbox NDJSON to ActivityNet format
├── requirements.txt          # Python dependencies
└── [other files and folders] # Videos, outputs, and intermediate data
```

## Setup Instructions

### 1. Create a Conda Environment
It is recommended to use a Conda environment for dependency management, especially for GPU support with PyTorch.

```bash
# Create a new environment (Python 3.11)
conda create -n rallyenv python=3.11 -y
conda activate rallyenv
```

### 2. Install PyTorch
Install the correct version of PyTorch for your CUDA version.
See: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (for CUDA 12.1):
```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
Or for CPU-only:
```bash
pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Other Requirements
```bash
pip install -r requirements.txt
```
> **Note:**
> The OpenAI CLIP library will be installed from GitHub.
> If you encounter issues, try:
> ```bash
> pip install git+https://github.com/openai/CLIP.git
> ```

## Script Usage

### 1. XCLIP Activity Detection (`xclip.py`)
Detects rally car activities (crash, drift, jump, normal driving) in a video using the X-CLIP model from HuggingFace.

- **Input**: Path to a rally video (edit `VIDEO_PATH` in the script).
- **Output**: CSV log of detected activities with timestamps in `events_detection/`.

**Run:**
```bash
python xclip.py
```

### 2. On-Road/Off-Road Detection (`normal_clip.py`)
Classifies each frame as on-road or off-road using OpenAI CLIP, annotates the video, and saves both the annotated video and a CSV log.

- **Input**: Path to a rally video (edit `VIDEO_PATH` in the script).
- **Output**: Annotated video and CSV log in the `road/` directory.

**Run:**
```bash
python normal_clip.py
```

### 3. Car Movement Tracking (`track_with_ema.py`, `track.py`)
Tracks the main car in the video using YOLOv8 and determines if it is moving or stationary using optical flow.

- **Input**: Path to a video and YOLOv8 model (edit `VIDEO_PATH` and `MODEL_PATH` in the script).
- **Output**: Real-time display with bounding boxes and movement status.

**Run:**
```bash
python track_with_ema.py
# or
python track.py
```

### 4. ActivityNet Annotation Preprocessing (`activitynet_preprocess.ipynb`)
Processes Labelbox NDJSON exports and converts them into ActivityNet-style JSON annotations for use with action recognition models (e.g., ActionFormer).

- **Input**: NDJSON export from Labelbox.
- **Output**: `activitynet_annotations.json` and optionally downloaded videos.

**Open and run in Jupyter:**
```bash
jupyter notebook activitynet_preprocess.ipynb
```

## Requirements
See `requirements.txt` for all dependencies.

## Notes
This toolkit is designed for modular use. Each script can be run independently, but they share common dependencies managed through `requirements.txt`. Ensure all paths in the scripts are updated to match your local file structure.