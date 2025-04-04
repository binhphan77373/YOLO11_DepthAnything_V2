---

# Object Detection and Depth Estimation App – Depth-MonocularVision

This application combines object detection using **YOLOv8** and depth estimation using the **Depth-Anything-V2** model.

## Requirements

- Python 3.8+
- PyTorch  
- OpenCV  
- Ultralytics (YOLOv8)

## Project Structure

```
Depth-MonocularVision/
├── checkpoints/ - Folder containing pre-trained models
│   ├── depth_anything_v2_vits.pth - Depth estimation model
│   └── yolo11n.pt - YOLOv8 model for object detection
├── depth_anything_v2/ - Source code for the Depth-Anything-V2 model
├── camera_utils.py - Utilities for handling camera and video
├── config.py - Application configuration
├── depth_model.py - Handles depth estimation
├── draw_utils.py - Drawing utilities for displaying results
├── main.py - Main application entry point
└── yolo_model.py - Handles object detection
```

## Usage

### Video Input

You can run the application with a video file input:

```bash
python main.py --video path_to_video_file
```

Example:

```bash
python main.py --video AriaEverydayActivities_1.0.0_loc1_script1_seq1_rec1_preview_rgb.mp4
```

To save the output video:

```bash
python main.py --video path_to_video_file --output path_to_output_video
```

Example:

```bash
python main.py --video AriaEverydayActivities_1.0.0_loc1_script1_seq1_rec1_preview_rgb.mp4 --output result.mp4
```

### Using Webcam

To run the application using the webcam:

```bash
python main.py --webcam
```

## Controls

- Press `q` to exit the application while it's running.

## Output Display

The application will display:
- Bounding boxes around detected objects
- Detection confidence score
- Depth value (in meters) at the center of each object

--- 