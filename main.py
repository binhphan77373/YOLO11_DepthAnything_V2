from yolo_model import load_yolo_model, detect_objects
from depth_model import load_depth_model, estimate_depth
from camera_utils import initialize_camera, is_camera_available, initialize_video
from draw_utils import draw_detection
import cv2
import torch
import os
import argparse

# Constants
CONFIDENCE_THRESHOLD = 0.7
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def process_webcam():
    yolo_model = load_yolo_model(device)
    depth_model = load_depth_model(device)

    cap = initialize_camera()
    if not cap:
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        detections = detect_objects(yolo_model, resized_frame, device)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        depth_map = estimate_depth(depth_model, rgb_frame, device)

        for detection in detections:
            x1, y1, x2, y2, conf = detection
            if conf >= CONFIDENCE_THRESHOLD:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth_value = depth_map[center_y, center_x] if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0] else 0.0
                draw_detection(resized_frame, x1, y1, x2, y2, conf, depth_value)

        cv2.imshow("Webcam Object Detection with Depth", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path, output_path=None):
    """
    Process a video file for object detection with depth estimation.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path to save the output video. If None, no output video is saved.
    """
    yolo_model = load_yolo_model(device)
    depth_model = load_depth_model(device)

    cap = initialize_video(video_path)
    if not cap:
        print(f"Error: Could not open video file {video_path}.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer if output path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
    
    print(f"Processing video with {total_frames} frames...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
        
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        detections = detect_objects(yolo_model, resized_frame, device)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        depth_map = estimate_depth(depth_model, rgb_frame, device)

        for detection in detections:
            x1, y1, x2, y2, conf = detection
            if conf >= CONFIDENCE_THRESHOLD:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth_value = depth_map[center_y, center_x] if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0] else 0.0
                draw_detection(resized_frame, x1, y1, x2, y2, conf, depth_value)
        
        # Display the frame
        cv2.imshow("Video Object Detection with Depth", resized_frame)
        
        # Write frame to output video if specified
        if out:
            out.write(resized_frame)
            
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed. Processed {frame_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection with Depth Estimation")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to save output video file (optional)")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of video file")
    
    args = parser.parse_args()
    
    if args.webcam:
        if is_camera_available():
            process_webcam()
        else:
            print("Error: No camera detected.")
    elif args.video:
        if os.path.exists(args.video):
            process_video(args.video, args.output)
        else:
            print(f"Error: Video file {args.video} not found.")
    else:
        print("No input source specified. Use --video or --webcam option.")
        print("Example: python main.py --video AriaEverydayActivities_1.0.0_loc1_script1_seq1_rec1_preview_rgb.mp4")
