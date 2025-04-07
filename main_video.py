from yolo_model import load_yolo_model, detect_objects
from depth_model import load_depth_model, estimate_depth
from camera_utils import initialize_camera, is_camera_available, initialize_video
from draw_utils import draw_detection
import cv2
import torch
import os
import argparse
import multiprocessing as mp
from queue import Empty
import time

# Constants
CONFIDENCE_THRESHOLD = 0.7
FRAME_WIDTH = 640
FRAME_HEIGHT = 640

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

def process_frame(frame_data):
    frame, frame_idx, yolo_model, depth_model, device = frame_data
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
    
    return frame_idx, resized_frame

def process_video(video_path, output_path=None):
    """
    Process a video file for object detection with depth estimation using multiprocessing.
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
    
    # Create a pool of workers
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes")
    pool = mp.Pool(processes=num_processes)
    
    # Create a queue for processed frames
    result_queue = mp.Queue()
    frame_buffer = {}
    next_frame_to_write = 0
    
    # FPS calculation variables
    start_time = time.time()
    frame_times = []
    current_fps = 0
    fps_update_interval = 1.0  # Update FPS every second
    last_fps_update = start_time
    
    try:
        frame_count = 0
        while True:
            # Read frames and submit to pool
            frames_to_process = []
            for _ in range(num_processes):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_to_process.append((frame, frame_count, yolo_model, depth_model, device))
                frame_count += 1
            
            if not frames_to_process:
                break
            
            # Process frames in parallel
            results = pool.map(process_frame, frames_to_process)
            
            # Put results in queue
            for frame_idx, processed_frame in results:
                result_queue.put((frame_idx, processed_frame))
            
            # Write frames in order
            while not result_queue.empty():
                try:
                    frame_idx, processed_frame = result_queue.get_nowait()
                    frame_buffer[frame_idx] = processed_frame
                    
                    # Write frames in order
                    while next_frame_to_write in frame_buffer:
                        frame_to_write = frame_buffer.pop(next_frame_to_write)
                        
                        # Calculate and display FPS
                        current_time = time.time()
                        frame_times.append(current_time)
                        
                        # Remove frame times older than 1 second
                        while frame_times and current_time - frame_times[0] > 1.0:
                            frame_times.pop(0)
                        
                        # Update FPS display every second
                        if current_time - last_fps_update >= fps_update_interval:
                            current_fps = len(frame_times)
                            last_fps_update = current_time
                        
                        # Add FPS text to frame
                        fps_text = f"FPS: {current_fps:.1f}"
                        cv2.putText(frame_to_write, fps_text, (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow("Video Object Detection with Depth", frame_to_write)
                        if out:
                            out.write(frame_to_write)
                        next_frame_to_write += 1
                        
                        if next_frame_to_write % 10 == 0:
                            print(f"Processed frame {next_frame_to_write}/{total_frames} ({(next_frame_to_write/total_frames)*100:.1f}%) - FPS: {current_fps:.1f}")
                except Empty:
                    break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error during video processing: {str(e)}")
    finally:
        # Clean up
        pool.close()
        pool.join()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Calculate average FPS
        total_time = time.time() - start_time
        average_fps = next_frame_to_write / total_time if total_time > 0 else 0
        print(f"Video processing completed. Processed {next_frame_to_write} frames.")
        print(f"Average FPS: {average_fps:.2f}")

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
