import cv2

def is_camera_available(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        return False
    cap.release()
    return True

def initialize_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    return cap

def initialize_video(video_path):
    """
    Initialize a video capture object from a video file.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        cv2.VideoCapture: Video capture object if successful, None otherwise
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    return cap
