from yolo_model import load_yolo_model, detect_objects
from depth_model import load_depth_model, estimate_depth
from draw_utils import draw_detection
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from concurrent.futures import ThreadPoolExecutor
import threading

# Constants
CONFIDENCE_THRESHOLD = 0.7
FRAME_WIDTH = 640
FRAME_HEIGHT = 640

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DepthMonocularVision(Node):
    def __init__(self):
        super().__init__('depth_monocular_vision')
        self.subscription = self.create_subscription(
            Image,
            'aria/rgb_image',
            self.listener_callback,
            1
        )
        self.bridge = CvBridge()
        self.yolo_model = load_yolo_model(device)
        self.depth_model = load_depth_model(device)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None

    def process_frame(self, frame):
        resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        detections = detect_objects(self.yolo_model, resized_frame, device)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        depth_map = estimate_depth(self.depth_model, rgb_frame, device)

        for detection in detections:
            x1, y1, x2, y2, conf = detection
            if conf >= CONFIDENCE_THRESHOLD:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth_value = depth_map[center_y, center_x] if 0 <= center_x < depth_map.shape[1] and 0 <= center_y < depth_map.shape[0] else 0.0
                draw_detection(resized_frame, x1, y1, x2, y2, conf, depth_value)
        
        return resized_frame

    def display_frame(self):
        while rclpy.ok():
            with self.lock:
                if self.processed_frame is not None:
                    cv2.imshow("ROS Image Object Detection with Depth", self.processed_frame)
                    cv2.waitKey(1)
            rclpy.spin_once(self, timeout_sec=0.01)

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        with self.lock:
            self.current_frame = frame

        # Xử lý frame trong một thread riêng
        future = self.executor.submit(self.process_frame, frame)
        future.add_done_callback(self._process_frame_callback)

    def _process_frame_callback(self, future):
        try:
            processed_frame = future.result()
            with self.lock:
                self.processed_frame = processed_frame
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

def main(args=None):
    rclpy.init(args=args)
    depth_monocular_vision = DepthMonocularVision()
    
    # Tạo thread riêng cho việc hiển thị
    display_thread = threading.Thread(target=depth_monocular_vision.display_frame)
    display_thread.daemon = True
    display_thread.start()
    
    try:
        rclpy.spin(depth_monocular_vision)
    except KeyboardInterrupt:
        pass
    finally:
        depth_monocular_vision.executor.shutdown()
        depth_monocular_vision.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
