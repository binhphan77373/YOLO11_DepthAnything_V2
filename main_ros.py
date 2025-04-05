from yolo_model import load_yolo_model, detect_objects
from depth_model import load_depth_model, estimate_depth
from draw_utils import draw_detection
import cv2
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node

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

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

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

        cv2.imshow("ROS Image Object Detection with Depth", resized_frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    depth_monocular_vision = DepthMonocularVision()
    rclpy.spin(depth_monocular_vision)
    depth_monocular_vision.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
