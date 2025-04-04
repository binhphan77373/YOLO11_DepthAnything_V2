import cv2

def draw_detection(frame, x1, y1, x2, y2, confidence, depth_value):
    color = (0, 255, 0)  # Green
    label = f"Conf: {confidence:.2f} | Depth: {depth_value:.2f}m"

    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    cv2.rectangle(frame, (int(x1), int(y1) - text_size[1] - 5), (int(x1) + text_size[0], int(y1)), (0, 0, 0), -1)
    cv2.putText(frame, label, (int(x1), int(y1) - 5), font, font_scale, (255, 255, 255), thickness)
