from ultralytics import YOLO

def load_yolo_model(device):
    print("Loading YOLOv8 model...")
    model = YOLO("./checkpoints/yolo11n.pt").to(device)
    return model

def detect_objects(model, frame, device):
    results = model.predict(frame, device=device)
    detections = []

    for detection in results[0].boxes:
        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
        conf = detection.conf.cpu().item()
        detections.append((x1, y1, x2, y2, conf))
    
    return detections
