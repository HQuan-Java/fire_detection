from ultralytics import YOLO


class YOLOFireDetector:
    def __init__(self, model_path="models/fire_yolo.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })

        return detections