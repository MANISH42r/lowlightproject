import os
import time
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class YOLOObjectDetector:
    def __init__(self, model_path=None, config_path="config.yaml"):
        self.config = load_config(config_path)
        cfg_det = self.config["detection"]

        if model_path is None:
            model_path = os.path.join(cfg_det["save_dir"], "baseline", "weights", "best.pt")
            if not os.path.exists(model_path):
                model_path = cfg_det.get("model_name", "yolov8n.pt")

        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.categories = self.config["dataset"]["categories"]

    def predict(self, image_input, conf_threshold=0.25):
        """
        image_input: PIL Image, numpy array (RGB or BGR), or file path.
        Returns:
            detections: list of dicts [{'class_id', 'class_name', 'confidence', 'bbox': [x1, y1, x2, y2]}]
            latency_ms: inference latency in milliseconds
            annotated_img: PIL Image with rendered bounding boxes and labels
        """
        if isinstance(image_input, str):
            img_pil = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                img_pil = Image.fromarray(image_input)
            else:
                img_pil = Image.fromarray(image_input)
        else:
            img_pil = image_input.convert('RGB')

        start_time = time.time()
        results = self.model.predict(
            source=img_pil,
            conf=conf_threshold,
            imgsz=self.config["detection"].get("image_size", 384),
            device=self.device,
            verbose=False
        )[0]
        latency_ms = (time.time() - start_time) * 1000.0

        detections = []
        boxes = results.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            cls_name = results.names.get(cls_id, self.categories[cls_id] if cls_id < len(self.categories) else f"class_{cls_id}")

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

        # Render annotated image
        res_plotted = results.plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(res_rgb)

        return detections, round(latency_ms, 2), annotated_pil
