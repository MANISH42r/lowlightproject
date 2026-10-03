import pytest
import numpy as np
from PIL import Image
from detection.inference import YOLOObjectDetector

def test_yolo_inference_dummy_image():
    detector = YOLOObjectDetector()
    dummy_img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
    
    dets, latency_ms, annotated_pil = detector.predict(dummy_img)

    assert isinstance(dets, list)
    assert latency_ms > 0.0
    assert isinstance(annotated_pil, Image.Image)
