import os
import pytest
from scripts.prepare_dataset import coco_to_yolo_bbox

def test_coco_to_yolo_bbox():
    # Normalized bbox [x_min, y_min, w, h]
    coco_bbox = [0.2, 0.3, 0.4, 0.5]
    xc, yc, w, h = coco_to_yolo_bbox(coco_bbox, 384, 384)

    assert abs(xc - 0.4) < 1e-4
    assert abs(yc - 0.55) < 1e-4
    assert abs(w - 0.4) < 1e-4
    assert abs(h - 0.5) < 1e-4
    assert 0 <= xc <= 1 and 0 <= yc <= 1
