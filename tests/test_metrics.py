import numpy as np
import pytest
from evaluation.image_quality import ImageQualityEvaluator

def test_no_reference_metrics():
    evaluator = ImageQualityEvaluator()
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    metrics = evaluator.calculate_no_reference_metrics(dummy_img)

    assert "mean_brightness" in metrics
    assert "contrast" in metrics
    assert "entropy" in metrics
    assert "spatial_frequency" in metrics

def test_reference_metrics():
    evaluator = ImageQualityEvaluator()
    ref_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    tar_img = np.full((100, 100, 3), 128, dtype=np.uint8)

    metrics = evaluator.calculate_reference_metrics(ref_img, tar_img)
    assert metrics["psnr"] > 40.0
    assert metrics["ssim"] == 1.0
