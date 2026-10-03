import os
import glob
import json
import yaml
import shutil
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from enhancement.inference import ZeroDCEEnhancer
from evaluation.image_quality import ImageQualityEvaluator

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_enhancement(quick_test=False):
    config = load_config()
    enhancer = ZeroDCEEnhancer()
    evaluator = ImageQualityEvaluator()

    test_img_dir = os.path.join(config["dataset"]["processed_dir"], "test", "images")
    test_lbl_dir = os.path.join(config["dataset"]["processed_dir"], "test", "labels")

    enhanced_test_img_dir = os.path.join(config["dataset"]["processed_dir"], "enhanced_test", "images")
    enhanced_test_lbl_dir = os.path.join(config["dataset"]["processed_dir"], "enhanced_test", "labels")

    os.makedirs(enhanced_test_img_dir, exist_ok=True)
    os.makedirs(enhanced_test_lbl_dir, exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    if quick_test:
        img_paths = img_paths[:50]

    print(f"\nEvaluating enhancement on {len(img_paths)} test images...")

    orig_stats = []
    enh_stats = []

    for idx, path in enumerate(tqdm(img_paths, desc="Enhancing test images")):
        base_name = os.path.basename(path)
        with Image.open(path) as img:
            orig_pil = img.convert('RGB')
            enh_pil = enhancer.enhance_pil(orig_pil)

        # Save enhanced image
        save_path = os.path.join(enhanced_test_img_dir, base_name)
        enh_pil.save(save_path)

        # Copy corresponding label file
        lbl_name = os.path.splitext(base_name)[0] + ".txt"
        lbl_src = os.path.join(test_lbl_dir, lbl_name)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(enhanced_test_lbl_dir, lbl_name))

        # Metrics
        orig_stats.append(evaluator.calculate_no_reference_metrics(np.array(orig_pil)))
        enh_stats.append(evaluator.calculate_no_reference_metrics(np.array(enh_pil)))

        # Save sample visualizations (first 10)
        if idx < 10:
            sample_dir = "results/visualizations/enhancement_samples"
            os.makedirs(sample_dir, exist_ok=True)
            w, h = orig_pil.size
            combined = Image.new('RGB', (w * 2, h))
            combined.paste(orig_pil, (0, 0))
            combined.paste(enh_pil, (w, 0))
            combined.save(os.path.join(sample_dir, f"sample_{idx+1}_{base_name}"))

    avg_orig = {k: round(float(np.mean([m[k] for m in orig_stats])), 2) for k in orig_stats[0]}
    avg_enh = {k: round(float(np.mean([m[k] for m in enh_stats])), 2) for k in enh_stats[0]}

    report = {
        "num_images_evaluated": len(img_paths),
        "original_images": avg_orig,
        "enhanced_images": avg_enh,
        "improvement_pct": {
            "brightness": round(((avg_enh["mean_brightness"] - avg_orig["mean_brightness"]) / max(1e-5, avg_orig["mean_brightness"])) * 100, 2),
            "contrast": round(((avg_enh["contrast"] - avg_orig["contrast"]) / max(1e-5, avg_orig["contrast"])) * 100, 2),
            "spatial_frequency": round(((avg_enh["spatial_frequency"] - avg_orig["spatial_frequency"]) / max(1e-5, avg_orig["spatial_frequency"])) * 100, 2)
        }
    }

    print("\n" + "=" * 60)
    print("ENHANCEMENT EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Original Brightness: {avg_orig['mean_brightness']} -> Enhanced: {avg_enh['mean_brightness']} (+{report['improvement_pct']['brightness']}%)")
    print(f"Original Contrast:   {avg_orig['contrast']} -> Enhanced: {avg_enh['contrast']} (+{report['improvement_pct']['contrast']}%)")
    print(f"Spatial Frequency:   {avg_orig['spatial_frequency']} -> Enhanced: {avg_enh['spatial_frequency']} (+{report['improvement_pct']['spatial_frequency']}%)")
    print(f"Original Entropy:    {avg_orig['entropy']} -> Enhanced: {avg_enh['entropy']}")

    with open("results/enhancement_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    # Create enhanced dataset.yaml for YOLO testing
    abs_enhanced_dir = os.path.abspath(config["dataset"]["processed_dir"]).replace("\\", "/")
    enhanced_yaml_cfg = {
        "path": abs_enhanced_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "enhanced_test/images",
        "nc": config["dataset"]["num_classes"],
        "names": {idx: cat for idx, cat in enumerate(config["dataset"]["categories"])}
    }
    with open(os.path.join(config["dataset"]["processed_dir"], "enhanced_dataset.yaml"), "w") as f:
        yaml.dump(enhanced_yaml_cfg, f, default_flow_style=False)

    print(f"\nEnhanced dataset configuration saved to data/enhanced_dataset.yaml")
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()
    evaluate_enhancement(quick_test=args.quick_test)
