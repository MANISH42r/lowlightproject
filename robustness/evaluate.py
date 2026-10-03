import os
import sys
import glob
import json
import shutil
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from enhancement.inference import ZeroDCEEnhancer
from robustness.brightness import apply_brightness_reduction
from robustness.noise import apply_gaussian_noise
from robustness.blur import apply_gaussian_blur
from robustness.motion_blur import apply_motion_blur
from robustness.contrast import apply_contrast_reduction
from robustness.occlusion import apply_partial_occlusion

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

DEGRADATION_FUNCS = {
    "Brightness Reduction": apply_brightness_reduction,
    "Gaussian Noise": apply_gaussian_noise,
    "Gaussian Blur": apply_gaussian_blur,
    "Motion Blur": apply_motion_blur,
    "Contrast Reduction": apply_contrast_reduction,
    "Partial Occlusion": apply_partial_occlusion
}

def create_degraded_dataset(test_img_dir, test_lbl_dir, target_dir, func):
    img_target = os.path.join(target_dir, "images")
    lbl_target = os.path.join(target_dir, "labels")
    os.makedirs(img_target, exist_ok=True)
    os.makedirs(lbl_target, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    for path in img_paths:
        base_name = os.path.basename(path)
        with Image.open(path) as img:
            img_np = np.array(img.convert('RGB'))
            deg_np = func(img_np)
            deg_pil = Image.fromarray(deg_np)
            deg_pil.save(os.path.join(img_target, base_name))

        lbl_name = os.path.splitext(base_name)[0] + ".txt"
        lbl_src = os.path.join(test_lbl_dir, lbl_name)
        if os.path.exists(lbl_src):
            shutil.copy2(lbl_src, os.path.join(lbl_target, lbl_name))

def run_robustness_experiments(quick_test=False, detector_path=None):
    config = load_config()
    enhancer = ZeroDCEEnhancer()

    if detector_path is None:
        detector_path = os.path.join(config["detection"]["save_dir"], "baseline", "weights", "best.pt")
        if not os.path.exists(detector_path):
            detector_path = config["detection"].get("model_name", "yolov8n.pt")

    yolo_model = YOLO(detector_path)
    device = "0" if torch.cuda.is_available() else "cpu"

    test_img_dir = os.path.join(config["dataset"]["processed_dir"], "test", "images")
    test_lbl_dir = os.path.join(config["dataset"]["processed_dir"], "test", "labels")

    temp_rob_dir = "data/robustness_temp"
    os.makedirs(temp_rob_dir, exist_ok=True)

    results = []

    print("\n" + "=" * 60)
    print("RUNNING ROBUSTNESS EXPERIMENTS UNDER SYNTHETIC DEGRADATIONS")
    print("=" * 60)

    for deg_name, deg_fn in DEGRADATION_FUNCS.items():
        print(f"\n--- Testing Degradation: {deg_name} ---")

        # 1. Create degraded dataset
        deg_dir = os.path.join(temp_rob_dir, deg_name.replace(" ", "_").lower())
        create_degraded_dataset(test_img_dir, test_lbl_dir, deg_dir, deg_fn)

        # Build dataset.yaml for degraded baseline
        abs_deg_dir = os.path.abspath(deg_dir).replace("\\", "/")
        deg_yaml = {
            "path": abs_deg_dir,
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": config["dataset"]["num_classes"],
            "names": {idx: c for idx, c in enumerate(config["dataset"]["categories"])}
        }
        deg_yaml_path = os.path.join(deg_dir, "dataset.yaml")
        with open(deg_yaml_path, "w") as f:
            yaml.dump(deg_yaml, f)

        # Evaluate Baseline YOLO on degraded images
        val_base = yolo_model.val(data=deg_yaml_path, split="test", device=device, verbose=False, workers=0)
        base_map50 = float(val_base.results_dict.get("metrics/mAP50(B)", val_base.box.map50))
        base_map50_95 = float(val_base.results_dict.get("metrics/mAP50-95(B)", val_base.box.map))

        # 2. Enhance degraded images with Zero-DCE++
        deg_enh_dir = os.path.join(deg_dir, "enhanced")
        deg_enh_imgs = os.path.join(deg_enh_dir, "images")
        deg_enh_lbls = os.path.join(deg_enh_dir, "labels")
        os.makedirs(deg_enh_imgs, exist_ok=True)
        os.makedirs(deg_enh_lbls, exist_ok=True)

        deg_img_paths = sorted(glob.glob(os.path.join(deg_dir, "images", "*.jpg")))
        if quick_test:
            deg_img_paths = deg_img_paths[:20]

        for p in deg_img_paths:
            b_name = os.path.basename(p)
            with Image.open(p) as im:
                enh_p = enhancer.enhance_pil(im)
                enh_p.save(os.path.join(deg_enh_imgs, b_name))
            l_name = os.path.splitext(b_name)[0] + ".txt"
            l_src = os.path.join(deg_dir, "labels", l_name)
            if os.path.exists(l_src):
                shutil.copy2(l_src, os.path.join(deg_enh_lbls, l_name))

        # Build dataset.yaml for enhanced degraded images
        abs_enh_dir = os.path.abspath(deg_enh_dir).replace("\\", "/")
        enh_yaml = {
            "path": abs_enh_dir,
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": config["dataset"]["num_classes"],
            "names": {idx: c for idx, c in enumerate(config["dataset"]["categories"])}
        }
        enh_yaml_path = os.path.join(deg_enh_dir, "dataset.yaml")
        with open(enh_yaml_path, "w") as f:
            yaml.dump(enh_yaml, f)

        val_enh = yolo_model.val(data=enh_yaml_path, split="test", device=device, verbose=False, workers=0)
        enh_map50 = float(val_enh.results_dict.get("metrics/mAP50(B)", val_enh.box.map50))
        enh_map50_95 = float(val_enh.results_dict.get("metrics/mAP50-95(B)", val_enh.box.map))

        diff_map50 = enh_map50 - base_map50

        print(f"  Baseline mAP@50: {base_map50:.4f} | Enhanced mAP@50: {enh_map50:.4f} (Diff: {diff_map50:+.4f})")

        results.append({
            "Degradation": deg_name,
            "Baseline mAP@50": round(base_map50, 4),
            "Enhanced mAP@50": round(enh_map50, 4),
            "Difference mAP@50": round(diff_map50, 4),
            "Baseline mAP@50-95": round(base_map50_95, 4),
            "Enhanced mAP@50-95": round(enh_map50_95, 4)
        })

    # Cleanup temp dir
    try:
        shutil.rmtree(temp_rob_dir)
    except Exception:
        pass

    # Save summary CSV and PNG
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/robustness.csv", index=False)

    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df["Baseline mAP@50"], width, label="Baseline (Degraded + YOLO)", color="coral")
    ax.bar(x + width/2, df["Enhanced mAP@50"], width, label="Enhanced (Degraded + Zero-DCE++ + YOLO)", color="teal")

    ax.set_ylabel("mAP@50")
    ax.set_title("Robustness Evaluation Under Synthetic Degradations")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Degradation"], rotation=15)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/figures/robustness_comparison.png")
    plt.close()

    print("\nRobustness experiments completed! Saved results to results/robustness.csv and results/figures/robustness_comparison.png")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    args = parser.parse_args()
    run_robustness_experiments(quick_test=args.quick_test)
