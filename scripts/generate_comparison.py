import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from detection.inference import YOLOObjectDetector
from enhancement.inference import ZeroDCEEnhancer

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_comparison(quick_test=False):
    config = load_config()

    base_metrics_path = os.path.join(config["detection"]["baseline_results_dir"], "metrics.json")
    enh_metrics_path = os.path.join(config["detection"]["enhanced_results_dir"], "metrics.json")

    base_m = {}
    enh_m = {}

    if os.path.exists(base_metrics_path):
        with open(base_metrics_path, "r") as f:
            base_m = json.load(f)
    else:
        print(f"Warning: {base_metrics_path} not found.")

    if os.path.exists(enh_metrics_path):
        with open(enh_metrics_path, "r") as f:
            enh_m = json.load(f)
    else:
        print(f"Warning: {enh_metrics_path} not found.")

    metrics_keys = ["precision", "recall", "mAP50", "mAP50_95", "latency_ms", "fps"]
    row_names = ["Precision", "Recall", "mAP@50", "mAP@50:95", "Latency (ms)", "FPS"]

    data = []
    for key, name in zip(metrics_keys, row_names):
        val_base = base_m.get(key, 0.0)
        val_enh = enh_m.get(key, 0.0)
        diff = val_enh - val_base
        data.append({
            "Metric": name,
            "Baseline YOLO": round(val_base, 4),
            "Enhanced Zero-DCE++ + YOLO": round(val_enh, 4),
            "Difference": round(diff, 4)
        })

    df = pd.DataFrame(data)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/comparison.csv", index=False)
    print("\n" + "=" * 60)
    print("CONTROLLED COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))

    # Bar plot comparison
    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_to_plot = ["Precision", "Recall", "mAP@50", "mAP@50:95"]
    df_plot = df[df["Metric"].isin(metrics_to_plot)]

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    ax.bar(x - width/2, df_plot["Baseline YOLO"], width, label="Baseline (Direct YOLO)", color="coral")
    ax.bar(x + width/2, df_plot["Enhanced Zero-DCE++ + YOLO"], width, label="Enhanced (Zero-DCE++ + YOLO)", color="teal")

    ax.set_ylabel("Score")
    ax.set_title("Object Detection Performance Comparison (Baseline vs Enhanced)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/comparison.png")
    plt.savefig("results/figures/mAP_comparison.png")
    plt.close()

    # Generate Visual Detections Comparison
    print("\nGenerating side-by-side detection visualizations...")
    detector = YOLOObjectDetector()
    enhancer = ZeroDCEEnhancer()

    test_img_dir = os.path.join(config["dataset"]["processed_dir"], "test", "images")
    sample_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))[:10]

    out_vis_dir = "results/visualizations/detection_comparison"
    os.makedirs(out_vis_dir, exist_ok=True)

    for idx, path in enumerate(sample_paths):
        b_name = os.path.basename(path)
        with Image.open(path) as img:
            orig_pil = img.convert('RGB')
            enh_pil = enhancer.enhance_pil(orig_pil)

        # Baseline detection
        dets_base, lat_base, vis_base = detector.predict(orig_pil)

        # Enhanced detection
        dets_enh, lat_enh, vis_enh = detector.predict(enh_pil)

        # Side by side concatenation
        w, h = orig_pil.size
        side_by_side = Image.new('RGB', (w * 2, h))
        side_by_side.paste(vis_base, (0, 0))
        side_by_side.paste(vis_enh, (w, 0))

        side_by_side.save(os.path.join(out_vis_dir, f"cmp_{idx+1}_{b_name}"))

    print(f"Visual detection samples saved to {out_vis_dir}")
    print("Comparison generation complete.")
    return df

if __name__ == "__main__":
    generate_comparison()
