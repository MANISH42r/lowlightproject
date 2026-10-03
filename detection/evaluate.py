import os
import json
import time
import yaml
import torch
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_detector(model_path=None, dataset_yaml=None, output_dir=None, is_enhanced=False):
    config = load_config()
    cfg_det = config["detection"]

    if dataset_yaml is None:
        dataset_yaml = os.path.join(
            config["dataset"]["processed_dir"], 
            "enhanced_dataset.yaml" if is_enhanced else "dataset.yaml"
        )

    if model_path is None:
        model_path = os.path.join(cfg_det["save_dir"], "baseline", "weights", "best.pt")
        if not os.path.exists(model_path):
            model_path = cfg_det.get("model_name", "yolov8n.pt")

    if output_dir is None:
        output_dir = cfg_det["enhanced_results_dir"] if is_enhanced else cfg_det["baseline_results_dir"]

    os.makedirs(output_dir, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print(f"EVALUATING YOLO DETECTOR ({'ENHANCED' if is_enhanced else 'BASELINE'})")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Output Directory: {output_dir}")

    model = YOLO(model_path)

    start_time = time.time()
    val_results = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=cfg_det.get("image_size", 384),
        device=device,
        verbose=True,
        workers=0
    )
    total_val_time = time.time() - start_time

    # Extract metrics
    metrics_dict = val_results.results_dict
    
    precision = float(metrics_dict.get("metrics/precision(B)", val_results.box.mp))
    recall = float(metrics_dict.get("metrics/recall(B)", val_results.box.mr))
    map50 = float(metrics_dict.get("metrics/mAP50(B)", val_results.box.map50))
    map50_95 = float(metrics_dict.get("metrics/mAP50-95(B)", val_results.box.map))

    # Latency calculation
    speed = val_results.speed # dict with preprocess, inference, loss, postprocess in ms
    latency_ms = speed.get("inference", 0.0) + speed.get("preprocess", 0.0) + speed.get("postprocess", 0.0)
    fps = round(1000.0 / latency_ms, 2) if latency_ms > 0 else 0.0

    eval_summary = {
        "pipeline": "Enhanced (Zero-DCE++ + YOLO)" if is_enhanced else "Baseline (Original + YOLO)",
        "model_path": model_path,
        "dataset_yaml": dataset_yaml,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "latency_ms": round(latency_ms, 2),
        "fps": fps,
        "speed_breakdown_ms": speed
    }

    print("\nRESULTS SUMMARY:")
    print(f"  Precision: {eval_summary['precision']:.4f}")
    print(f"  Recall:    {eval_summary['recall']:.4f}")
    print(f"  mAP@50:    {eval_summary['mAP50']:.4f}")
    print(f"  mAP@50-95: {eval_summary['mAP50_95']:.4f}")
    print(f"  Latency:   {eval_summary['latency_ms']:.2f} ms ({eval_summary['fps']} FPS)")

    summary_file = os.path.join(output_dir, "metrics.json")
    with open(summary_file, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSaved metrics to {summary_file}")
    return eval_summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--dataset-yaml", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    evaluate_detector(
        model_path=args.model_path,
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        is_enhanced=args.enhanced
    )
