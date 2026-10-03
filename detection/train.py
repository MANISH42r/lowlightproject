import os
import yaml
import torch
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_detector(quick_test=False, epochs=None, dataset_yaml=None):
    config = load_config()
    cfg_det = config["detection"]

    if dataset_yaml is None:
        dataset_yaml = os.path.join(config["dataset"]["processed_dir"], "dataset.yaml")

    num_epochs = epochs if epochs is not None else (2 if quick_test else cfg_det["epochs"])
    batch_size = 8 if quick_test else cfg_det["batch_size"]
    img_size = cfg_det.get("image_size", 384)
    model_name = cfg_det.get("model_name", "yolov8n.pt")
    save_dir = cfg_det.get("save_dir", "checkpoints/detection")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device for YOLO training: {device}")

    print("\n" + "=" * 60)
    print(f"STARTING YOLO OBJECT DETECTOR TRAINING ({model_name})")
    print("=" * 60)

    model = YOLO(model_name)

    results = model.train(
        data=dataset_yaml,
        epochs=num_epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=save_dir,
        name="baseline",
        exist_ok=True,
        seed=config["system"].get("seed", 42),
        verbose=True,
        workers=0
    )

    best_ckpt = os.path.join(save_dir, "baseline", "weights", "best.pt")
    print(f"\nYOLO training complete! Best checkpoint: {best_ckpt}")
    return best_ckpt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_detector(quick_test=args.quick_test, epochs=args.epochs)
