import os
import json
import shutil
import yaml
import random
from pathlib import Path
from PIL import Image

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def coco_to_yolo_bbox(bbox, img_width=384, img_height=384):
    """
    COCO bbox format: [x_min, y_min, w, h] (normalized in ExDark json)
    YOLO bbox format: [x_center, y_center, width, height] (all normalized 0-1)
    """
    x_min, y_min, w, h = bbox
    
    # If bbox coordinates are in pixels (w/h > 1), normalize them
    if x_min > 1.0 or y_min > 1.0 or w > 1.0 or h > 1.0:
        x_min /= img_width
        y_min /= img_height
        w /= img_width
        h /= img_height

    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0

    # Clip values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return x_center, y_center, w, h

def process_split(split_name, image_list, annotations_by_img, cat_id_to_idx, dest_dir, raw_dir):
    img_dest_dir = os.path.join(dest_dir, split_name, "images")
    lbl_dest_dir = os.path.join(dest_dir, split_name, "labels")

    # Clean destination directories
    if os.path.exists(img_dest_dir):
        shutil.rmtree(img_dest_dir)
    if os.path.exists(lbl_dest_dir):
        shutil.rmtree(lbl_dest_dir)

    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)

    copied_count = 0
    skipped_count = 0
    total_valid_boxes = 0
    total_invalid_boxes = 0

    for img_info in image_list:
        file_path = img_info.get("file_name")
        if not file_path or not os.path.exists(file_path):
            alt_path = os.path.join(raw_dir, os.path.basename(file_path))
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                skipped_count += 1
                continue

        # Target image name
        img_id = img_info["id"]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        parent_dir = os.path.basename(os.path.dirname(file_path))
        target_img_name = f"{parent_dir}_{base_name}.jpg"
        target_txt_name = f"{parent_dir}_{base_name}.txt"

        target_img_path = os.path.join(img_dest_dir, target_img_name)
        target_txt_path = os.path.join(lbl_dest_dir, target_txt_name)

        # Copy image
        shutil.copy2(file_path, target_img_path)
        copied_count += 1

        # Process annotations
        origin = img_info.get("_origin", "default")
        img_anns = annotations_by_img.get((origin, img_id), annotations_by_img.get(img_id, []))
        yolo_lines = []

        for ann in img_anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_idx:
                total_invalid_boxes += 1
                continue

            class_idx = cat_id_to_idx[cat_id]
            bbox = ann["bbox"]

            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                total_invalid_boxes += 1
                continue

            xc, yc, w, h = coco_to_yolo_bbox(bbox, img_info.get("width", 384), img_info.get("height", 384))
            if w <= 0.001 or h <= 0.001:
                total_invalid_boxes += 1
                continue

            yolo_lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            total_valid_boxes += 1

        with open(target_txt_path, "w") as txt_f:
            txt_f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    return copied_count, skipped_count, total_valid_boxes, total_invalid_boxes

def prepare_dataset(quick_test=False):
    config = load_config()
    seed = config["system"].get("seed", 42)
    random.seed(seed)

    processed_dir = config["dataset"]["processed_dir"]
    train_json_path = config["dataset"]["train_annotation_json"]
    val_json_path = config["dataset"]["val_annotation_json"]
    raw_images_dir = config["dataset"]["raw_images_dir"]

    print("Loading COCO annotations...")
    with open(train_json_path, "r") as f:
        train_coco = json.load(f)
    with open(val_json_path, "r") as f:
        val_coco = json.load(f)

    # Categories
    categories = train_coco["categories"]
    categories_sorted = sorted(categories, key=lambda x: x["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories_sorted)}
    cat_names = [cat["name"] for cat in categories_sorted]

    print(f"Categories ({len(cat_names)}): {cat_names}")

    combined_anns_map = {}
    
    train_imgs = train_coco["images"]
    for img in train_imgs:
        img["_origin"] = "train"
    for a in train_coco["annotations"]:
        combined_anns_map.setdefault(("train", a["image_id"]), []).append(a)

    val_imgs = val_coco["images"]
    for img in val_imgs:
        img["_origin"] = "val"
    for a in val_coco["annotations"]:
        combined_anns_map.setdefault(("val", a["image_id"]), []).append(a)

    random.shuffle(train_imgs)
    random.shuffle(val_imgs)

    if quick_test:
        print("\n*** QUICK TEST MODE: Using a small subset of the dataset ***")
        n_train = 100
        n_val = 30
        n_test = 30
        train_split_imgs = train_imgs[:n_train]
        val_split_imgs = train_imgs[n_train:n_train + n_val]
        test_split_imgs = val_imgs[:n_test]
    else:
        # Full split: 85% train_imgs -> train, 15% train_imgs -> val, val_imgs -> test
        n_train = int(len(train_imgs) * 0.85)
        train_split_imgs = train_imgs[:n_train]
        val_split_imgs = train_imgs[n_train:]
        test_split_imgs = val_imgs

    print(f"\nDataset Splits:")
    print(f"  - Train: {len(train_split_imgs)} images")
    print(f"  - Val:   {len(val_split_imgs)} images")
    print(f"  - Test:  {len(test_split_imgs)} images")

    print("\nProcessing Train split...")
    tr_c, tr_s, tr_vb, tr_ib = process_split("train", train_split_imgs, combined_anns_map, cat_id_to_idx, processed_dir, raw_images_dir)

    print("Processing Val split...")
    va_c, va_s, va_vb, va_ib = process_split("val", val_split_imgs, combined_anns_map, cat_id_to_idx, processed_dir, raw_images_dir)

    print("Processing Test split...")
    te_c, te_s, te_vb, te_ib = process_split("test", test_split_imgs, combined_anns_map, cat_id_to_idx, processed_dir, raw_images_dir)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Train set: {tr_c} images, {tr_vb} valid bboxes (Skipped: {tr_s} imgs, {tr_ib} invalid bboxes)")
    print(f"Val set:   {va_c} images, {va_vb} valid bboxes (Skipped: {va_s} imgs, {va_ib} invalid bboxes)")
    print(f"Test set:  {te_c} images, {te_vb} valid bboxes (Skipped: {te_s} imgs, {te_ib} invalid bboxes)")

    # Write dataset.yaml
    dataset_yaml_path = os.path.join(processed_dir, "dataset.yaml")
    abs_processed_dir = os.path.abspath(processed_dir).replace("\\", "/")

    dataset_cfg = {
        "path": abs_processed_dir,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(cat_names),
        "names": {idx: name for idx, name in enumerate(cat_names)}
    }

    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    print(f"\nSaved dataset config to {dataset_yaml_path}")
    print("Dataset preparation complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", help="Run with small subset for fast verification")
    args = parser.parse_args()
    prepare_dataset(quick_test=args.quick_test)
