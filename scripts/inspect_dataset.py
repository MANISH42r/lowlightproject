import os
import json
import yaml
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def inspect_split(json_path, raw_images_dir="images"):
    if not os.path.exists(json_path):
        print(f"Error: Annotation JSON file not found: {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_map = {cat["id"]: cat["name"] for cat in categories}
    ann_counts = Counter(ann["category_id"] for ann in annotations)

    missing_images = []
    corrupt_images = []
    resolutions = []
    aspect_ratios = []

    for img in images:
        file_path = img.get("file_name")
        if not file_path:
            continue
        
        # Check standard location or fallback path
        full_path = file_path
        if not os.path.exists(full_path):
            alt_path = os.path.join(raw_images_dir, os.path.basename(file_path))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                missing_images.append(file_path)
                continue

        try:
            with Image.open(full_path) as im:
                im.verify()
            w, h = img.get("width", 0), img.get("height", 0)
            if w > 0 and h > 0:
                resolutions.append((w, h))
                aspect_ratios.append(round(w / h, 3))
        except Exception as e:
            corrupt_images.append((full_path, str(e)))

    cat_breakdown = {category_map.get(cat_id, f"ID-{cat_id}"): count for cat_id, count in sorted(ann_counts.items())}

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "categories": [cat["name"] for cat in categories],
        "category_breakdown": cat_breakdown,
        "missing_count": len(missing_images),
        "corrupt_count": len(corrupt_images),
        "missing_images": missing_images[:10],
        "corrupt_images": corrupt_images[:10],
        "resolutions": resolutions,
        "aspect_ratios": aspect_ratios,
    }

def main():
    config = load_config()
    print("=" * 60)
    print("EXDARK DATASET INSPECTION")
    print("=" * 60)

    train_json = config["dataset"]["train_annotation_json"]
    val_json = config["dataset"]["val_annotation_json"]
    raw_dir = config["dataset"]["raw_images_dir"]

    print(f"Inspecting Train Split ({train_json})...")
    train_stats = inspect_split(train_json, raw_dir)

    print(f"Inspecting Val Split ({val_json})...")
    val_stats = inspect_split(val_json, raw_dir)

    total_images = (train_stats["num_images"] if train_stats else 0) + (val_stats["num_images"] if val_stats else 0)
    total_annos = (train_stats["num_annotations"] if train_stats else 0) + (val_stats["num_annotations"] if val_stats else 0)

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total Images: {total_images}")
    print(f"Total Annotations: {total_annos}")
    
    if train_stats:
        print(f"\nTrain Images: {train_stats['num_images']}")
        print(f"Train Annotations: {train_stats['num_annotations']}")
        print(f"Train Missing Images: {train_stats['missing_count']}")
        print(f"Train Corrupt Images: {train_stats['corrupt_count']}")
        print(f"Categories ({train_stats['num_categories']}): {', '.join(train_stats['categories'])}")
        print("\nCategory Breakdown (Train):")
        for cat, count in train_stats["category_breakdown"].items():
            print(f"  - {cat:12s}: {count} annotations")

    if val_stats:
        print(f"\nVal Images: {val_stats['num_images']}")
        print(f"Val Annotations: {val_stats['num_annotations']}")
        print(f"Val Missing Images: {val_stats['missing_count']}")
        print(f"Val Corrupt Images: {val_stats['corrupt_count']}")
        print("\nCategory Breakdown (Val):")
        for cat, count in val_stats["category_breakdown"].items():
            print(f"  - {cat:12s}: {count} annotations")

    # Resolution statistics
    all_res = (train_stats["resolutions"] if train_stats else []) + (val_stats["resolutions"] if val_stats else [])
    if all_res:
        widths = [r[0] for r in all_res]
        heights = [r[1] for r in all_res]
        print("\nImage Resolution Statistics:")
        print(f"  - Width range: {min(widths)} to {max(widths)} (Mean: {sum(widths)/len(widths):.1f})")
        print(f"  - Height range: {min(heights)} to {max(heights)} (Mean: {sum(heights)/len(heights):.1f})")

    # Save summary report to results/dataset_summary.json
    os.makedirs("results", exist_ok=True)
    summary = {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "total_images": total_images,
        "total_annotations": total_annos
    }
    with open("results/dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDataset inspection complete. Summary saved to results/dataset_summary.json")

if __name__ == "__main__":
    main()
