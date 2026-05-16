import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_annotation(path):
    root = ET.parse(path).getroot()
    width = float(root.findtext("size/width", "1"))
    height = float(root.findtext("size/height", "1"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")
        if name is None or box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        objects.append({
            "name": name,
            "area_ratio": area / max(width * height, 1.0),
        })
    return objects


def summarize(root, split):
    annotation_root = Path(root) / "Annotations" / "CLS-LOC" / split
    if not annotation_root.exists():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_root}")

    class_counts = Counter()
    object_counts = []
    area_ratios = []
    files = sorted(annotation_root.rglob("*.xml"))
    for path in files:
        objects = parse_annotation(path)
        object_counts.append(len(objects))
        for obj in objects:
            class_counts[obj["name"]] += 1
            area_ratios.append(obj["area_ratio"])

    total_images = len(files)
    total_objects = sum(object_counts)
    return {
        "split": split,
        "images": total_images,
        "objects": total_objects,
        "classes": len(class_counts),
        "avg_objects_per_image": total_objects / max(total_images, 1),
        "avg_box_area_ratio": sum(area_ratios) / max(len(area_ratios), 1),
        "top_classes": class_counts.most_common(20),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize ImageNet LOC XML annotations.")
    parser.add_argument("--root", type=Path, default=Path("data/imagenet"))
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", type=Path, default=Path("imagenet-dataset-stats.json"))
    args = parser.parse_args()

    stats = summarize(args.root, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
