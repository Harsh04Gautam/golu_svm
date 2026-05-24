import argparse
import json
from collections import Counter
from pathlib import Path


def load_records(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def summarize(records, iou_threshold):
    records = list(records)
    total = len(records)
    class_correct = sum(1 for item in records if item["class_correct"])
    localized = sum(1 for item in records if item["iou"] >= iou_threshold)
    both_correct = sum(
        1 for item in records
        if item["class_correct"] and item["iou"] >= iou_threshold
    )

    low_iou = sorted(records, key=lambda item: item["iou"])[:20]
    high_conf_wrong = sorted(
        [item for item in records if not item["class_correct"]],
        key=lambda item: item["score"],
        reverse=True,
    )[:20]
    confused_pairs = Counter(
        (item["target_class"], item["pred_class"])
        for item in records
        if not item["class_correct"]
    )

    return {
        "total": total,
        "classification_accuracy": class_correct / max(total, 1),
        "localization_accuracy": localized / max(total, 1),
        "classification_and_localization_accuracy": both_correct / max(total, 1),
        "mean_iou": sum(item["iou"] for item in records) / max(total, 1),
        "top_confusions": [
            {"target": pair[0], "prediction": pair[1], "count": count}
            for pair, count in confused_pairs.most_common(20)
        ],
        "lowest_iou_examples": [
            {
                "image": item["image"],
                "target": item["target_class"],
                "prediction": item["pred_class"],
                "score": item["score"],
                "iou": item["iou"],
            }
            for item in low_iou
        ],
        "high_confidence_mistakes": [
            {
                "image": item["image"],
                "target": item["target_class"],
                "prediction": item["pred_class"],
                "score": item["score"],
                "iou": item["iou"],
            }
            for item in high_conf_wrong
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze exported ImageNet localization predictions.")
    parser.add_argument("--predictions", type=Path, default=Path("runs/imagenet-sparse/val-predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/imagenet-sparse/error-report.json"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    report = summarize(load_records(args.predictions), args.iou_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
