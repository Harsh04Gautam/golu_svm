import json

import torch

from checkpoint import load_checkpoint
from config import Config
from data import ImageNetLocalizationDataset
from metrics import box_iou
from model import GoluImageNet


cfg = Config()


def move_targets(targets, device):
    return {key: value.to(device) for key, value in targets.items()}


def load_class_maps():
    if not cfg.class_map_path.exists():
        return None, None
    with cfg.class_map_path.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)
    return class_to_idx, {idx: name for name, idx in class_to_idx.items()}


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_to_idx, idx_to_class = load_class_maps()
    dataset = ImageNetLocalizationDataset(
        cfg.data_root,
        "val",
        cfg.image_size,
        class_to_idx=class_to_idx,
    )
    model = GoluImageNet().to(device)
    load_checkpoint(cfg.checkpoint_path, model, device=device)
    model.eval()

    cfg.prediction_export_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.prediction_export_path.open("w", encoding="utf-8") as file:
        for index in range(len(dataset)):
            image, target = dataset[index]
            image_path, annotation_path = dataset.samples[index]
            images = image.unsqueeze(0).to(device)
            targets = move_targets({
                "label": target["label"].unsqueeze(0),
                "box": target["box"].unsqueeze(0),
            }, device)

            outputs, _ = model(images)
            probs = outputs["class_logits"].softmax(dim=-1)
            score, pred_label = probs.max(dim=-1)
            iou = box_iou(outputs["box"], targets["box"])

            pred_idx = int(pred_label.item())
            target_idx = int(targets["label"].item())
            record = {
                "image": str(image_path),
                "annotation": str(annotation_path),
                "pred_label": pred_idx,
                "target_label": target_idx,
                "pred_class": idx_to_class.get(pred_idx, str(pred_idx)) if idx_to_class else str(pred_idx),
                "target_class": idx_to_class.get(target_idx, str(target_idx)) if idx_to_class else str(target_idx),
                "score": float(score.item()),
                "iou": float(iou.item()),
                "pred_box": [float(value) for value in outputs["box"][0].detach().cpu()],
                "target_box": [float(value) for value in targets["box"][0].detach().cpu()],
                "class_correct": pred_idx == target_idx,
            }
            file.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"wrote predictions: {cfg.prediction_export_path}")


if __name__ == "__main__":
    main()
