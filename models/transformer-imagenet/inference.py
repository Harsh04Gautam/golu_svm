import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T

from checkpoint import load_checkpoint
from config import Config
from model import GoluImageNet


cfg = Config()


def preprocess(image):
    transform = T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0)


def load_class_names(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        class_to_idx = json.load(file)
    return {idx: name for name, idx in class_to_idx.items()}


def draw_prediction(image, box, label, score, output_path):
    width, height = image.size
    scaled = [
        box[0] * width,
        box[1] * height,
        box[2] * width,
        box[3] * height,
    ]
    draw = ImageDraw.Draw(image)
    draw.rectangle(scaled, outline="red", width=3)
    draw.text((scaled[0], max(0, scaled[1] - 14)), f"{label} {score:.3f}", fill="red")
    image.save(output_path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Run ImageNet localization inference.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=cfg.checkpoint_path)
    parser.add_argument("--class-map", type=Path, default=cfg.class_map_path)
    parser.add_argument("--output", type=Path, default=Path("prediction.jpg"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(args.image).convert("RGB")
    model = GoluImageNet().to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    tensor = preprocess(image).to(device)
    outputs, _ = model(tensor)
    probs = outputs["class_logits"].softmax(dim=-1)
    score, label_idx = probs.max(dim=-1)

    class_names = load_class_names(args.class_map)
    idx = int(label_idx.item())
    label = class_names.get(idx, str(idx)) if class_names else str(idx)
    box = outputs["box"][0].detach().cpu().tolist()

    draw_prediction(image, box, label, float(score.item()), args.output)
    print(f"label: {label}")
    print(f"score: {score.item():.4f}")
    print(f"box: {[round(value, 4) for value in box]}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
