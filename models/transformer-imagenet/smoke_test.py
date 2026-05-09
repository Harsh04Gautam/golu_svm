import torch

from metrics import localization_metrics
from model import GoluImageNet


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GoluImageNet().to(device)
    model.train()

    images = torch.randn(2, 3, 224, 224, device=device)
    targets = {
        "label": torch.tensor([0, 1], dtype=torch.long, device=device),
        "box": torch.tensor(
            [[0.10, 0.10, 0.55, 0.55], [0.25, 0.20, 0.75, 0.80]],
            dtype=torch.float32,
            device=device,
        ),
    }

    outputs, losses = model(images, targets)
    losses["loss"].backward()
    metrics = localization_metrics(outputs, targets)

    print(f"class_logits: {tuple(outputs['class_logits'].shape)}")
    print(f"box: {tuple(outputs['box'].shape)}")
    print(f"loss: {losses['loss'].item():.4f}")
    print(f"giou_loss: {losses['giou_loss'].item():.4f}")
    print(f"top1: {metrics['top1'].item():.4f}")
    print(f"top5: {metrics['top5'].item():.4f}")
    print(f"mean_iou: {metrics['mean_iou'].item():.4f}")
    print(f"mean_giou: {metrics['mean_giou'].item():.4f}")
    print(f"center_error: {metrics['center_error'].item():.4f}")


if __name__ == "__main__":
    main()
