import time

import torch

from checkpoint import load_checkpoint, save_checkpoint
from config import Config
from data import build_dataloaders
from metrics import localization_metrics
from model import GoluImageNet


cfg = Config()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.matmul.fp32_precision = "tf32"


def move_targets(targets):
    return {key: value.to(device) for key, value in targets.items()}


def main():
    train_loader, val_loader = build_dataloaders(cfg)
    model = GoluImageNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_val_loss = float("inf")
    start_epoch = 0

    if cfg.checkpoint_path.exists():
        print("\nLoading existing model\n")
        checkpoint = load_checkpoint(
            cfg.checkpoint_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["loss"]

    model.print_model_info()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        print(f"\nEPOCH {epoch + 1}:")

        optimizer.zero_grad(set_to_none=True)
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = move_targets(targets)

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, losses = model(images, targets)
                loss = losses["loss"] / cfg.grad_accum_steps
            loss.backward()

            should_step = (
                (step + 1) % cfg.grad_accum_steps == 0
                or (step + 1) == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += losses["loss"].item()
            if step % 10 == 9:
                elapsed = f"{time.time() - start_time:.2f}s"
                avg_loss = running_loss / 10
                print(
                    f"step: {step + 1:<10} time: {elapsed:<10} "
                    f"loss: \033[1;92m{avg_loss:<10.4f}\033[0m"
                )
                running_loss = 0.0
                start_time = time.time()

        val_metrics = evaluate(model, val_loader)
        scheduler.step()
        print_validation(val_metrics)

        if cfg.save_model and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            print(f"New best loss {best_val_loss:.4f}")
            save_checkpoint(
                cfg.checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                val_metrics,
            )


def print_validation(metrics):
    print(
        f"val_loss: \033[1;92m{metrics['loss']:<10.4f}\033[0m "
        f"top1: \033[1;92m{metrics['top1']:<8.4f}\033[0m "
        f"top5: \033[1;92m{metrics['top5']:<8.4f}\033[0m "
        f"mean_iou: \033[1;92m{metrics['mean_iou']:<8.4f}\033[0m "
        f"mean_giou: \033[1;92m{metrics['mean_giou']:<8.4f}\033[0m "
        f"center_error: \033[1;92m{metrics['center_error']:<8.4f}\033[0m "
        f"loc@{cfg.iou_threshold}: \033[1;92m{metrics['loc_at_iou']:<8.4f}\033[0m"
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    totals = {
        "loss": 0.0,
        "cls_loss": 0.0,
        "box_loss": 0.0,
        "giou_loss": 0.0,
        "top1": 0.0,
        "top5": 0.0,
        "mean_iou": 0.0,
        "mean_giou": 0.0,
        "center_error": 0.0,
        "loc_at_iou": 0.0,
    }
    total_batches = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device)
        targets = move_targets(targets)
        outputs, losses = model(images, targets)
        metrics = localization_metrics(outputs, targets, iou_threshold=cfg.iou_threshold)
        batch_size = images.shape[0]

        totals["loss"] += losses["loss"].item()
        totals["cls_loss"] += losses["cls_loss"].item()
        totals["box_loss"] += losses["box_loss"].item()
        totals["giou_loss"] += losses["giou_loss"].item()
        for name, value in metrics.items():
            totals[name] += value.item() * batch_size
        total_batches += 1
        total_examples += batch_size

    batches = max(total_batches, 1)
    examples = max(total_examples, 1)
    return {
        "loss": totals["loss"] / batches,
        "cls_loss": totals["cls_loss"] / batches,
        "box_loss": totals["box_loss"] / batches,
        "giou_loss": totals["giou_loss"] / batches,
        "top1": totals["top1"] / examples,
        "top5": totals["top5"] / examples,
        "mean_iou": totals["mean_iou"] / examples,
        "mean_giou": totals["mean_giou"] / examples,
        "center_error": totals["center_error"] / examples,
        "loc_at_iou": totals["loc_at_iou"] / examples,
    }


if __name__ == "__main__":
    main()
