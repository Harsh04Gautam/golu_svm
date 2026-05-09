from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "loss": best_val_loss,
        "metrics": metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", float("inf")),
        "metrics": checkpoint.get("metrics", {}),
    }
