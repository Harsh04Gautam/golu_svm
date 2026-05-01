import os
import time

import torch

from config import Config
from data import build_dataloaders
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

    if os.path.exists("imagenet-model.pt"):
        print("\nLoading existing model\n")
        checkpoint = torch.load("imagenet-model.pt", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_val_loss = checkpoint["loss"]

    model.print_model_info()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        print(f"\nEPOCH {epoch + 1}:")

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = move_targets(targets)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                _, losses = model(images, targets)
            losses["loss"].backward()
            optimizer.step()

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

        val_loss = evaluate(model, val_loader)
        scheduler.step()
        print(f"val_loss: \033[1;92m{val_loss:<10.4f}\033[0m")

        if cfg.save_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best loss {best_val_loss:.4f}")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": best_val_loss,
            }, "imagenet-model.pt")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    for images, targets in loader:
        images = images.to(device)
        targets = move_targets(targets)
        _, losses = model(images, targets)
        total_loss += losses["loss"].item()
        total += 1
    return total_loss / max(total, 1)


if __name__ == "__main__":
    main()
