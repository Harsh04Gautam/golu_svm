import torch

from checkpoint import load_checkpoint
from config import Config
from data import build_dataloaders
from main import evaluate, print_validation
from model import GoluImageNet


cfg = Config()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader = build_dataloaders(cfg)
    model = GoluImageNet().to(device)
    load_checkpoint(cfg.checkpoint_path, model, device=device)
    metrics = evaluate(model, val_loader)
    print_validation(metrics)


if __name__ == "__main__":
    main()
