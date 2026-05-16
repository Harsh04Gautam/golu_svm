from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Data config. Use the ImageNet LOC/CLS-LOC layout with XML box annotations.
    data_root: Path = Path("data/imagenet")
    image_size: int = 224
    num_classes: int = 1000
    num_workers: int = 4
    class_map_path: Path = Path("imagenet-class-map.json")
    crop_attempts: int = 10
    min_crop_box_area: float = 0.25

    # Train config.
    epochs: int = 100
    batch: int = 32
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    cls_loss_weight: float = 1.0
    box_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    iou_threshold: float = 0.5
    save_model: bool = True
    checkpoint_path: Path = Path("imagenet-model.pt")
    run_dir: Path = Path("runs/imagenet-sparse")
    metrics_path: Path = Path("runs/imagenet-sparse/metrics.jsonl")
    config_snapshot_path: Path = Path("runs/imagenet-sparse/config.json")
    seed: int = 42

    # Model config.
    patch_size: int = 16
    num_head: int = 8
    num_embed: int = 256
    head: int = 32 * num_head
    kernel: int = 2
    num_layer: int = 10
    dropout: float = 0.1

    def print_config(self):
        for name, value in self.__dict__.items():
            print(f"{name:<20}: \033[1;92m{value}\033[0m")
        print("\n")
