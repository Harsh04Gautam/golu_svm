import json
import random
from dataclasses import asdict
from pathlib import Path

import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_config_snapshot(cfg):
    path = Path(cfg.config_snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(asdict(cfg)), file, indent=2, sort_keys=True)


def append_metrics(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.total += float(value) * count
        self.count += count

    @property
    def avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count
