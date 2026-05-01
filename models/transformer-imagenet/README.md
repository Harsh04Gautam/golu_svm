# Sparse Vision Transformer for ImageNet Localization

This module adapts the CIFAR-100 sparse local attention idea for ImageNet-style object localization.

The model has:

- Sparse local patch attention controlled by `Config.kernel`
- A class token for 1000-way classification
- A box token for normalized `[xmin, ymin, xmax, ymax]` object localization
- Combined cross-entropy and Smooth L1 loss

## Expected Data Layout

Use the ImageNet LOC / CLS-LOC structure with XML annotations:

```text
data/imagenet/
  Data/CLS-LOC/train/...
  Data/CLS-LOC/val/...
  Annotations/CLS-LOC/train/...
  Annotations/CLS-LOC/val/...
```

Each annotation XML should contain ImageNet-style `object/name` and `object/bndbox` entries.

## Run

```bash
cd /home/harsh/Dev/golu_svm
PYTHONPATH=models/transformer-imagenet uv run python models/transformer-imagenet/main.py
```

Adjust `Config.data_root`, `batch`, `image_size`, and `kernel` in `config.py` before full training.
