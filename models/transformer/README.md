# Vision Transformer with Sparse Local Attention

This module implements a **Vision Transformer (ViT)** with **Sparse Local Attention**, designed to improve computational efficiency while maintaining competitive performance on image classification tasks.

The model has been trained and evaluated on the **CIFAR-100** dataset.

---

## Overview

Standard Vision Transformers rely on **global self-attention**, which scales quadratically with the number of tokens:

O(N²)

This implementation replaces global attention with a **structured sparse local attention mechanism**, reducing computational complexity while preserving important spatial relationships.

---

## Key Idea: Sparse Local Attention

Instead of attending to all tokens, each query attends only to a **local spatial neighborhood**.

### Benefits
- **Linear Complexity**: Reduces attention cost from O(N²) → O(N)
- **Efficiency**: Scales better to higher-resolution inputs
- **Inductive Bias**: Introduces locality similar to CNN receptive fields
- **Improved Training Stability** on smaller datasets like CIFAR-100

---

## Model Architecture

A 12-layer Transformer encoder with the following configuration:

| Component | Specification |
|----------|--------------|
| Layers | 12 |
| Embedding Dimension (n_embd) | 128 |
| Attention Heads | 8 |
| MLP Hidden Size | 256 |
| Patch Size | 16×16 |
| Total Parameters | ~2.4M |

---

## Training Details

The model is trained **from scratch** on CIFAR-100.

| Parameter | Value |
|----------|-------|
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 5e-5 |
| Dropout | 0.2 |

### Additional Details
- Patch embeddings are created via **linear projection of flattened patches**
- Uses **learnable 1D positional encodings**

---

## Results

| Metric | Value |
|--------|------|
| Validation Accuracy | 68.73% |
| Validation Loss | 1.1491 |
| Training Loss | 0.4724 |

---

## Baseline Comparison

| Model | Parameters | Training Data | Accuracy |
|------|------------|--------------|----------|
| ViT-Huge/14 | 632M | JFT-300M (pretrained) | 94.55% |
| This Model | 2.4M | CIFAR-100 (scratch) | 68.73% |

---

## Implementation Notes

- Sparse attention is implemented using a **local windowed masking strategy**
- Designed to be easily extensible to higher-resolution datasets (e.g., ImageNet)
- Code is structured for experimentation with attention mechanisms

---

## Next Steps

- Scale training to higher-resolution datasets (e.g., ImageNet- Explore a DeepSeek-inspired Multi-Head Latent Attention (MLA) mechanism to compress token representations before attention and improve efficiency.)
