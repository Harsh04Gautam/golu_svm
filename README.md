# Vision Transformer with Sparse Local Attention

This repository implements a **Vision Transformer (ViT)** architecture optimized with **Sparse Local Attention** for image recognition on the CIFAR-100 dataset. The implementation addresses the computational bottlenecks of standard Transformers by reducing attention complexity from $O(N^2)$ to $O(N)$.

---

## Technical Overview

The core objective of this project is to mitigate the **Quadratic Attention Cost** inherent in global self-attention. While standard ViTs scale quadratically with the input sequence length ($N$), this model utilizes a structured sparsity pattern where each pixel attends only to its local neighborhood.

### Sparse Local Attention
* **Mechanism**: Instead of global interaction, queries are multiplied with a limited set of keys within a local spatial window.
* **Inductive Bias**: Re-introduces CNN-like local receptive fields, aiding generalization on smaller datasets like CIFAR-100 without the massive pre-training requirements of standard ViTs.
* **Complexity**: Reduces time complexity to $O(N)$.

---

## Model Architecture

The implemented model is a 12-layer Transformer Encoder.

| Component | Specification |
| :--- | :--- |
| **Layers** | 12 |
| **Embedding Dimension ($n_{embd}$)** | 128 |
| **Attention Heads** | 8 |
| **MLP Size** | 256 |
| **Patch Size** | $16\times16$ |
| **Total Parameters** | 2.4M |


---

## Training & Hyperparameters

The model was trained from scratch on CIFAR-100.

* **Epochs**: 100 
* **Batch Size**: 64 
* **Learning Rate**: $5\times10^{-5}$ 
* **Dropout**: 0.2 
* **Optimizer**: Linear projection of flattened patches with learnable 1D positional encodings.

---

## Performance Comparison

| Model | Parameters | Data | CIFAR-100 Accuracy |
| :--- | :--- | :--- | :--- |
| **ViT-Huge/14** | 632M | JFT-300M (Pre-trained) | **94.55%** |
| **This Model** | **2.4M** | CIFAR-100 (Scratch) | **68.73%** |

### Results Summary
* **Validation Accuracy**: 68.73% 
* **Validation Loss**: 1.1491 
* **Training Loss**: 0.4724

---

## Research & Future Exploration
The project also explored alternative architectures for sequence modeling in vision:
* **State Space Models (Mamba)**: Linear-time complexity via recurrent state updates.
* **Diffusion/Flow Models**: Generative transformations increasingly competitive for stable training.
