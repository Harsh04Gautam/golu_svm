# Vision Models 

This repository contains implementations and experiments with different approaches to classifcation/generative models.

The codebase is organized into separate modules, each exploring a different class of models. The goal is to build, test, and iterate on these approaches over time.

---

## Repository Structure

├── models/
│ ├── transformer/ # Vision Transformer with sparse local attention
│ ├── world/ # World models / state-space approaches (in progress)
│ └── diffusion/ # Diffusion and flow-based models (planned)
└── README.md

---

## Modules

### Transformer (Implemented)

Implementation of a Vision Transformer variant with **sparse local attention**.

- Designed to reduce attention complexity  
- Trained on CIFAR-100  
- Includes model, training pipeline, and evaluation  

See: `models/transformer/README.md`

---

### World (Planned)

This module is intended for experiments with **world models and state-space architectures**.

---

### Diffusion (Planned)

This module will contain implementations of **diffusion and flow-based models**.

---

## Notes

- This repository is under active development
- Modules will be updated and expanded over time
- Structure may evolve as new ideas and experiments are added
