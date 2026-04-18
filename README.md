# LIBS-Based Medical Image Classification

A clean project comparing **4 models** on **2 medical imaging datasets**.

## Overview

This project compares different architectural approaches for medical image classification:

1. **baseline_cnn** - Simple CNN baseline
2. **libs_cnn** - LIBS (Local Interpretable Basis Space) with CNN backbone
3. **densenet121** - Pretrained DenseNet121 backbone
4. **libs_densenet121** - LIBS with pretrained DenseNet121 backbone

## Datasets

- **PathMNIST** - 9-class tissue classification (93,636 training samples)
- **BloodMNIST** - 8-class blood cell classification (11,959 training samples)

## Project Structure

```
project/
├── main.py                      # Entry point for training and evaluation
├── data/                        # Dataset handling
│   ├── pathmnist.py
│   ├── pathmnist.npz
│   ├── bloodmnist.npz
│   └── dermamnist.npz
├── models/                      # Model implementations
│   ├── baseline_cnn.py          # Simple CNN baseline
│   ├── libs_model.py            # LIBS models (LIBSCNN, LIBSDenseNet121)
│   ├── pretrained_models.py     # PretrainedDenseNet
│   └── factory.py               # Model builder
├── training/                    # Training loop
│   ├── train.py
│   ├── evaluate.py
│   └── study.py                 # Study runner
├── utils/                       # Utilities
│   ├── metrics.py               # Evaluation metrics
│   ├── visualization.py         # Plotting utilities
│   ├── gradcam.py               # Grad-CAM visualization
│   └── statistics.py            # Statistical tests
├── analysis/                    # Analysis (placeholder)
├── experiments/                 # Configs and results
│   ├── config.yaml              # Default config
│   ├── config_pathmnist_4models.yaml   # PathMNIST comparison
│   ├── config_bloodmnist_4models.yaml  # BloodMNIST comparison
│   └── results/                 # Saved metrics and checkpoints
└── notebooks/                   # Jupyter notebooks
    └── full_project_workflow.ipynb
```

## Quick Start

### Train all 4 models on PathMNIST

```bash
python main.py --config experiments/config_pathmnist_4models.yaml
```

### Train all 4 models on BloodMNIST

```bash
python main.py --config experiments/config_bloodmnist_4models.yaml
```

### Train a single model

```bash
python main.py --config experiments/config.yaml
```

## Results

Results are saved in `experiments/results/` with the following structure:

```
results/
├── pathmnist/
│   ├── baseline_cnn_metrics.json
│   ├── libs_cnn_metrics.json
│   ├── densenet121_metrics.json
│   ├── libs_densenet121_metrics.json
│   ├── comparison_summary.json
│   └── checkpoints/
└── bloodmnist/
    ├── baseline_cnn_metrics.json
    ├── libs_cnn_metrics.json
    ├── densenet121_metrics.json
    ├── libs_densenet121_metrics.json
    ├── comparison_summary.json
    └── checkpoints/
```

## Configuration

Edit `experiments/config_*.yaml` to customize:

- **Dataset**: pathmnist, bloodmnist, dermamnist
- **Models**: baseline_cnn, libs_cnn, densenet121, libs_densenet121
- **Training parameters**: learning_rate, batch_size, epochs, etc.
- **LIBS options**: use_sobel, use_fusion, sobel_mode

## Key Findings

### BloodMNIST Results

| Model | Accuracy | F1 (Macro) | Notes |
|-------|----------|-----------|-------|
| **DenseNet121** | **96.99%** | 0.9675 | Best performer |
| Baseline CNN | 95.32% | 0.9481 | Strong baseline |
| LIBS-DenseNet121 | 94.07% | 0.9335 | LIBS adds slight overhead |
| LIBS-CNN | 88.66% | 0.8764 | Frozen Sobel edges suboptimal |

### PathMNIST Results

| Model | Accuracy | F1 (Macro) | Notes |
|-------|----------|-----------|-------|
| **DenseNet121** | **92.01%** | 0.8993 | Best performer |
| Baseline CNN | 91.11% | 0.8802 | Good baseline |
| LIBS-DenseNet121 | 91.50% | 0.8776 | Marginal improvement |
| LIBS-CNN | 91.23% | 0.8705 | Slight benefit over baseline |

## Requirements

```
torch
torchvision
numpy
scikit-learn
matplotlib
seaborn
pyyaml
medmnist
```

Install with:
```bash
pip install -r project/requirements.txt
```

## Analysis

For detailed analysis, see [ANALYSIS.md](ANALYSIS.md) and [BLOODMNIST_ANALYSIS.md](BLOODMNIST_ANALYSIS.md).

Key insights:
- **Pretrained DenseNet121 is the best performer** (~97% on BloodMNIST, ~92% on PathMNIST)
- **LIBS provides marginal benefits** on simple CNN, slight overhead on pretrained models
- **Fixed Sobel filters limit adaptability** - making them learnable could improve performance
- **ImageNet pretraining is crucial** for medical image classification tasks

## Future Work

1. Make Sobel filters learnable instead of fixed
2. Explore learnable fusion strategies
3. Test on larger medical imaging datasets (PCAM, etc.)
4. Multi-task learning with auxiliary tasks
5. Explainability analysis (Grad-CAM, attention visualizations)

## Author Notes

This project demonstrates that:
- Classical computer vision edge detection (Sobel) can be integrated with deep learning
- Pretrained models significantly outperform from-scratch training
- Hand-crafted features can provide interpretability but at a performance cost
- Dataset-specific tuning is essential for medical imaging

---

Last updated: April 18, 2026