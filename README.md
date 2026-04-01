# GA-Inspired Representation Learning on MedMNIST

This repository provides a complete and reproducible thesis-oriented framework to evaluate whether a **Geometric Algebra (GA)-inspired input representation** improves image classification behavior compared with a standard CNN baseline.

## Research objective

We compare controlled pipelines on medical image datasets:

1. **Baseline CNN**: standard RGB input
2. **GA-CNN**: input transformed into a multivector-inspired tensor (scalar + gradient-based geometric components)
3. **DenseNet121**: SOTA transfer-learning baseline on raw input
4. **GA-DenseNet121**: DenseNet121 with GA-inspired input embedding

The comparison enforces fairness:

- Same dataset split
- Same optimizer/loss/training setup
- Same CNN backbone design
- Only difference: input representation layer

## Method summary

### Baseline

- Lightweight CNN with Conv + BatchNorm + ReLU + MaxPool + FC classifier

### GA-inspired representation (practical)

Input tensor $(B, C, H, W)$ is transformed to enriched channels:

- Scalar component: original intensities
- Vector-like components: Sobel gradients $(\partial_x, \partial_y)$
- Optional higher-order terms:
  - Gradient magnitude: $\sqrt{\partial_x^2 + \partial_y^2}$
  - Local orientation: $\mathrm{atan2}(\partial_y, \partial_x)$

Output is concatenated as a **multivector-inspired tensor** and then passed to the same CNN architecture.

### Formal encoding specification

For each channel $c$ and pixel $(u,v)$ with normalized input $I_c(u,v)$:

$$
S_x=
\begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1
\end{bmatrix},
\qquad
S_y=
\begin{bmatrix}
-1 & -2 & -1\\
0 & 0 & 0\\
1 & 2 & 1
\end{bmatrix}
$$

$$
G_{x,c}=I_c * S_x,
\quad
G_{y,c}=I_c * S_y,
\quad
M_c=\sqrt{G_{x,c}^2+G_{y,c}^2+\varepsilon},
\quad
\Theta_c=\frac{\operatorname{atan2}(G_{y,c},G_{x,c}+\varepsilon)}{\pi}
$$

Concatenated feature tensor:

- Without higher-order terms: $(B,3C,H,W)$ from $[I,G_x,G_y]$
- With higher-order terms: $(B,5C,H,W)$ from $[I,G_x,G_y,M,\Theta]$

## Folder structure

- [data](data)
- [models](models)
  - [cnn_baseline.py](models/cnn_baseline.py)
  - [ga_representation.py](models/ga_representation.py)
  - [ga_cnn_model.py](models/ga_cnn_model.py)
- [training](training)
  - [train.py](training/train.py)
  - [evaluate.py](training/evaluate.py)
- [utils](utils)
  - [metrics.py](utils/metrics.py)
  - [visualization.py](utils/visualization.py)
  - [gradcam.py](utils/gradcam.py)
- [experiments/config.yaml](experiments/config.yaml)
- [main.py](main.py)
- [requirements.txt](requirements.txt)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run single experiment:

```bash
python main.py --config experiments/config.yaml
```

The default config runs all four models:

- `baseline_cnn`
- `ga_cnn`
- `densenet121`
- `ga_densenet121`

and supports dataset switching between `pathmnist` and `pcam`.

3. Run full methodological study (multi-dataset + multi-seed + data fractions):

```bash
python main.py --config experiments/config.yaml --study
```

Single-run mode automatically:

- Download PathMNIST
- Train baseline and GA-CNN with identical settings
- Save best checkpoints
- Evaluate test metrics
- Generate confusion matrices, training curves, Grad-CAM overlays

Study mode automatically:

- Runs controlled experiments across datasets, seeds, and train-data fractions
- Saves per-run metrics in structured folders
- Computes mean ± std, 95% CI, and paired permutation-test p-values

## Outputs

Saved under [experiments/results](experiments/results):

- `*_metrics.json`
- `*_history.json`
- `*_training_curves.png`
- `*_confusion_matrix.png`
- `*_gradcam.png`
- `model_comparison.png`
- `comparison_summary.json`

`comparison_summary.json` now includes `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, and `ece`.

Study outputs (if `--study`):

- `study/raw_results.json`
- `study/summary_statistics.json`

## Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix

## Interpretability

Grad-CAM is applied to both models to compare spatial attention and focus consistency over pathology regions.

## Limitations

- This is a representation-learning experiment, **not a clinical decision system**.
- GA module is intentionally simplified and does not implement full Clifford algebra neural operators.
- Improvements may depend on architecture depth, augmentation policy, and random seed.

## Methodological rigor checklist

- ✅ Multi-dataset evaluation supported (e.g., PathMNIST, BloodMNIST, DermaMNIST)
- ✅ Multi-seed evaluation supported (recommend 3–5 seeds minimum)
- ✅ Statistical reporting (mean ± std, 95% CI)
- ✅ Significance testing (paired permutation test)
- ✅ Data-efficiency protocol supported (e.g., 25%, 50%, 75%, 100%)

## Reproducibility notes

- Global seed is fixed in configuration.
- Deterministic settings are enabled for PyTorch backends.
- Hyperparameters are centralized in [experiments/config.yaml](experiments/config.yaml).
# GA-Based-Representation-for-Image-Based-Learning-in-Medical-Imaging
