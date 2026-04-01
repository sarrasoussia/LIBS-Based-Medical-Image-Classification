from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) using confidence bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y_true), 1)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_conf >= lo) & (y_conf <= hi)
        else:
            mask = (y_conf >= lo) & (y_conf < hi)
        if not np.any(mask):
            continue

        bin_acc = np.mean((y_pred[mask] == y_true[mask]).astype(np.float64))
        bin_conf = float(np.mean(y_conf[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

    return float(ece)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
) -> Dict[str, object]:
    """Compute core classification metrics.

    - Accuracy: overall fraction of correct predictions.
    - Precision (macro): average class-wise positive predictive value.
    - Recall (macro): average class-wise sensitivity.
    - F1-score (macro): harmonic mean of precision and recall per class.
    - ECE: expected calibration error over confidence bins.
    - Confusion matrix: count table of true vs predicted classes.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    ece = expected_calibration_error(y_true, y_pred, y_conf, n_bins=15)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "ece": float(ece),
        "confusion_matrix": cm.tolist(),
    }
