from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
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


def reliability_diagram_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, object]:
    """Collect reliability diagram bin statistics."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)
    bin_count = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_conf >= lo) & (y_conf <= hi)
        else:
            mask = (y_conf >= lo) & (y_conf < hi)
        if not np.any(mask):
            continue
        bin_count[i] = int(np.sum(mask))
        bin_acc[i] = float(np.mean((y_pred[mask] == y_true[mask]).astype(np.float64)))
        bin_conf[i] = float(np.mean(y_conf[mask]))

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_acc.tolist(),
        "bin_confidence": bin_conf.tolist(),
        "bin_count": bin_count.tolist(),
    }


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute multiclass Brier score over one-hot targets."""
    if y_prob.ndim != 2:
        raise ValueError("y_prob must have shape [N, C].")
    n, c = y_prob.shape
    if n == 0:
        return 0.0
    one_hot = np.zeros((n, c), dtype=np.float64)
    idx = np.clip(y_true.astype(np.int64), 0, c - 1)
    one_hot[np.arange(n), idx] = 1.0
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def one_vs_rest_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    """Compute one-vs-rest ROC-AUC per class plus macro/micro averages."""
    if y_prob.ndim != 2:
        raise ValueError("y_prob must have shape [N, C].")
    n, c = y_prob.shape
    if n == 0:
        return {
            "per_class_auc": [],
            "macro_auc": None,
            "micro_auc": None,
        }

    one_hot = np.zeros((n, c), dtype=np.int64)
    idx = np.clip(y_true.astype(np.int64), 0, c - 1)
    one_hot[np.arange(n), idx] = 1

    per_class_auc = []
    for class_index in range(c):
        binary_true = one_hot[:, class_index]
        try:
            auc_value = roc_auc_score(binary_true, y_prob[:, class_index])
        except ValueError:
            auc_value = float("nan")
        per_class_auc.append(float(auc_value))

    try:
        macro_auc = float(roc_auc_score(one_hot, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = None
    try:
        micro_auc = float(roc_auc_score(one_hot, y_prob, average="micro", multi_class="ovr"))
    except ValueError:
        micro_auc = None

    return {
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
    }


def predictive_mutual_information(prob_samples: np.ndarray) -> np.ndarray:
    """Estimate mutual information from repeated predictive probability samples.

    prob_samples should have shape [S, N, C].
    Returns a vector of length N.
    """
    if prob_samples.ndim != 3:
        raise ValueError("prob_samples must have shape [S, N, C].")
    mean_prob = np.mean(prob_samples, axis=0)
    mean_entropy = -np.sum(mean_prob * np.log(np.clip(mean_prob, 1e-12, 1.0)), axis=1)
    sample_entropy = -np.sum(prob_samples * np.log(np.clip(prob_samples, 1e-12, 1.0)), axis=2)
    expected_entropy = np.mean(sample_entropy, axis=0)
    return mean_entropy - expected_entropy


def support_weighting_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    """Quantify per-class support and its impact on macro metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    support = cm.sum(axis=1).astype(np.float64)
    total = max(float(support.sum()), 1.0)
    support_fraction = support / total
    class_correct = np.diag(cm).astype(np.float64)
    class_accuracy = np.divide(class_correct, support, out=np.zeros_like(class_correct), where=support > 0)
    weighted_impact = support_fraction * class_accuracy
    return {
        "support": support.astype(int).tolist(),
        "support_fraction": support_fraction.tolist(),
        "class_accuracy": class_accuracy.tolist(),
        "weighted_impact": weighted_impact.tolist(),
    }


def histogram_data(
    values: np.ndarray,
    bins: int,
    value_range: tuple[float, float] | None = None,
) -> Dict[str, object]:
    hist, edges = np.histogram(values, bins=bins, range=value_range)
    total = max(int(np.sum(hist)), 1)
    density = hist.astype(np.float64) / total
    return {
        "bin_edges": edges.tolist(),
        "counts": hist.astype(int).tolist(),
        "density": density.tolist(),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_conf: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    entropy: Optional[np.ndarray] = None,
    logit_abs: Optional[np.ndarray] = None,
    n_bins: int = 15,
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
    ece = expected_calibration_error(y_true, y_pred, y_conf, n_bins=n_bins)
    cm = confusion_matrix(y_true, y_pred)
    labels = np.arange(cm.shape[0])
    row_totals = cm.sum(axis=1)
    per_class_accuracy = np.divide(
        np.diag(cm),
        row_totals,
        out=np.zeros_like(row_totals, dtype=np.float64),
        where=row_totals > 0,
    )

    precision_per_class = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
    )
    recall_per_class = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
    )
    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0,
    )

    reliability = reliability_diagram_data(y_true, y_pred, y_conf, n_bins=n_bins)

    brier = None
    if y_prob is not None:
        brier = multiclass_brier_score(y_true, y_prob)

    result: Dict[str, object] = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "ece": float(ece),
        "precision_per_class": precision_per_class.astype(float).tolist(),
        "recall_per_class": recall_per_class.astype(float).tolist(),
        "f1_per_class": f1_per_class.astype(float).tolist(),
        "accuracy_per_class": per_class_accuracy.astype(float).tolist(),
        "reliability_diagram": reliability,
        "confusion_matrix": cm.tolist(),
    }

    if brier is not None:
        result["brier_score"] = float(brier)

    # Distribution logging for calibration/stability analysis.
    result["confidence_histogram"] = histogram_data(y_conf, bins=20, value_range=(0.0, 1.0))
    if entropy is not None:
        result["entropy_histogram"] = histogram_data(entropy, bins=20)
    if logit_abs is not None:
        result["logit_magnitude_histogram"] = histogram_data(logit_abs, bins=20)

    return result
