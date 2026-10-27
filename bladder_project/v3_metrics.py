from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _safe(callable_, *args, **kwargs) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(callable_(*args, **kwargs))
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 15
) -> float:
    confidence = np.max(y_prob, axis=1)
    prediction = np.argmax(y_prob, axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(1, len(y_true))
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.sum() / total * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    labels = np.arange(len(class_names), dtype=int)
    if y_prob.ndim != 2 or y_prob.shape[1] != len(labels):
        raise ValueError(f"Expected probabilities [N,{len(labels)}], got {y_prob.shape}")
    if not np.isfinite(y_prob).all():
        raise FloatingPointError("Non-finite probability detected")
    sums = y_prob.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=2e-4):
        raise ValueError(f"Probability rows do not sum to one: range={sums.min():.6f}..{sums.max():.6f}")

    y_bin = label_binarize(y_true, classes=labels)
    if len(labels) == 2 and y_bin.shape[1] == 1:
        y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
        "cohen_kappa": _safe(cohen_kappa_score, y_true, y_pred),
        "log_loss": _safe(log_loss, y_true, y_prob, labels=labels.tolist()),
        "brier_score": multiclass_brier(y_true, y_prob, len(labels)),
        "ece_15": expected_calibration_error(y_true, y_prob, bins=15),
        "jaccard_macro": _safe(jaccard_score, y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "jaccard_weighted": _safe(jaccard_score, y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": _safe(roc_auc_score, y_bin, y_prob, average="macro", multi_class="ovr"),
        "roc_auc_ovr_weighted": _safe(roc_auc_score, y_bin, y_prob, average="weighted", multi_class="ovr"),
        "roc_auc_ovr_micro": _safe(roc_auc_score, y_bin, y_prob, average="micro", multi_class="ovr"),
        "auprc_macro": _safe(average_precision_score, y_bin, y_prob, average="macro"),
        "auprc_weighted": _safe(average_precision_score, y_bin, y_prob, average="weighted"),
        "auprc_micro": _safe(average_precision_score, y_bin, y_prob, average="micro"),
        "confusion_matrix": cm.tolist(),
        "support": total,
    }

    for average in ("macro", "weighted", "micro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class: List[Dict[str, Any]] = []
    for index, class_name in enumerate(class_names):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        npv = tn / (tn + fn) if tn + fn else float("nan")
        fpr = fp / (fp + tn) if fp + tn else float("nan")
        fnr = fn / (fn + tp) if fn + tp else float("nan")
        per_class.append(
            {
                "class": str(class_name),
                "support": int(support[index]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(precision[index]),
                "recall_sensitivity": float(recall[index]),
                "specificity": float(specificity),
                "npv": float(npv),
                "fpr": float(fpr),
                "fnr": float(fnr),
                "f1": float(f1[index]),
                "roc_auc": _safe(roc_auc_score, y_bin[:, index], y_prob[:, index]),
                "auprc": _safe(average_precision_score, y_bin[:, index], y_prob[:, index]),
            }
        )

    specificity = np.asarray([row["specificity"] for row in per_class], dtype=float)
    npv = np.asarray([row["npv"] for row in per_class], dtype=float)
    weights = np.asarray([row["support"] for row in per_class], dtype=float)
    metrics["specificity_macro"] = float(np.nanmean(specificity))
    metrics["specificity_weighted"] = float(np.nansum(specificity * weights) / max(1.0, np.nansum(weights)))
    metrics["npv_macro"] = float(np.nanmean(npv))
    metrics["npv_weighted"] = float(np.nansum(npv * weights) / max(1.0, np.nansum(weights)))
    return metrics, per_class, cm
