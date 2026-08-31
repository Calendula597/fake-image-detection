from __future__ import annotations
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_metrics(labels, probs, threshold: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    n_unique = len(np.unique(labels))
    auc = roc_auc_score(labels, probs) if n_unique > 1 else float("nan")
    preds = (probs >= threshold).astype(int)
    out = {
        "roc_auc": auc,
        "pr_auc": average_precision_score(labels, probs) if n_unique > 1 else float("nan"),
        "log_loss": (
            log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1]) if n_unique > 1 else float("nan")
        ),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "mcc": matthews_corrcoef(labels, preds) if n_unique > 1 else float("nan"),
        "brier": (
            brier_score_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7)) if n_unique > 1 else float("nan")
        ),
    }
    return out
