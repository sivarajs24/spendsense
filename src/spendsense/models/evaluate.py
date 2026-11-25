"""Model evaluation metrics calculation and reporting."""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Computes comprehensive multiclass classification metrics.

    Macro-F1 is the primary model-selection metric to prevent dominant categories
    from distorting performance on minority classes.
    """
    if labels is None:
        labels = sorted(list(set(y_true).union(set(y_pred))))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "labels": labels,
        "per_class": {
            lbl: {
                "precision": round(report_dict[lbl]["precision"], 4),
                "recall": round(report_dict[lbl]["recall"], 4),
                "f1_score": round(report_dict[lbl]["f1-score"], 4),
                "support": int(report_dict[lbl]["support"]),
            }
            for lbl in labels
            if lbl in report_dict
        },
        "confusion_matrix": conf_mat,
        "classification_report_str": classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        ),
    }
