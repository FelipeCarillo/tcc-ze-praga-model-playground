"""Cálculo de métricas de classificação."""

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """
    Calcula métricas macro e weighted.

    Returns:
        Dict com accuracy, f1_macro, f1_weighted, precision_macro, recall_macro.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: list[str],
) -> dict:
    """Retorna o classification_report como dict (inclui precisão/recall por classe)."""
    return classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
