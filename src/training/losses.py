"""Funções de perda: Cross-Entropy com label smoothing e pesos de classe."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def compute_class_weights(train_csv: str | Path, num_classes: int) -> list[float]:
    """
    Calcula pesos de classe balanceados: weight_i = n_total / (n_classes * count_i).

    Returns:
        Lista de floats com len == num_classes.
    """
    df = pd.read_csv(train_csv)
    counts = np.bincount(df["label_idx"].values, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return weights.tolist()


def build_loss(
    label_smoothing: float = 0.1,
    class_weights: list[float] | None = None,
    device: torch.device | None = None,
) -> nn.CrossEntropyLoss:
    """
    Cria CrossEntropyLoss com label smoothing e pesos de classe opcionais.

    Args:
        label_smoothing: Fator de suavização de rótulos.
        class_weights: Lista de pesos por classe (saída de compute_class_weights).
        device: Dispositivo onde os pesos serão alocados.

    Returns:
        nn.CrossEntropyLoss configurado.
    """
    weight_tensor: torch.Tensor | None = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
