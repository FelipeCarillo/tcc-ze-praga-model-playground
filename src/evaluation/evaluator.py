"""Avaliação completa no test set: métricas + relatório por classe."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..training.metrics import compute_metrics, per_class_report
from ..utils.logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    label_names: list[str],
    device: torch.device | None = None,
) -> dict:
    """
    Avalia o modelo no loader fornecido.

    Returns:
        Dict com accuracy, f1_macro, f1_weighted, precision_macro, recall_macro,
        per_class, y_true, y_pred, y_probs.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_probs: list = []

    for images, labels in tqdm(loader, desc="Avaliando"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_probs.extend(probs.cpu().numpy())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    report = per_class_report(y_true_arr, y_pred_arr, label_names)

    logger.info(f"Acurácia: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f}")

    return {
        **metrics,
        "per_class": report,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "y_probs": np.array(y_probs),
    }


def save_metrics(results: dict, save_path: str | Path, model_name: str) -> None:
    """Salva as métricas escalares (sem arrays numpy) em JSON."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "model": model_name,
        **{k: v for k, v in results.items() if not isinstance(v, np.ndarray)},
    }
    save_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    logger.info(f"Métricas salvas em {save_path}")
