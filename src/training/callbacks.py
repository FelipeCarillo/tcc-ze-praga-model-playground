"""Callbacks de treinamento: checkpoint e early stopping."""

import copy
from pathlib import Path

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointCallback:
    """
    Salva o melhor checkpoint com base em uma métrica monitorada.

    Args:
        save_dir: Diretório onde o checkpoint será salvo.
        model_name: Nome base do arquivo (.pth).
        monitor: Nome da métrica a monitorar (ex: 'val_f1_macro').
        mode: 'max' ou 'min'.
    """

    def __init__(
        self,
        save_dir: str | Path,
        model_name: str,
        monitor: str = "val_f1_macro",
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_weights: dict | None = None

    def __call__(self, model: nn.Module, metrics: dict[str, float], epoch: int) -> bool:
        """
        Verifica se houve melhora e salva checkpoint se sim.

        Returns:
            True se este foi o melhor epoch até agora.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return False

        improved = (self.mode == "max" and value > self.best_value) or (
            self.mode == "min" and value < self.best_value
        )
        if improved:
            self.best_value = value
            self.best_weights = copy.deepcopy(model.state_dict())
            ckpt_path = self.save_dir / f"best_{self.model_name}.pth"
            torch.save(self.best_weights, ckpt_path)
            logger.info(
                f"Checkpoint salvo (epoch {epoch}) | {self.monitor}={value:.4f} → {ckpt_path}"
            )
        return improved

    def restore_best(self, model: nn.Module) -> None:
        """Carrega os melhores pesos no modelo."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Melhores pesos restaurados | {self.monitor}={self.best_value:.4f}")


class EarlyStoppingCallback:
    """
    Interrompe o treinamento quando a métrica monitorada para de melhorar.

    Args:
        patience: Épocas sem melhora antes de parar.
        monitor: Métrica a monitorar.
        mode: 'max' ou 'min'.
        min_delta: Mudança mínima para considerar melhora.
    """

    def __init__(
        self,
        patience: int = 7,
        monitor: str = "val_f1_macro",
        mode: str = "max",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def __call__(self, metrics: dict[str, float]) -> bool:
        """
        Atualiza contador e retorna True se o treinamento deve ser interrompido.
        """
        value = metrics.get(self.monitor)
        if value is None:
            return False

        improved = (self.mode == "max" and value > self.best_value + self.min_delta) or (
            self.mode == "min" and value < self.best_value - self.min_delta
        )
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(f"Early stopping: {self.patience} épocas sem melhora em {self.monitor}.")
            return True
        return False
