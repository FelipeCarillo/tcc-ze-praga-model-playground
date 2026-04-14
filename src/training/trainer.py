"""
Loop de treinamento com AMP, TensorBoard, gradient clipping e two-phase fine-tuning.
Monitora val_f1_macro para checkpoint e early stopping.
"""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .callbacks import CheckpointCallback, EarlyStoppingCallback
from .metrics import compute_metrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Treina um modelo PyTorch com suporte a:
    - Mixed precision (AMP)
    - TensorBoard logging
    - Gradient clipping
    - Two-phase fine-tuning (warmup congelado + fine-tuning completo)
    - Checkpoint por val_f1_macro
    - Early stopping

    Args:
        model: Modelo a ser treinado.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        optimizer: Otimizador (AdamW com parameter groups).
        scheduler: LR scheduler (step a cada batch).
        criterion: Função de perda.
        device: Dispositivo de treino.
        checkpoint_dir: Diretório para checkpoints.
        tensorboard_dir: Diretório para logs do TensorBoard.
        model_name: Nome do modelo (usado no checkpoint e logs).
        mixed_precision: Se True, usa torch.cuda.amp.
        gradient_clip_norm: Norma máxima para gradient clipping (0 = desabilitado).
        log_every_n_steps: Frequência de log no TensorBoard (em batches).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        device: torch.device | None = None,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
        tensorboard_dir: str | Path = "artifacts/tensorboard",
        model_name: str = "model",
        mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        log_every_n_steps: int = 20,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.gradient_clip_norm = gradient_clip_norm
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_dir = Path(checkpoint_dir)

        self.model.to(self.device)
        self.scaler = GradScaler(enabled=self.mixed_precision)
        self.writer = SummaryWriter(log_dir=str(Path(tensorboard_dir) / model_name))
        self._global_step = 0

    def _train_epoch(self) -> tuple[float, list[int], list[int]]:
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in tqdm(self.train_loader, leave=False, desc="train"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.detach().argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            if self._global_step % self.log_every_n_steps == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self._global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self._global_step
                )

            self._global_step += 1

        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss, all_labels, all_preds

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, list[int], list[int]]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        for images, labels in tqdm(self.val_loader, leave=False, desc="val"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss, all_labels, all_preds

    def fit(
        self,
        epochs_total: int = 30,
        epochs_warmup: int = 3,
        patience: int = 7,
        freeze_fn: Callable | None = None,
        unfreeze_fn: Callable | None = None,
    ) -> nn.Module:
        """
        Executa o loop de treino completo com two-phase fine-tuning.

        Args:
            epochs_total: Número máximo de épocas.
            epochs_warmup: Épocas com backbone congelado (Fase 1).
            patience: Paciência para early stopping.
            freeze_fn: Função que congela o backbone (chamada antes da Fase 1).
            unfreeze_fn: Função que descongela o backbone (início da Fase 2).

        Returns:
            Modelo com os melhores pesos (melhor val_f1_macro).
        """
        checkpoint_cb = CheckpointCallback(
            save_dir=self.checkpoint_dir,
            model_name=self.model_name,
            monitor="val_f1_macro",
            mode="max",
        )
        early_stop_cb = EarlyStoppingCallback(
            patience=patience,
            monitor="val_f1_macro",
            mode="max",
        )

        if freeze_fn and epochs_warmup > 0:
            freeze_fn(self.model)
            logger.info(f"Fase 1: backbone congelado por {epochs_warmup} épocas.")

        for epoch in range(1, epochs_total + 1):
            if unfreeze_fn and epoch == epochs_warmup + 1:
                unfreeze_fn(self.model)
                logger.info(
                    f"Fase 2 (epoch {epoch}): backbone descongelado para fine-tuning completo."
                )

            train_loss, train_true, train_pred = self._train_epoch()
            val_loss, val_true, val_pred = self._val_epoch()

            train_metrics = compute_metrics(train_true, train_pred)
            val_metrics = compute_metrics(val_true, val_pred)
            val_metrics_log = {f"val_{k}": v for k, v in val_metrics.items()}
            val_metrics_log["val_loss"] = val_loss

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            for k, v in train_metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            logger.info(
                f"Epoch {epoch:03d}/{epochs_total} | "
                f"train_loss={train_loss:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_f1_macro={val_metrics['f1_macro']:.4f}"
            )

            checkpoint_cb(self.model, val_metrics_log, epoch)

            if early_stop_cb(val_metrics_log):
                break

        self.writer.close()
        checkpoint_cb.restore_best(self.model)
        return self.model
