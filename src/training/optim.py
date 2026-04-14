"""Otimizador AdamW + scheduler cosine com warmup linear."""

import math

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module,
    lr_backbone: float = 3e-5,
    lr_head: float = 3e-4,
    weight_decay: float = 1e-4,
) -> AdamW:
    """
    Cria AdamW com parameter groups separados para backbone e head.

    Args:
        lr_backbone: Learning rate para parâmetros do backbone.
        lr_head: Learning rate para parâmetros do head de classificação.
        weight_decay: Penalização L2.

    Returns:
        Otimizador AdamW com dois parameter groups.
    """
    head_keywords = {"head", "fc", "classifier"}
    backbone_params, head_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".")[0]
        if top in head_keywords:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
    return AdamW(param_groups, weight_decay=weight_decay)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Scheduler com warmup linear seguido de decaimento cosine.

    Args:
        warmup_steps: Número de steps de warmup linear (0 → lr_max).
        total_steps: Total de steps de treinamento.

    Returns:
        LambdaLR aplicado ao otimizador.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
