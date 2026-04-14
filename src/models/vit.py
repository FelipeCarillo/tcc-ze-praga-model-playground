"""ViT-Base/16 via timm para classificação de doenças de soja."""

import torch.nn as nn

from .factory import build_model, freeze_backbone, unfreeze_backbone


def build_vit_b16(num_classes: int = 29, pretrained: bool = True) -> nn.Module:
    """Constrói ViT-B/16 com head para num_classes."""
    return build_model("vit_b16", num_classes=num_classes, pretrained=pretrained)


__all__ = ["build_vit_b16", "freeze_backbone", "unfreeze_backbone"]
