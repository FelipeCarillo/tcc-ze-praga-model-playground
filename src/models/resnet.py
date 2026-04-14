"""ResNet-50 via timm para classificação de doenças de soja."""

import torch.nn as nn

from .factory import build_model, freeze_backbone, unfreeze_backbone


def build_resnet50(num_classes: int = 29, pretrained: bool = True) -> nn.Module:
    """Constrói ResNet-50 com head substituído para num_classes."""
    return build_model("resnet50", num_classes=num_classes, pretrained=pretrained)


__all__ = ["build_resnet50", "freeze_backbone", "unfreeze_backbone"]
