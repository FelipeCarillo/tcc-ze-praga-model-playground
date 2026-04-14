"""EfficientNet-B4 (tf_efficientnet_b4_ns) via timm para classificação de doenças de soja."""

import torch.nn as nn

from .factory import build_model, freeze_backbone, unfreeze_backbone


def build_efficientnet_b4(num_classes: int = 29, pretrained: bool = True) -> nn.Module:
    """Constrói EfficientNet-B4 (noisy-student) com head para num_classes."""
    return build_model("efficientnet_b4", num_classes=num_classes, pretrained=pretrained)


__all__ = ["build_efficientnet_b4", "freeze_backbone", "unfreeze_backbone"]
