"""
Fábrica de modelos — cria qualquer arquitetura via timm com interface uniforme.

Mapeamento:
  resnet50        → timm 'resnet50'
  efficientnet_b4 → timm 'tf_efficientnet_b4_ns'
  vit_b16         → timm 'vit_base_patch16_224'
"""

import timm
import torch.nn as nn

_TIMM_NAMES: dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b4": "tf_efficientnet_b4_ns",
    "vit_b16": "vit_base_patch16_224",
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Constrói um modelo de classificação via timm.

    Args:
        name: Nome amigável ('resnet50', 'efficientnet_b4', 'vit_b16') ou nome timm direto.
        num_classes: Número de classes de saída.
        pretrained: Se True, carrega pesos ImageNet pré-treinados.

    Returns:
        Modelo nn.Module configurado para fine-tuning.
    """
    timm_name = _TIMM_NAMES.get(name, name)
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model: nn.Module) -> None:
    """
    Congela todos os parâmetros exceto o head de classificação.
    Compatível com arquiteturas timm (head, fc ou classifier).
    """
    head_names = {"head", "fc", "classifier"}
    for name, param in model.named_parameters():
        top_level = name.split(".")[0]
        param.requires_grad = top_level in head_names


def unfreeze_backbone(model: nn.Module) -> None:
    """Descongela todos os parâmetros para fine-tuning completo."""
    for param in model.parameters():
        param.requires_grad = True
