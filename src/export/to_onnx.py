"""Exporta modelo PyTorch para ONNX com opset 17 e dynamic batch axis."""

from pathlib import Path

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_size: int = 224,
    opset_version: int = 17,
) -> Path:
    """
    Exporta model para ONNX com batch size dinâmico.

    Args:
        model: Modelo PyTorch em modo eval() com melhores pesos carregados.
        output_path: Caminho de saída (.onnx).
        input_size: Tamanho H=W da imagem de entrada.
        opset_version: Versão do opset ONNX (padrão: 17).

    Returns:
        Path para o arquivo ONNX gerado.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"Modelo exportado para ONNX: {output_path} ({size_mb:.1f} MB)")
    return output_path
