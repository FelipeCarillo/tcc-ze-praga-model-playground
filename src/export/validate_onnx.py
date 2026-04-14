"""
Valida paridade numérica entre PyTorch e ONNX Runtime.
Roda N_SAMPLES amostras do test set e compara logits com np.allclose.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logger import get_logger

logger = get_logger(__name__)

N_SAMPLES = 50
RTOL = 1e-3
ATOL = 1e-5


def validate_onnx(
    model: nn.Module,
    onnx_path: str | Path,
    test_loader: DataLoader,
    device: torch.device | None = None,
) -> bool:
    """
    Compara saídas do modelo PyTorch com ONNX Runtime em N_SAMPLES imagens.

    Args:
        model: Modelo PyTorch (eval mode, melhores pesos).
        onnx_path: Caminho para o arquivo .onnx.
        test_loader: DataLoader do test set.
        device: Dispositivo PyTorch.

    Returns:
        True se todas as comparações passarem; False caso contrário.
    """
    device = device or torch.device("cpu")
    model.eval().to(device)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    collected = 0
    all_passed = True

    for images, _ in test_loader:
        if collected >= N_SAMPLES:
            break
        images = images.to(device)

        with torch.no_grad():
            torch_out = model(images).cpu().numpy()

        ort_out = sess.run(None, {input_name: images.cpu().numpy()})[0]

        if not np.allclose(torch_out, ort_out, rtol=RTOL, atol=ATOL):
            max_diff = float(np.abs(torch_out - ort_out).max())
            logger.error(f"FALHA na paridade ONNX! Diferença máxima: {max_diff:.6f}")
            all_passed = False
            break

        collected += images.size(0)

    if all_passed:
        logger.info(
            f"Paridade ONNX validada em {min(collected, N_SAMPLES)} amostras "
            f"(rtol={RTOL}, atol={ATOL})."
        )
    return all_passed
