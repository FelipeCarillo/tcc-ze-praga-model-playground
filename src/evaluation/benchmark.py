"""
Benchmark de latência de inferência (CPU e GPU) e tamanho em disco.
Mede tempo para batches de 1, 8, 32 com 100 runs após warmup.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZES = [1, 8, 32]
N_WARMUP = 10
N_RUNS = 100


def _measure_latency(
    model: nn.Module,
    input_shape: tuple[int, int, int],
    batch_size: int,
    device: torch.device,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
) -> dict[str, float]:
    """Mede latência média e desvio padrão (ms) para um batch size específico."""
    model.eval().to(device)
    dummy = torch.randn(batch_size, *input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
    }


def benchmark_model(
    model: nn.Module,
    input_size: int,
    checkpoint_path: str | Path | None = None,
    onnx_path: str | Path | None = None,
) -> dict:
    """
    Benchmarka o modelo em CPU (e GPU se disponível) para múltiplos batch sizes.

    Returns:
        Dict com resultados de latência por device/batch_size e tamanhos em disco (MB).
    """
    input_shape = (3, input_size, input_size)
    results: dict = {"latency": {}, "size_mb": {}}

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        device_key = device.type
        results["latency"][device_key] = {}
        for bs in BATCH_SIZES:
            stats = _measure_latency(model, input_shape, bs, device)
            results["latency"][device_key][f"batch_{bs}"] = stats
            logger.info(
                f"[benchmark] {device_key} | batch={bs} | "
                f"mean={stats['mean_ms']:.2f}ms | p95={stats['p95_ms']:.2f}ms"
            )

    if checkpoint_path and Path(checkpoint_path).exists():
        results["size_mb"]["pth"] = round(Path(checkpoint_path).stat().st_size / 1e6, 2)

    if onnx_path and Path(onnx_path).exists():
        results["size_mb"]["onnx"] = round(Path(onnx_path).stat().st_size / 1e6, 2)

    return results
