"""
CLI de exportação para ONNX + validação de paridade.

Uso:
    python scripts/export.py --config configs/resnet50.yaml \
        --checkpoint artifacts/checkpoints/best_resnet50.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import create_dataloaders
from src.data.transforms import get_val_transforms
from src.export.to_onnx import export_to_onnx
from src.export.validate_onnx import validate_onnx
from src.models.factory import build_model
from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta modelo para ONNX e valida paridade")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base_config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    cfg = load_model_config(args.config, args.base_config)
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    input_size = model_cfg["input_size"]
    num_classes = cfg["num_classes"]
    device = torch.device("cpu")  # exportação sempre em CPU para portabilidade

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    onnx_path = Path("artifacts/onnx") / f"{model_name}.onnx"
    export_to_onnx(model, onnx_path, input_size=input_size)

    val_tf = get_val_transforms(input_size)
    _, _, test_loader = create_dataloaders(
        processed_dir=args.data_dir,
        train_transform=val_tf,
        val_transform=val_tf,
        batch_size=cfg["batch_size"],
        num_workers=0,
    )

    passed = validate_onnx(model, onnx_path, test_loader, device=device)
    if not passed:
        raise RuntimeError("Validação ONNX falhou! Verifique os logs acima.")

    logger.info(f"Export concluído: {onnx_path}")


if __name__ == "__main__":
    main()
