"""
CLI de avaliação no test set.

Uso:
    python scripts/evaluate.py --config configs/resnet50.yaml \
        --checkpoint artifacts/checkpoints/best_resnet50.pth
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import create_dataloaders
from src.data.transforms import get_val_transforms
from src.evaluation.confusion import plot_confusion_matrix
from src.evaluation.evaluator import evaluate, save_metrics
from src.models.factory import build_model
from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia modelo no test set")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map_path = args.data_dir / "label_map.csv"
    label_names: list[str] = []
    with label_map_path.open() as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: int(r["label_idx"]))
        label_names = [r["label"] for r in rows]

    val_tf = get_val_transforms(input_size)
    _, _, test_loader = create_dataloaders(
        processed_dir=args.data_dir,
        train_transform=val_tf,
        val_transform=val_tf,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results = evaluate(model, test_loader, label_names, device=device)
    save_metrics(results, Path("artifacts/metrics") / f"metrics_{model_name}.json", model_name)

    fig_path = Path("artifacts/figures") / f"confusion_{model_name}.png"
    plot_confusion_matrix(results["y_true"], results["y_pred"], label_names, save_path=fig_path)
    logger.info(f"Confusion matrix salva em {fig_path}")


if __name__ == "__main__":
    main()
