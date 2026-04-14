"""
CLI de treinamento.

Uso:
    python scripts/train.py --config configs/resnet50.yaml
    python scripts/train.py --config configs/vit_b16.yaml --data_dir /content/data/processed
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import create_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import build_model, freeze_backbone, unfreeze_backbone
from src.training.losses import build_loss, compute_class_weights
from src.training.optim import build_optimizer, build_warmup_cosine_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelo de classificação de doenças de soja")
    parser.add_argument("--config", type=Path, required=True, help="YAML do modelo")
    parser.add_argument("--base_config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--data_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    cfg = load_model_config(args.config, args.base_config)
    set_seed(cfg["seed"])

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]
    input_size = model_cfg["input_size"]
    num_classes = cfg["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Iniciando treino: {model_name} | input={input_size}px | device={device}")
    logger.info(f"Git commit: {_get_git_hash()}")

    train_tf = get_train_transforms(input_size)
    val_tf = get_val_transforms(input_size)
    train_loader, val_loader, _ = create_dataloaders(
        processed_dir=args.data_dir,
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    model = build_model(model_name, num_classes=num_classes, pretrained=not args.no_pretrained)

    class_weights = compute_class_weights(args.data_dir / "train.csv", num_classes)
    criterion = build_loss(
        label_smoothing=cfg["loss"]["label_smoothing"],
        class_weights=class_weights,
        device=device,
    )

    optimizer = build_optimizer(
        model,
        lr_backbone=3e-5,
        lr_head=3e-4,
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs_total"] * steps_per_epoch
    warmup_steps = int(cfg["scheduler"]["warmup_steps_ratio"] * total_steps)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        model_name=model_name,
        mixed_precision=cfg.get("mixed_precision", True),
        gradient_clip_norm=cfg.get("gradient_clip_norm", 1.0),
        tensorboard_dir=cfg["logging"]["tensorboard_dir"],
        log_every_n_steps=cfg["logging"]["log_every_n_steps"],
    )

    trainer.fit(
        epochs_total=cfg["epochs_total"],
        epochs_warmup=cfg["epochs_warmup"],
        patience=cfg["patience_early_stop"],
        freeze_fn=freeze_backbone,
        unfreeze_fn=unfreeze_backbone,
    )
    logger.info("Treino concluído.")


if __name__ == "__main__":
    main()
