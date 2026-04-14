"""
SoybeanLeafDataset — carrega imagens a partir de CSV gerado por splits.py.

CSV esperado: colunas filepath, label, label_idx
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A


class SoybeanLeafDataset(Dataset):
    """
    Dataset de doenças foliares de soja carregado a partir de CSV.

    Args:
        csv_path: Caminho para o CSV (train.csv, val.csv ou test.csv).
        transform: Pipeline albumentations aplicado a cada imagem.
    """

    def __init__(self, csv_path: str | Path, transform: A.Compose | None = None) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        if not {"filepath", "label", "label_idx"}.issubset(self.df.columns):
            raise ValueError("CSV deve ter colunas: filepath, label, label_idx")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = np.array(Image.open(row["filepath"]).convert("RGB"))
        label_idx = int(row["label_idx"])

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label_idx


def create_dataloaders(
    processed_dir: str | Path,
    train_transform: A.Compose,
    val_transform: A.Compose,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria DataLoaders para train, val e test a partir dos CSVs em processed_dir.

    Returns:
        Tupla (train_loader, val_loader, test_loader).
    """
    processed_dir = Path(processed_dir)

    train_ds = SoybeanLeafDataset(processed_dir / "train.csv", transform=train_transform)
    val_ds = SoybeanLeafDataset(processed_dir / "val.csv", transform=val_transform)
    test_ds = SoybeanLeafDataset(processed_dir / "test.csv", transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[dataset] train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")
    return train_loader, val_loader, test_loader
