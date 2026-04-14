"""
Gera splits estratificados train/val/test e salva em CSV.

Uso:
    python src/data/splits.py --raw_dir data/raw/digipathos --out_dir data/processed
"""

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = 1 - TRAIN_RATIO - VAL_RATIO = 0.15
RANDOM_STATE = 42


def generate_splits(raw_dir: Path, out_dir: Path) -> None:
    """
    Varre raw_dir/<classe>/*.jpg, gera splits 70/15/15 estratificados e
    salva train.csv, val.csv, test.csv em out_dir.

    Colunas CSV: filepath, label, label_idx
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    if not classes:
        raise FileNotFoundError(f"Nenhuma subpasta de classe encontrada em {raw_dir}")

    label_to_idx = {cls: i for i, cls in enumerate(classes)}

    filepaths: list[Path] = []
    labels: list[str] = []

    for cls in classes:
        cls_dir = raw_dir / cls
        imgs = [p for p in cls_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS]
        filepaths.extend(imgs)
        labels.extend([cls] * len(imgs))

    label_idxs = [label_to_idx[lbl] for lbl in labels]

    X_train, X_tmp, y_train, y_tmp, yi_train, yi_tmp = train_test_split(
        filepaths, labels, label_idxs,
        test_size=(1 - TRAIN_RATIO),
        stratify=label_idxs,
        random_state=RANDOM_STATE,
    )

    val_ratio_of_tmp = VAL_RATIO / (1 - TRAIN_RATIO)
    X_val, X_test, y_val, y_test, yi_val, yi_test = train_test_split(
        X_tmp, y_tmp, yi_tmp,
        test_size=(1 - val_ratio_of_tmp),
        stratify=yi_tmp,
        random_state=RANDOM_STATE,
    )

    splits = {
        "train": (X_train, y_train, yi_train),
        "val": (X_val, y_val, yi_val),
        "test": (X_test, y_test, yi_test),
    }

    for split_name, (fps, lbls, idxs) in splits.items():
        csv_path = out_dir / f"{split_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "label_idx"])
            for fp, lbl, idx in zip(fps, lbls, idxs):
                writer.writerow([str(fp), lbl, idx])
        print(f"[splits] {split_name}: {len(fps)} amostras → {csv_path}")

    label_map_path = out_dir / "label_map.csv"
    with label_map_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label_idx", "label"])
        for cls, idx in label_to_idx.items():
            writer.writerow([idx, cls])
    print(f"[splits] label_map salvo em {label_map_path}")
    print(f"[splits] {len(classes)} classes: {classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera splits estratificados")
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    generate_splits(args.raw_dir, args.out_dir)
