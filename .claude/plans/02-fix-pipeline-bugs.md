# 02 — Fix Pipeline Bugs

> **Status**: TODO | **Esforço**: 3h | **Bloqueia**: não (mas evita warnings e dor de cabeça em Colab atualizado)

## Context

Code review do pipeline em `tcc-ze-praga-model-playground/` identificou 5 issues técnicos: APIs deprecated do albumentations e torch.amp, LRs hardcoded em vez de virem do YAML, config `class_weights` ignorado, e um filtro defensivo faltando em `splits.py`.

Aplicar antes que alguém atualize as libs e tudo quebre.

---

## Issues a corrigir

### B1 — `src/data/transforms.py:13` e `:20`

API antiga do albumentations (vai ser removida em v2.0):

```python
# antes
A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0))
A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3)

# depois
A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0))
A.CoarseDropout(
    num_holes_range=(1, 8),
    hole_height_range=(8, 32),
    hole_width_range=(8, 32),
    fill_value=0,
    p=0.3,
)
```

### B2 — `src/training/trainer.py:11`

`torch.cuda.amp` deprecated desde torch 2.3:

```python
# antes
from torch.cuda.amp import GradScaler, autocast
# ...
self.scaler = GradScaler(enabled=self.mixed_precision)
with autocast(enabled=self.mixed_precision):

# depois
from torch.amp import GradScaler, autocast
# ...
self.scaler = GradScaler("cuda", enabled=self.mixed_precision)
with autocast(device_type="cuda", enabled=self.mixed_precision):
```

### B3 — `scripts/train.py:79–80` (LRs hardcoded)

`lr_backbone=3e-5, lr_head=3e-4` estão no código em vez de no config:

```python
# scripts/train.py — antes
optimizer = build_optimizer(
    model,
    lr_backbone=3e-5,
    lr_head=3e-4,
    weight_decay=cfg["optimizer"]["weight_decay"],
)

# depois
optimizer = build_optimizer(
    model,
    lr_backbone=cfg["optimizer"]["lr_backbone"],
    lr_head=cfg["optimizer"]["lr_head"],
    weight_decay=cfg["optimizer"]["weight_decay"],
)
```

Adicionar em `configs/base.yaml`:

```yaml
optimizer:
  name: adamw
  weight_decay: 1.0e-4
  lr_backbone: 3.0e-5
  lr_head: 3.0e-4
```

### B4 — `scripts/train.py:70` (`class_weights` ignorado)

Sempre computa pesos, ignorando `cfg.loss.class_weights`:

```python
# antes
class_weights = compute_class_weights(args.data_dir / "train.csv", num_classes)

# depois
class_weights = None
if cfg["loss"].get("class_weights") == "balanced":
    class_weights = compute_class_weights(args.data_dir / "train.csv", num_classes)
```

### B5 — `src/data/splits.py:44` (filtro defensivo)

**PULAR** — este arquivo será substituído pelo plano 01 (`scripts/ingest/04_split.py`). Não vale tocar.

---

## Verificação

```bash
cd tcc-ze-praga-model-playground

# 1. Sintaxe
python -m py_compile src/data/transforms.py src/training/trainer.py scripts/train.py

# 2. Sem deprecation warnings na importação
python -W error::DeprecationWarning -c "from src.data.transforms import get_train_transforms; get_train_transforms(224)"
python -W error::DeprecationWarning -c "from src.training.trainer import Trainer"

# 3. Smoke train: 1 época sem warnings de deprecation
python scripts/train.py --config configs/resnet50.yaml --data_dir data/processed 2>&1 | grep -i "deprecat" && echo "AINDA TEM WARNING" || echo "OK"

# 4. Conferir que LRs vêm do config (alterar base.yaml e ver o log mudar)
```

**Aceite**: smoke train roda sem `DeprecationWarning` e `optimizer.param_groups[0]["lr"]` reflete o valor de `base.yaml`.
