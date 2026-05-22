# 06 — TTA + Ensemble

> **Status**: TODO | **Esforço**: 4h | **Bloqueia**: não, ganho fácil

## Context

Dois ganhos baratos de acurácia, sem retreino, comuns na literatura:

1. **TTA (Test-Time Augmentation)**: roda inferência em N variações da imagem (flip, crops), média softmax. +1–3% F1.
2. **Ensemble**: média de logits dos 3 modelos (ResNet-50 + EfficientNet-B4 + ViT-B/16). +1–2% F1 adicional sobre o melhor individual.

Custo: latência (3x para ensemble, 4–8x para TTA). Justificável em servidor; **não no edge**.

---

## TTA

### Implementação — `src/inference/tta.py`

```python
import torch

def tta_inference(model, image, n_augs: int = 4):
    """
    Roda inferência em N variações: original + hflip + vflip + hvflip.
    Para n_augs=8: adiciona 4 crops (5-crop center + 4 corners).
    Retorna softmax médio.
    """
    augs = [image, image.flip(-1)]  # original + hflip
    if n_augs >= 4:
        augs.extend([image.flip(-2), image.flip(-1).flip(-2)])  # +vflip, +hvflip
    if n_augs == 8:
        augs.extend(_5crop(image))

    with torch.no_grad():
        probs = torch.stack([torch.softmax(model(a), dim=-1) for a in augs])
    return probs.mean(dim=0)
```

Trade-off n_augs:
- `n=2` (orig + hflip) — +0.5%, 2x latência
- `n=4` (+ vflip + hvflip) — +1.5%, 4x latência (default)
- `n=8` (+ crops) — +2.5%, 8x latência

---

## Ensemble dos 3 modelos

### Implementação — `src/inference/ensemble.py`

Cada modelo já exportado em ONNX (via `src/export/to_onnx.py`). Carrega 3 sessões `onnxruntime`, faz média de **logits** (mais estável que média de probs).

```python
import numpy as np
import onnxruntime as ort

class OnnxEnsemble:
    def __init__(self, onnx_paths: list[str], weights: list[float] | None = None):
        self.sessions = [ort.InferenceSession(p) for p in onnx_paths]
        self.weights = weights or [1.0 / len(onnx_paths)] * len(onnx_paths)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (B, 3, H, W) — atenção: cada modelo pode ter H diferente
        # Pré-processar cada um separado se input_size diferir
        logits_list = [s.run(None, {"input": x_i})[0] for s, x_i in zip(self.sessions, self._preprocess_per_model(x))]
        logits_avg = sum(w * l for w, l in zip(self.weights, logits_list))
        return logits_avg
```

**Detalhe importante**: ResNet-50 e ViT-B/16 esperam 224×224, EfficientNet-B4 espera 380×380. **Não dá para passar o mesmo tensor**. Cada modelo recebe seu input redimensionado, mas a média de logits é straightforward (saída é `(B, num_classes)` em todos).

### Pesos do ensemble

Default: média uniforme `[1/3, 1/3, 1/3]`. Melhor: pesos proporcionais a `val_f1_macro` de cada modelo, normalizados:

```python
def compute_ensemble_weights(val_metrics: dict[str, float]) -> dict[str, float]:
    # val_metrics: {"resnet50": 0.82, "efficientnet_b4": 0.85, "vit_b16": 0.84}
    total = sum(val_metrics.values())
    return {k: v / total for k, v in val_metrics.items()}
```

---

## Combinar TTA + Ensemble

`tta_inference` por modelo → ensemble dos 3 outputs já com TTA aplicado:

```python
def predict_tta_ensemble(image, models, weights, n_augs=4):
    probs_per_model = [tta_inference(m, image, n_augs) for m in models]
    return sum(w * p for w, p in zip(weights, probs_per_model))
```

Custo total: 3 × n_augs forward passes (12 para n_augs=4).

---

## Verificação

```bash
# 1. Treinar e exportar os 3 modelos (planos 01–03 concluídos)
for cfg in resnet50 efficientnet_b4 vit_b16; do
    python scripts/train.py --config configs/$cfg.yaml
    python scripts/export.py --config configs/$cfg.yaml --checkpoint artifacts/checkpoints/best_$cfg.pth
done

# 2. Comparação em test set
python scripts/compare_inference_modes.py --test_csv data/processed/test.csv > artifacts/comparison.md

# Esperado em comparison.md (formato):
# | Modo                     | F1-macro | Latência batch=1 (ms) |
# | resnet50 single          |   0.82   |   45                  |
# | resnet50 + TTA(4)        |   0.835  |   180                 |
# | ensemble(3) single       |   0.86   |   135                 |
# | ensemble(3) + TTA(4)     |   0.875  |   540                 |
```

**Aceite**:
- TTA(4) > single em ≥ 1% F1
- Ensemble > melhor single em ≥ 1% F1
- TTA + Ensemble > Ensemble em ≥ 0.5% F1
- Latência ensemble + TTA(4) ≤ 1.5s em CPU batch=1 (para Lambda)
