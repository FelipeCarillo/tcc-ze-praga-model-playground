# 04 — OOD Detection + Calibração

> **Status**: TODO | **Esforço**: 6h | **Bloqueia**: crítico para produto (não para TCC defensável)

## Context

Sem OOD: usuário envia foto de gato, modelo devolve "97% mancha_alvo" com straight face. Inaceitável para produto.

Sem calibração: CNNs modernos são sistematicamente overconfident — confiança 99% ≠ accuracy 99%. Métrica de produto ruim.

Ambos resolvem-se com técnicas baratas pós-treino.

---

## OOD Detection

### Implementação — `src/inference/ood.py`

Dois scorers em paralelo (combinar via OR), thresholds calibrados em set de val OOD:

**MSP** (Maximum Softmax Probability — Hendrycks & Gimpel 2017):
```python
def msp_score(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).max(dim=-1).values  # alto = in-dist
```

**Energy score** (Liu et al. 2020 — mais discriminativo que MSP):
```python
def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return -T * torch.logsumexp(logits / T, dim=-1)  # alto = OOD
```

### Calibração do threshold

```python
# scripts/calibrate_ood.py
# 1. Roda modelo em val set in-dist → distribuição de scores
# 2. Roda modelo em "val_ood/" (50–100 imagens não-folha: gatos, paisagens, papel, telas) → distribuição
# 3. Escolhe threshold no quantil que dá 95% TPR (true positive rate em in-dist)
# 4. Salva em artifacts/ood_threshold.json: {"msp": 0.42, "energy": -8.3}
```

### Set OOD de calibração

Criar `data/ood_eval/` com ~100 imagens variadas:
- Animais (gato, cachorro, vaca) — 20
- Paisagens — 20
- Objetos cotidianos (mesa, livro) — 20
- Outras plantas (não-soja) — 20
- Borrões/ruído — 10
- Fotos de tela/papel — 10

Fontes: ImageNet samples, Unsplash, fotos próprias.

---

## Temperature Scaling (calibração)

### Implementação — `src/inference/calibration.py`

Pós-treino, otimiza único escalar `T > 0` que divide logits antes do softmax. Não toca em pesos.

```python
import torch
from torch.optim import LBFGS

def fit_temperature(logits_val: torch.Tensor, labels_val: torch.Tensor) -> float:
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = LBFGS([T], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_val / T, labels_val)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(T.detach())
```

Salvar em `artifacts/temperature_<model>.json`.

### Métrica de calibração — ECE

**Expected Calibration Error** (Guo et al. 2017): divide predições em M=15 bins por confiança, soma `|acc(bin) - conf(bin)|` ponderado pelo tamanho do bin.

```python
def expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        if in_bin.sum() > 0:
            acc_bin = correct[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += in_bin.float().mean() * abs(acc_bin - conf_bin)
    return float(ece)
```

---

## Fluxo de inferência produto

```
Imagem → forward → logits
     ├─ energy_score(logits) > threshold? → "unknown" + log para HITL
     └─ logits / T → softmax → top-1 class + confidence calibrada
```

---

## Verificação

```bash
# 1. Treinar normal (planos 01–03 concluídos)
python scripts/train.py --config configs/resnet50.yaml

# 2. Coletar logits + labels do val set
python scripts/dump_logits.py --split val --output artifacts/val_logits.pt

# 3. Calibrar temperatura
python scripts/calibrate_temperature.py --logits artifacts/val_logits.pt
# Output esperado: T entre 1.0 e 3.0 (típico para CNN)

# 4. Calibrar threshold OOD
python scripts/calibrate_ood.py --ood_dir data/ood_eval
# Output: thresholds salvos em artifacts/ood_threshold.json

# 5. Reliability diagram pre/post
python scripts/plot_reliability.py --before artifacts/val_logits.pt --T <T_calibrada>
# Conferir: ECE pós ≤ 50% do ECE pré

# 6. Smoke OOD: rodar inferência em data/ood_eval/ → 95%+ devem ser rejeitadas
```

**Aceite**: ECE_pos / ECE_pre ≤ 0.5; rejeição em OOD ≥ 90% com FPR em in-dist ≤ 10%.
