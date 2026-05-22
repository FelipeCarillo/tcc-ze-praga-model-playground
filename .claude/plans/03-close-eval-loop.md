# 03 — Close Eval Loop

> **Status**: TODO | **Esforço**: 2h | **Bloqueia**: parcial (sem isso, métricas finais ficam em arquivo separado)

## Context

O `plan-pipeline.md` §8 diz "Ao final: carrega melhor checkpoint, avalia no test set, salva `metrics_<modelo>.json`". Isso **não está implementado** — `trainer.fit()` termina após restaurar best weights, sem chamar test eval. O usuário tem que rodar `scripts/evaluate.py` separadamente.

Adicionalmente: `evaluator.evaluate()` exige `label_names: list[str]` mas ninguém lê `label_map.csv` automaticamente.

---

## Mudanças

### M1 — `trainer.fit()` retorna ou chama avaliação no test

Opção escolhida: **callback opcional em `fit()`**, mantém o trainer agnóstico de test loader.

```python
# src/training/trainer.py — assinatura nova
def fit(
    self,
    epochs_total: int = 30,
    epochs_warmup: int = 3,
    patience: int = 7,
    freeze_fn: Callable | None = None,
    unfreeze_fn: Callable | None = None,
    test_callback: Callable[[nn.Module], None] | None = None,  # NOVO
) -> nn.Module:
    # ... loop existente ...
    self.writer.close()
    checkpoint_cb.restore_best(self.model)
    if test_callback is not None:
        test_callback(self.model)
    return self.model
```

### M2 — `scripts/train.py` passa um test_callback

```python
# após o fit, antes de exit
from src.evaluation.evaluator import evaluate, save_metrics
import pandas as pd

def _test_eval(trained_model: nn.Module) -> None:
    test_ds = SoybeanLeafDataset(args.data_dir / "test.csv", transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=cfg["num_workers"], pin_memory=True)
    label_map = pd.read_csv(args.data_dir / "label_map.csv").sort_values("label_idx")
    label_names = label_map["label"].tolist()

    results = evaluate(trained_model, test_loader, label_names, device=device)
    save_metrics(
        results,
        save_path=Path("artifacts/metrics") / f"metrics_{model_name}.json",
        model_name=model_name,
    )

trainer.fit(..., test_callback=_test_eval)
```

### M3 — `evaluator.evaluate()` recebe label_names mas `scripts/evaluate.py` standalone também deve ler `label_map.csv`

Atualizar `scripts/evaluate.py` (se existir) para carregar `label_map.csv` automaticamente em vez de exigir argumento. Estado atual desconhecido — verificar antes de mexer.

---

## Verificação

```bash
cd tcc-ze-praga-model-playground

# 1. Smoke train completo
python scripts/train.py --config configs/resnet50.yaml --data_dir data/processed

# 2. Verificar que metrics_resnet50.json existe e tem per_class populado
test -f artifacts/metrics/metrics_resnet50.json && echo "OK arquivo existe"
python -c "import json; d = json.load(open('artifacts/metrics/metrics_resnet50.json')); assert 'per_class' in d and len(d['per_class']) > 0, 'per_class vazio'; print('OK per_class:', list(d['per_class'].keys())[:3])"

# 3. Verificar que scripts/evaluate.py standalone ainda funciona
python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint artifacts/checkpoints/best_resnet50.pth
```

**Aceite**: ao rodar `train.py`, ao final é gerado `artifacts/metrics/metrics_<model>.json` com `accuracy`, `f1_macro`, `f1_weighted`, `precision_macro`, `recall_macro` e `per_class` (dict com 1 entrada por classe contendo precision/recall/f1).
