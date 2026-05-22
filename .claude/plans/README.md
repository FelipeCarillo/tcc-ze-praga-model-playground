# Planos — TCC Zé Praga

Planos de evolução do projeto de classificação de doenças foliares de soja. Cada arquivo é um plano executável; alguns são detalhados (01) e outros são lista de tarefas (02–03). Os planos 04–07 viram detalhados quando chegar a fase deles.

## Índice

| # | Plano | Status | Bloqueia? | Esforço |
|---|---|---|---|---|
| 01 | [Dataset foundation — PlantVillage + Kaggle + dedup](01-dataset-foundation.md) | TODO | bloqueia todo treino | 25–30h |
| 02 | [Fix pipeline bugs (deprecated APIs, LRs hardcoded)](02-fix-pipeline-bugs.md) | TODO | não | 3h |
| 03 | [Close eval loop (trainer chama evaluate ao final)](03-close-eval-loop.md) | TODO | parcial | 2h |
| 04 | [OOD detection + calibração (temperature scaling)](04-ood-and-calibration.md) | TODO | crítico para produto | 6h |
| 05 | [Leaf segmentation (SAM ou U-Net para fundo)](05-leaf-segmentation.md) | TODO | melhora robustez | 8h |
| 06 | [TTA + ensemble dos 3 modelos](06-tta-and-ensemble.md) | TODO | ganho fácil | 4h |
| 07 | [Produtização (MLflow + registry + Lambda + monitoring)](07-productization.md) | TODO | pós-modelo | 30h+ |
| 08 | [Roadmap em fases (0 → 3)](08-roadmap-phases.md) | doc viva | — | — |

## Ordem recomendada

```
01 (dataset) → 02 (bugs) → 03 (eval loop) → [04, 05, 06 em paralelo] → 07
                                            └─ 08 é doc viva, atualizar conforme evolui
```

## Princípios

- **Nada de treinar nos 233 originais** — métrica é ruído. Plano 01 destrava tudo.
- **Test set proibido de conter imagens de laboratório** — métrica deve refletir uso real do app.
- **Sintética é apoio, não substituto** — só usar com `source=synthetic` + nunca no test.
- **Cada plano tem critério de aceite mensurável** — sem "fica melhor", só thresholds.
- **Mudanças de escopo (ex.: 8 → 5 classes ativas) passam por orientador antes de codar.**

## Contexto rápido do projeto

- Repo do modelo: `tcc-ze-praga-model-playground/` (clonado do GitHub do Felipe Carillo).
- Dataset bruto extraído em: `datasets/soja/<classe>/*.jpg` — 233 imagens, 8 classes, desbalanceado 4–77.
- Pipeline alvo: 3 modelos (ResNet-50, EfficientNet-B4, ViT-B/16) → ONNX → Lambda.
- KPI do plan-pipeline original: F1-macro ≥ 0.85, recall ferrugem ≥ 0.90 — **impossível de medir com dataset atual**.
