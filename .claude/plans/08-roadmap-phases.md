# 08 — Roadmap em Fases

> **Status**: doc viva | atualizar conforme o projeto evolui

## Context

Visão temporal do projeto. Cada fase tem objetivos, planos vinculados e critério de saída. Documento de referência para alinhamento com orientador e tomada de decisão.

---

## Fase 0 — Corrigir Base

**Janela**: 1–2 semanas
**Objetivo**: ter pipeline que treina em dados honestos, com métricas confiáveis e bugs corrigidos.

**Planos vinculados**:
- [01 — Dataset foundation](01-dataset-foundation.md) (crítico, gargalo)
- [02 — Fix pipeline bugs](02-fix-pipeline-bugs.md)
- [03 — Close eval loop](03-close-eval-loop.md)

**Critério de saída**:
- ✅ Audit do dataset C1–C9 todos verdes
- ✅ Smoke train ResNet-50 roda sem warnings
- ✅ `artifacts/metrics/metrics_resnet50.json` é gerado automaticamente ao final do train
- ✅ Mudança de escopo (8 → 5–6 classes ativas) aprovada pelo orientador

---

## Fase 1 — TCC Defensável

**Janela**: 1 mês após Fase 0
**Objetivo**: ter os 3 modelos treinados, comparação honesta, ONNX exportados, PoC de Lambda.

**Atividades**:
- Treinar ResNet-50, EfficientNet-B4, ViT-B/16 em dataset Fase 0.
- K-fold cross-validation (em vez de split único) — opcional se N ainda baixo.
- [Plano 04 — OOD + calibração](04-ood-and-calibration.md).
- [Plano 06 — TTA + ensemble](06-tta-and-ensemble.md).
- Exportar ONNX, validar paridade.
- PoC Lambda: 1 endpoint funcional com 1 modelo.

**Critério de saída**:
- ✅ Tabela comparativa dos 3 modelos (F1, precision, recall por classe, latência CPU/GPU, tamanho).
- ✅ ECE reduzido em ≥ 50% pós calibração.
- ✅ OOD rejection ≥ 90% em set sintético.
- ✅ Lambda PoC respondendo < 1s p95.
- ✅ Defesa de TCC1 e/ou TCC2 viável.

**Risco**: dataset Fase 0 pode ainda ser insuficiente para bater KPIs originais (F1 ≥ 0.85). **Decisão honesta**: ajustar KPIs no relatório ou rodar Fase 2 antes de defender.

---

## Fase 2 — MVP

**Janela**: 3–6 meses
**Objetivo**: app na mão de ~10 agricultores parceiros, coletando dados reais e ciclando feedback.

**Atividades**:
- App mobile (React Native ou Flutter — decisão fora deste repo).
- Backend FastAPI/Lambda com:
  - [Plano 05 — Segmentação](05-leaf-segmentation.md) integrada.
  - [Plano 04 — OOD + calibração](04-ood-and-calibration.md).
  - [Plano 06 — TTA + ensemble](06-tta-and-ensemble.md).
- [Plano 07 — Produtização](07-productization.md) — camadas: tracking, registry, serving, coleta, HITL.
- Parceria com 1–2 agrônomos para rotulagem.
- Coletar **1.000+ imagens em condições reais**.

**Critério de saída**:
- ✅ App em produção (TestFlight / Play Console internal).
- ✅ Pipeline HITL funcionando: foto → predict → review humano → re-label se necessário.
- ✅ Métricas de produto baseline: DAU, retenção, taxa de OOD, NPS qualitativo.
- ✅ Primeiro retreino disparado por novos dados.

---

## Fase 3 — Escala

**Janela**: 12+ meses
**Objetivo**: produto que sobrevive sem o usuário inicial, escalável a outras culturas.

**Atividades**:
- Multi-cultura: milho, algodão, feijão (replicar pipeline com novos datasets).
- Segmentação **por lesão** (não só por folha) — instance segmentation.
- Modelo edge (TFLite/ONNX-mobile) para uso offline.
- Drift monitoring automatizado, retreino agendado.
- Compliance LGPD revisada com advogado.
- Estudo de impacto agronômico (parceria acadêmica para medir outcome real).

**Critério de saída**:
- ✅ ≥ 3 culturas suportadas.
- ✅ Modo offline funcional em smartphone modesto.
- ✅ Métrica agronômica publicável (paper / relatório técnico).

---

## Decisões macro pendentes

| # | Decisão | Quem | Prazo |
|---|---|---|---|
| 1 | Mudança 8 → 5–6 classes ativas | orientador | antes da Fase 0 |
| 2 | Dataset complementar definitivo (Kaggle Karim?) | orientador + agrônomo | Fase 0 |
| 3 | Stack mobile (RN vs Flutter) | grupo | Fase 2 |
| 4 | Hosting (AWS vs GCP) | grupo + financeiro | Fase 2 |
| 5 | Parceria agronômica formal | coordenação | Fase 2 |

---

## Princípios duradouros

1. **Métrica não-honesta é pior que sem métrica.** Test set sempre sem laboratório, sem leakage, sem sintético.
2. **Dataset > modelo.** Antes de tunar hiperparâmetro, perguntar: "tenho dados suficientes para detectar essa diferença?".
3. **Produto > publicação.** TCC ganha mais com 1 app de verdade na mão de agricultor do que com benchmark sintético inflado.
4. **Documentar trade-offs.** Toda decisão registrada com "por que sim" e "por que não".
5. **HITL desde cedo.** Modelo só fica bom em produção se humano corrigir erros sistematicamente.
