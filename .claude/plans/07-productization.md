# 07 — Produtização

> **Status**: TODO | **Esforço**: 30h+ | **Bloqueia**: pós-modelo (não bloqueia TCC)

## Context

Hoje o repo é um playground acadêmico bem estruturado. Para virar produto (app na mão de agricultor), faltam: serving, observabilidade, feedback loop, retreino, compliance e CI/CD.

Este plano lista as camadas necessárias. Cada camada vira plano detalhado próprio quando chegar a fase de implementar.

---

## Camadas

### Tracking de experimentos

- **Hoje**: TensorBoard local, sem registry.
- **Alvo**: **MLflow** com artifact store em S3.
- Registrar: hiperparâmetros (do YAML), métricas finais, modelo `.pth` e `.onnx`, dataset version (hash do `manifest_unified.parquet`).

### Model Registry

- **MLflow Model Registry** com stages: `Staging` → `Production` → `Archived`.
- Promoção manual via UI; produção sempre via API consultando `client.get_latest_versions(name, stages=["Production"])`.

### Serving

Opções:

| Opção | Latência | Custo | Cold start | Veredito |
|---|---|---|---|---|
| **AWS Lambda Container (ONNX Runtime)** | 200–800ms | $ | 2–5s | **default p/ v1** |
| **ECS Fargate** | <200ms | $$ | sempre quente | se latência < 500ms exigida |
| **SageMaker Endpoint** | <200ms | $$$ | sempre quente | overkill p/ MVP |

Lambda Container Image é o caminho — usa o ONNX já exportado, runtime leve. Plano original já considera (mencionado em §10 do `plan-pipeline.md`).

### Coleta de dados em produção

- App envia: foto + (opt-in) geo + (obrigatório) consentimento LGPD.
- **S3**: `s3://zepraga-prod-images/yyyy/mm/dd/<uuid>.jpg`.
- **DynamoDB** `predictions`: PK=`prediction_id`, attrs: `user_id_hash`, `model_version`, `class_predicted`, `confidence_calibrated`, `ood_flag`, `timestamp`, `image_s3_uri`, `feedback` (preenchido depois).
- Anonimização: stripar EXIF (GPS, dispositivo) na ingestão.

### Rotulagem (HITL — Human-in-the-Loop)

- **Label Studio open-source**, deploy próprio.
- Workflow: imagens com `ood_flag=true` OU `confidence < 0.7` vão para fila de rotulagem.
- Agrônomos parceiros rotulam → label vira ground-truth.
- Threshold de N novas labels para gatilhar retreino (ex.: 200 novas por classe).

### Treino on-demand

Opções:
- **Modal.com**: GPU H100 sob demanda, Python-native, ótimo p/ MVP.
- **AWS SageMaker Training**: mais integrado, mais burocrático.
- **Colab Pro**: barato, limitado a sessões manuais.

Para CI de retreino: **Modal + Step Functions** (gateway).

### Observabilidade

- **CloudWatch**:
  - Métricas: latência p50/p95/p99, taxa de OOD, taxa de erro.
  - Logs: input shape, model version, predicted class, confidence.
- **Drift detection** custom:
  - Embeddings (penúltima camada) das últimas N imagens.
  - KS test ou MMD vs. baseline (val set fixo no momento do deploy).
  - Alarme se drift > threshold (semanal).
- **Grafana** opcional para dashboard de produto.

### Retraining loop

```
[N novas labels acumuladas] → trigger Step Functions
   ↓
1. Snapshot do dataset atual + novas labels
2. Re-rodar pipeline ingest (planos 01)
3. Treinar 3 modelos (atualizar artifacts)
4. Avaliar em test set HOLDOUT (não mudou!)
5. Comparar F1: novo > antigo + margem 0.5%?
   - Sim → push to Staging, alertar humano
   - Não → log + abortar
6. [HUMANO] revisa em Staging, promove para Production
7. App refresca config (model version cached, TTL 1h)
```

### LGPD

- Consentimento explícito no app (modal antes de tirar foto).
- Retenção: imagens originais por 90 dias (anonimizadas); embeddings por 2 anos.
- Direito de deleção: endpoint que apaga `predictions` por `user_id_hash`.
- DPA com hosting (AWS) e parceiros.

### CI/CD

GitHub Actions, 3 jobs:

1. **lint**: `ruff check` + `ruff format --check`.
2. **test**: `pytest` (smoke tests em data sintética pequena para não baixar dataset real no CI).
3. **onnx_validation**: roda `src/export/validate_onnx.py` em modelo dummy (paridade torch vs onnxruntime) — falha se `np.allclose(rtol=1e-3)` falhar.

Bonus: workflow scheduled (cron weekly) que checa drift e abre Issue se detectar.

### Threshold de "não sei"

App **sempre** mostra disclaimer + confiança. Quando:
- `ood_flag=true` → "Não consegui identificar — confira foto/iluminação".
- `confidence_calibrated < 0.65` → "Possível: <classe> (não tenho certeza). Considere consultar agrônomo."
- `confidence_calibrated ≥ 0.65` → predição + link "saiba mais sobre <classe>".

Threshold 0.65 = palpite inicial; refinar empiricamente com agrônomos.

---

## Stack resumo

```
┌─────────────────────────────────────────────────────────────────┐
│                            App (Mobile)                          │
│  câmera → foto → preview → submit (com consentimento)            │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              API Gateway → Lambda Container                      │
│  segment → classify (ensemble) → ood check → calibrate → return  │
└─────────────────────────────────────────────────────────────────┘
            │                                       │
            ▼                                       ▼
   ┌───────────────────┐                  ┌──────────────────┐
   │ S3 (imgs raw)     │                  │ DynamoDB (preds) │
   └───────────────────┘                  └──────────────────┘
            │                                       │
            └────────── HITL fila ──────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │  Label Studio        │
                  │  (agrônomos parceiros)│
                  └──────────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │  Retraining gate     │
                  │  (Step Functions)    │
                  └──────────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │  Modal (treino) →    │
                  │  MLflow → Lambda     │
                  └──────────────────────┘
```

---

## Verificação por camada

Por enquanto este plano é doc viva. Quando virar implementação, cada camada terá seu próprio sub-plano com critério de aceite.

Verificação macro para "produto v1": app + Lambda + MLflow + CI minimal rodando end-to-end com 1 modelo, 1 endpoint, latência p95 < 1s, LGPD OK.
