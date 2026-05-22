# 01 — Dataset Foundation: PlantVillage + Kaggle + Dedup

> **Status**: TODO | **Esforço**: 25–30h | **Bloqueia**: todo o treino subsequente

## Context

Treinar nos 233 originais é tempo perdido. Stratified 70/15/15 em classes com 4–9 imagens dá val/test com 1–2 amostras → métrica vira ruído estatístico. Adicionalmente: Digipathos publica múltiplas fotos da mesma folha (leakage entre splits) e a classe `ferrugem_mancha_alvo` é contaminada (rust + target spot juntos).

Este plano resolve **escala, leakage, balanceamento e qualidade do test set** antes de qualquer treino real.

---

## 1. Diagnóstico do dataset atual

| Classe | Imagens | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|
| oidio | 77 | 53 | 12 | 12 |
| mancha_alvo | 62 | 43 | 9 | 10 |
| ferrugem_mancha_alvo | 32 | 22 | 5 | 5 |
| folha_carijo | 22 | 15 | 3 | 4 |
| septoria | 21 | 14 | 3 | 4 |
| saudavel | 9 | 6 | 1 | 2 |
| mancha_mirotecio | 6 | 4 | 1 | 1 |
| murcha_esclerocio | 4 | 2 | 1 | 1 |

**Problemas:**
- Imbalance 19:1 (oidio vs murcha_esclerocio).
- `murcha_esclerocio` e `mancha_mirotecio` matematicamente não-avaliáveis.
- Risco alto de leakage (Digipathos = múltiplas fotos por folha).
- `ferrugem_mancha_alvo` é uma classe contaminada — não é "ferrugem" nem "mancha alvo" puros.
- 100% das imagens são "in-field controlado" pela Embrapa; **zero amostras de smartphone real**.

## 2. Inventário de fontes externas

| Fonte | Acesso | Domínio | Soja? | Volume útil | Decisão |
|---|---|---|---|---|---|
| **PlantVillage** (Hughes & Salathé 2015, Mendeley v2) | livre | laboratório | só `Soybean___healthy` | ~5k healthy | **Integrar em `saudavel` (cap 200, `is_lab=true`)** |
| **Kaggle Soybean Diseased Leaf** (Karim 2022) | Kaggle API | misto | sim, várias | ~6.4k: Healthy/Mosaic/Rust/Yellow Mosaic/Caterpillar/Diabrotica | **Integrar Rust + Mosaic + Yellow Mosaic** após mapeamento c/ agrônomo |
| **PlantDoc** (Singh et al. NYU) | livre | in-the-wild | sim | ~65 (poucas) | **Fase 3**, validação OOD |
| **iNaturalist research-grade** | API | in-the-wild | sim | variável | **Fase 6 (pós-deploy HITL)** |
| **Synthetic** (SD + ControlNet/LoRA) | gerado | sintético | sim | sob demanda | **Só se classe ficar < 50 reais**, `source=synthetic`, **nunca no test** |

**Achado crítico:** PlantVillage **não tem doenças de soja**, só `Soybean___healthy`. A hipótese inicial de "ganho geral via PlantVillage" cai por terra.

## 3. Taxonomia canônica

**8 classes brutas → 5–6 ativas + 2 training-only:**

| Canônica | Origem | Ação |
|---|---|---|
| `oidio` | Digipathos (77) | mantém |
| `mancha_alvo` | Digipathos (62) + parte de `ferrugem_mancha_alvo` | re-rotular |
| `ferrugem_asiatica` | parte de `ferrugem_mancha_alvo` + Kaggle Rust | re-rotular + suplementar |
| `folha_carijo` (mosaico) | Digipathos (22) + Kaggle Mosaic/Yellow Mosaic | suplementar |
| `septoria` | Digipathos (21) | mantém |
| `saudavel` | Digipathos (9) + Kaggle Healthy + PlantVillage Healthy (cap 200, lab) | suplementar |
| `mancha_mirotecio` | Digipathos (6) | **training-only**, não no test |
| `murcha_esclerocio` | Digipathos (4) | **training-only**, não no test |
| `outro_descartar` | catch-all de externos não-mapeados | descarte |

**Mudança de escopo**: 8 classes → 5–6 ativas (avaliadas) + 2 training-only. **Precisa alinhar com orientador antes de codar.**

## 4. `configs/class_mapping.yaml` — fonte única da verdade

Mapeia rótulo de cada fonte para taxonomia canônica. Exemplo de estrutura:

```yaml
canonical_classes:
  - oidio
  - mancha_alvo
  - ferrugem_asiatica
  - folha_carijo
  - septoria
  - saudavel
  - mancha_mirotecio
  - murcha_esclerocio

sources:
  digipathos:
    "Oidio": oidio
    "Mancha Alvo": mancha_alvo
    "Septoria": septoria
    "Folha Carijo": folha_carijo
    "Saudavel": saudavel
    "Mancha de Mirotecio": mancha_mirotecio
    "Murcha Esclerocio": murcha_esclerocio
    "Ferrugem-Mancha Alvo": _MANUAL_REVIEW_  # re-rotular
  plantvillage:
    "Soybean___healthy": saudavel  # cap 200, is_lab=true
  kaggle_soybean_karim:
    "Rust": ferrugem_asiatica
    "Mosaic": folha_carijo
    "Yellow Mosaic": folha_carijo
    "Healthy": saudavel
    "Caterpillar": outro_descartar
    "Diabrotica": outro_descartar

active_classes:  # apenas estas no val/test
  - oidio
  - mancha_alvo
  - ferrugem_asiatica
  - folha_carijo
  - septoria
  - saudavel

training_only_classes:  # entram no train mas não no val/test
  - mancha_mirotecio
  - murcha_esclerocio
```

## 5. Pipeline de ingestão — 5 scripts

Criar em `tcc-ze-praga-model-playground/scripts/ingest/`:

### `01_download.py`
- Baixa cada fonte para `data/raw/<source>/`.
- Digipathos: já temos local em `datasets/soja/` → cópia/symlink.
- PlantVillage: download via `requests` do Mendeley (URL: https://data.mendeley.com/datasets/tywbtsjrjv/1).
- Kaggle Karim: `kagglehub.dataset_download("sagarghimire/soybean-diseased-leaf-dataset")` (verificar slug).
- Gera `manifest_raw.parquet` com 1 linha por imagem.

### `02_unify.py`
- Lê `class_mapping.yaml`.
- Symlinka (não copia) para `data/interim/unified/<canonical>/<source>__<id>.<ext>`.
- Calcula: `sha1` (conteúdo), `phash`, `dhash`, `whash` via `imagehash`.
- Detecta `is_lab` heurística (PlantVillage = lab, resto = field; configurável por fonte).
- Gera `manifest_unified.parquet`.

### `03_dedup.py`
- Voting de 3 hashes: dois pares são "iguais" se Hamming distance ≤ 6 em **pHash E dHash E wHash** (AND, mais conservador).
- Constrói grafo via `networkx`, `connected_components` → atribui `group_id`.
- Passada **cross-class**: imagens muito similares mas em classes diferentes → reporta em `cross_class_collisions.csv` para revisão humana (não remove automaticamente).
- Output: `manifest_dedup.parquet` com `group_id` populado.

### `04_split.py`
Substitui `src/data/splits.py`:
- `StratifiedGroupKFold(n_splits=7)` para 86/7/7 ou ajustar para alvo ~70/15/15 via combinação de folds.
- Garante: `set(train.group_id) ∩ set(val.group_id) == ∅` e idem para test.
- Filtro **rigoroso**: `is_lab=true` proibido em val e test (apenas train).
- Filtro: `training_only_classes` (mirotecio, esclerocio) excluídas de val/test.
- Gera `data/processed/{train,val,test}.csv` + `label_map.csv`.

### `05_audit.py`
- Verifica os critérios C1–C9 (ver §7).
- Gera `data/processed/audit_report.md` + `sanity_grid.png` (16 imagens random por classe).
- Falha com exit 1 se qualquer C# não passar (uso em CI futuro).

## 6. Schema do manifest Parquet

Uma linha por imagem, evolui ao longo dos scripts:

| Coluna | Tipo | Origem | Descrição |
|---|---|---|---|
| `image_id` | str | sha1 | ID único (sha1 do conteúdo) |
| `source` | str | 01 | "digipathos", "plantvillage", "kaggle_soybean_karim", "synthetic" |
| `source_class` | str | 01 | rótulo bruto da fonte |
| `canonical_class` | str | 02 | rótulo canônico pós-mapping |
| `filepath_raw` | str | 01 | caminho original |
| `filepath_unified` | str | 02 | symlink em `data/interim/unified/` |
| `width`, `height` | int | 02 | dimensões |
| `phash`, `dhash`, `whash` | str | 02 | hashes perceptuais (hex 16 chars) |
| `group_id` | int | 03 | grupo de duplicatas |
| `split` | str | 04 | "train", "val", "test" ou "excluded" |
| `is_lab` | bool | 02 | true = laboratório (PlantVillage); false = field |
| `license` | str | 01 | CC-BY-4.0, MIT, etc. |
| `download_ts` | datetime | 01 | timestamp de download |
| `notes` | str | manual | comentários (re-rotulagem, exclusões) |

## 7. Critérios de aceite C1–C9

Mensuráveis, com threshold definido:

| # | Critério | Threshold | Justificativa |
|---|---|---|---|
| C1 | classes ativas com ≥ 50 grupos | ≥ 5 das 6 | viabilidade estatística |
| C2 | val e test cada classe ativa | ≥ 8 grupos | mínimo p/ F1 estável |
| C3 | grupos atômicos | `train ∩ val ∩ test == ∅` | zero leakage |
| C4 | dedup efetiva | `N_pre > N_post` se há similares | dedup funcionou |
| C5 | test sem lab | `%(is_lab=true) == 0%` no test | métrica de uso real |
| C6 | integridade do manifest | sha1 confere com conteúdo | sem corrupção |
| C7 | cross-class collisions revisadas | `human_reviewed=true` em todas | sem rótulos conflitantes |
| C8 | balanceamento train | `max_class / min_class ≤ 30` | viável c/ class_weights |
| C9 | sanity visual | grid revisado por humano | rótulos batem visualmente |

## 8. Tratamento de `ferrugem_mancha_alvo` (32 imagens)

**Processo manual** (~1–2h, fora de código):
1. Abrir as 32 imagens.
2. Re-rotular cada uma em 3 buckets:
   - `ferrugem_asiatica` (pústulas marrons, lado abaxial)
   - `mancha_alvo` (lesões circulares concêntricas tipo alvo)
   - `multilabel` (genuinamente ambos — esperado ~10 imagens)
3. Bucket `multilabel` vai para `data/raw/digipathos_multilabel/`:
   - Excluído de single-label train.
   - Usado como **set sentinela** no relatório: rodar inferência single-label, mostrar top-2 logits, demonstrar que o modelo "vê" ambas as doenças. **Achado interessante para o TCC.**
4. Salvar planilha de re-rotulagem em `data/raw/digipathos_relabel.csv` com colunas: `original_filename`, `new_class`, `confidence` (0/1/2), `notes`.

## 9. Dependências novas

Adicionar em `requirements.txt`:

```
imagehash==4.3.1
networkx==3.3
pyarrow==16.1.0
kagglehub==0.2.5
```

## 10. Decisões e trade-offs registrados

| # | Decisão | Trade-off |
|---|---|---|
| D1 | PlantVillage **só** para `saudavel`, capped em 200 | Perde volume mas evita drift lab→field na classe healthy |
| D2 | Kaggle Karim como fonte principal de Rust/Mosaic | Necessita validação agrônomo (rótulos podem divergir) |
| D3 | `is_lab=true` proibido em val/test | Reduz N de avaliação mas dá métrica honesta |
| D4 | Voting AND de 3 hashes (não OR) | Menos falsos positivos de dedup; pode passar duplicatas borderline |
| D5 | `StratifiedGroupKFold` em vez de simples split | Mais complexo mas única forma de honrar grupos |
| D6 | Sintética só se `< 50` reais | Sintética enviesa; usar como último recurso |
| D7 | Mirotecio/Esclerocio training-only | Mantém info no train, evita avaliar em N=4-6 |
| D8 | Manifest Parquet (não CSV) | Tipagem, performance, mais complexo de inspecionar |

## 11. Sequenciamento de tarefas

| # | Tarefa | Esforço | Bloqueia | Paraleliz. |
|---|---|---|---|---|
| 1 | Criar `class_mapping.yaml` | 1h | tudo | — |
| 2 | **[HUMANO]** Alinhar com orientador escopo 8→5+2 | — | tudo | — |
| 3 | **[HUMANO]** Re-rotular 32 ferrugem_mancha_alvo | 1.5h | #5, #6 | // #4 |
| 4 | Implementar `01_download.py` | 3h | #5–#8 | // #3 |
| 5 | Implementar `02_unify.py` | 4h | #6 | — |
| 6 | Implementar `03_dedup.py` | 4h | #7 | — |
| 7 | Implementar `04_split.py` | 4h | #8 | — |
| 8 | Implementar `05_audit.py` | 3h | #9 | — |
| 9 | Rodar pipeline end-to-end + iterar nos audits | 3h | #10 | — |
| 10 | Smoke train (1 época ResNet-50) | 1h | — | — |
| 11 | Atualizar `docs/plan-pipeline.md` (§4, §13) | 1h | — | // #5–#9 |
| 12 | Criar `docs/datasets-inventory.md` | 1h | — | // #5–#9 |
| 13 | Adicionar deps em `requirements.txt` | 0.5h | #4 | — |

**Caminho crítico:** #2 → #1 → #4 → #5 → #6 → #7 → #8 → #9 → #10 (~25h sem contar humanos).

## 12. Mudanças em arquivos existentes

| Arquivo | Mudança |
|---|---|
| `src/data/splits.py` | Deprecar; manter shim que delega para `scripts/ingest/04_split.py` |
| `src/data/download.py` | Substituir `NotImplementedError` por chamada a `scripts/ingest/01_download.py` |
| `configs/base.yaml` | `num_classes: 8` → `num_classes: 6` (provável, após decisão D2) + comentário |
| `docs/plan-pipeline.md` | Reescrever §4 (dataset) e §13 (critérios de sucesso) refletindo escopo real |
| `requirements.txt` | Adicionar 4 deps de §9 |
| `docs/datasets-inventory.md` | **Novo** — inventário formal com licenças, contagens, mapeamentos |

## Verification

End-to-end:

```bash
cd tcc-ze-praga-model-playground

# 1. Download
python scripts/ingest/01_download.py --sources digipathos,plantvillage,kaggle_soybean_karim

# 2. Unify
python scripts/ingest/02_unify.py

# 3. Dedup
python scripts/ingest/03_dedup.py
# Abrir cross_class_collisions.csv, marcar human_reviewed=true em cada linha

# 4. Split
python scripts/ingest/04_split.py

# 5. Audit
python scripts/ingest/05_audit.py
# Conferir audit_report.md: todos os critérios C1–C9 com ✅

# 6. Sanity visual
open data/processed/sanity_grid.png
# Verificar: rótulos batem com imagens; sem fotos óbvias do mesmo objeto em splits diferentes

# 7. Smoke train
python scripts/train.py --config configs/resnet50.yaml --data_dir data/processed
# Esperar: 1 época sem erro, val_f1_macro > random (1/6 = 0.167)
```

**Critério final**: audit_report.md verde + smoke train passa = plano 01 concluído. Liberado para planos 02+.
