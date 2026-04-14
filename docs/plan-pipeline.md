# Pipeline de Treinamento — Zé Praga

> Especificação técnica do pipeline de treinamento, avaliação e exportação dos modelos de classificação de doenças da soja.
> **Destino:** prompt para Claude Code gerar a estrutura de notebooks Jupyter + scripts Python para execução em Google Colab.
> **Projeto:** TCC IMT — Engenharia de Computação — Grupo B (Zé Praga).

---

## 1. Contexto e Objetivo

Treinar e comparar **três arquiteturas de deep learning** para classificação multiclasse de doenças foliares da soja, exportar os modelos para **ONNX** e disponibilizar artefatos prontos para inferência via FastAPI/AWS Lambda.

**Arquiteturas comparadas:**
1. **ResNet-50** (CNN baseline, ~25M params)
2. **EfficientNet-B4** (CNN otimizada, ~19M params)
3. **ViT-B/16** (Vision Transformer, ~86M params — risco de exceder 250MB Lambda; mitigado via container images)

**Métricas de comparação:** acurácia top-1, F1-score macro/weighted, precisão e recall por classe, matriz de confusão, tempo de inferência (CPU + GPU), tamanho do modelo em disco e tamanho ONNX.

---

## 2. Stack Tecnológica

| Camada | Ferramenta |
|---|---|
| Linguagem | Python 3.11 |
| Framework DL | PyTorch 2.3 + torchvision |
| Modelos pré-treinados | `timm` (PyTorch Image Models) |
| Transforms / Augmentation | `albumentations` |
| Tracking | TensorBoard (local) + opcional Weights & Biases |
| Exportação | `torch.onnx` + `onnx` + `onnxruntime` (validação) |
| Notebooks | Jupyter / Google Colab (T4 ou A100) |
| Versionamento de dados | Google Drive montado no Colab |
| Reprodutibilidade | `seed_everything` + `requirements.txt` pinado |

---

## 3. Estrutura do Repositório

```
ze-praga-training/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml                  # hiperparâmetros comuns
│   ├── resnet50.yaml
│   ├── efficientnet_b4.yaml
│   └── vit_b16.yaml
├── notebooks/
│   ├── 00_setup_colab.ipynb       # mount drive, instala deps, baixa dataset
│   ├── 01_eda_dataset.ipynb       # análise exploratória, balanceamento, exemplos
│   ├── 02_train_resnet50.ipynb
│   ├── 03_train_efficientnet_b4.ipynb
│   ├── 04_train_vit_b16.ipynb
│   ├── 05_evaluation_compare.ipynb # consolida métricas das 3 arquiteturas
│   └── 06_export_onnx.ipynb       # exporta os 3 modelos + valida via onnxruntime
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py            # baixa Digipathos / PlantVillage
│   │   ├── dataset.py             # SoybeanLeafDataset (torch.utils.data.Dataset)
│   │   ├── transforms.py          # pipelines de augmentation por split
│   │   └── splits.py              # gera train/val/test estratificados (70/15/15)
│   ├── models/
│   │   ├── factory.py             # build_model(name, num_classes, pretrained=True)
│   │   ├── resnet.py
│   │   ├── efficientnet.py
│   │   └── vit.py
│   ├── training/
│   │   ├── trainer.py             # loop de treino com early stopping
│   │   ├── losses.py              # CE + label smoothing + pesos por classe
│   │   ├── optim.py               # AdamW + scheduler cosine c/ warmup
│   │   ├── metrics.py             # acc, f1, precision, recall por classe
│   │   └── callbacks.py           # checkpoint, early stopping, log
│   ├── evaluation/
│   │   ├── evaluator.py           # avaliação no test set
│   │   ├── confusion.py           # plota e salva matriz de confusão
│   │   └── benchmark.py           # mede latência CPU/GPU + tamanho
│   ├── export/
│   │   ├── to_onnx.py             # exporta modelo PyTorch -> ONNX
│   │   └── validate_onnx.py       # confere paridade numérica torch vs onnxruntime
│   └── utils/
│       ├── seed.py                # set_seed(seed)
│       ├── config.py              # carrega YAML
│       └── logger.py
├── scripts/
│   ├── train.py                   # CLI: python scripts/train.py --config configs/resnet50.yaml
│   ├── evaluate.py
│   └── export.py
└── artifacts/                     # gitignore
    ├── checkpoints/
    ├── onnx/
    ├── metrics/
    └── figures/
```

---

## 4. Dataset

### 4.1 Fonte primária

**Digipathos (Embrapa)** — ~3.000 imagens de doenças em soja capturadas em campo no Brasil, 29 classes (incluindo saudável). URL base: `https://www.digipathos-rep.cnptia.embrapa.br`.

> ⚠️ Reavaliar inclusão complementar do **PlantVillage** apenas para a classe `healthy` se o Digipathos apresentar desbalanceamento crítico em folhas saudáveis. Decisão registrada em `configs/base.yaml` via flag `use_plantvillage_healthy: bool`.

### 4.2 Organização em disco

```
data/raw/digipathos/
  ├── ferrugem_asiatica/
  ├── oidio/
  ├── antracnose/
  ├── crestamento_bacteriano/
  ├── cercosporiose/
  ├── mancha_alvo/
  ├── ...
  └── saudavel/

data/processed/
  ├── train.csv     # colunas: filepath, label, label_idx
  ├── val.csv
  └── test.csv
```

### 4.3 Split

- Estratificado por classe: **70% train / 15% val / 15% test**
- `random_state = 42` fixo
- Gerado uma única vez por `src/data/splits.py` e salvo em CSV para garantir reprodutibilidade entre as 3 arquiteturas

---

## 5. Pré-processamento e Data Augmentation

### 5.1 Resoluções de entrada por modelo

| Modelo | Input size |
|---|---|
| ResNet-50 | 224×224 |
| EfficientNet-B4 | 380×380 |
| ViT-B/16 | 224×224 |

### 5.2 Pipeline de augmentation (treino) — `albumentations`

```python
A.Compose([
    A.RandomResizedCrop(size, size, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
    A.HueSaturationValue(10, 15, 10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

### 5.3 Pipeline val/test (determinístico)

```python
A.Compose([
    A.Resize(size, size),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

---

## 6. Modelos

Todos via `timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)` para padronizar.

| Config key | timm name |
|---|---|
| `resnet50` | `resnet50` |
| `efficientnet_b4` | `tf_efficientnet_b4_ns` |
| `vit_b16` | `vit_base_patch16_224` |

**Estratégia de fine-tuning (todas as arquiteturas):**
- **Fase 1 (3 épocas):** congela backbone, treina apenas o classifier head com `lr = 1e-3`
- **Fase 2 (até 30 épocas):** descongela tudo, treina end-to-end com `lr = 3e-5` (backbone) e `lr = 3e-4` (head) via parameter groups

---

## 7. Configuração de Treinamento

### 7.1 `configs/base.yaml`

```yaml
seed: 42
num_classes: 29  # confirmar após EDA do Digipathos
batch_size: 32
num_workers: 4
epochs_warmup: 3
epochs_total: 30
patience_early_stop: 7

optimizer:
  name: adamw
  weight_decay: 1.0e-4

scheduler:
  name: cosine_with_warmup
  warmup_steps_ratio: 0.1

loss:
  name: cross_entropy
  label_smoothing: 0.1
  class_weights: balanced  # calculado a partir do train.csv

mixed_precision: true       # torch.cuda.amp
gradient_clip_norm: 1.0

logging:
  tensorboard_dir: artifacts/tensorboard
  log_every_n_steps: 20

checkpoint:
  monitor: val_f1_macro
  mode: max
  save_top_k: 1
```

### 7.2 Overrides por modelo

`configs/vit_b16.yaml`:
```yaml
batch_size: 16          # ViT consome mais VRAM
input_size: 224
optimizer:
  weight_decay: 0.05    # recomendado para ViT
```

---

## 8. Loop de Treinamento (`src/training/trainer.py`)

Responsabilidades:

1. Inicializar modelo, otimizador, scheduler, scaler (AMP), loss
2. Para cada época:
   - **Train step:** forward, loss, backward, clip, optimizer step, scheduler step
   - **Val step:** forward sem grad, agrega predições, calcula métricas
   - Loga em TensorBoard: `train/loss`, `val/loss`, `val/acc`, `val/f1_macro`, `lr`
   - Salva checkpoint se `val_f1_macro` melhorou
   - Early stopping após `patience` épocas sem melhora
3. Ao final: carrega melhor checkpoint, avalia no test set, salva `metrics_<modelo>.json`

---

## 9. Avaliação e Comparação

`notebooks/05_evaluation_compare.ipynb` consolida:

- Tabela com acurácia, F1 macro, F1 weighted, precisão e recall **por classe** para os 3 modelos
- Matriz de confusão normalizada (heatmap) para cada modelo
- Curva PR por classe (matplotlib)
- **Benchmark de latência:** mede tempo de inferência em CPU e GPU para batches de 1, 8, 32 (média de 100 runs após warmup)
- **Tamanho:** PyTorch state_dict (.pth) e ONNX (.onnx)
- Recomendação final: qual modelo seguir para produção (considerando trade-off acurácia vs latência vs tamanho para AWS Lambda)

Output: `artifacts/metrics/comparison_report.md` + `artifacts/figures/*.png`.

---

## 10. Exportação para ONNX

`notebooks/06_export_onnx.ipynb` + `src/export/to_onnx.py`:

```python
torch.onnx.export(
    model.eval(),
    dummy_input,                       # shape (1, 3, H, W)
    f"artifacts/onnx/{model_name}.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)
```

**Validação de paridade:** `validate_onnx.py` roda 50 imagens do test set em PyTorch e em `onnxruntime`, compara via `np.allclose(rtol=1e-3, atol=1e-5)` e falha o build se divergir.

---

## 11. Setup do Colab (`notebooks/00_setup_colab.ipynb`)

Sequência fixa de células:

1. **Verificar GPU:** `!nvidia-smi`
2. **Montar Drive:** `from google.colab import drive; drive.mount('/content/drive')`
3. **Clonar repo:** `!git clone <repo_url> /content/ze-praga-training`
4. **Instalar deps:** `!pip install -q -r /content/ze-praga-training/requirements.txt`
5. **Baixar dataset:** `!python /content/ze-praga-training/src/data/download.py --target /content/data/raw`
6. **Gerar splits:** `!python /content/ze-praga-training/src/data/splits.py`
7. **Sanity check:** carrega 1 batch, plota imagens com labels

> Os notebooks de treino (02–04) começam executando `00_setup_colab.ipynb` via `%run` para não duplicar setup.

---

## 12. Reprodutibilidade

- `set_seed(42)` em **todo** notebook (afeta `random`, `numpy`, `torch`, `torch.cuda`, `torch.backends.cudnn.deterministic = True`)
- Splits salvos em CSV versionado
- `requirements.txt` com versões pinadas (`torch==2.3.1`, `timm==1.0.7`, `albumentations==1.4.10`, ...)
- Configs YAML salvas junto a cada checkpoint
- Hash do commit logado no início de cada run

---

## 13. Critérios de Sucesso (alvos para TCC 1)

| Métrica | Alvo mínimo |
|---|---|
| Acurácia top-1 (test) | ≥ 90% no melhor modelo |
| F1 macro (test) | ≥ 0.85 |
| Recall em ferrugem-asiática | ≥ 0.90 (alta relevância epidemiológica) |
| Latência inferência CPU (batch=1) | ≤ 500 ms para o modelo escolhido para produção |
| Tamanho ONNX do modelo escolhido | ≤ 250 MB (ou plano de container Lambda documentado) |

---

## 14. Próximas Etapas (fora do escopo deste pipeline)

- Pipeline de fine-tuning incremental com fotos enviadas pelos usuários (Human-in-the-Loop) — documentado no `relatorio_dataset_ze_praga.docx`
- Integração do `.onnx` do modelo escolhido com a API FastAPI (módulo `inference/` do backend)
- Deploy serverless via Lambda container image

---

## Instruções para o Claude Code

Ao receber este documento:

1. Crie a estrutura completa do repositório descrita na **Seção 3**.
2. Implemente todos os módulos em `src/` com docstrings e type hints.
3. Crie os notebooks listados em `notebooks/` com células markdown explicativas em **português brasileiro** entre as células de código.
4. Preencha o `requirements.txt` com versões pinadas compatíveis com Colab (CUDA 12.x).
5. Use `pathlib.Path` em vez de strings para paths.
6. Siga PEP 8, formate com `black` (line length 100).
7. Não implemente o download real do Digipathos sem confirmar a URL final — deixe um TODO claro em `src/data/download.py`.
8. Entregue um `README.md` na raiz com instruções de execução passo a passo no Colab.