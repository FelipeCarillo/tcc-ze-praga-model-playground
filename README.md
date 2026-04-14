# Zé Praga — Pipeline de Treinamento

Repositório de experimentação e treinamento dos modelos de classificação de doenças foliares de soja do projeto **Zé Praga** (TCC). O pipeline cobre desde o download do dataset Digipathos até a exportação dos modelos em ONNX para servir no backend FastAPI.

---

## Modelos

| Modelo | Config | Backbone timm | Input |
|--------|--------|---------------|-------|
| ResNet-50 | `configs/resnet50.yaml` | `resnet50` | 224×224 |
| EfficientNet-B4 | `configs/efficientnet_b4.yaml` | `tf_efficientnet_b4_ns` | 380×380 |
| ViT-B/16 | `configs/vit_b16.yaml` | `vit_base_patch16_224` | 224×224 |

Todos os modelos são carregados via **timm** com pesos pré-treinados no ImageNet e passam por _two-phase fine-tuning_: backbone congelado nas primeiras épocas (warmup) e descongelado para fine-tuning completo em seguida.

---

## Estrutura do Repositório

```
tcc-ze-praga-model-playground/
├── configs/
│   ├── base.yaml               # hiperparâmetros compartilhados
│   ├── resnet50.yaml           # overrides do ResNet-50
│   ├── efficientnet_b4.yaml    # overrides do EfficientNet-B4
│   └── vit_b16.yaml            # overrides do ViT-B/16
├── scripts/
│   ├── train.py                # CLI de treinamento
│   ├── evaluate.py             # CLI de avaliação no test set
│   └── export.py               # CLI de exportação ONNX + validação
├── src/
│   ├── data/
│   │   ├── download.py         # placeholder para download do Digipathos
│   │   ├── splits.py           # geração de splits 70/15/15 estratificados
│   │   ├── transforms.py       # pipelines albumentations (train / val)
│   │   └── dataset.py          # SoybeanLeafDataset + create_dataloaders
│   ├── models/
│   │   ├── factory.py          # build_model / freeze_backbone / unfreeze_backbone
│   │   ├── resnet.py           # build_resnet50
│   │   ├── efficientnet.py     # build_efficientnet_b4
│   │   └── vit.py              # build_vit_b16
│   ├── training/
│   │   ├── metrics.py          # compute_metrics / per_class_report
│   │   ├── losses.py           # build_loss + compute_class_weights
│   │   ├── optim.py            # AdamW + cosine scheduler com warmup
│   │   ├── callbacks.py        # CheckpointCallback + EarlyStoppingCallback
│   │   └── trainer.py          # Trainer (AMP, TensorBoard, gradient clipping)
│   ├── evaluation/
│   │   ├── evaluator.py        # evaluate + save_metrics
│   │   ├── confusion.py        # plot_confusion_matrix
│   │   └── benchmark.py        # latência CPU/GPU por batch size
│   ├── export/
│   │   ├── to_onnx.py          # export_to_onnx (opset 17, dynamic batch)
│   │   └── validate_onnx.py    # validação de paridade PyTorch × ONNX Runtime
│   └── utils/
│       ├── seed.py             # set_seed
│       ├── config.py           # load_config / load_model_config (deep merge)
│       └── logger.py           # get_logger
├── notebooks/
│   ├── 00_setup_colab.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_train_resnet50.ipynb
│   ├── 03_train_efficientnet_b4.ipynb
│   ├── 04_train_vit_b16.ipynb
│   ├── 05_evaluation_compare.ipynb
│   └── 06_export_onnx.ipynb
├── data/
│   ├── raw/digipathos/         # imagens brutas: <classe>/*.jpg
│   └── processed/              # train.csv, val.csv, test.csv, label_map.csv
├── artifacts/
│   ├── checkpoints/            # best_<modelo>.pth
│   ├── tensorboard/            # logs do TensorBoard
│   ├── metrics/                # metrics_<modelo>.json
│   ├── figures/                # confusion_<modelo>.png
│   └── onnx/                   # <modelo>.onnx
└── requirements.txt
```

---

## Guia de Execução no Google Colab

### Passo 1 — Verificar GPU

```python
import torch
print(torch.cuda.get_device_name(0))  # deve mostrar T4, A100, etc.
```

Se não houver GPU disponível: `Ambiente de execução → Alterar tipo de ambiente de execução → GPU T4`.

### Passo 2 — Montar o Drive e clonar o repositório

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run([
    "git", "clone",
    "https://github.com/<seu-usuario>/tcc-ze-praga-model-playground.git",
    "/content/repo"
])
```

```python
import os
os.chdir("/content/repo")
```

### Passo 3 — Instalar dependências

```bash
pip install -r requirements.txt
```

### Passo 4 — Baixar o dataset Digipathos

O dataset Digipathos (Embrapa) deve ser baixado manualmente em:
[https://www.digipathos-rep.cnptia.embrapa.br](https://www.digipathos-rep.cnptia.embrapa.br)

Organize as imagens no formato:

```
data/raw/digipathos/
    <nome_da_doenca>/
        imagem_001.jpg
        imagem_002.jpg
        ...
```

Ou carregue a partir do Google Drive:

```python
import shutil
shutil.copytree(
    "/content/drive/MyDrive/digipathos",
    "/content/repo/data/raw/digipathos"
)
```

### Passo 5 — Gerar splits estratificados

```bash
python src/data/splits.py \
    --raw_dir data/raw/digipathos \
    --out_dir data/processed
```

Isso cria `data/processed/train.csv`, `val.csv`, `test.csv` e `label_map.csv` com splits 70/15/15 estratificados por classe.

### Passo 6 — Executar os notebooks em ordem

| Notebook | Descrição |
|----------|-----------|
| `00_setup_colab.ipynb` | Verificação de ambiente e dependências |
| `01_eda.ipynb` | Análise exploratória do Digipathos (distribuição de classes, amostras) |
| `02_train_resnet50.ipynb` | Treinamento do ResNet-50 |
| `03_train_efficientnet_b4.ipynb` | Treinamento do EfficientNet-B4 |
| `04_train_vit_b16.ipynb` | Treinamento do ViT-B/16 |
| `05_evaluation_compare.ipynb` | Comparação dos 3 modelos no test set + confusion matrices |
| `06_export_onnx.ipynb` | Exportação ONNX + validação de paridade numérica |

### Passo 7 — Ou usar os scripts CLI diretamente

**Treinar:**

```bash
python scripts/train.py --config configs/resnet50.yaml --data_dir data/processed
python scripts/train.py --config configs/efficientnet_b4.yaml --data_dir data/processed
python scripts/train.py --config configs/vit_b16.yaml --data_dir data/processed
```

**Avaliar no test set:**

```bash
python scripts/evaluate.py \
    --config configs/resnet50.yaml \
    --checkpoint artifacts/checkpoints/best_resnet50.pth
```

**Exportar para ONNX:**

```bash
python scripts/export.py \
    --config configs/resnet50.yaml \
    --checkpoint artifacts/checkpoints/best_resnet50.pth
```

---

## Monitoramento com TensorBoard

```bash
tensorboard --logdir artifacts/tensorboard
```

---

## Integração com o Backend

Após exportar os modelos em ONNX, copie os arquivos `artifacts/onnx/*.onnx` para o repositório do backend FastAPI (`tcc-ze-praga-backend`) e configure o path nos settings de cada modelo.
