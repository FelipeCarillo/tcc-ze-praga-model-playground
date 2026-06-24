# Resultados - Classificacao de Doencas Foliares de Soja (Ze Praga)

Dataset: **ASDID** (Auburn Soybean Disease Image Dataset, Zenodo 7304859, CC0) - 8.130 imagens limpas, 6 classes, split 70/15/15 (train 5690 / val 1219 / test 1221).
Treino local: RTX 3060 Laptop (6GB), PyTorch 2.3.1+cu121, two-phase fine-tuning, AMP, pesos de classe + label smoothing.

## Comparacao no test set (1.221 imagens)

| Modelo | Backbone (timm) | Acuracia (top-1) | F1-macro | ONNX |
|---|---|---|---|---|
| ResNet-50 | resnet50 | 95,99% | 96,13% | resnet50.onnx (90 MB) |
| ViT-B/16 | vit_base_patch16_224 | 98,03% | 98,06% | vit_b16.onnx (327 MB) |
| **EfficientNet-B4** | tf_efficientnet_b4_ns | **98,77%** | **98,82%** | efficientnet_b4.onnx (67 MB) |

**Melhor modelo: EfficientNet-B4** (98,77% acc, 98,82% F1-macro), tambem o ONNX mais leve (67 MB) - recomendado para o backend.

## Classes (6)
cercosporiose, ferrugem-asiatica, mancha-alvo, mancha-olho-de-ra (frogeye, substitui antracnose - ausente no ASDID), mildio, saudavel.

## Artefatos (artifacts/)
- checkpoints/best_<modelo>.pth - pesos PyTorch
- onnx/<modelo>.onnx - exportados (paridade validada)
- metrics/metrics_<modelo>.json - metricas completas + por classe
- figures/confusion_<modelo>.png - matrizes de confusao

## Reproduzir
```
python src/data/splits.py --raw_dir data/raw/asdid --out_dir data/processed
python scripts/train.py    --config configs/resnet50.yaml --data_dir data/processed
python scripts/evaluate.py --config configs/resnet50.yaml --checkpoint artifacts/checkpoints/best_resnet50.pth
python scripts/export.py   --config configs/resnet50.yaml --checkpoint artifacts/checkpoints/best_resnet50.pth
```
(troque resnet50 por vit_b16 ou efficientnet_b4). Notebooks didaticos em notebooks/ (00 prep, 01-03 treino, 04 export).
