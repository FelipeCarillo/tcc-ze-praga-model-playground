# 05 — Leaf Segmentation

> **Status**: TODO | **Esforço**: 8h | **Bloqueia**: não, mas é o maior ganho de robustez em campo

## Context

Modelo treinado em Digipathos aprende correlações espúrias com o **fundo** (chão, mão do agricultor, sombra, cartela de cor). Em produção com fotos de smartphone, isso desmorona.

Solução: segmentar folha → mascarar fundo → classificar só a folha.

Ganho típico em literatura: +5–10% F1 em imagens "in the wild" quando treino é "in lab/controlled".

---

## Opções comparadas

| Abordagem | Tamanho | Latência (CPU) | Qualidade | Esforço |
|---|---|---|---|---|
| **SAM ViT-H** (Meta 2023) | ~2.5 GB | lentíssimo | excelente | baixo (zero-shot) |
| **MobileSAM** | ~10 MB | ~200ms | bom | baixo |
| **SAM2-tiny** | ~80 MB | ~150ms | ótimo | baixo |
| **U-Net leve treinado** | ~3 MB | ~50ms | bom (precisa dataset) | médio (anotar 200 folhas) |
| **GrabCut/Otsu** | 0 | <10ms | só com fundo limpo | baixo (clássico) |

**Recomendação inicial**: **MobileSAM com prompt central (point=centro da imagem)** — zero-shot, leve, sem precisar treinar nada. Validar em set de campo antes de partir para U-Net próprio.

---

## Arquitetura

```
Imagem (raw)
   │
   ▼
[Segmenter] ──→ Mask binária (HxW)
   │
   ├─ se mask cobre < 5% da imagem → fallback: usa imagem inteira (mask pode ter falhado)
   │
   ▼
Aplicar mask:
   - Background = preto OU média ImageNet (testar ambos)
   - Crop bbox da mask + 10% padding
   │
   ▼
Resize para input_size do classificador
   │
   ▼
[Classifier] → logits → predição
```

---

## Implementação

### `src/segmentation/mobile_sam.py`

```python
from mobile_sam import sam_model_registry, SamPredictor

class LeafSegmenter:
    def __init__(self, checkpoint: str = "weights/mobile_sam.pt"):
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def segment(self, image: np.ndarray) -> np.ndarray:
        self.predictor.set_image(image)
        h, w = image.shape[:2]
        # Prompt: ponto no centro (heurística simples; melhorar com YOLO leaf-detector futuramente)
        input_point = np.array([[w // 2, h // 2]])
        input_label = np.array([1])
        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        # pega a maior mask conexa
        return masks[scores.argmax()]
```

### Inferência integrada — `src/inference/pipeline.py`

```python
class InferencePipeline:
    def __init__(self, segmenter, classifier, ood_scorer, temperature):
        self.segmenter = segmenter
        self.classifier = classifier
        self.ood = ood_scorer
        self.T = temperature

    def __call__(self, image: np.ndarray) -> dict:
        mask = self.segmenter.segment(image)
        if mask.sum() / mask.size < 0.05:
            cropped = image  # fallback
        else:
            cropped = apply_mask_and_crop(image, mask)

        x = preprocess(cropped)
        logits = self.classifier(x)
        if self.ood.score(logits) > self.ood.threshold:
            return {"class": "unknown", "confidence": None, "ood": True}
        probs = torch.softmax(logits / self.T, dim=-1)
        return {
            "class": LABELS[probs.argmax()],
            "confidence": float(probs.max()),
            "ood": False,
        }
```

---

## Decisão importante: **segmentar no treino também?**

Duas opções:

| Opção | Prós | Contras |
|---|---|---|
| **A**: treinar em imagens originais, segmentar só em inferência | Sem precisar reprocessar dataset | Treino-inferência mismatch (modelo aprende com fundo, prediz sem) |
| **B**: treinar com imagens já segmentadas | Treino-inferência consistente | Precisa rodar segmentação em todo o dataset |

**Recomendação**: **opção B**. Adicionar etapa em `02_unify.py` (plano 01) ou criar `02b_segment.py` que produz `data/interim/segmented/`. Treino e inferência usam o mesmo pipeline.

Custo: ~233 imagens × 200ms = ~1min. Insignificante.

---

## Dependências novas

```
mobile_sam==0.0.1  # ou git install do repo do CSAILVision/MobileSAM
```

---

## Verificação

```bash
# 1. Baixar pesos MobileSAM
wget -O weights/mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# 2. Smoke segment em 10 imagens random
python scripts/test_segmenter.py --n 10 --save_dir artifacts/seg_preview/
# Abrir as imagens e conferir: folha bem destacada, fundo limpo

# 3. Re-treinar com dataset segmentado (após plano 01 + segmenter wired)
python scripts/ingest/02b_segment.py
python scripts/train.py --config configs/resnet50.yaml

# 4. A/B: comparar F1 em set "in the wild" (PlantDoc soja)
python scripts/eval_wild.py --checkpoint best_resnet50.pth --with_seg
python scripts/eval_wild.py --checkpoint best_resnet50.pth --no_seg
# Esperar: with_seg supera no_seg em ≥ 3% F1
```

**Aceite**: F1 em set in-the-wild com segmentação > F1 sem segmentação em ≥ 3 pontos absolutos.
