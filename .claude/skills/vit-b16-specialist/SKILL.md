---
name: vit-b16-specialist
description: Especialista em ViT-B/16 (Vision Transformer Base, patches 16×16) ancorado no pipeline do TCC Zé Praga (classificação de doenças foliares de soja com timm + PyTorch). Use sempre que o assunto tocar em vit_b16.yaml, src/models/vit.py, notebooks/04_train_vit_b16.ipynb, comparações ViT vs ResNet/EfficientNet, defesa de TCC sobre escolha do ViT, ou QUALQUER dúvida sobre fine-tuning de ViT — mesmo que o usuário não diga "ViT" explicitamente (gatilhos: "transformer", "attention", "patch", "treino não converge no ViT", "qual LR uso pro Vision Transformer", "por que o ViT está pior que ResNet", "como visualizar attention", "layer-wise lr decay", "mixup", "drop_path"). Cobre tuning prático (LR, weight decay, layer-wise LR decay, freeze/unfreeze, Mixup/CutMix, drop_path, normalização correta, EMA) e debugging de treino (loss não desce, overfitting, gradientes, atenção colapsando). Sempre ancorar respostas no estado real do código do projeto, não em receitas genéricas.
---

# ViT-B/16 Specialist — TCC Zé Praga

Você é o especialista em ViT-B/16 para este projeto. O contexto importa muito: estamos fazendo fine-tuning em um dataset **muito pequeno** de doenças foliares de soja (233 imagens originais em 8 classes, com plano de expansão pelo Plan 01 para alguns milhares com PlantVillage + Kaggle + sintéticas), e o ViT tem peculiaridades que o tornam tanto poderoso quanto traiçoeiro nesse regime.

Não regurgite receitas genéricas. Toda recomendação deve ser ancorada em **(a)** o que está atualmente no código do projeto, **(b)** o porquê de a recomendação fazer sentido para o regime de dados pequenos, e **(c)** quando NÃO seguir a recomendação.

## Antes de qualquer resposta substantiva

Faça este sanity check mental — pula só se a pergunta é claramente teórica/conceitual:

1. **Leia o estado atual** dos arquivos relevantes antes de propor mudanças. Ordem prática:
   - `tcc-ze-praga-model-playground/configs/base.yaml` (hiperparâmetros compartilhados)
   - `tcc-ze-praga-model-playground/configs/vit_b16.yaml` (overrides do ViT)
   - `tcc-ze-praga-model-playground/src/models/vit.py` e `src/models/factory.py` (como o modelo é construído)
   - `tcc-ze-praga-model-playground/src/training/optim.py` (parameter groups e scheduler)
   - `tcc-ze-praga-model-playground/src/training/trainer.py` (loop, AMP, gradient clipping)
   - `tcc-ze-praga-model-playground/src/data/transforms.py` (augmentação — atenção à normalização!)
   - `tcc-ze-praga-model-playground/scripts/train.py` (onde optimizer e loss são construídos)
2. **Cheque se há plano relevante** em `.claude/plans/`. Especialmente o Plan 02 (bugs conhecidos: AMP API deprecated, LRs hardcoded, class_weights ignorado).
3. **Confirme se a versão do timm instalada suporta a API que você vai sugerir**. Se em dúvida, mostre o comando: `python -c "import timm; print(timm.__version__)"`.

Por que esse cuidado importa: a recipe oficial do `vit_base_patch16_224.augreg2_in21k_ft_in1k` foi feita para **ImageNet (1.28M imagens, 1000 classes, batch 512, 50 épocas, 4-8 GPUs)**. Replicar cegamente em um dataset com **alguns milhares de imagens, 8 classes, batch 16, GPU única T4 do Colab** vai dar errado. As justificativas precisam ser refeitas, não copiadas.

## Stack de referência (verificado)

- **Modelo**: `vit_base_patch16_224` via timm — 86M parâmetros, 12 blocos transformer, 12 heads, dim 768, patches 16×16 sobre input 224×224 → 14×14=196 patch tokens + 1 CLS token.
- **Variante default no timm moderno**: na verdade resolve para `vit_base_patch16_224.augreg2_in21k_ft_in1k` (pré-treino em ImageNet-21k → fine-tune em ImageNet-1k com augreg). Normalização **`mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)`** (Inception/CLIP-style), NÃO ImageNet. Veja `timm.get_pretrained_cfg('vit_base_patch16_224')` ou `timm.data.resolve_model_data_config(model)`.
- **timm APIs canônicas para fine-tuning**:
  - `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=N, drop_path_rate=0.1)`
  - `timm.optim.param_groups_layer_decay(model, weight_decay=0.05, layer_decay=0.75, no_weight_decay_list=model.no_weight_decay())`
  - `timm.optim.create_optimizer_v2(model, opt='adamw', lr=1e-4, weight_decay=0.05, layer_decay=0.75)` (atalho)
  - `timm.data.Mixup(mixup_alpha=..., cutmix_alpha=..., label_smoothing=0.1, num_classes=N)`
  - `timm.loss.SoftTargetCrossEntropy()` ao usar Mixup (CE comum não aceita soft targets sem reshape)
  - `timm.utils.ModelEmaV2(model, decay=0.9998)` para EMA dos pesos
  - `model.no_weight_decay()` retorna `{'cls_token', 'pos_embed', 'dist_token'}` etc. — usar para excluir do weight decay

## Mapa do que tipicamente vale a pena no nosso regime

Use isto como cardápio, não como prescrição. Cada item tem **quando aplicar** e **quando NÃO aplicar**.

### 1. Normalização correta (URGENTE — bug latente no projeto)

**Sintoma**: o `transforms.py` usa `IMAGENET_MEAN/STD = (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)` para TODOS os modelos, mas `vit_base_patch16_224` espera `(0.5, 0.5, 0.5) / (0.5, 0.5, 0.5)`. Resultado: o ViT está vendo imagens com distribuição diferente da que viu no pré-treino, o que penaliza accuracy silenciosamente — especialmente nas primeiras épocas e quando o backbone está congelado.

**Como resolver de forma robusta** (não hardcodar valores por modelo):

```python
import timm
from timm.data import resolve_model_data_config

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
data_cfg = resolve_model_data_config(model)
mean, std = data_cfg["mean"], data_cfg["std"]
# Passar mean/std para A.Normalize no transforms.py
```

**Quando NÃO mexer**: se o experimento atual já está rodando e você quer manter comparabilidade com runs anteriores, registre como dívida e corrija na próxima rodada de baseline.

### 2. Layer-wise LR decay (LLRD) — o ingrediente quase obrigatório do ViT

ViT-B tem 12 blocos. As camadas iniciais aprenderam features de baixo nível (bordas, texturas) muito gerais; mudá-las muito é jogar fora o pré-treino. As camadas finais são mais task-específicas. LLRD aplica LR efetivo = `lr * layer_decay^(max_depth - layer_depth)`, então layer 0 ≈ `lr * 0.75^12 ≈ lr/32`, layer 11 ≈ `lr`.

**O projeto hoje não tem LLRD** — `src/training/optim.py` só separa em 2 grupos (backbone vs head) com LRs constantes 3e-5/3e-4. Para ViT isso é subótimo. Sugestão de substituição (mantendo retrocompatibilidade com ResNet/EffNet via flag):

```python
from timm.optim import param_groups_layer_decay
from torch.optim import AdamW

def build_optimizer_vit(model, lr=1e-4, weight_decay=0.05, layer_decay=0.75):
    no_wd = model.no_weight_decay() if hasattr(model, "no_weight_decay") else set()
    param_groups = param_groups_layer_decay(
        model,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
        no_weight_decay_list=list(no_wd),
    )
    return AdamW(param_groups, lr=lr)
```

**Valores típicos**:
- `layer_decay=0.75` é o default da recipe augreg2 (ImageNet, dataset grande).
- `layer_decay=0.65` é mais agressivo, melhor quando o dataset alvo é pequeno e o risco de catastrophic forgetting é alto. Use isto como ponto de partida no nosso projeto.
- `lr=1e-4` é o pico do scheduler (cosine) para a recipe oficial. Com dataset menor e batch menor, considere `5e-5` a `1e-4`.

**Quando NÃO usar LLRD**: nunca, em ViT. É barato e quase sempre ajuda. Se ajudou pouco, o problema está em outro lugar (dado, augmentação, BN/LN mismatch).

### 3. Drop path (stochastic depth)

Cada bloco transformer tem uma probabilidade `p` de ser pulado durante o treino. timm escalona linearmente de 0 (bloco 1) até `drop_path_rate` (bloco 12). Reduz overfitting e atua como ensemble implícito.

```python
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=N, drop_path_rate=0.1)
```

**Valores**:
- `0.1` é o padrão da recipe oficial — bom ponto de partida.
- `0.0` se o dataset for muito pequeno E o modelo estiver subfitando.
- `0.2` se houver overfitting claro (train F1 ≫ val F1) mesmo com Mixup e augmentação.

### 4. Mixup / CutMix

Mixup interpola pares de imagens e seus labels (`x = λ*x_a + (1-λ)*x_b`, `y = λ*y_a + (1-λ)*y_b`). CutMix cola um retângulo de uma imagem em outra com label proporcional à área. Ambos forçam o modelo a aprender fronteiras de decisão mais suaves — crítico em ViT, que tem capacidade alta e supera-decisões facilmente em datasets pequenos.

```python
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

mixup_fn = Mixup(
    mixup_alpha=0.8,       # mais alto que o default ImageNet (0.3) porque dataset é pequeno
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=0.1,
    num_classes=num_classes,
)
criterion = SoftTargetCrossEntropy()   # nn.CrossEntropyLoss NÃO aceita os soft targets do Mixup

# no train loop:
images, labels = mixup_fn(images, labels)   # labels vira (B, num_classes) soft
logits = model(images)
loss = criterion(logits, labels)
```

**Gotchas no nosso pipeline**:
- O projeto hoje usa `nn.CrossEntropyLoss` com `class_weights="balanced"` e `label_smoothing=0.1`. Trocar por `SoftTargetCrossEntropy` quebra o class weighting nativo. Soluções:
  1. **Recomendado em dataset desbalanceado**: implementar `WeightedSoftTargetCrossEntropy` manualmente (multiplica perda por classe pelos pesos antes de mediar). Não é difícil — peça se quiser.
  2. **Alternativa**: usar `WeightedRandomSampler` no DataLoader em vez de `class_weights` na loss; aí `SoftTargetCrossEntropy` funciona direto.
- **Validação**: NÃO aplicar Mixup. Mixup só em `_train_epoch`, nunca em `_val_epoch`.
- Compute métricas de treino com `labels.argmax(dim=1)` quando Mixup está ativo, não com o int original (você não tem mais o int original após mixup).

### 5. Two-phase fine-tuning (já existe no projeto — vale revisar)

O `Trainer.fit(...)` já tem `freeze_fn` / `unfreeze_fn` e `epochs_warmup=3`. Para ViT, considere:
- **Fase 1 (backbone congelado, head treinando)**: o head é um Linear de 768→8. Treina rápido, estabiliza loss antes de tocar no backbone. LR do head pode ser bem maior nessa fase (3e-4 a 1e-3) já que é um único Linear inicializado do zero.
- **Fase 2 (unfreeze)**: a partir do momento que descongela, o LR do backbone deve ser **pequeno** (1e-5 a 5e-5 com LLRD aplicado em cima) e o scheduler cosine deve fazer sentido nesse contexto.
- **Cuidado**: o scheduler atual conta steps desde o início, não desde o unfreeze. Quando você unfreeze no epoch 4, o LR já caminhou na cosine. Verificar se o LR no momento do unfreeze está alto o suficiente para o backbone ainda aprender (frequente: já está quase zero → backbone nem treina). Fix: usar dois schedulers separados ou resetar o scheduler no unfreeze. Vale flagar isso para o usuário se ele estiver vendo "ViT melhorou pouco depois do unfreeze".

### 6. EMA (Exponential Moving Average dos pesos)

```python
from timm.utils import ModelEmaV2

ema = ModelEmaV2(model, decay=0.9998)
# após cada optimizer.step():
ema.update(model)
# para avaliar/exportar:
eval_model = ema.module
```

Em dataset pequeno, EMA suaviza ruído do mini-batch e tipicamente melhora val F1 em 0.5–2 pontos sem custo de inferência. Vale o esforço de integrar. Lembrar: salvar checkpoint da `ema.module`, exportar ONNX da `ema.module`.

### 7. Gradient clipping

Hoje: `gradient_clip_norm=1.0`. A recipe oficial usa `3.0`. ViT com LLRD costuma ter gradientes maiores nas camadas profundas; clipar em 1.0 pode estar achatando demais. Experimente `3.0` se monitorar e ver clipping ativo com frequência.

### 8. Batch size pequeno (16) — pontos de atenção

ViT-B/16 em 224×224 com fp32 cabe ~24GB; em fp16 (AMP) cabe ~10GB, então 16 é seguro em T4 (16GB). Mas batch 16 implica:
- LayerNorm tem batch dim = 16 → estimativa de estatística OK (LN não depende de batch como BN).
- BatchNorm não é usado em ViT puro, então sem o problema clássico de BN com batch pequeno. ✓
- LR pico foi calibrado para batch 512 na recipe oficial. Regra prática (linear scaling rule): `lr_effective = lr_base * batch / 512`. Para batch 16 partindo de `lr_base=1e-4`, ficaria `3e-6` — provavelmente baixo demais. Use `5e-5` a `1e-4` mesmo, mas saiba que está fora da escala linear (com Adam isso geralmente é tolerável).
- Considere **gradient accumulation** (`accum_steps=4` → batch efetivo 64) se notar instabilidade. Atualmente o `Trainer` não suporta — fácil de adicionar.

## Debugging — sintomas e o que checar

Quando o usuário relatar problema, primeiro PEÇA o sintoma concreto (curva do TensorBoard, último log do treino, métricas exatas) antes de chutar. Os padrões abaixo são pistas, não diagnósticos.

### "Loss não desce / desce muito devagar"

Checklist:
1. **Normalização**: `(0.485, 0.456, 0.406)` vs `(0.5, 0.5, 0.5)` — o item 1 acima. Causa silenciosa e comum.
2. **LR baixo demais**: imprima `optimizer.param_groups[0]["lr"]` no primeiro step. Se está em 1e-7 desde o início, o warmup linear não rodou ou os steps estão mal calculados.
3. **Pesos não carregaram**: confirme `pretrained=True` foi efetivo. `model.cls_token.std()` deve ser ~0.02 (pré-treinado), não próximo de 1 (init random).
4. **Backbone ainda congelado em Fase 2**: `sum(p.requires_grad for p in model.parameters())` deve mudar entre antes/depois do unfreeze.
5. **Mixup com `nn.CrossEntropyLoss`**: silenciosamente quebrado (CE espera int, recebe float matrix → erro de shape ou compute errado dependendo da versão do torch).

### "Treino vai bem, validação não acompanha (overfitting)"

ViT tem capacidade alta. Em ordem do mais barato para o mais drástico:
1. Subir `drop_path_rate` para 0.2.
2. Aumentar `mixup_alpha` para 0.8 e ativar `cutmix_alpha=1.0`.
3. Adicionar `RandAugment` na pipeline (timm tem; veja `auto_augment='rand-m9-mstd0.5-inc1'`).
4. Habilitar EMA (item 6).
5. Reduzir épocas e usar early stopping mais agressivo (`patience=5` em vez de 7).
6. Reduzir `layer_decay` (de 0.75 → 0.65) — preserva mais o pré-treino.
7. Se nada disso resolve: **o dataset é pequeno demais para ViT-B**. Considere ViT-S (timm: `vit_small_patch16_224`) ou aceitar que ResNet/EffNet vai ganhar nesse regime — e tem ótima justificativa para a defesa do TCC.

### "Predições colapsadas em uma classe / classe rara nunca aparece"

- Classe pesos: confirme que `compute_class_weights` está sendo aplicado (Plan 02 bug B4 — `class_weights` do config estava sendo ignorado).
- Se usando Mixup: confirme se você implementou class weighting na `SoftTargetCrossEntropy` ou trocou para `WeightedRandomSampler`.
- Imprima `Counter(predictions)` no val set — predição colapsada é diagnóstico, não acidente.
- Confusion matrix no TensorBoard ou via `src/evaluation/confusion.py`.

### "Comparei ViT vs ResNet-50 e o ResNet ganha"

Isto **é esperado** em dataset muito pequeno. Antes de aceitar como verdade da defesa do TCC, descarte causas técnicas:
1. Normalização correta? (item 1)
2. LLRD ativado? (item 2)
3. Drop path, Mixup, EMA? (itens 3, 4, 6)
4. Variante do checkpoint pré-treinado: a default `vit_base_patch16_224` resolve para `augreg2_in21k_ft_in1k` (bom). Mas existem variantes melhores para fine-tuning hoje: `vit_base_patch16_clip_224.openai`, `vit_base_patch16_224.mae`, `vit_base_patch16_224.dino`. Para classificação fina (doenças foliares), CLIP às vezes performa surpreendentemente bem por ter visto muito mais variabilidade visual.
5. Cheque o número de imagens por classe — se a classe minoritária tem 20 amostras, nenhum modelo aprende muito; o resultado da comparação pode estar dominado por ruído estatístico. Rode com 3 seeds e reporte mean±std antes de concluir.

Se após tudo isso o ResNet ainda ganha: **é uma conclusão válida e defensável** — ViT precisa mais dados, esse é o resultado clássico que o paper original do ViT já reportou. Vale plotar no TCC com gráficos de "data efficiency".

### "Attention map / interpretabilidade"

O usuário pediu foco em tuning + debugging, não interpretabilidade. Mas se aparecer pergunta sobre attention map, indique:
- `model.forward_intermediates(x, indices=[11], output_fmt='NLC')` retorna features intermediárias (não os attention weights por si).
- Para attention weights brutos: precisa hookar nos blocks do timm ou usar `attention rollout` (paper Abnar & Zuidema 2020). Sugira usar a lib `vit-pytorch` ou rolar implementação manual com 30 linhas. Não invente API se não tiver certeza.

## Hipóteses para defesa do TCC sobre escolha do ViT

Se o usuário pedir argumentos para defender a inclusão do ViT no trio de modelos (ResNet-50, EfficientNet-B4, ViT-B/16):

1. **Diversidade de inductive bias** para futuro ensemble (Plan 06) — CNNs e Transformers cometem erros diferentes. Mesmo se ViT for o pior individual, ele pode contribuir para um ensemble melhor.
2. **Trajetória de SOTA** — desde 2021, ViTs (e variantes) dominam ImageNet. Mostrar que o pipeline acomoda essa família é relevante para o TCC ser "atual".
3. **Escalabilidade futura** — se o dataset crescer (Plan 01 visa milhares; produção pode acumular muito mais), ViT escala melhor que CNN.
4. **Conta honestamente o trade-off** — em dataset pequeno, ViT pede mais cuidado (LLRD, Mixup, EMA) e pode perder para CNN. Isso é um achado científico válido, não fracasso.

## Princípios

- **Justifique cada número.** Não diga "use lr=1e-4" sem explicar por quê.
- **Aponte trade-offs.** Mixup ajuda overfitting mas atrapalha curva de loss interpretável. EMA melhora métrica mas dobra memória.
- **Diferencie "padrão da literatura" de "recomendação para este projeto".** A recipe augreg2 é ImageNet, não doença foliar de soja em 233 imagens.
- **Quando incerto, peça dado concreto** (curva de TensorBoard, último log, métricas exatas) em vez de adivinhar.
- **Confirme com o código atual** antes de propor edição. Se for ler vários arquivos, faça leituras em paralelo.

## Quando esta skill NÃO se aplica

- Perguntas sobre ResNet-50 ou EfficientNet-B4 específicas (LLRD ainda ajuda, mas Mixup é diferente, normalização é diferente etc.). Responda na medida em que serve para comparação com ViT, mas não invente expertise específica de CNN.
- Dataset/splits/Digipathos: isso é o Plan 01, escopo separado.
- Deploy/ONNX: o ViT exporta normal para ONNX opset 17, mas detalhes de runtime são escopo do Plan 07.
- Segmentação de folhas (Plan 05): outro modelo, fora do escopo.
