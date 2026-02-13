# CadQuery Code Generator from Images

## 👁‍📚 Overview

This project tackles the task of generating CadQuery (Python-based CAD) scripts from images of 3D objects. The system takes a rendered image of a shape and generates a syntactically correct Python script that recreates it using the CadQuery library.

It is inspired by research in image captioning, code generation, and vision-language modeling.

---

## 📊 Baseline Model Architecture

### Components

* **Image Encoder:** ResNet-18 (pretrained on ImageNet, last FC layer removed)
* **Token Embedding:** Embeds CadQuery tokens to vectors
* **Sequence Decoder:** GRU that receives token embeddings + image features at each step
* **Output Layer:** Linear projection to predict the next token in the vocabulary

### Input/Output

* Input: 2D rendered image of the 3D object + previously predicted tokens
* Output: Sequence of CadQuery tokens

> Inspired by: *Show and Tell: A Neural Image Caption Generator* (Vinyals et al., 2015)

---

## ✅ State-of-the-Art Upgrade (implemented)

The repository now includes a production-ready **SOTA-oriented pipeline** in `src/mecagent/` that upgrades the baseline while staying trainable on modest hardware:

1. **ViT-style image encoder + Transformer decoder**
   - `PatchImageEncoder`: patch embedding + Transformer encoder stack.
   - `CadQuerySOTAModel`: autoregressive Transformer decoder conditioned on image memory.

2. **Better decoding strategies**
   - Top-k + temperature sampling (`decode_top_k`) for diversity.
   - Length-normalized beam search (`decode_beam_search`) for stronger deterministic generation.
   - Optional token constraints to block invalid vocabulary ids during inference.

3. **Training stability features**
   - Label smoothing loss (`cadquery_loss`).
   - Scheduled sampling ratio helper.
   - Curriculum helper to ramp CAD operation complexity over early epochs.

This is designed to directly target the original project goal: improve geometric fidelity and syntax validity compared with the GRU baseline.

---

## 🚀 How to use the new model

```python
import torch
from mecagent import CadQuerySOTAModel, ModelConfig, decode_beam_search

cfg = ModelConfig(vocab_size=2048, max_length=512, image_size=224)
model = CadQuerySOTAModel(cfg)

image = torch.randn(1, 3, 224, 224)
sequence = decode_beam_search(
    model=model,
    image=image,
    bos_token_id=1,
    eos_token_id=2,
    max_length=256,
    beam_size=4,
)
```

---

## ⚠️ Constraints

* The original baseline was trained **without GPU** due to hardware limitations.
* As a result, model complexity and training time were constrained.
* The previous notebook only used greedy decoding.

---

## 📈 Metrics (Current baseline results)

* **Valid Syntax Rate:** \~80%
* **Mean IOU with Ground Truth:** \~1%

These reflect low-resource training limits and motivate the SOTA upgrade path included in this repo.

---

## 🧭 Recommended next run plan

1. Re-train using `CadQuerySOTAModel` with the same train/validation split.
2. Use curriculum (`curriculum_max_ops`) for the first 5 epochs.
3. Add scheduled sampling starting at epoch 2.
4. Evaluate with:
   - `metrics/valid_syntax_rate.py`
   - `metrics/best_iou.py`
5. Compare greedy vs top-k vs beam search for IoU and syntax validity.

---

## 🧪 Project structure

- `good_luck.ipynb`: original baseline notebook.
- `src/mecagent/model.py`: SOTA model architecture.
- `src/mecagent/decoding.py`: advanced decoding strategies.
- `src/mecagent/training.py`: training utilities.
- `metrics/`: evaluation scripts.

---

> Author: Joseph Elias Al Khoury  
> Enhanced with SOTA-ready architecture and decoding pipeline.
