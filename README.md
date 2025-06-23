# CadQuery Code Generator from Images

## 👁‍📚 Overview

This project tackles the task of generating CadQuery (Python-based CAD) scripts from images of 3D objects. The system takes a rendered image of a shape and generates a syntactically correct Python script that recreates it using the CadQuery library.

It is inspired by research in image captioning, code generation, and vision-language modeling.

---

## 📊 Model Architecture

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

## ⚠️ Constraints

* The model was trained **without GPU** due to hardware limitations.
* As a result, model complexity and training time were constrained.
* No attention, transformers, or beam search used.

---

## ✅ Post-processing Enhancements (CPU-friendly)

### 1. Token Repair & Filtering

A `tokens_to_code()` function was built to:

* Fix unmatched parentheses
* Skip malformed instructions or undefined variables
* Add `.extrude()` when needed
* Remove syntactically invalid constructs

> Inspired by: *DrRepair: Neural Program Repair by Jointly Learning to Localize and Repair* (Lachaux et al., 2021)

### 2. Code Decoding Strategy

* Currently uses greedy decoding
* Future options (CPU-possible):

  * Top-k sampling
  * Temperature sampling

> Based on: *Fan et al., 2018 - Hierarchical Neural Story Generation*

### 3. Data Augmentation Ideas

* Multiple views of the same object
* Render style variations (e.g., shadows, colors, edges)
* Synthetic noise injection

> Referenced from: *Data Augmentation for Vision and Language Tasks* (Wang et al., 2020)

---

## ⚡ What Could Be Enhanced (with GPU Access)

### 🛠️ Architecture Improvements

* **Transformer Decoder** instead of GRU
  ➔ Based on: *CodeT5* (Wang et al., 2021)

* **Add Attention (Luong / Bahdanau)** to focus on relevant features per token
  ➔ From: *Neural Machine Translation by Jointly Learning to Align and Translate* (Bahdanau et al., 2014)

* **Replace ResNet with Vision Transformer (ViT/DeiT)** for image encoding
  ➔ From: *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2020)

### ⚖️ Training Improvements

* **Curriculum Learning**: start from simple shapes and gradually increase difficulty
  ➔ Bengio et al., 2009

* **Scheduled Sampling** to reduce exposure bias
  ➔ Bengio et al., 2015

* **Token Dropout & Masking** for robustness
  ➔ Inspired by: *T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (Raffel et al., 2020)

---

## 📈 Metrics (Current Results)

* **Valid Syntax Rate:** \~80%
* **Mean IOU with Ground Truth:** \~1%

These results reflect the limitations of low-resource training, but offer a strong foundation for iterative enhancement.

---

## 🚀 Final Notes

This project is a functional and extensible baseline. All components are designed to run on CPU with low memory, making it deployable on constrained systems.

With access to more computing power, the architecture could be expanded and improved using state-of-the-art sequence modeling techniques.

> Author: Joseph Elias Al Khoury
> M2 AI in Health @ École Centrale de Lille
