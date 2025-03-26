
# Tensor Factorization PEFT (TLoRA)

This repository provides a collection of **low-rank adaptation** techniques for adapting large Vision Transformers (ViTs) or similar neural networks to new tasks or domains. Traditional *full fine-tuning* requires retraining all model parameters, which can be computationally expensive. Low-rank methods—such as **LoRA**, and the tensor based methods implemented in this project—reduce the number of tunable parameters while still achieving competitive performance.

The project tests these techniques on image classification tasks (e.g., **CIFAR-10** and **Caltech-UCSD Birds 200**), comparing their parameter counts and final accuracies.

---

## Table of Contents

- [Tensor Factorization PEFT (TLoRA)](#tensor-factorization-peft-tlora)
  - [Table of Contents](#table-of-contents)
  - [Techniques Overview](#techniques-overview)
    - [LoRA (Low-Rank Adaptation)](#lora-low-rank-adaptation)
    - [CP (Canonical Polyadic)](#cp-canonical-polyadic)
    - [Tucker (Tucker2, Tucker3)](#tucker-tucker2-tucker3)
    - [HTTucker](#httucker)
    - [HTFTucker](#htftucker)
    - [Multi-Layer Tucker](#multi-layer-tucker)
  - [Usage](#usage)
    - [Basic Training Command](#basic-training-command)
  - [Results](#results)
    - [CIFAR-10](#cifar-10)
    - [Caltech-UCSD Birds 200](#caltech-ucsd-birds-200)
    - [Observations and Interpretations](#observations-and-interpretations)
  - [References](#references)

---

## Techniques Overview

Below is a brief summary of each factorization approach implemented in this project.

### LoRA (Low-Rank Adaptation)
Introduced in [1], **LoRA** replaces the full matrix \(\mathbf{W}\) with
\[
\mathbf{W} + \mathbf{A}\mathbf{B},
\]
where \(\mathbf{A}\in\mathbb{R}^{d\times r}\) and \(\mathbf{B}\in\mathbb{R}^{r\times d}\) have a small rank \(r \ll d\). Instead of modifying all of \(\mathbf{W}\), only \(\mathbf{A}\) and \(\mathbf{B}\) are learned, drastically reducing trainable parameters.

### CP (Canonical Polyadic)
**CP decomposition** (also known as PARAFAC) factorizes a 3D tensor into a sum of rank-1 outer products. For adaptation, we treat Q, K, V as slices of a 3D tensor and decompose them into three low-rank factor matrices.

### Tucker (Tucker2, Tucker3)
**Tucker decomposition** [2] is a higher-order extension of SVD that factorizes a tensor into a core tensor multiplied by factor matrices along each mode. This can be done as:
- **Tucker2**: a core \(\mathbf{G}\in\mathbb{R}^{r\times r \times 3}\) (for Q, K, V) and two factor matrices (\(\mathbf{U},\mathbf{V}\)).
- **Tucker3**: a core \(\mathbf{G}\in\mathbb{R}^{r_m \times r_r \times r_c}\) and three factor matrices for the modes, rows, and columns.

### HTTucker
**HTTucker** is designed to incorporate per-head updates for **Q, K, V** individually. Each head and each matrix (Q/K/V) gets its own small Tucker core and set of factor matrices, enabling the adapter to learn head-specific adjustments with a modest parameter overhead.

### HTFTucker
**HTFTucker** extends the idea above but merges **Q, K, V** into one combined core along an extra dimension, while also splitting out per-head factors.

### Multi-Layer Tucker
Extends Tucker factorization across multiple layers. Instead of a separate set of low-rank factors per layer, **Multi-Layer Tucker** attempts to share a global core with layer-specific factors (i.e. 1 single tensor for all the MHA in the model), further reducing the number of parameters needed for adaptation.

---

## Usage

### Basic Training Command

Use `main.py` with the following arguments (shown with their defaults):

```bash
python main.py \
  --dataset cifar10 \
  --factorization cp \
  --rank 8 \
  --batch-size 128 \
  --num-epochs 20 \
  --learning-rate 1e-4 \
  --eta-min 1e-6 \
  --weight-decay 1e-2 \
  --num-workers 4 \
  --seed 123
```

Common flags:

- `--dataset`: Choose among `cifar10`, `caltech_birds`, `cifar100` (or others implemented).  
- `--factorization`: One of `[lora, cp, tucker2, tucker3, httucker, htftucker, multi_layer_tucker]`.  
- `--rank`: Integer or tuple (e.g. `8` or `8,8,8`).  
- `--batch-size`: Batch size for training.  
- `--num-epochs`: Total epochs to train.  
- `--checkpoint-path`: If provided, loads a saved checkpoint.  
- `--compile-model`: Enables PyTorch compilation

Other scripts:

- **`main_classifier.py`**: Fine-tunes only a classifier head.  
- **`main_full.py`**: Demonstration of fully fine-tuning Q, K, V parameters.  
- **`optuna_search.py`**: Automated hyperparameter search with [Optuna](https://optuna.org).  

---

## Results

We conducted experiments on **CIFAR-10** and **Caltech-UCSD Birds 200** fine-tuning the `google/vit-base-patch16-224` model from HuggingFace model hub. All experiments used a single NVIDIA T4 GPU and a **cosine annealing** learning-rate scheduler:

- **CIFAR-10**: 20 epochs, batch size = 128, LR = 1e-4 → 1e-6, weight decay = 1e-2  
- **CUB-200**: 50 epochs, batch size = 64, LR = 1e-4 → 1e-6, weight decay = 1e-2  

### CIFAR-10

| Method               | R=1 (Trainable Params / %)  | Accuracy  | R=4 (Trainable Params / %)   | Accuracy  | R=8 (Trainable Params / %)    | Accuracy  |
|----------------------|-----------------------------|-----------|------------------------------|-----------|-------------------------------|-----------|
| **Classifier Only**  | –                           | –         | 7,690 (0.01%)               | 93.59%    | –                             | –         |
| **LoRA (Hugging Face)** | –                         | –         | –                            | –         | 450,058 (0.52%)               | 98.11%    |
| **LoRA**             | 62,986 (0.07%)             | 97.13%    | 228,874 (0.27%)             | 98.11%    | 450,058 (0.52%)              | 98.33%    |
| **CP**               | 26,158 (0.03%)             | 98.11%    | 81,562 (0.09%)              | 98.10%    | 155,434 (0.18%)              | 98.31%    |
| **Tucker**           | 26,170 (0.03%)             | 98.35%    | 82,330 (0.10%)              | 98.50%    | 161,578 (0.19%)              | 98.54%    |
| **HTTucker**         | 38,110 (0.04%)             | 98.15%    | 131,530 (0.15%)             | 98.48%    | 269,194 (0.31%)              | 98.58%    |
| **HTFTucker**        | 17,866 (0.02%)             | 98.08%    | 51,418 (0.06%)              | 98.38%    | 138,154 (0.16%)              | 98.45%    |
| **Multi-Layer Tucker** | 9,242 (0.01%)             | 97.91%    | 14,150 (0.02%)              | 98.25%    | 24,194 (0.03%)               | 98.41%    |

### Caltech-UCSD Birds 200

| Method                    | R=1 (Trainable Params / %)  | Accuracy  | R=4 (Trainable Params / %)   | Accuracy   | R=8 (Trainable Params / %)    | Accuracy   |
|---------------------------|-----------------------------|-----------|------------------------------|------------|-------------------------------|------------|
| **QKV Full fine tuning**  | –                           | –         | 21,415,112 (24.92%)         | 85.40%     | –                             | –          |
| **Classifier Only**       | –                           | –         | 153,800 (0.18%)             | 85.35%     | –                             | –          |
| **LoRA**                  | 209,096 (0.24%)            | 85.05%    | 374,984 (0.44%)             | 86.50%     | 596,168 (0.69%)              | 86.24%     |
| **CP**                    | 172,268 (0.20%)            | 87.45%    | 227,672 (0.26%)             | 86.81%     | 301,544 (0.35%)              | 87.16%     |
| **Tucker**                | 172,280 (0.20%)            | 87.09%    | 228,440 (0.27%)             | 87.02%     | 307,688 (0.36%)              | 86.73%     |
| **HTTucker**              | 184,220 (0.21%)            | 86.87%    | 277,640 (0.32%)             | 86.57%     | 415,304 (0.48%)              | 86.50%     |
| **HTFTucker**             | 163,976 (0.19%)            | 86.23%    | 197,528 (0.23%)             | 86.85%     | 284,264 (0.33%)              | 86.87%     |
| **Multi-Layer Tucker**    | 155,352 (0.18%)            | 86.37%    | 160,260 (0.19%)             | 86.80%     | 170,304 (0.20%)              | 86.76%     |

> **Notes**:  
> - **Trainable Params / %**: Number of parameters being updated vs. total model parameters, expressed also as a percentage of the original ViT backbone.  
> - Training used **cosine annealing** scheduler from `--learning-rate 1e-4` down to `--eta-min 1e-6`.  
> - **“Classifier Only”** means only the final classification layer was updated.  
> - **“QKV Full Fine Tuning”** (Birds 200 table) adjusts all Q, K, V parameters at every layer (shown only for reference with R=4).  
> - **Single-run experiments**: Due to limited GPU resources, results reflect a single training run per configuration and may vary with subsequent runs.
> 
---

### Observations and Interpretations

- **CIFAR-10**:  
  - All factorization methods can achieve high accuracy (>98%), with **HTTucker** at \(R=8\) slightly leading at **98.58%**.  
  - Tucker and CP are also strong contenders; Tucker at \(R=8\) reaches **98.54%** with moderate parameter overhead.  
  - Even **Multi-Layer Tucker** retains strong performance with very few added parameters.

- **Caltech-UCSD Birds 200**:  
  - Notably, **CP** at \(R=8\) yields **87.16%**, outperforming full QKV fine-tuning (85.40%) while using far fewer trainable parameters.  
  - Tucker-based methods are also competitive, hovering around 86–87% accuracy.  
  - **Classifier Only** sits at ~85.35%, highlighting that rank-based methods significantly improve upon naive partial fine-tuning.

In general, these results demonstrate that **low-rank adaptations** (especially CP or Tucker-based approaches) can match or exceed classical fine-tuning accuracy while training only a small fraction of the parameters. This is highly beneficial for resource-constrained scenarios and for rapid domain adaptation with minimal overhead.

---

## References

1. Hu, Edward J., et al. "**LoRA: Low-Rank Adaptation of Large Language Models**." *arXiv preprint arXiv:2106.09685 (2021).*  
2. Kolda, Tamara G., and Brett W. Bader. "**Tensor decompositions and applications**." *SIAM review 51.3 (2009): 455-500.*  
3. Bershatsky, Daniel, et al. "**LoTR: Low Tensor Rank Weight Adaptation**." *arXiv preprint arXiv:2402.01376 (2024).*  
4. Jie, Shibo, and Zhi-Hong Deng. "**FACT: Factor-Tuning for Lightweight Adaptation on Vision Transformer**." *AAAI Conference on Artificial Intelligence (2023).*  
