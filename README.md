# Aegis-Lite: AI Model Integrity & Backdoor Detection Toolkit

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research--prototype-yellow)

---

## Abstract

Aegis-Lite is a research-grade Python toolkit for **detecting backdoor attacks** embedded in
deep neural network classifiers. Given a suspect model and a clean reference model, Aegis-Lite
extracts intermediate activation representations, measures their statistical divergence, and
probes the suspect model's sensitivity to a known trigger pattern. The combined signals are
distilled into an **Integrity Score** (0–100) that summarises whether a model is likely to have
been backdoored. The project ships with a full CIFAR-10 training pipeline, a 3×3 white-square
backdoor attack implementation, and a Jupyter demo notebook.

---

## Problem Statement

Modern machine learning pipelines increasingly rely on third-party datasets, pre-trained
weights, and automated training infrastructure. Any of these supply-chain components can be
tampered with by an adversary to embed a *backdoor*: a hidden behaviour that causes the model
to misclassify inputs that contain a small, imperceptible trigger while maintaining high
accuracy on clean data. Such attacks are extremely difficult to detect by standard accuracy
benchmarks alone because the poisoned model achieves near-normal test performance.

---

## Threat Model

| Aspect | Assumption |
|---|---|
| **Adversary goal** | Cause the target model to predict a fixed *target class* whenever a trigger is present |
| **Adversary capability** | Can poison a fraction of the training data (data-poisoning threat model) |
| **Defender knowledge** | Has access to a clean reference model and a small set of clean evaluation data |
| **Trigger type** | Static, input-space trigger (3×3 white square, bottom-right corner) |
| **Poison rate** | 10% of training samples (configurable) |

---

## Backdoor Attack Description

The backdoor is injected via **data poisoning**:

1. A random 10% subset of training images are selected.
2. Each selected image has a **3×3 white-pixel square** stamped onto its bottom-right corner.
3. The label of every poisoned sample is overwritten to **class 0** (*airplane*).
4. The model is trained normally on this poisoned dataset.

At inference time, the model classifies clean images accurately but consistently predicts
class 0 for any image containing the trigger—achieving an Attack Success Rate (ASR) typically
exceeding 90%.

---

## Detection Methodology

Aegis-Lite uses four detection modules:

### Module 1 — Activation Extraction (`detection/extract_activations.py`)
Registers a PyTorch forward hook on a user-specified layer (default: `fc1`) and collects the
activation vectors for up to 2,000 clean test samples from both the reference model and the
suspect model.

### Module 2 — Divergence Metrics (`detection/divergence_metrics.py`)
Computes three statistical distances between the clean and suspect activation distributions:

| Metric | Description |
|---|---|
| Mean Activation Difference (MAD) | Mean absolute diff of per-feature means |
| L2 Norm Distance | Euclidean distance between mean vectors |
| KL Divergence | Average per-feature KL divergence of histograms |

These are combined into a single normalised **Divergence Score** in [0, 1].

### Module 3 — Trigger Sensitivity (`detection/trigger_sensitivity.py`)
Evaluates how much the suspect model's confidence on the target class increases when the known
trigger pattern is applied to clean images. A large confidence shift (>0.5 on average) is a
strong indicator of a backdoor. This is normalised into a **Sensitivity Score** in [0, 1].

### Module 4 — Integrity Score (`detection/integrity_score.py`)
Combines the two sub-scores with equal weights:

```
Integrity Score = 100 − (0.5 × Divergence Score + 0.5 × Sensitivity Score) × 100
```

A score **≥ 70** is considered *likely clean*; below 70 is *suspicious*.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/aegis-lite.git
cd aegis-lite

# Install dependencies
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch 2.0+, torchvision, scikit-learn, scipy, matplotlib, tqdm.

---

## How to Run

### Train the clean model
```bash
python training/train_clean.py
# → results/clean_model.pth
```

### Train the backdoored model
```bash
python training/train_backdoor.py
# → results/backdoor_model.pth
```

### Run the full detection pipeline (Jupyter)
```bash
jupyter notebook notebooks/demo.ipynb
```
Set `SKIP_TRAINING = True` in the first cell to load the pre-trained weights.

### Run detection from Python directly
```python
import torch
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T

from models.cnn import SimpleCNN
from detection.integrity_score import compute_integrity_score, compare_and_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
test_set = torchvision.datasets.CIFAR10(root="results/data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

clean_model = SimpleCNN().to(DEVICE)
clean_model.load_state_dict(torch.load("results/clean_model.pth", map_location=DEVICE))

suspect_model = SimpleCNN().to(DEVICE)
suspect_model.load_state_dict(torch.load("results/backdoor_model.pth", map_location=DEVICE))

result = compute_integrity_score(clean_model, suspect_model, test_loader, device=DEVICE)
print(f"Integrity Score: {result['integrity_score']:.2f} / 100")
```

---

## Expected Results

After 10 epochs of training on CIFAR-10:

| Model | Clean Test Accuracy | Attack Success Rate (ASR) | Integrity Score |
|---|---|---|---|
| Clean CNN | ~72 % | < 15 % | ~85–95 |
| Backdoored CNN | ~70 % | > 90 % | ~20–45 |

> Exact values vary by run due to dataset shuffling and hardware differences.
> The key signal is the **large gap** in Integrity Score between the clean and backdoored models.

Example console output (detection run):

```
[IntegrityScore] Extracting activations from layer 'fc1'...
[IntegrityScore] Computing divergence metrics...
[IntegrityScore] Computing trigger sensitivity...

==================================================
  Divergence Score:   0.4821
  Sensitivity Score:  0.8734
  INTEGRITY SCORE:    31.78 / 100
==================================================
```

---

## Folder Structure

```
aegis-lite/
├── models/
│   ├── __init__.py
│   └── cnn.py                  # SimpleCNN for CIFAR-10
├── training/
│   ├── __init__.py
│   ├── train_clean.py          # Train reference clean model
│   └── train_backdoor.py       # Train poisoned backdoor model
├── attacks/
│   ├── __init__.py
│   └── backdoor_trigger.py     # 3×3 white-square trigger injection
├── detection/
│   ├── __init__.py
│   ├── extract_activations.py  # Forward-hook activation extraction
│   ├── divergence_metrics.py   # MAD, L2, KL divergence
│   ├── trigger_sensitivity.py  # Confidence-shift sensitivity probe
│   └── integrity_score.py      # Combined integrity score + plots
├── notebooks/
│   └── demo.ipynb              # End-to-end demonstration notebook
├── results/
│   └── visualizations/         # Auto-generated plots
├── requirements.txt
└── README.md
```

---

## Visualizations

The toolkit automatically generates the following plots in `results/visualizations/`:

| File | Description |
|---|---|
| `pca_activations.png` | 2-D PCA scatter of clean vs backdoor activations |
| `activation_distributions.png` | Per-feature histogram comparison |
| `trigger_sensitivity.png` | Confidence distribution before/after trigger |
| `integrity_score_comparison.png` | Bar chart of final integrity scores |

---

## Future Work

- **LLM backdoors**: Extend the detection framework to large language models by hooking into
  transformer attention layers and measuring token-level activation divergence under prompt
  triggers (e.g., "BadGPT"-style attacks).
- **Adaptive attacks**: Evaluate robustness of the detector against adaptive adversaries who
  minimise activation divergence during training.
- **Automated trigger reverse-engineering**: Integrate Neural Cleanse or TABOR-style optimisation
  to reconstruct the trigger pattern without prior knowledge.
- **Multi-trigger scenarios**: Support detection of models backdoored with multiple independent
  triggers targeting different classes.
- **Certification**: Explore randomised-smoothing approaches to provide provable integrity
  guarantees rather than heuristic scores.

---

## License

This project is released under the [MIT License](LICENSE).

> **Disclaimer**: This toolkit is intended solely for research and educational purposes.
> It must not be used to develop or deploy malicious models in production systems.
