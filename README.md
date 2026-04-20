# Memristor-Aware Neural Network Training Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

This project is a generalized research workflow for training ideal and memristor-aware neural networks from raw LTP/LTD device measurements. It is designed to produce publication-style outputs such as accuracy/loss versus epoch plots, confusion matrices, device conductance curves, Excel tables, checkpoints, and a combined `training_history.png` similar to the reference training-history figure.

The goal is not to exactly reproduce one paper figure. The paper/manuals are used as guidance for the expected workflow: extract device conductance behavior, choose a dataset/model/epoch count, run an ideal neural network and a memristor-effect evaluation, then export clean graphs for comparison.

## Main Workflow

1. Read raw device files from `data/raw_data` or a path selected by the user.
2. Detect resistance, conductance, current, voltage, and pulse columns from Excel/CSV/TXT tables.
3. Convert resistance to conductance when the direct conductance column is missing or unnamed.
4. Classify traces as LTP, LTD, IV, or unknown.
5. Extract device statistics:
   - `g_on`
   - `g_off`
   - ON/OFF ratio
   - conductance states
   - LTP/LTD nonlinearity
   - cycle/read variation estimate
6. Ask the user only for dataset, compatible model, and epochs; choose the rest from sensible automatic defaults.
7. Train/evaluate in one of three modes:
   - `ideal`: normal PyTorch training.
   - `hardware_aware`: trains with quantization/noise/ADC effects.
   - `dual`: runs ideal and hardware-aware training/evaluation, then exports comparison graphs.
8. Export plots, Excel files, checkpoints, and JSON histories every epoch.

## Recommended Environment

The project is intended to run in your existing `snn_data` conda environment.

```powershell
conda activate snn_data
python main.py
```

If dependencies are missing:

```powershell
pip install -r requirements.txt
```

## Interactive Run

```powershell
python main.py
```

Interactive mode asks for the key research choices:

- Dataset: `MNIST`, `FMNIST`, `CIFAR10`, `CIFAR100`, `KMNIST`, `EMNIST`, `SVHN`, `TinyImageNet`
- Compatible model for the selected dataset
- Epoch count

Everything else is selected automatically from sensible research defaults:

- `dual` training mode
- `ideal_then_hardware` dual strategy by default: full ideal run first, then full hardware-aware run, then final comparison
- target accuracy `95%`; training stops when ideal or memristor test accuracy reaches it
- batch size
- learning rate with automatic `ReduceLROnPlateau` scheduling
- weight decay
- early stopping patience
- hardware precision/noise settings
- live progress display
- raw data path `data/raw_data`

Advanced overrides are still available through non-interactive CLI flags.

## Supported Datasets and Models

The CLI validates dataset/model compatibility before training starts.

| Dataset | Compatible models |
|---|---|
| `MNIST`, `KMNIST` | `MLP`, `ANN`, `CNN`, `LeNet`, `SNN` |
| `FMNIST` | `MLP`, `ANN`, `CNN`, `DeepCNN`, `LeNet`, `SNN` |
| `EMNIST` | `MLP`, `ANN`, `CNN`, `DeepCNN`, `LeNet` |
| `CIFAR10`, `CIFAR100` | `CNN`, `DeepCNN`, `VGGSmall`, `VGG11`, `VGG16`, `ResNet18`, `ResNet34` |
| `SVHN` | `CNN`, `DeepCNN`, `VGGSmall`, `VGG11`, `ResNet18` |
| `TinyImageNet` | `DeepCNN`, `VGGSmall`, `VGG11`, `ResNet18`, `ResNet34` |

Dataset-aware transforms are selected automatically. CIFAR/TinyImageNet runs use augmentation such as crop, flip, and color jitter; digit datasets avoid transforms that would change label identity.

## Quick Decision Table

| Goal | Suggested run |
|---|---|
| Full pipeline sanity check | `FMNIST` + `CNN` + 3 epochs, `--mode dual` |
| Strong FMNIST result | `FMNIST` + `DeepCNN` + 40 epochs |
| CIFAR-10 comparison | `CIFAR10` + `ResNet18` + 80 epochs |
| Harder 100-class run | `CIFAR100` + `ResNet34` + 100 epochs |
| Hardware degradation study | Start with defaults, then vary `--read-noise`, `--weight-bits`, and `--adc-bits` |
| Per-class degradation analysis | Use `class_wise_comparison.png` and `confusion_matrix_diff_ideal_vs_hardware.png` |
