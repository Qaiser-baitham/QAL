# iMC Usage Guide - A to Z

This file is a practical, end-user reference for the iMC (in-memory-compute) training framework. It explains *what the project can calculate*, *what plots/graphs it can produce*, *how to choose a dataset and model*, and *how the outputs are organized*. For architecture/internals see `README.md`.

---

## 1. What this project does

The framework trains a neural network twice on the same dataset:

1. An **ideal** software-floating-point run (no hardware effects).
2. A **hardware-aware (memristor)** run that applies weight quantization, conductance-range mapping, read noise, cycle variation, stuck-at faults, and ADC quantization of logits - parameters derived from the raw Ag/GeTe/Pt memristor measurements in `data/raw`.

The two runs are then compared, and the framework exports plots, Excel tables, checkpoints, and Markdown reports that describe training behaviour, hardware impact, per-class accuracy, energy/latency, and device characterization.

---

## 2. Datasets, models, and what changes under the hood

### 2.1 Supported datasets

| Dataset       | Image size | Channels | Classes | Class labels used in plots                |
|---------------|------------|----------|---------|-------------------------------------------|
| `MNIST`       | 28x28      | 1        | 10      | `0`..`9`                                  |
| `FMNIST`      | 28x28      | 1        | 10      | `T-shirt/top`, `Trouser`, `Pullover`, ... |
| `KMNIST`      | 28x28      | 1        | 10      | Kuzushiji syllables `o`, `ki`, `su`, ...  |
| `EMNIST`      | 28x28      | 1        | 47      | digits + letters (balanced split)         |
| `CIFAR10`     | 32x32      | 3        | 10      | `airplane`, `automobile`, ...             |
| `CIFAR100`    | 32x32      | 3        | 100     | full 100-class list                       |
| `SVHN`        | 32x32      | 3        | 10      | `0`..`9`                                  |
| `TinyImageNet`| 64x64      | 3        | 200     | `class_000`..`class_199`                  |

Class labels are pulled from `src/data_loader/datasets.CLASS_NAMES`. The confusion matrix, per-class F1 bar, per-class accuracy bar, top-confused-pair bar and class-wise ideal-vs-hardware comparison all use these real labels instead of numeric indices.

### 2.2 Supported models (and what the framework will allow per dataset)

| Dataset        | Allowed models                                                  |
|----------------|-----------------------------------------------------------------|
| MNIST / KMNIST | MLP, ANN, CNN, LeNet, SNN                                       |
| FMNIST         | MLP, ANN, CNN, DeepCNN, LeNet, SNN                              |
| EMNIST         | MLP, ANN, CNN, DeepCNN, LeNet                                   |
| CIFAR10/100    | CNN, DeepCNN, VGGSmall, VGG11, VGG16, ResNet18, ResNet34        |
| SVHN           | CNN, DeepCNN, VGGSmall, VGG11, ResNet18                         |
| TinyImageNet   | DeepCNN, VGGSmall, VGG11, ResNet18, ResNet34                    |

### 2.3 Dataset-aware training-time augmentation

`create_loaders` applies different `torchvision.transforms` depending on the dataset so the ideal accuracy is actually close to published baselines:

- **CIFAR10 / CIFAR100**: `RandomCrop(pad=4, reflect) + HorizontalFlip + ColorJitter` + normalize.
- **SVHN**: crop + pad (no flip - 6 and 9 must not be mirrored).
- **TinyImageNet**: crop-from-72 + flip + color jitter.
- **FMNIST / KMNIST**: mild reflect-padded crop (no flip).
- **MNIST / EMNIST**: no augmentation (preserves digit identity).

Evaluation loaders always use plain resize + normalize.

### 2.4 Auto-selected hyperparameters per (dataset, model)

The CLI looks up the chosen (dataset, model) in a table of best-known recipes (`RECIPES` in `src/cli.py`) and uses those values for batch size, learning rate, weight decay and target accuracy. The target accuracy values are set from published baselines for that architecture, e.g.:

- FMNIST + DeepCNN -> target ~0.94
- CIFAR10 + ResNet18 -> target ~0.93
- CIFAR10 + VGG16 -> target ~0.925
- CIFAR100 + ResNet34 -> target ~0.76
- TinyImageNet + ResNet34 -> target ~0.60

For deep image runs (CIFAR/TinyImageNet + ResNet/VGG) with epochs >= 20, the scheduler is automatically switched to `CosineAnnealingWarmRestarts`; otherwise `ReduceLROnPlateau` is used. Harder datasets (`CIFAR100`, `TinyImageNet`) also bump `adc_bits` to 7 to preserve logit resolution under quantization.

If a (dataset, model) pair is not in the table, the framework falls back to coarse heuristics.

---

## 3. How to run

### Interactive

```powershell
python main.py
```

You are asked for three things only:

1. Dataset (e.g. `cifar10`)
2. Model (from the compatible list the framework prints)
3. Max epochs

Everything else - batch size, LR, weight decay, target accuracy, scheduler, ADC bits - is filled in by the recipe for that pair.

### Non-interactive

```powershell
python main.py --config configs/default.json --non-interactive \
    --dataset cifar10 --model ResNet18 --mode dual --epochs 80
```

Optional overrides: `--batch-size`, `--learning-rate`, `--weight-decay`, `--weight-bits`, `--activation-bits`, `--adc-bits`, `--read-noise`, `--cycle-variation-scale`, `--target-accuracy`, `--early-stopping-patience`, `--gradient-clip`, `--compare-models`, `--comparison-epochs`, `--run-dse`.

### Resume a run

```powershell
python main.py --non-interactive --dataset fmnist --model DeepCNN --mode dual --epochs 30 --resume resume_best
```

---

## 4. What the framework calculates

### Per epoch, per run mode (ideal / hardware-aware)

- Training loss, training accuracy
- Test loss, test accuracy
- Best-so-far test accuracy
- Train-test generalization gap (accuracy + loss)
- Test error rate (1 - accuracy)
- Confusion matrix (raw counts and row-normalized)
- Per-class precision, recall, F1 and support
- Macro-averaged and weighted-averaged precision/recall/F1
- Per-class accuracy (diagonal of normalized confusion)
- Top-K most-confused class pairs
- Learning rate and LR changes
- Parameter-norm and gradient-norm trajectories
- Trainable-parameter count
- Epoch duration, cumulative elapsed time, ETA, per-mode runtime timing table

### For hardware-aware runs only

- MACs per sample (hook-counted)
- Estimated energy per epoch (pJ) using C*V^2 line model + ADC energy
- Estimated inference latency (ns) using RC + clock-period model
- Energy-per-MAC (pJ)
- Throughput proxy (samples/second)
- TOPS/W proxy
- Energy-delay product (pJ*ns)
- ADC / weight bit width used

### Dual-mode comparison

- Per-class accuracy and F1 gap between ideal and hardware
- Confusion-matrix shift heatmap (hardware - ideal, row-normalized)
- Accuracy-drop vs epoch, loss-increase vs epoch
- Accuracy-per-energy and accuracy-per-latency efficiency curves

### Device characterization (from raw LTP/LTD measurements)

- g_on, g_off, ON/OFF ratio, dynamic range (dB)
- Number of conductance states, state means and stds
- LTP and LTD nonlinearity, symmetry index
- Cycle variation sigma, endurance stability
- Polynomial fits (degree 3) of raw LTP/LTD
- Moving-average smoothed LTP/LTD curves

### Optional: design-space exploration (`--run-dse` or always in `dual` mode)

Sweeps key knobs (weight bits, ADC bits, noise, array size) and writes a sweep Excel.

---

## 5. Plots produced (by category)

All plots are saved as PNG in `outputs/<run_id>/plots/` and mirrored as Excel tables in `outputs/<run_id>/excel/`.

### Device plots (one set per run)

- `device_conductance_vs_pulse.png`
- `device_ltp_curve.png`, `device_ltd_curve.png`
- `device_ltp_fitted.png`, `device_ltd_fitted.png` (raw + MA-5 smoothed + polynomial fit)
- `device_characterization_dashboard.png` (6-panel: LTP, LTD, normalized update, state bars, metrics, histogram)
- `device_histogram.png`

### Training curves (per mode: `ideal` / `hardware_aware`)

- `<mode>_accuracy_vs_epoch.png`, `<mode>_loss_vs_epoch.png`
- `<mode>_train_test_accuracy_vs_epoch.png`, `<mode>_train_test_loss_vs_epoch.png`
- `<mode>_generalization_gap_vs_epoch.png`, `<mode>_loss_gap_vs_epoch.png`
- `<mode>_test_error_rate_vs_epoch.png`
- `<mode>_best_accuracy_vs_epoch.png`
- `<mode>_learning_rate_vs_epoch.png`
- `<mode>_test_accuracy_vs_learning_rate.png`, `<mode>_test_loss_vs_learning_rate.png`
- `<mode>_epoch_time_vs_epoch.png`
- `<mode>_parameter_norm_vs_epoch.png`, `<mode>_gradient_norm_vs_epoch.png`
- `<mode>_trainable_parameters_vs_epoch.png`

### Per-class diagnostics (per mode)

- `<mode>_confusion_matrix.png` - with dataset-aware class labels
- `<mode>_confusion_matrix_normalized.png` - row-normalized version
- `<mode>_class_metrics.png` - per-class F1 bar, color-coded by quality
- `<mode>_per_class_accuracy.png` - per-class accuracy (recall) bar
- `<mode>_top_confused_pairs.png` - top K (true -> predicted) confusions

### Comparison (dual mode only)

- `training_history.png` - main figure, train + test accuracy and loss for both modes
- `comparison_accuracy.png`, `comparison_loss.png`
- `comparison_accuracy_smoothed.png`, `comparison_loss_smoothed.png`
- `comparison_best_accuracy_vs_epoch.png`, `comparison_error_rate_vs_epoch.png`
- `comparison_accuracy_drop_vs_epoch.png`, `comparison_loss_increase_vs_epoch.png`
- `comparison_epoch_time_vs_epoch.png`
- Dashboards: `01_accuracy_loss_dashboard.png`, `02_generalization_dashboard.png`, `03_memristor_impact_dashboard.png`, `04_learning_rate_dashboard.png`, `05_hardware_cost_dashboard.png`, `06_curve_quality_dashboard.png`
- `class_wise_comparison.png` - per-class ideal vs hardware accuracy + drop
- `confusion_matrix_diff_ideal_vs_hardware.png` - row-normalized (hw - ideal) heatmap

### Hardware / efficiency plots

- `hardware_energy_vs_accuracy.png`
- `hardware_latency_vs_accuracy.png`
- `hardware_energy_delay_product.png`
- `hardware_throughput_vs_epoch.png`
- `hardware_tops_w_vs_epoch.png`
- `runtime_epoch_time.png`

### Weight / distribution plots

- `ai_weight_distribution.png` (legacy layout)

---

## 6. Output layout

```
outputs/
|-- ideal/
|   |-- logs/            # best/latest checkpoints, hw metrics JSON, trace CSVs
|   |-- plots/           # PNGs
|   `-- excel/           # mirrored Excel for every plot
|-- hardware/
|   `-- ...
|-- reports/
|   |-- run.log
|   |-- final_summary.md
|   |-- paramInfo.md
|   |-- plot_explanations.md
|   `-- artifact_index.md
`-- excel/
    |-- comparison_ideal_vs_hardware.xlsx
    |-- training_history.xlsx
    |-- paramInfo.xlsx
    |-- runtime_timing.xlsx
    |-- artifact_index.xlsx
    `-- ...
```

`final_summary.md` and `plot_explanations.md` are plain Markdown and can be read without any tools. `paramInfo.xlsx` and `paramInfo.md` give you the full per-epoch parameter history for both modes.

---

## 7. Tips for best possible accuracy

- **Pick the right model for the dataset** - the CLI restricts you to compatible pairs, but within those use the deeper variants for CIFAR/TinyImageNet (ResNet18/ResNet34, VGG16).
- **Use at least 40-80 epochs for CIFAR/TinyImageNet** - the cosine scheduler needs room to decay.
- **Don't override the recipe values unless you know what you're doing** - the defaults come from published baselines and the auto-augmentation settings.
- **Use `--compare-models` first** to quickly estimate which architecture will work best for a budget of `--comparison-epochs`.
- **Increase `--adc-bits` to 8 and reduce `--read-noise` to 0.005** to see the "best-case" hardware ceiling; the defaults are deliberately conservative.

---

## 8. Quick decision table

| I want to ...                                  | Use this                                                    |
|------------------------------------------------|-------------------------------------------------------------|
| Sanity-check the full pipeline                 | `FMNIST` + `CNN` + 3 epochs, `--mode dual`                  |
| Best FMNIST number                             | `FMNIST` + `DeepCNN` + 40 epochs                            |
| Publishable CIFAR10 comparison                 | `CIFAR10` + `ResNet18` + 80 epochs (cosine auto-selected)   |
| Hardest run (100-class)                        | `CIFAR100` + `ResNet34` + 100 epochs                        |
| Minimize energy on device                      | drop `--weight-bits`, drop `--adc-bits`, watch EDP plot     |
| Understand per-class degradation from device   | `class_wise_comparison.png` + `confusion_matrix_diff*.png`  |
