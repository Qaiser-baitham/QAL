# Memristor-Aware Neural Network Training Framework

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

## Non-Interactive Examples

Quick FMNIST smoke run:

```powershell
python main.py --config configs/run-local.json --non-interactive --dataset fmnist --model CNN --mode dual --epochs 1 --batch-size 512 --raw-data data/raw_data --resume fresh
```

Paper-style CIFAR-10 direction with six-convolution `DeepCNN`:

```powershell
python main.py --config configs/run-local.json --non-interactive --dataset cifar10 --model DeepCNN --mode dual --epochs 200 --batch-size 128 --raw-data data/raw_data --resume fresh
```

More severe hardware non-ideality:

```powershell
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --epochs 20 --read-noise 0.03 --cycle-variation-scale 0.5
```

Disable the live CMD progress display for background runs:

```powershell
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --epochs 20 --no-progress
```

Explicit dual strategy selection:

```powershell
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --dual-strategy ideal_then_hardware --epochs 20
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --dual-strategy independent --epochs 20
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --dual-strategy shared_model_eval --epochs 20
```

Dual strategies:

- `ideal_then_hardware`: default. Runs the full ideal phase first, saves ideal artifacts, then runs the full hardware-aware phase, saves hardware artifacts, then exports final comparison.
- `independent`: kept as an explicit independent dual run path. It runs separate ideal and hardware-aware models sequentially.
- `shared_model_eval`: trains one ideal model and evaluates that same model under memristor effects each epoch. Use this only when you intentionally want interleaved shared-model comparison.

## Outputs

Main outputs are written under `outputs/`.

```text
outputs/
  checkpoints/   latest and best PyTorch checkpoints
  history/       ideal_history.json and hardware_aware_history.json
  plots/         PNG plots
  excel/         Origin/Excel-friendly data files
  reports/       run.log with parsing, device-model, and training details
```

Important generated files:

- `outputs/plots/training_history.png`: combined ideal-vs-memristor accuracy/loss figure.
- `outputs/excel/training_history.xlsx`: source data for the combined figure.
- `outputs/plots/01_accuracy_loss_dashboard.png`: grouped accuracy/loss evidence.
- `outputs/plots/02_generalization_dashboard.png`: train-test gap and error-rate evidence.
- `outputs/plots/03_memristor_impact_dashboard.png`: accuracy drop and loss penalty from device effects.
- `outputs/plots/04_learning_rate_dashboard.png`: learning-rate setting versus epoch/accuracy.
- `outputs/plots/05_hardware_cost_dashboard.png`: estimated energy/latency cost evidence.
- `outputs/plots/06_curve_quality_dashboard.png`: raw and smoothed accuracy/loss curves for cleaner interpretation.
- `outputs/reports/plot_explanations.md`: short justification for what each graph supports.
- `outputs/excel/plot_explanations.xlsx`: Excel version of the graph justification table.
- `outputs/reports/final_summary.md`: final model parameters, performance, optimization, hardware metrics, and quality notes.
- `outputs/excel/final_summary.xlsx`: Excel version of the final summary.
- `outputs/reports/paramInfo.md`: end-of-run epoch-by-epoch model parameter and optimization report.
- `outputs/excel/paramInfo.xlsx`: Excel version of the parameter information report.
- `outputs/plots/runtime_epoch_time.png`: continuously updated epoch-duration and elapsed-runtime graph.
- `outputs/excel/runtime_timing.csv`: runtime timing history per epoch.
- `outputs/excel/runtime_timing.xlsx`: Excel version of runtime timing history.
- `outputs/reports/runtime_timing.json`: JSON runtime timing history.
- `outputs/reports/quality_check.md`: automated artifact and data validation report.
- `outputs/excel/quality_check.xlsx`: Excel version of the quality check.
- `outputs/plots/comparison_accuracy.png`
- `outputs/plots/comparison_loss.png`
- `outputs/plots/ideal_accuracy_vs_epoch.png`
- `outputs/plots/hardware_aware_accuracy_vs_epoch.png`
- `outputs/plots/device_ltp_curve.png`
- `outputs/plots/device_ltd_curve.png`
- `outputs/plots/device_conductance_vs_pulse.png`
- `outputs/plots/device_histogram.png`
- `outputs/excel/device_state_distribution.xlsx`

Every generated PNG graph has a matching Excel file in `outputs/excel`. The dashboards group supporting plots together, while the individual plots remain available for Origin, paper figures, or deeper analysis.

Additional generated analysis includes:

- train accuracy vs epoch
- test accuracy vs epoch
- train/test accuracy together
- train/test loss together
- error rate vs epoch
- best-so-far accuracy vs epoch
- generalization gap vs epoch
- loss gap vs epoch
- learning rate vs epoch
- test accuracy vs learning rate
- test loss vs learning rate
- epoch time vs epoch
- model parameter norm vs epoch
- gradient norm vs epoch
- trainable parameters vs epoch
- ideal-vs-memristor accuracy drop
- ideal-vs-memristor loss increase
- raw and smoothed accuracy/loss curves
- annotated confusion matrices with count and percentage per cell
- hardware energy/latency vs accuracy

The learning rate is dynamic by default. `ReduceLROnPlateau` reduces LR automatically when validation loss stalls, and the LR schedule is saved in the history, plots, Excel files, and checkpoints.

Checkpoints include model state, optimizer state, scheduler state, config, history, and the extracted device model. If training is interrupted or crashes, emergency latest checkpoints are saved; continue from them with `--resume resume_latest`.

Run a final artifact quality check:

```powershell
python -m src.utils.quality_check
```

During training, the CMD screen shows a clean research dashboard:

- `iMC RESEARCH RUN` header with dataset, model, mode, dual strategy, device, input shape, memristor summary, and output directory.
- Phase banners such as `PHASE 1: IDEAL RUN`, `PHASE 2: HARDWARE RUN`, and `FINAL COMPARISON`.
- Clear runtime prefixes: `[IDEAL][TRAIN]`, `[IDEAL][TEST ]`, `[HARDWARE][TRAIN]`, `[HARDWARE][TEST ]`, `[CHECKPOINT]`, `[SAVE]`, and `[COMPARE]`.
- Timing metrics: `Time` for the current train/test or epoch step, `Elapsed` for total wall-clock runtime, `Avg` for average completed epoch time, and `ETA` for estimated remaining time in the current phase.
- One active tqdm progress bar at a time with the same label as the active phase.

The runtime graph is not opened as a GUI window by default because Windows and VS Code terminals are more stable when training runs headless. Instead, `outputs/plots/runtime_epoch_time.png` is rewritten after every completed epoch from the timing history. The compact per-epoch summary is not printed to the CMD screen; the final epoch-by-epoch parameter report is written to `outputs/reports/paramInfo.md`, and detailed configuration, model architecture, parsing logs, and formulas are still written to `outputs/reports/run.log`.

## Current Raw Data Notes

For the provided `data/raw_data/LTP.xlsx` and `data/raw_data/LTD.xlsx`, the parser now detects the unnamed conductance column as `1 / RESISTANCE`. The current extracted device model is approximately:

```text
g_on      = 0.008817 S
g_off     = 0.000245 S
ON/OFF    = 35.96
states    = 19
LTP nl    = 0.365
LTD nl    = 0.401
sigma     = 0.142
```

Because the current files appear to contain one LTP and one LTD trace, the cycle variation is an estimate rather than true repeated-cycle statistics. The default `cycle_variation_scale` is therefore conservative at `0.1`. Increase it only when you want a harsher hardware-degradation experiment.

## Model Guidance

Use `CNN` for fast FMNIST/MNIST runs. Use `DeepCNN` for CIFAR-10-style experiments because it contains six convolution layers followed by two fully connected layers, matching the general architecture direction used in the reference paper.

Compatibility is checked automatically. Unknown datasets/models raise an error instead of silently falling back.

## Resume

Each mode writes latest and best checkpoints:

```text
outputs/checkpoints/ideal_latest.pt
outputs/checkpoints/ideal_best.pt
outputs/checkpoints/hardware_aware_latest.pt
outputs/checkpoints/hardware_aware_best.pt
```

Resume examples:

```powershell
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --epochs 50 --resume resume_latest
python main.py --non-interactive --dataset fmnist --model CNN --mode dual --epochs 50 --resume resume_best
```

## Full Usage Guide

For a complete A-to-Z operating guide, see `README_USAGE.md`. It covers:

- every supported dataset and model pair
- auto-selected recipes and training hyperparameters
- all calculated metrics
- every generated plot category
- output folder layout
- accuracy tips and recommended experiment choices
- resume and checkpoint behavior

## Scope

This is a transparent PyTorch-side behavioral simulator. It applies memristor-inspired finite conductance states, quantization, cycle/read noise, optional stuck-at faults, activation quantization, ADC quantization, and simple energy/latency estimates. It is not a drop-in replacement for the full C++ NeuroSim backend, but it follows the same research workflow shape: measured device curves, algorithm training/evaluation, and separate hardware-facing metrics.
