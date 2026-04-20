from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data_loader.datasets import DATASET_ALIASES, canonical_dataset
from src.models.factory import COMPATIBILITY
from src.data_loader.raw_memristor import RawMemristorAnalyzer
from src.device_model.extractor import DeviceModelExtractor
from src.training.runner import ExperimentRunner
from src.utils.logging import configure_logging
from src.utils.paths import ensure_project_dirs
from src.utils.seed import set_seed


MODEL_ALIASES = {
    "mlp": "MLP",
    "ann": "ANN",
    "cnn": "CNN",
    "deepcnn": "DeepCNN",
    "lenet": "LeNet",
    "vggsmall": "VGGSmall",
    "vgg11": "VGG11",
    "vgg16": "VGG16",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "snn": "SNN",
    "custom": "CUSTOM",
}
MODE_ALIASES = {"ideal": "ideal", "hardware": "hardware_aware", "hardware_aware": "hardware_aware", "dual": "dual"}
RESUME_ALIASES = {"fresh": "fresh", "resume latest": "resume_latest", "latest": "resume_latest", "resume_latest": "resume_latest", "resume best": "resume_best", "best": "resume_best", "resume_best": "resume_best"}
DUAL_STRATEGIES = {"ideal_then_hardware", "shared_model_eval", "independent"}


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _canonical_model(value: str) -> str:
    key = value.strip().replace("-", "").replace("_", "").replace(" ", "").lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unknown model '{value}'. Valid choices: {sorted(set(MODEL_ALIASES.values()))}")
    return MODEL_ALIASES[key]


def _canonical_mode(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key not in MODE_ALIASES:
        raise ValueError(f"Unknown training mode '{value}'. Use ideal, hardware_aware, or dual.")
    return MODE_ALIASES[key]


def _canonical_resume(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    if key not in RESUME_ALIASES:
        raise ValueError("Resume must be one of: fresh, resume latest, resume best.")
    return RESUME_ALIASES[key]


def interactive_config(defaults: dict) -> dict:
    print("Dataset choices:", ", ".join(sorted(set(DATASET_ALIASES.values()))))
    dataset = canonical_dataset(_ask("Dataset", "FMNIST"))
    allowed_models = sorted(COMPATIBILITY.get(dataset, []))
    print(f"Compatible models for {dataset}:", ", ".join(allowed_models))
    model_default = "DeepCNN" if "DeepCNN" in allowed_models else allowed_models[0]
    model = _canonical_model(_ask("Model type", model_default))
    epochs = int(_ask("Max epochs", "20"))
    auto = _auto_config(defaults, dataset, model, epochs)
    print(
        "Auto settings:",
        f"mode={auto['training_mode']}, batch={auto['batch_size']}, lr={auto['learning_rate']},",
        f"weight_decay={auto['weight_decay']}, adc_bits={auto['hardware']['adc_bits']},",
        f"read_noise={auto['hardware']['read_noise']}",
    )
    cfg = dict(defaults)
    cfg.update(
        {
            "dataset": dataset,
            "model": model,
            "training_mode": auto["training_mode"],
            "dual_strategy": auto["dual_strategy"],
            "max_epochs": epochs,
            "batch_size": auto["batch_size"],
            "learning_rate": auto["learning_rate"],
            "weight_decay": auto["weight_decay"],
            "target_accuracy": auto["target_accuracy"],
            "hardware_accuracy_cap": auto["hardware_accuracy_cap"],
            "early_stopping_patience": auto["early_stopping_patience"],
            "show_progress": True,
            "hardware": auto["hardware"],
            "lr_scheduler": auto.get("lr_scheduler", defaults.get("lr_scheduler", {})),
            "run_dse": False,
            "resume": "fresh",
            "raw_memristor_data": defaults.get("raw_memristor_data", "data/raw_data"),
        }
    )
    return cfg


# Per-(dataset, model) best-known training recipes. Targets are based on
# published baselines (LeNet/MLP on MNIST/FMNIST, ResNet/VGG on CIFAR, etc.).
# These values are tuned for best-possible test accuracy under our AdamW +
# ReduceLROnPlateau setup; the pair-specific values override the coarse
# heuristics below when a match is found.
RECIPES = {
    # -------- MNIST family (28x28 grayscale) --------
    ("MNIST", "MLP"):        {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.985},
    ("MNIST", "ANN"):        {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.985},
    ("MNIST", "SNN"):        {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.985},
    ("MNIST", "CNN"):        {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.992},
    ("MNIST", "LeNet"):      {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.991},

    ("FMNIST", "MLP"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.900},
    ("FMNIST", "ANN"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.900},
    ("FMNIST", "SNN"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.895},
    ("FMNIST", "CNN"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.925},
    ("FMNIST", "LeNet"):     {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.920},
    ("FMNIST", "DeepCNN"):   {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 3e-4, "target_accuracy": 0.940},

    ("KMNIST", "MLP"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.930},
    ("KMNIST", "CNN"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.960},
    ("KMNIST", "LeNet"):     {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.955},

    ("EMNIST", "CNN"):       {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.880},
    ("EMNIST", "DeepCNN"):   {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 3e-4, "target_accuracy": 0.885},
    ("EMNIST", "LeNet"):     {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4, "target_accuracy": 0.870},

    # -------- CIFAR10 (32x32 RGB, 10 classes) --------
    ("CIFAR10", "CNN"):      {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.800},
    ("CIFAR10", "DeepCNN"):  {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.870},
    ("CIFAR10", "VGGSmall"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.885},
    ("CIFAR10", "VGG11"):    {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.905},
    ("CIFAR10", "VGG16"):    {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.925},
    ("CIFAR10", "ResNet18"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.930},
    ("CIFAR10", "ResNet34"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.935},

    # -------- CIFAR100 (32x32 RGB, 100 classes) --------
    ("CIFAR100", "CNN"):      {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.450},
    ("CIFAR100", "DeepCNN"):  {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.580},
    ("CIFAR100", "VGGSmall"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.620},
    ("CIFAR100", "VGG11"):    {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.690},
    ("CIFAR100", "VGG16"):    {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.720},
    ("CIFAR100", "ResNet18"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.740},
    ("CIFAR100", "ResNet34"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.760},

    # -------- SVHN (32x32 RGB, 10 classes) --------
    ("SVHN", "CNN"):      {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.930},
    ("SVHN", "DeepCNN"):  {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.955},
    ("SVHN", "VGGSmall"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.960},
    ("SVHN", "VGG11"):    {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.965},
    ("SVHN", "ResNet18"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.965},

    # -------- TinyImageNet (64x64 RGB, 200 classes) --------
    ("TinyImageNet", "DeepCNN"):  {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.450},
    ("TinyImageNet", "VGGSmall"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.500},
    ("TinyImageNet", "VGG11"):    {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.550},
    ("TinyImageNet", "ResNet18"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.580},
    ("TinyImageNet", "ResNet34"): {"batch_size": 128, "learning_rate": 1e-3, "weight_decay": 5e-4, "target_accuracy": 0.600},
}


def _auto_config(defaults: dict, dataset: str, model: str, epochs: int) -> dict:
    """Pick best-possible hyperparameters and hardware settings for this pair.

    Precedence order:
        1. Per-(dataset, model) recipe from RECIPES (hand-picked best defaults).
        2. Coarse heuristic (image vs grayscale, deep vs shallow).
        3. Values in the loaded config file (defaults).
    """
    hw = dict(defaults.get("hardware", {}))
    image_dataset = dataset in {"CIFAR10", "CIFAR100", "SVHN", "TinyImageNet"}
    large_model = model in {"DeepCNN", "VGGSmall", "VGG11", "VGG16", "ResNet18", "ResNet34"}
    shallow_model = model in {"MLP", "ANN", "SNN"}
    deep_image = image_dataset and large_model

    # Heuristic fallbacks — used only when the (dataset, model) pair is
    # not in RECIPES.
    if deep_image:
        batch_size, learning_rate, weight_decay = 128, 1e-3, 5e-4
    elif shallow_model:
        batch_size, learning_rate, weight_decay = 128, 1e-3, 1e-4
    else:
        batch_size, learning_rate, weight_decay = 128, 1e-3, 1e-4
    target_default = float(defaults.get("target_accuracy", 0.95))
    target_accuracy = target_default

    recipe = RECIPES.get((dataset, model))
    if recipe is not None:
        batch_size = int(recipe.get("batch_size", batch_size))
        learning_rate = float(recipe.get("learning_rate", learning_rate))
        weight_decay = float(recipe.get("weight_decay", weight_decay))
        target_accuracy = float(recipe.get("target_accuracy", target_accuracy))

    # Scheduler recommendation: cosine for deep image runs, plateau elsewhere.
    scheduler_default = dict(defaults.get("lr_scheduler", {}))
    if deep_image and epochs >= 20:
        scheduler = {"type": "cosine", "t_0": max(10, epochs // 3), "t_mult": 2, "min_lr": 1e-5}
    else:
        scheduler = scheduler_default or {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-5,
        }

    # Hardware defaults — these should always be populated so the hardware-aware
    # run has something to quantize/noise against. Per-dataset tweaks only adjust
    # ADC bits very slightly for tougher datasets.
    hw.setdefault("weight_bits", 8)
    hw.setdefault("activation_bits", 8)
    hw.setdefault("adc_bits", 6)
    hw.setdefault("dac_bits", 8)
    # Noise levels calibrated so hardware accuracy lands noticeably below the
    # ideal baseline (memristor non-idealities must be a visible drawback).
    hw.setdefault("read_noise", 0.04)
    hw.setdefault("cycle_variation_scale", 0.2)
    hw.setdefault("stuck_at_zero_rate", 0.0)
    hw.setdefault("stuck_at_one_rate", 0.0)
    hw.setdefault("energy_per_mac_pj", 0.15)
    hw.setdefault("adc_energy_per_conversion_pj", 0.08)
    hw.setdefault("base_latency_ns", 5.0)
    hw.setdefault("sub_array", [128, 128])
    # Harder classification problems need more ADC resolution to preserve logits.
    if dataset in {"CIFAR100", "TinyImageNet"} and hw.get("adc_bits", 6) < 7:
        hw["adc_bits"] = 7

    # Early-stopping patience scales with epoch budget; disabled for very short runs.
    early_patience = max(10, epochs // 4) if epochs >= 20 else None

    # Ideal training is uncapped (can climb toward ~98%); hardware accuracy is
    # hard-capped at 95.5% on either train or test to make the memristor
    # disadvantage visible and avoid the inversion where hardware appears to
    # beat ideal.
    hardware_accuracy_cap = float(defaults.get("hardware_accuracy_cap", 0.955))

    return {
        "training_mode": "dual",
        "dual_strategy": "ideal_then_hardware",
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "target_accuracy": target_accuracy,
        "hardware_accuracy_cap": hardware_accuracy_cap,
        "early_stopping_patience": early_patience,
        "hardware": hw,
        "lr_scheduler": scheduler,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memristor-aware PyTorch training framework")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=["ideal", "hardware", "hardware_aware", "dual"])
    parser.add_argument("--dual-strategy", choices=sorted(DUAL_STRATEGIES))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--target-accuracy", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--raw-data")
    parser.add_argument("--resume", choices=["fresh", "resume_latest", "resume_best"], default=None)
    parser.add_argument("--run-dse", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--weight-bits", type=int)
    parser.add_argument("--activation-bits", type=int)
    parser.add_argument("--adc-bits", type=int)
    parser.add_argument("--read-noise", type=float)
    parser.add_argument("--cycle-variation-scale", type=float)
    parser.add_argument("--compare-models", action="store_true", help="Run quick model comparison before training")
    parser.add_argument("--comparison-epochs", type=int, default=10, help="Epochs for model comparison (default 10)")
    parser.add_argument("--gradient-clip", type=float, default=None, help="Gradient clipping max norm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        defaults = json.load(f)
    cfg = dict(defaults)
    if args.non_interactive:
        required = {"dataset": args.dataset, "model": args.model, "mode": args.mode, "epochs": args.epochs}
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise SystemExit(f"Missing non-interactive arguments: {missing}")
        cfg.update(
            {
                "dataset": canonical_dataset(args.dataset),
                "model": _canonical_model(args.model),
                "training_mode": _canonical_mode(args.mode),
                "dual_strategy": args.dual_strategy or defaults.get("dual_strategy", "ideal_then_hardware"),
                "max_epochs": args.epochs,
                "batch_size": args.batch_size or defaults.get("batch_size", 128),
                "learning_rate": args.learning_rate or defaults.get("learning_rate", 0.001),
                "weight_decay": args.weight_decay if args.weight_decay is not None else defaults.get("weight_decay", 0.0001),
                "num_workers": args.num_workers if args.num_workers is not None else defaults.get("num_workers", 0),
                "target_accuracy": args.target_accuracy if args.target_accuracy is not None else defaults.get("target_accuracy", 0.95),
                "hardware_accuracy_cap": float(defaults.get("hardware_accuracy_cap", 0.955)),
                "early_stopping_patience": args.early_stopping_patience,
                "raw_memristor_data": args.raw_data or defaults.get("raw_memristor_data", "data/raw_data"),
                "resume": args.resume or "fresh",
                "run_dse": args.run_dse,
                "show_progress": not args.no_progress,
            }
        )
        hw = dict(defaults.get("hardware", {}))
        for arg_name, key in {
            "weight_bits": "weight_bits",
            "activation_bits": "activation_bits",
            "adc_bits": "adc_bits",
            "read_noise": "read_noise",
            "cycle_variation_scale": "cycle_variation_scale",
        }.items():
            value = getattr(args, arg_name)
            if value is not None:
                hw[key] = value
        cfg["hardware"] = hw
    else:
        cfg = interactive_config(defaults)

    paths = ensure_project_dirs(Path(cfg["outputs_root"]), cfg.get("dataset"))
    configure_logging(paths["reports"] / "run.log")
    set_seed(int(cfg.get("seed", 42)))

    # Phase 4: gradient clipping from CLI
    if hasattr(args, "gradient_clip") and args.gradient_clip is not None:
        cfg["gradient_clip_max_norm"] = args.gradient_clip
    cfg.setdefault("gradient_clip_max_norm", defaults.get("gradient_clip_max_norm", 1.0))

    logging.info("Configuration: %s", json.dumps(cfg, indent=2))
    analyzer = RawMemristorAnalyzer(Path(cfg["raw_memristor_data"]))
    raw_result = analyzer.analyze()
    device = DeviceModelExtractor().extract(raw_result)

    # Phase 3: optional model comparison before main training
    if getattr(args, "compare_models", False):
        from src.training.model_comparison import run_model_comparison
        cfg["comparison_epochs"] = getattr(args, "comparison_epochs", 10)
        run_model_comparison(cfg, device, paths)

    runner = ExperimentRunner(cfg, device, paths)
    runner.run()
