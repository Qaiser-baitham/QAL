"""Phase 3: Evidence-based model comparison.

Runs a controlled quick-evaluation of multiple model architectures
on the same dataset with identical hyperparameters, then exports
a comparison table so model selection is data-driven, not assumed.

Usage in CLI:
    python main.py --non-interactive --compare-models --dataset FMNIST --epochs 10
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd


def run_model_comparison(cfg: dict, device_model, paths: dict[str, Path]) -> dict:
    """Run a quick comparison of compatible models and return ranked results."""
    import torch
    from torch import nn, optim
    from src.data_loader.datasets import create_loaders
    from src.models.factory import create_model, COMPATIBILITY

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = cfg["dataset"]
    compatible = COMPATIBILITY.get(dataset, [])
    # Select representative subset for comparison
    candidates = []
    for model_name in ["MLP", "CNN", "LeNet", "DeepCNN", "VGGSmall"]:
        if model_name in compatible:
            candidates.append(model_name)

    if not candidates:
        logging.warning("No compatible models found for %s", dataset)
        return {}

    epochs = min(int(cfg.get("comparison_epochs", 10)), int(cfg.get("max_epochs", 20)))
    batch_size = int(cfg.get("batch_size", 128))
    lr = float(cfg.get("learning_rate", 0.001))

    train_loader, test_loader, spec = create_loaders(
        dataset, cfg.get("data_root", "datasets"), batch_size,
        int(cfg.get("num_workers", 0)), device,
    )

    results = []
    logging.info("=" * 60)
    logging.info("MODEL COMPARISON: %s on %s (%d epochs each)", candidates, dataset, epochs)
    logging.info("=" * 60)

    for model_name in candidates:
        logging.info("--- Evaluating %s ---", model_name)
        model = create_model(model_name, dataset, spec).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 1e-4)))
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        final_acc = 0.0
        final_loss = 0.0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            total_loss = 0.0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    total_loss += loss.item() * y.size(0)
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
            final_acc = correct / max(total, 1)
            final_loss = total_loss / max(total, 1)
            best_acc = max(best_acc, final_acc)

        elapsed = time.time() - start_time
        logging.info("%s: final_acc=%.4f, best_acc=%.4f, params=%d, time=%.1fs",
                     model_name, final_acc, best_acc, n_params, elapsed)

        results.append({
            "model": model_name,
            "parameters": n_params,
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "final_loss": final_loss,
            "training_time_s": elapsed,
            "epochs": epochs,
            "accuracy_per_param": final_acc / max(n_params, 1) * 1e6,  # acc per million params
        })

    # Sort by best accuracy
    results.sort(key=lambda r: r["best_accuracy"], reverse=True)

    # Export
    out_dir = paths.get("excel", paths.get("root", Path("outputs/excel")))
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_excel(out_dir / "model_comparison.xlsx", index=False)

    # Log recommendation
    if results:
        best = results[0]
        logging.info("=" * 60)
        logging.info("RECOMMENDATION: %s (best_acc=%.4f, params=%d)", best["model"], best["best_accuracy"], best["parameters"])
        # Check if simpler model is within 1% of best
        for r in results[1:]:
            if r["parameters"] < best["parameters"] and (best["best_accuracy"] - r["best_accuracy"]) < 0.01:
                logging.info(
                    "NOTE: %s is within 1%% of best with fewer params (%d vs %d). "
                    "Consider using it for lower hardware cost.",
                    r["model"], r["parameters"], best["parameters"]
                )
        logging.info("=" * 60)

    return {"comparison_results": results}
