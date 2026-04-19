"""
Publication-grade plotting utilities.
Every plot is saved both as PNG and as an Excel workbook containing the
underlying numerical data.
"""
from __future__ import annotations
import os
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.io import save_excel

sns.set_theme(style="whitegrid", context="paper")
_FIG = {"dpi": 200, "bbox_inches": "tight"}


def _save_pair(fig, plots_dir, excel_dir, name, sheets: Dict[str, pd.DataFrame]):
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(excel_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, f"{name}.png"), **_FIG)
    plt.close(fig)
    save_excel(sheets, os.path.join(excel_dir, f"{name}.xlsx"))


# ---------------- DEVICE LEVEL ----------------
def plot_conductance_vs_pulse(traces, plots_dir, excel_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    sheets = {}
    for t in traces:
        if t.conductance is None: continue
        x = t.pulse if t.pulse is not None else np.arange(len(t.conductance))
        ax.plot(x, t.conductance, label=t.name[:30], lw=1.1)
        sheets[t.name[:30]] = pd.DataFrame({"pulse": x, "G": t.conductance})
    ax.set_xlabel("Pulse / Cycle #"); ax.set_ylabel("Conductance (G)")
    ax.set_title("Conductance vs Pulse"); ax.legend(fontsize=6, ncol=2)
    _save_pair(fig, plots_dir, excel_dir, "device_conductance_vs_pulse", sheets)


def plot_ltp_ltd(traces, plots_dir, excel_dir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sheets = {}
    for t in traces:
        if t.ltp is not None and t.ltp.size > 1:
            ax[0].plot(t.ltp, lw=1.0, label=t.name[:20])
            sheets[f"LTP_{t.name[:24]}"] = pd.DataFrame({"G_LTP": t.ltp})
        if t.ltd is not None and t.ltd.size > 1:
            ax[1].plot(t.ltd, lw=1.0, label=t.name[:20])
            sheets[f"LTD_{t.name[:24]}"] = pd.DataFrame({"G_LTD": t.ltd})
    ax[0].set_title("LTP"); ax[1].set_title("LTD")
    for a in ax: a.set_xlabel("Step"); a.set_ylabel("G"); a.legend(fontsize=6)
    _save_pair(fig, plots_dir, excel_dir, "device_ltp_ltd", sheets)


def plot_onoff_histogram(traces, plots_dir, excel_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ratios = [t.meta.get("on_off_ratio", np.nan) for t in traces
              if t.meta.get("on_off_ratio", 0) < 1e6]
    ratios = [r for r in ratios if np.isfinite(r)]
    if ratios:
        ax.hist(ratios, bins=min(20, len(ratios)), color="steelblue", edgecolor="k")
    ax.set_xlabel("ON/OFF ratio"); ax.set_ylabel("Count")
    ax.set_title("ON/OFF Ratio Distribution")
    _save_pair(fig, plots_dir, excel_dir, "device_onoff_hist",
               {"on_off": pd.DataFrame({"on_off_ratio": ratios})})


def plot_state_histogram(traces, plots_dir, excel_dir):
    fig, ax = plt.subplots(figsize=(6, 4)); sheets = {}
    for t in traces:
        if t.conductance is None: continue
        g = t.conductance[np.isfinite(t.conductance)]
        ax.hist(g, bins=50, alpha=0.4, label=t.name[:20])
        sheets[t.name[:30]] = pd.DataFrame({"G": g})
    ax.set_xlabel("Conductance"); ax.set_ylabel("Count")
    ax.set_title("Conductance-State Histogram"); ax.legend(fontsize=6)
    _save_pair(fig, plots_dir, excel_dir, "device_state_hist", sheets)


# ---------------- AI LEVEL ----------------
def plot_training_curves(logs: pd.DataFrame, plots_dir, excel_dir, tag=""):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(logs["epoch"], logs["train_acc"], label="train")
    ax[0].plot(logs["epoch"], logs["test_acc"],  label="test")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy"); ax[0].legend()
    ax[0].set_title(f"Accuracy vs Epochs {tag}")
    ax[1].plot(logs["epoch"], logs["train_loss"], label="train")
    ax[1].plot(logs["epoch"], logs["test_loss"],  label="test")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss"); ax[1].legend()
    ax[1].set_title(f"Loss vs Epochs {tag}")
    _save_pair(fig, plots_dir, excel_dir, f"ai_training_curves_{tag}".strip("_"),
               {"training_log": logs})


def plot_confusion(cm: np.ndarray, plots_dir, excel_dir, tag=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap="viridis", ax=ax)
    ax.set_xlabel("Pred"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix {tag}")
    _save_pair(fig, plots_dir, excel_dir, f"ai_confusion_{tag}".strip("_"),
               {"confusion": pd.DataFrame(cm)})


def plot_weight_distribution(model, plots_dir, excel_dir):
    import torch.nn as nn
    fig, ax = plt.subplots(figsize=(7, 4)); sheets = {}
    for name, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None and isinstance(m, (nn.Linear, nn.Conv2d)):
            w = m.weight.detach().cpu().numpy().ravel()
            ax.hist(w, bins=80, alpha=0.4, label=name[:22])
            sheets[name.replace(".", "_")[:30] or "w"] = pd.DataFrame({"w": w})
    ax.set_xlabel("Weight"); ax.set_ylabel("Count"); ax.legend(fontsize=6)
    ax.set_title("Weight Distribution")
    _save_pair(fig, plots_dir, excel_dir, "ai_weight_distribution", sheets)


# ---------------- COMPARISON (IDEAL vs HARDWARE-AWARE) ----------------
def plot_dual_comparison(logs_ideal: pd.DataFrame, logs_hw: pd.DataFrame,
                         plots_dir: str, excel_dir: str):
    """Overlay accuracy & loss curves for ideal vs hardware-aware runs."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(logs_ideal["epoch"], logs_ideal["test_acc"],  label="ideal (test)", lw=1.5)
    ax[0].plot(logs_hw["epoch"],    logs_hw["test_acc"],     label="hw-aware (test)", lw=1.5)
    ax[0].plot(logs_ideal["epoch"], logs_ideal["train_acc"], label="ideal (train)", lw=1.0, ls="--", alpha=0.7)
    ax[0].plot(logs_hw["epoch"],    logs_hw["train_acc"],    label="hw-aware (train)", lw=1.0, ls="--", alpha=0.7)
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy vs Epochs: Ideal vs Hardware-Aware"); ax[0].legend()

    ax[1].plot(logs_ideal["epoch"], logs_ideal["test_loss"], label="ideal (test)", lw=1.5)
    ax[1].plot(logs_hw["epoch"],    logs_hw["test_loss"],    label="hw-aware (test)", lw=1.5)
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Loss")
    ax[1].set_title("Loss vs Epochs: Ideal vs Hardware-Aware"); ax[1].legend()

    n = min(len(logs_ideal), len(logs_hw))
    merged = pd.DataFrame({
        "epoch":            logs_ideal["epoch"].values[:n],
        "ideal_train_acc":  logs_ideal["train_acc"].values[:n],
        "ideal_test_acc":   logs_ideal["test_acc"].values[:n],
        "ideal_train_loss": logs_ideal["train_loss"].values[:n],
        "ideal_test_loss":  logs_ideal["test_loss"].values[:n],
        "hw_train_acc":     logs_hw["train_acc"].values[:n],
        "hw_test_acc":      logs_hw["test_acc"].values[:n],
        "hw_train_loss":    logs_hw["train_loss"].values[:n],
        "hw_test_loss":     logs_hw["test_loss"].values[:n],
    })
    _save_pair(fig, plots_dir, excel_dir, "comparison_accuracy_loss",
               {"curves": merged})


def plot_dual_summary(summary_df: pd.DataFrame, plots_dir: str, excel_dir: str):
    """Bar plot: final accuracy + energy for ideal vs hardware-aware."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    modes = summary_df["mode"].tolist()
    ax[0].bar(modes, summary_df["acc"], color=["#4C72B0", "#DD8452"], edgecolor="k")
    ax[0].set_ylabel("Final Test Accuracy"); ax[0].set_ylim(0, 1.0)
    ax[0].set_title("Final Accuracy")
    for i, v in enumerate(summary_df["acc"]):
        ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    if "energy_J" in summary_df.columns:
        ax[1].bar(modes, summary_df["energy_J"], color=["#4C72B0", "#DD8452"], edgecolor="k")
        ax[1].set_ylabel("Energy (J) per inference"); ax[1].set_yscale("log")
        ax[1].set_title("Energy Impact")
    _save_pair(fig, plots_dir, excel_dir, "comparison_summary", {"summary": summary_df})


# ---------------- HARDWARE LEVEL ----------------
def plot_dse(df: pd.DataFrame, plots_dir, excel_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sc = ax[0].scatter(df["energy_J"], df["accuracy"],
                       c=df["weight_bits"], cmap="plasma", s=50, edgecolor="k")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Energy (J)")
    ax[0].set_ylabel("Accuracy"); ax[0].set_title("Energy vs Accuracy")
    plt.colorbar(sc, ax=ax[0], label="weight_bits")

    sc2 = ax[1].scatter(df["latency_s"], df["accuracy"],
                        c=df["noise_std"], cmap="viridis", s=50, edgecolor="k")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Latency (s)")
    ax[1].set_ylabel("Accuracy"); ax[1].set_title("Latency vs Accuracy")
    plt.colorbar(sc2, ax=ax[1], label="noise_std")
    _save_pair(fig, plots_dir, excel_dir, "hw_dse_scatter", {"dse": df})
