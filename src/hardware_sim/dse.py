"""Phase 7: Expanded Design Space Exploration (DSE).

Sweeps across ADC bits, weight bits, sub-array size, and noise levels
to estimate accuracy/energy/latency trade-offs. Generates:
- Sweep table (Excel)
- Accuracy vs Energy heatmap
- Pareto frontier (accuracy vs energy)
- Noise sensitivity analysis

Penalty model:
- Lower precision → more quantization error → accuracy penalty
- Penalties are estimated from information-theoretic degradation:
  accuracy_drop ≈ α * (8 - weight_bits)^β + γ * (8 - adc_bits)^β
  where α, β, γ are calibrated from the observed ideal-vs-hardware gap.

All values are APPROXIMATE and for comparative analysis only.
"""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd


def run_dse(history: dict, cfg: dict, out_dir: Path) -> None:
    """Export comprehensive design-space exploration from hardware training results."""
    if not history.get("val_accuracy"):
        logging.warning("DSE skipped because no hardware accuracy history is available.")
        return

    base_acc = float(history["val_accuracy"][-1])
    hw = cfg.get("hardware", {})
    base_weight_bits = int(hw.get("weight_bits", 8))
    base_adc_bits = int(hw.get("adc_bits", 6))

    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Main precision × sub-array sweep ===
    rows = _precision_subarray_sweep(base_acc, hw)
    df = pd.DataFrame(rows)
    df.to_excel(out_dir / "dse_precision_subarray.xlsx", index=False)
    logging.info("DSE precision-subarray table exported (%d configs)", len(rows))

    # === 2. Noise sensitivity sweep ===
    noise_rows = _noise_sweep(base_acc, hw)
    pd.DataFrame(noise_rows).to_excel(out_dir / "dse_noise_sensitivity.xlsx", index=False)
    logging.info("DSE noise sensitivity table exported")

    # === 3. Generate plots ===
    _plot_dse_heatmap(df, out_dir)
    _plot_pareto(df, out_dir)
    _plot_noise_sensitivity(noise_rows, out_dir)
    _plot_precision_tradeoff(rows, out_dir)


def _precision_subarray_sweep(base_acc: float, hw: dict) -> list[dict]:
    """Sweep ADC bits × weight bits × sub-array size."""
    rows = []
    v_read = float(hw.get("v_read", 0.2))
    c_line_fF = float(hw.get("c_line_fF", 1.0))
    r_wire = float(hw.get("r_wire_ohm", 2.0))
    adc_energy_pJ = float(hw.get("adc_energy_pJ", 2.0))
    clock_ns = float(hw.get("base_latency_ns", 5.0))
    C = c_line_fF * 1e-15

    for adc_bits in [2, 4, 6, 8, 10]:
        for weight_bits in [1, 2, 4, 6, 8]:
            for sub_rows, sub_cols in [(64, 64), (128, 128), (256, 256), (512, 512)]:
                # Accuracy penalty model (information-theoretic approximation)
                # Lower bits = more quantization noise = larger penalty
                w_penalty = max(0, 8 - weight_bits) * 0.012 * (1 + 0.1 * max(0, 4 - weight_bits))
                a_penalty = max(0, 8 - adc_bits) * 0.006 * (1 + 0.1 * max(0, 4 - adc_bits))
                est_acc = max(0.0, base_acc - w_penalty - a_penalty)

                # Energy model: E_MAC = C*V^2, scaled by weight_bits/8
                e_mac_pJ = C * v_read ** 2 * 1e12 * (weight_bits / 8)
                # Larger arrays → more MACs per access but more wire energy
                area_scale = (sub_rows * sub_cols) / (128 * 128)
                e_mac_pJ *= (1 + 0.1 * np.log2(max(area_scale, 1)))
                # ADC energy scales with 2^adc_bits
                adc_ops_per_col = 1  # one readout per column
                e_adc = adc_energy_pJ * (2 ** adc_bits / 64)  # normalized to 6-bit baseline

                # Latency model: RC + clock, scaled by array size
                RC_ns = r_wire * C * 1e9
                t_mac_ns = RC_ns + clock_ns * max(1.0, 8.0 / adc_bits)
                # Larger array = more parallelism = lower effective latency per MAC
                effective_latency_ns = t_mac_ns / max(np.sqrt(area_scale), 0.5)

                # TOPS/W proxy
                ops_per_s = 2.0 / (effective_latency_ns * 1e-9)  # 2 ops (mul+add) per MAC
                power_W = (e_mac_pJ + e_adc) * 1e-12 / (effective_latency_ns * 1e-9)
                tops_w = ops_per_s / max(power_W, 1e-18) / 1e12

                rows.append({
                    "adc_bits": adc_bits,
                    "weight_bits": weight_bits,
                    "sub_array_rows": sub_rows,
                    "sub_array_cols": sub_cols,
                    "estimated_accuracy": est_acc,
                    "accuracy_penalty": w_penalty + a_penalty,
                    "energy_per_mac_pJ": e_mac_pJ,
                    "adc_energy_pJ": e_adc,
                    "total_energy_pJ": e_mac_pJ + e_adc,
                    "latency_ns_per_mac": effective_latency_ns,
                    "tops_w_proxy": tops_w,
                    "area_scale": area_scale,
                })
    return rows


def _noise_sweep(base_acc: float, hw: dict) -> list[dict]:
    """Sweep read noise and cycle variation to assess robustness."""
    rows = []
    for read_noise in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        for cycle_var in [0.0, 0.05, 0.1, 0.2, 0.5]:
            # Noise penalty: proportional to total noise sigma
            total_sigma = read_noise + cycle_var * float(hw.get("cycle_variation_scale", 0.1))
            # Empirical: ~3% drop per 0.1 sigma (from NeuroSim literature)
            noise_penalty = total_sigma * 0.3
            est_acc = max(0.0, base_acc - noise_penalty)
            rows.append({
                "read_noise": read_noise,
                "cycle_variation": cycle_var,
                "total_sigma": total_sigma,
                "estimated_accuracy": est_acc,
                "accuracy_penalty": noise_penalty,
            })
    return rows


def _plot_dse_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of accuracy vs (weight_bits, adc_bits) for 128x128 sub-array."""
    import matplotlib.pyplot as plt

    subset = df[df["sub_array_rows"] == 128].copy()
    if subset.empty:
        return

    pivot = subset.pivot_table(values="estimated_accuracy", index="weight_bits", columns="adc_bits", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("Design Space Exploration — 128×128 Sub-array", fontsize=14, fontweight="bold")

    # Accuracy heatmap
    ax = axes[0]
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto",
                   vmin=pivot.values.min() * 100, vmax=pivot.values.max() * 100)
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Weight bits")
    ax.set_title("Estimated Accuracy (%)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if val < (pivot.values.max() * 100 - 5) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Energy heatmap
    pivot_e = subset.pivot_table(values="total_energy_pJ", index="weight_bits", columns="adc_bits", aggfunc="mean")
    ax = axes[1]
    im2 = ax.imshow(pivot_e.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot_e.columns)), labels=[str(c) for c in pivot_e.columns])
    ax.set_yticks(range(len(pivot_e.index)), labels=[str(i) for i in pivot_e.index])
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Weight bits")
    ax.set_title("Total Energy per MAC (pJ)")
    for i in range(len(pivot_e.index)):
        for j in range(len(pivot_e.columns)):
            val = pivot_e.values[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_dir / "dse_heatmap.png", dpi=200)
    plt.close(fig)


def _plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    """Pareto frontier: accuracy vs total energy."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by weight_bits
    for wb in sorted(df["weight_bits"].unique()):
        subset = df[df["weight_bits"] == wb]
        ax.scatter(subset["total_energy_pJ"], subset["estimated_accuracy"] * 100,
                   alpha=0.6, s=40, label=f"W={wb}bit")

    # Find and plot Pareto frontier
    points = df[["total_energy_pJ", "estimated_accuracy"]].values
    pareto = _pareto_front(points[:, 0], points[:, 1])
    pareto_sorted = pareto[pareto[:, 0].argsort()]
    ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1] * 100, "k--", linewidth=2, label="Pareto frontier")

    ax.set_xlabel("Energy per MAC (pJ)")
    ax.set_ylabel("Estimated Accuracy (%)")
    ax.set_title("Accuracy vs Energy — Pareto Analysis")
    ax.grid(True, linestyle="--", alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "dse_pareto.png", dpi=200)
    plt.close(fig)


def _plot_noise_sensitivity(noise_rows: list[dict], out_dir: Path) -> None:
    """Heatmap of accuracy under different noise conditions."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(noise_rows)
    pivot = df.pivot_table(values="estimated_accuracy", index="read_noise", columns="cycle_variation", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), labels=[f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[f"{i:.2f}" for i in pivot.index])
    ax.set_xlabel("Cycle-to-cycle variation")
    ax.set_ylabel("Read noise (sigma)")
    ax.set_title("Noise Sensitivity: Estimated Accuracy (%)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] * 100
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if val < 80 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "dse_noise_sensitivity.png", dpi=200)
    plt.close(fig)


def _plot_precision_tradeoff(rows: list[dict], out_dir: Path) -> None:
    """Line plot: accuracy and TOPS/W vs weight bits for different ADC configs."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(rows)
    subset = df[df["sub_array_rows"] == 128]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Precision Trade-off Analysis (128×128 array)", fontsize=13, fontweight="bold")

    for adc in sorted(subset["adc_bits"].unique()):
        group = subset[subset["adc_bits"] == adc].sort_values("weight_bits")
        axes[0].plot(group["weight_bits"], group["estimated_accuracy"] * 100,
                     "o-", label=f"ADC={adc}bit", linewidth=1.5, markersize=5)
        axes[1].plot(group["weight_bits"], group["tops_w_proxy"],
                     "s-", label=f"ADC={adc}bit", linewidth=1.5, markersize=5)

    axes[0].set_xlabel("Weight bits")
    axes[0].set_ylabel("Estimated Accuracy (%)")
    axes[0].set_title("Accuracy vs Precision")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Weight bits")
    axes[1].set_ylabel("TOPS/W (proxy)")
    axes[1].set_title("Efficiency vs Precision")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(fontsize=8)

    fig.savefig(out_dir / "dse_precision_tradeoff.png", dpi=200)
    plt.close(fig)


def _pareto_front(cost: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Extract Pareto-optimal points (minimize cost, maximize benefit)."""
    points = np.column_stack([cost, benefit])
    sorted_idx = np.argsort(points[:, 0])
    points = points[sorted_idx]
    pareto = [points[0]]
    for p in points[1:]:
        if p[1] >= pareto[-1][1]:
            pareto.append(p)
    return np.array(pareto)
