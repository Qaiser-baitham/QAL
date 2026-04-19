from __future__ import annotations
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd

from src.device_model.extractor import DeviceModel
from src.utils.io import export_excel


class Exporter:
    def __init__(self, plots_dir: Path, excel_dir: Path, reports_dir: Path | None = None):
        self.plots_dir = plots_dir
        self.excel_dir = excel_dir
        self.reports_dir = reports_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.excel_dir.mkdir(parents=True, exist_ok=True)
        if self.reports_dir is not None:
            self.reports_dir.mkdir(parents=True, exist_ok=True)

    def device(self, device: DeviceModel) -> None:
        states = np.asarray(device.state_means, dtype=float)
        pulse = np.arange(len(states))
        self._line("device_conductance_vs_pulse", pulse, states, "Pulse/state index", "Conductance (S)")
        ltp_trace = _first_trace(device, "LTP")
        ltd_trace = _first_trace(device, "LTD")
        if ltp_trace is not None:
            self._trace_line("device_ltp_curve", ltp_trace, "Measured LTP Conductance")
        else:
            self._line("device_ltp_curve", pulse, states, "Pulse/state index", "Conductance (S)")
        if ltd_trace is not None:
            self._trace_line("device_ltd_curve", ltd_trace, "Measured LTD Conductance")
        else:
            self._line("device_ltd_curve", pulse, states[::-1], "Pulse/state index", "Conductance (S)")
        self._hist("device_histogram", states, "Conductance (S)")
        export_excel(self.excel_dir / "device_state_distribution.xlsx", {"state": pulse.tolist(), "mean_conductance_s": states.tolist(), "std_conductance_s": device.state_stds})
        # Phase 2: comprehensive device characterization dashboard
        self._device_characterization_dashboard(device)
        # Phase 2: raw + smoothed + fitted LTP/LTD
        self._device_fitted_curves(device)
        # Phase 2: device summary Excel
        self._device_summary_excel(device)

    def _device_characterization_dashboard(self, device: DeviceModel) -> None:
        """6-panel device characterization dashboard (publication-level)."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
        fig.suptitle("Device Characterization Dashboard — Ag/GeTe/Pt Memristor", fontsize=16, fontweight="bold")

        # Panel 1: LTP raw + smoothed + fitted
        ax = axes[0, 0]
        ltp_trace = _first_trace(device, "LTP")
        if ltp_trace and ltp_trace.get("conductance"):
            g = np.asarray(ltp_trace["conductance"], dtype=float)
            x = np.arange(len(g))
            ax.plot(x, g * 1e3, "o-", color="#1f77b4", markersize=3, linewidth=1.0, alpha=0.6, label="Raw")
            smoothed = _moving_average(g.tolist(), window=5)
            ax.plot(x, np.asarray(smoothed) * 1e3, "-", color="#ff7f0e", linewidth=2.0, label="Smoothed")
            if device.fitted_ltp_coeffs:
                xn = np.linspace(0, 1, len(g))
                fitted = np.polyval(device.fitted_ltp_coeffs, xn)
                ax.plot(x, fitted * 1e3, "--", color="#2ca02c", linewidth=2.0, label="Poly fit (deg 3)")
        ax.set_title("LTP: Raw / Smoothed / Fitted")
        ax.set_xlabel("Pulse number")
        ax.set_ylabel("Conductance (mS)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 2: LTD raw + smoothed + fitted
        ax = axes[0, 1]
        ltd_trace = _first_trace(device, "LTD")
        if ltd_trace and ltd_trace.get("conductance"):
            g = np.asarray(ltd_trace["conductance"], dtype=float)
            x = np.arange(len(g))
            ax.plot(x, g * 1e3, "o-", color="#d62728", markersize=3, linewidth=1.0, alpha=0.6, label="Raw")
            smoothed = _moving_average(g.tolist(), window=5)
            ax.plot(x, np.asarray(smoothed) * 1e3, "-", color="#ff7f0e", linewidth=2.0, label="Smoothed")
            if device.fitted_ltd_coeffs:
                xn = np.linspace(0, 1, len(g))
                fitted = np.polyval(device.fitted_ltd_coeffs, xn)
                ax.plot(x, fitted * 1e3, "--", color="#2ca02c", linewidth=2.0, label="Poly fit (deg 3)")
        ax.set_title("LTD: Raw / Smoothed / Fitted")
        ax.set_xlabel("Pulse number")
        ax.set_ylabel("Conductance (mS)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 3: Normalized weight update (LTP + LTD overlay)
        ax = axes[0, 2]
        for kind, color, label_prefix in [("LTP", "#1f77b4", "LTP"), ("LTD", "#d62728", "LTD")]:
            trace = _first_trace(device, kind)
            if trace and trace.get("conductance"):
                g = np.asarray(trace["conductance"], dtype=float)
                g_range = np.nanmax(g) - np.nanmin(g)
                if g_range > 1e-30:
                    normalized = (g - np.nanmin(g)) / g_range
                    x = np.linspace(0, 1, len(normalized))
                    ax.plot(x, normalized, "o-", color=color, markersize=3, linewidth=1.5, label=f"{label_prefix} measured")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Ideal linear")
        ax.set_title("Normalized Weight Update")
        ax.set_xlabel("Normalized pulse")
        ax.set_ylabel("Normalized conductance")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

        # Panel 4: Conductance state distribution with error bars
        ax = axes[1, 0]
        states = np.asarray(device.state_means, dtype=float)
        stds = np.asarray(device.state_stds, dtype=float)
        state_idx = np.arange(len(states))
        ax.bar(state_idx, states * 1e3, yerr=stds * 1e3, color="#9467bd", alpha=0.7, capsize=3, label=f"{len(states)} states")
        ax.set_title(f"Conductance States (n={len(states)})")
        ax.set_xlabel("State index")
        ax.set_ylabel("Conductance (mS)")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.legend(fontsize=8)

        # Panel 5: ON/OFF ratio and dynamic range annotation
        ax = axes[1, 1]
        metrics = {
            "ON/OFF ratio": device.on_off_ratio,
            "Dynamic range (dB)": device.dynamic_range_db,
            "LTP nonlinearity": device.ltp_nonlinearity,
            "LTD nonlinearity": device.ltd_nonlinearity,
            "Symmetry index": device.ltp_symmetry,
            "Cycle variation": device.cycle_variation_sigma,
            "Endurance stability": device.endurance_stability,
            "Num states": device.num_states,
        }
        y_pos = np.arange(len(metrics))
        labels = list(metrics.keys())
        values = list(metrics.values())
        colors = ["#2ca02c" if _metric_quality(k, v) == "good" else "#ff7f0e" if _metric_quality(k, v) == "fair" else "#d62728" for k, v in metrics.items()]
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02 * max(abs(v) for v in values), bar.get_y() + bar.get_height() / 2, f"{val:.4g}", va="center", fontsize=8)
        ax.set_title("Device Metrics Summary")
        ax.set_xlabel("Value")
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Panel 6: Conductance histogram
        ax = axes[1, 2]
        all_g = []
        for trace in getattr(device, "source_traces", []) or []:
            if trace.get("conductance"):
                all_g.extend(trace["conductance"])
        if all_g:
            all_g = np.asarray(all_g, dtype=float)
            all_g = all_g[np.isfinite(all_g) & (all_g > 0)]
            ax.hist(all_g * 1e3, bins=min(30, max(5, len(all_g) // 3)), color="#17becf", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title("Conductance Distribution")
        ax.set_xlabel("Conductance (mS)")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        fig.savefig(self.plots_dir / "device_characterization_dashboard.png", dpi=250)
        plt.close(fig)

    def _device_fitted_curves(self, device: DeviceModel) -> None:
        """Individual publication-quality LTP/LTD with raw + smoothed + fitted."""
        import matplotlib.pyplot as plt

        for kind, coeffs, color in [("LTP", device.fitted_ltp_coeffs, "#1f77b4"), ("LTD", device.fitted_ltd_coeffs, "#d62728")]:
            trace = _first_trace(device, kind)
            if not trace or not trace.get("conductance"):
                continue
            g = np.asarray(trace["conductance"], dtype=float)
            x = np.arange(len(g))

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, g * 1e3, "o", color=color, markersize=4, alpha=0.5, label="Raw data")
            smoothed = _moving_average(g.tolist(), window=5)
            ax.plot(x, np.asarray(smoothed) * 1e3, "-", color="#ff7f0e", linewidth=2.0, label="Smoothed (MA-5)")
            if coeffs:
                xn = np.linspace(0, 1, len(g))
                fitted = np.polyval(coeffs, xn)
                ax.plot(x, fitted * 1e3, "--", color="#2ca02c", linewidth=2.0, label="Polynomial fit (deg 3)")
            ax.set_title(f"{kind} Curve — Raw / Smoothed / Fitted")
            ax.set_xlabel("Pulse number")
            ax.set_ylabel("Conductance (mS)")
            ax.grid(True, linestyle="--", alpha=0.3)
            _style_legend(ax)
            fig.tight_layout()
            fig.savefig(self.plots_dir / f"device_{kind.lower()}_fitted.png", dpi=200)
            plt.close(fig)

            # Excel export for fitted curves
            excel_data = {"pulse": x.tolist(), "raw_conductance_S": g.tolist(), "smoothed_conductance_S": smoothed}
            if coeffs:
                xn = np.linspace(0, 1, len(g))
                excel_data["fitted_conductance_S"] = np.polyval(coeffs, xn).tolist()
            export_excel(self.excel_dir / f"device_{kind.lower()}_fitted.xlsx", excel_data)

    def _device_summary_excel(self, device: DeviceModel) -> None:
        """Comprehensive device characterization summary in Excel."""
        summary = {
            "parameter": [
                "g_on (S)", "g_off (S)", "ON/OFF ratio", "Dynamic range (dB)",
                "Num states", "LTP nonlinearity", "LTD nonlinearity",
                "LTP-LTD symmetry", "Cycle variation sigma", "Endurance stability",
            ],
            "value": [
                device.g_on, device.g_off, device.on_off_ratio, device.dynamic_range_db,
                device.num_states, device.ltp_nonlinearity, device.ltd_nonlinearity,
                device.ltp_symmetry, device.cycle_variation_sigma, device.endurance_stability,
            ],
            "formula": [
                device.formulas.get("g_on", ""), device.formulas.get("g_off", ""),
                device.formulas.get("on_off_ratio", ""), device.formulas.get("dynamic_range_db", ""),
                device.formulas.get("num_states", ""), device.formulas.get("ltp_ltd_nonlinearity", ""),
                device.formulas.get("ltp_ltd_nonlinearity", ""), device.formulas.get("symmetry", ""),
                device.formulas.get("cycle_variation_sigma", ""), device.formulas.get("endurance_stability", ""),
            ],
        }
        export_excel(self.excel_dir / "device_characterization_summary.xlsx", summary)

    def training(self, history: dict, prefix: str) -> None:
        epochs = history.get("epoch", [])
        if not epochs:
            return
        self._line(f"{prefix}_accuracy_vs_epoch", epochs, history.get("val_accuracy", []), "Epoch", "Accuracy")
        self._line(f"{prefix}_loss_vs_epoch", epochs, history.get("train_loss", []), "Epoch", "Loss")
        self._single_training_suite(history, prefix)
        export_excel(self.excel_dir / f"{prefix}_history.xlsx", history)
        cm = history.get("confusion_matrix")
        if cm is not None:
            cm_arr = np.asarray(cm)
            class_names = _class_names_from_history(history, cm_arr.shape[0])
            title_suffix = _title_suffix(history)
            self._matrix(f"{prefix}_confusion_matrix", cm_arr, "Predicted", "True", class_names=class_names, title_suffix=title_suffix)
            # Phase 6: normalized confusion matrix
            self._normalized_matrix(f"{prefix}_confusion_matrix_normalized", cm_arr, "Predicted", "True", class_names=class_names, title_suffix=title_suffix)
            # Phase 6: per-class precision/recall/F1
            self._class_metrics(f"{prefix}_class_metrics", cm_arr, history, class_names=class_names, title_suffix=title_suffix)
            # Phase 8: per-class accuracy bar, top-confused pairs, CM difference (if both CMs available)
            self._per_class_accuracy_bar(f"{prefix}_per_class_accuracy", cm_arr, class_names, title_suffix=title_suffix)
            self._top_confused_pairs(f"{prefix}_top_confused_pairs", cm_arr, class_names, title_suffix=title_suffix, top_k=min(15, cm_arr.shape[0] * (cm_arr.shape[0] - 1)))

    def comparison(self, ideal: dict | None, hardware: dict | None) -> None:
        if not ideal or not hardware:
            return
        epochs = ideal.get("epoch", [])
        n = min(len(epochs), len(hardware.get("epoch", [])))
        if n == 0:
            return
        data = {
            "epoch": epochs[:n],
            "dataset": [ideal.get("metadata", {}).get("dataset", "")],
            "model": [ideal.get("metadata", {}).get("model", "")],
            "ideal_label": [_curve_label(ideal, "Ideal")],
            "hardware_label": [_curve_label(hardware, "Memristor")],
            "ideal_accuracy": ideal.get("val_accuracy", [])[:n],
            "hardware_accuracy": hardware.get("val_accuracy", [])[:n],
            "ideal_train_accuracy": ideal.get("train_accuracy", [])[:n],
            "hardware_train_accuracy": hardware.get("train_accuracy", [])[:n],
            "ideal_train_loss": ideal.get("train_loss", [])[:n],
            "hardware_train_loss": hardware.get("train_loss", [])[:n],
            "ideal_loss": ideal.get("val_loss", [])[:n],
            "hardware_loss": hardware.get("val_loss", [])[:n],
            "ideal_learning_rate": _aligned(ideal.get("learning_rate", []), n),
            "hardware_learning_rate": _aligned(hardware.get("learning_rate", []), n),
            "ideal_seconds": _aligned(ideal.get("seconds", []), n),
            "hardware_seconds": _aligned(hardware.get("seconds", []), n),
        }
        data.update(_comparison_metrics(data, hardware, n))
        data.update(_smoothed_metrics(data))
        ideal_label = data["ideal_label"][0]
        hardware_label = data["hardware_label"][0]
        export_excel(self.excel_dir / "comparison_ideal_vs_hardware.xlsx", data)
        export_excel(self.excel_dir / "training_history.xlsx", data)
        self._multi_line("comparison_accuracy", data["epoch"], {ideal_label: data["ideal_accuracy"], hardware_label: data["hardware_accuracy"]}, "Epoch", "Accuracy", title="Test Accuracy Comparison")
        self._multi_line("comparison_loss", data["epoch"], {ideal_label: data["ideal_loss"], hardware_label: data["hardware_loss"]}, "Epoch", "Loss", title="Test Loss Comparison")
        self._multi_line("comparison_accuracy_smoothed", data["epoch"], {f"{ideal_label} raw": data["ideal_accuracy"], f"{ideal_label} smooth": data["ideal_accuracy_smooth"], f"{hardware_label} raw": data["hardware_accuracy"], f"{hardware_label} smooth": data["hardware_accuracy_smooth"]}, "Epoch", "Accuracy", title="Raw and Smoothed Accuracy")
        self._multi_line("comparison_loss_smoothed", data["epoch"], {f"{ideal_label} raw": data["ideal_loss"], f"{ideal_label} smooth": data["ideal_loss_smooth"], f"{hardware_label} raw": data["hardware_loss"], f"{hardware_label} smooth": data["hardware_loss_smooth"]}, "Epoch", "Loss", title="Raw and Smoothed Loss")
        self._training_history(data)
        self._comparison_suite(data)
        hw = {
            "accuracy": hardware.get("val_accuracy", []),
            "energy_pj": hardware.get("energy_pj", []),
            "latency_ns": hardware.get("latency_ns", []),
        }
        export_excel(self.excel_dir / "hardware_energy_latency_accuracy.xlsx", hw)
        self._line("hardware_energy_vs_accuracy", hw["accuracy"], hw["energy_pj"], "Accuracy", "Energy (pJ)")
        self._line("hardware_latency_vs_accuracy", hw["accuracy"], hw["latency_ns"], "Accuracy", "Latency (ns)")
        # Phase 8: energy-delay product and throughput-vs-epoch plots
        epochs_hw = list(range(1, len(hw["energy_pj"]) + 1))
        if hw["energy_pj"] and hw["latency_ns"] and len(hw["energy_pj"]) == len(hw["latency_ns"]):
            edp = [float(e) * float(l) for e, l in zip(hw["energy_pj"], hw["latency_ns"])]
            export_excel(self.excel_dir / "hardware_energy_delay_product.xlsx", {"epoch": epochs_hw, "edp_pj_ns": edp})
            self._line("hardware_energy_delay_product", epochs_hw, edp, "Epoch", "Energy x Latency (pJ*ns)")
        macs = hardware.get("macs_per_sample", []) or []
        if macs and hw["latency_ns"] and len(macs) == len(hw["latency_ns"]):
            # throughput (samples/sec) = 1 / per-sample-latency (s)
            throughput = [1.0 / max(float(l) * 1e-9, 1e-18) for l in hw["latency_ns"]]
            export_excel(self.excel_dir / "hardware_throughput_vs_epoch.xlsx", {"epoch": epochs_hw, "throughput_samples_per_sec": throughput})
            self._line("hardware_throughput_vs_epoch", epochs_hw, throughput, "Epoch", "Throughput (samples/s)")
        tops_w = hardware.get("tops_w_proxy", []) or []
        if tops_w:
            export_excel(self.excel_dir / "hardware_tops_w_vs_epoch.xlsx", {"epoch": list(range(1, len(tops_w) + 1)), "tops_w_proxy": tops_w})
            self._line("hardware_tops_w_vs_epoch", list(range(1, len(tops_w) + 1)), tops_w, "Epoch", "TOPS/W (proxy)")

    def param_info(self, histories: dict[str, dict], cfg: dict, device: DeviceModel) -> None:
        rows = []
        for mode, history in histories.items():
            rows.extend(_param_rows(mode, history, cfg))
        if not rows:
            return

        pd.DataFrame(rows).to_excel(self.excel_dir / "paramInfo.xlsx", index=False)
        if self.reports_dir is not None:
            (self.reports_dir / "paramInfo.md").write_text(_param_info_markdown(rows, histories, cfg, device), encoding="utf-8")
            self._write_artifact_index()

    def runtime_timing(self, rows: list[dict]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        frame.to_csv(self.excel_dir / "runtime_timing.csv", index=False)
        frame.to_excel(self.excel_dir / "runtime_timing.xlsx", index=False)
        if self.reports_dir is not None:
            import json

            (self.reports_dir / "runtime_timing.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self._runtime_timing_plot(frame)

    def _runtime_timing_plot(self, frame: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for mode, group in frame.groupby("mode"):
            group = group.sort_values("epoch")
            axes[0].plot(group["epoch"], group["epoch_duration_seconds"], marker="o", label=_legend_label(mode, group["epoch_duration_seconds"], "Seconds"))
            axes[1].plot(group["epoch"], group["elapsed_total_seconds"], marker="o", label=_legend_label(mode, group["elapsed_total_seconds"], "Seconds"))
        axes[0].set_title("Epoch Duration")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Seconds")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        axes[1].set_title("Elapsed Runtime")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Seconds")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        for ax in axes:
            _style_legend(ax)
        fig.savefig(self.plots_dir / "runtime_epoch_time.png", dpi=200)
        plt.close(fig)

    def _line(self, name: str, x, y, xlabel: str, ylabel: str, title: str | None = None) -> None:
        export_excel(self.excel_dir / f"{name}.xlsx", {xlabel: list(x), ylabel: list(y)})
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_y, plot_ylabel = _plot_series(y, ylabel)
        label = _single_line_legend_label(name, x, plot_y, xlabel, plot_ylabel)
        ax.plot(x, plot_y, marker="o", markevery=_markevery(x), linewidth=2.0, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_ylabel)
        ax.set_title(title or name.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.30)
        _style_legend(ax)
        plt.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _single_training_suite(self, history: dict, prefix: str) -> None:
        epochs = history.get("epoch", [])
        train_acc = history.get("train_accuracy", [])
        val_acc = history.get("val_accuracy", [])
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        lr = history.get("learning_rate", [])
        seconds = history.get("seconds", [])
        if not epochs:
            return

        if train_acc and val_acc:
            label = _history_label(history, prefix)
            self._multi_line(f"{prefix}_train_test_accuracy_vs_epoch", epochs, {f"{label} train": train_acc, f"{label} test": val_acc}, "Epoch", "Accuracy", title=f"{label} Train/Test Accuracy")
            self._line(f"{prefix}_generalization_gap_vs_epoch", epochs, _subtract(train_acc, val_acc), "Epoch", "Train - test accuracy")
            self._line(f"{prefix}_test_error_rate_vs_epoch", epochs, [1.0 - float(v) for v in val_acc], "Epoch", "Error rate")
            self._line(f"{prefix}_best_accuracy_vs_epoch", epochs, _cummax(val_acc), "Epoch", "Best accuracy so far")
        if train_loss and val_loss:
            label = _history_label(history, prefix)
            self._multi_line(f"{prefix}_train_test_loss_vs_epoch", epochs, {f"{label} train": train_loss, f"{label} test": val_loss}, "Epoch", "Loss", title=f"{label} Train/Test Loss")
            self._line(f"{prefix}_loss_gap_vs_epoch", epochs, _subtract(val_loss, train_loss), "Epoch", "Test - train loss")
        if lr:
            self._line(f"{prefix}_learning_rate_vs_epoch", epochs, lr, "Epoch", "Learning rate")
            if val_acc:
                self._scatter(f"{prefix}_test_accuracy_vs_learning_rate", lr, val_acc, "Learning rate", "Test accuracy")
            if val_loss:
                self._scatter(f"{prefix}_test_loss_vs_learning_rate", lr, val_loss, "Learning rate", "Test loss")
        if seconds:
            self._line(f"{prefix}_epoch_time_vs_epoch", epochs, seconds, "Epoch", "Seconds")
        if history.get("parameter_norm"):
            self._line(f"{prefix}_parameter_norm_vs_epoch", epochs, history["parameter_norm"], "Epoch", "Parameter norm")
        if history.get("grad_norm"):
            self._line(f"{prefix}_gradient_norm_vs_epoch", epochs, history["grad_norm"], "Epoch", "Gradient norm")
        if history.get("trainable_parameters"):
            self._line(f"{prefix}_trainable_parameters_vs_epoch", epochs, history["trainable_parameters"], "Epoch", "Trainable parameters")

    def _training_history(self, data: dict[str, list]) -> None:
        import matplotlib.pyplot as plt

        epochs = data["epoch"]
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
        ideal_label = data.get("ideal_label", ["Ideal"])[0]
        hardware_label = data.get("hardware_label", ["Memristor"])[0]
        fig.suptitle(f"Training History - {ideal_label} vs {hardware_label}", fontsize=16, fontweight="bold")

        ax = axes[0]
        if data.get("ideal_train_accuracy"):
            values = _percent(data["ideal_train_accuracy"])
            ax.plot(epochs, values, color="#9bbce0", linewidth=2.0, alpha=0.45, label=_legend_label(f"{ideal_label} - Train", values, "Accuracy (%)"))
        if data.get("hardware_train_accuracy"):
            values = _percent(data["hardware_train_accuracy"])
            ax.plot(epochs, values, color="#f2a39a", linewidth=2.0, alpha=0.45, label=_legend_label(f"{hardware_label} - Train", values, "Accuracy (%)"))
        ideal_accuracy = _percent(data["ideal_accuracy"])
        hardware_accuracy = _percent(data["hardware_accuracy"])
        ax.plot(epochs, ideal_accuracy, color="#1f77b4", marker="o", markevery=_markevery(epochs), linewidth=2.2, label=_legend_label(f"{ideal_label} - Test", ideal_accuracy, "Accuracy (%)"))
        ax.plot(epochs, hardware_accuracy, color="#d62728", marker="o", markevery=_markevery(epochs), linewidth=2.2, label=_legend_label(f"{hardware_label} - Test", hardware_accuracy, "Accuracy (%)"))
        ax.set_title("Accuracy vs Epoch (Train + Test)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle="--", alpha=0.25)
        _style_legend(ax)

        ax = axes[1]
        if data.get("ideal_train_loss"):
            ax.plot(epochs, data["ideal_train_loss"], color="#9bbce0", linewidth=2.0, alpha=0.45, label=_legend_label(f"{ideal_label} - Train", data["ideal_train_loss"], "Loss"))
        if data.get("hardware_train_loss"):
            ax.plot(epochs, data["hardware_train_loss"], color="#f2a39a", linewidth=2.0, alpha=0.45, label=_legend_label(f"{hardware_label} - Train", data["hardware_train_loss"], "Loss"))
        ax.plot(epochs, data["ideal_loss"], color="#1f77b4", marker="o", markevery=_markevery(epochs), linewidth=2.2, label=_legend_label(f"{ideal_label} - Test", data["ideal_loss"], "Loss"))
        ax.plot(epochs, data["hardware_loss"], color="#d62728", marker="o", markevery=_markevery(epochs), linewidth=2.2, label=_legend_label(f"{hardware_label} - Test", data["hardware_loss"], "Loss"))
        ax.set_title("Loss vs Epoch (Train + Test)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.25)
        _style_legend(ax)

        fig.savefig(self.plots_dir / "training_history.png", dpi=200)
        plt.close(fig)

    def _multi_line(self, name: str, x, series: dict[str, list], xlabel: str, ylabel: str, title: str | None = None) -> None:
        export_excel(self.excel_dir / f"{name}.xlsx", {xlabel: list(x), **{label: list(y) for label, y in series.items()}})
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for label, y in series.items():
            plot_y, plot_ylabel = _plot_series(y, ylabel)
            ax.plot(x, plot_y, marker="o", markevery=_markevery(x), linewidth=2.0, label=_legend_label(label, plot_y, plot_ylabel))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(_plot_ylabel(ylabel))
        ax.set_title(title or name.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.30)
        _style_legend(ax)
        plt.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _scatter(self, name: str, x, y, xlabel: str, ylabel: str) -> None:
        export_excel(self.excel_dir / f"{name}.xlsx", {xlabel: list(x), ylabel: list(y)})
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_y, plot_ylabel = _plot_series(y, ylabel)
        label = _single_line_legend_label(name, x, plot_y, xlabel, plot_ylabel)
        ax.scatter(x, plot_y, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(plot_ylabel)
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.25)
        _style_legend(ax)
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _trace_line(self, name: str, trace: dict, title: str) -> None:
        pulse = trace.get("pulse") or list(range(len(trace.get("conductance", []))))
        conductance = trace.get("conductance", [])
        n = min(len(pulse), len(conductance))
        pulse = pulse[:n]
        conductance = conductance[:n]
        export_excel(
            self.excel_dir / f"{name}.xlsx",
            {
                "source_file": trace.get("file", ""),
                "source_sheet": trace.get("sheet", ""),
                "classification": trace.get("kind", ""),
                "classification_reason": trace.get("reason", ""),
                "Pulse/state index": pulse,
                "Conductance (S)": conductance,
            },
        )
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        label = _wrap_legend_text(f"{trace.get('kind', 'trace')} {Path(trace.get('file', '')).name} ({trace.get('reason', '')})")
        ax.plot(pulse, conductance, marker="o", markevery=_markevery(pulse), linewidth=2.0, label=label)
        ax.set_xlabel("Pulse/state index")
        ax.set_ylabel("Conductance (S)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.30)
        _style_legend(ax)
        plt.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _comparison_suite(self, data: dict[str, list]) -> None:
        epochs = data["epoch"]
        ideal_label = _first(data.get("ideal_label"), "Ideal")
        hardware_label = _first(data.get("hardware_label"), "Memristor")
        grouped = {
            "01_accuracy_loss_dashboard": {
                "why": "Accuracy and loss together support whether learning is real: accuracy should rise while loss falls.",
                "plots": [
                    ("Accuracy", {f"{ideal_label} train": data["ideal_train_accuracy"], f"{ideal_label} test": data["ideal_accuracy"], f"{hardware_label} train": data["hardware_train_accuracy"], f"{hardware_label} test": data["hardware_accuracy"]}, "Accuracy"),
                    ("Loss", {f"{ideal_label} train": data["ideal_train_loss"], f"{ideal_label} test": data["ideal_loss"], f"{hardware_label} train": data["hardware_train_loss"], f"{hardware_label} test": data["hardware_loss"]}, "Loss"),
                ],
            },
            "02_generalization_dashboard": {
                "why": "Train-test gaps and error rates support whether the model is overfitting or generalizing.",
                "plots": [
                    ("Accuracy gap", {f"{ideal_label} train-test": data["ideal_accuracy_gap"], f"{hardware_label} train-test": data["hardware_accuracy_gap"]}, "Gap"),
                    ("Test error", {f"{ideal_label} error": data["ideal_error_rate"], f"{hardware_label} error": data["hardware_error_rate"]}, "Error rate"),
                ],
            },
            "03_memristor_impact_dashboard": {
                "why": "Accuracy drop and loss increase show how much device non-ideality changes the same trained model.",
                "plots": [
                    ("Memristor accuracy drop", {f"{ideal_label} - {hardware_label}": data["accuracy_drop"]}, "Accuracy drop"),
                    ("Memristor loss increase", {f"{hardware_label} - {ideal_label}": data["loss_increase"]}, "Loss increase"),
                ],
            },
            "04_learning_rate_dashboard": {
                "why": "Learning-rate plots document the optimization setting used to produce the training curves.",
                "plots": [
                    ("LR vs epoch", {f"{ideal_label} lr": data["ideal_learning_rate"], f"{hardware_label} lr": data["hardware_learning_rate"]}, "Learning rate"),
                    ("Accuracy vs LR", {f"{ideal_label} acc": data["ideal_accuracy"], f"{hardware_label} acc": data["hardware_accuracy"]}, "Accuracy"),
                ],
                "x_for_second": data["ideal_learning_rate"],
            },
            "06_curve_quality_dashboard": {
                "why": "Raw and smoothed curves together make noisy training trends easier to read while preserving the original data.",
                "plots": [
                    ("Smoothed accuracy", {f"{ideal_label} raw": data["ideal_accuracy"], f"{ideal_label} smooth": data["ideal_accuracy_smooth"], f"{hardware_label} raw": data["hardware_accuracy"], f"{hardware_label} smooth": data["hardware_accuracy_smooth"]}, "Accuracy"),
                    ("Smoothed loss", {f"{ideal_label} raw": data["ideal_loss"], f"{ideal_label} smooth": data["ideal_loss_smooth"], f"{hardware_label} raw": data["hardware_loss"], f"{hardware_label} smooth": data["hardware_loss_smooth"]}, "Loss"),
                ],
            },
        }

        if data.get("energy_pj") and data.get("latency_ns"):
            grouped["05_hardware_cost_dashboard"] = {
                "why": "Energy and latency plotted with accuracy support the hardware tradeoff behind the memristor result.",
                "plots": [
                    ("Energy vs epoch", {"energy pJ": data["energy_pj"]}, "Energy (pJ)"),
                    ("Latency vs epoch", {"latency ns": data["latency_ns"]}, "Latency (ns)"),
                ],
            }

        explanations = []
        for name, spec in grouped.items():
            self._dashboard(name, epochs, spec)
            export_excel(self.excel_dir / f"{name}.xlsx", _dashboard_excel_data(epochs, spec))
            explanations.append({"graph": f"{name}.png", "supports": spec["why"]})

        detailed = {
            "comparison_accuracy_drop_vs_epoch": ("Memristor accuracy drop across epochs supports the robustness claim.", data["accuracy_drop"], "Accuracy drop"),
            "comparison_loss_increase_vs_epoch": ("Memristor loss increase supports the non-ideality penalty analysis.", data["loss_increase"], "Loss increase"),
            "comparison_best_accuracy_vs_epoch": ("Best-so-far curves support convergence and final model selection.", None, "Accuracy"),
            "comparison_error_rate_vs_epoch": ("Error-rate curves support accuracy results from the opposite direction.", None, "Error rate"),
            "comparison_epoch_time_vs_epoch": ("Epoch time supports compute/runtime cost reporting.", None, "Seconds"),
            "comparison_accuracy_smoothed": ("Raw plus smoothed accuracy supports clean curve interpretation without hiding original values.", None, "Accuracy"),
            "comparison_loss_smoothed": ("Raw plus smoothed loss supports clean convergence interpretation without hiding original values.", None, "Loss"),
        }
        self._line("comparison_accuracy_drop_vs_epoch", epochs, data["accuracy_drop"], "Epoch", "Accuracy drop")
        self._line("comparison_loss_increase_vs_epoch", epochs, data["loss_increase"], "Epoch", "Loss increase")
        self._multi_line("comparison_best_accuracy_vs_epoch", epochs, {ideal_label: data["ideal_best_accuracy"], hardware_label: data["hardware_best_accuracy"]}, "Epoch", "Best accuracy")
        self._multi_line("comparison_error_rate_vs_epoch", epochs, {ideal_label: data["ideal_error_rate"], hardware_label: data["hardware_error_rate"]}, "Epoch", "Error rate")
        self._multi_line("comparison_epoch_time_vs_epoch", epochs, {ideal_label: data["ideal_seconds"], hardware_label: data["hardware_seconds"]}, "Epoch", "Seconds")
        for graph, (why, _, _) in detailed.items():
            explanations.append({"graph": f"{graph}.png", "supports": why})
        explanations.extend(_standard_explanations())

        pd.DataFrame(explanations).to_excel(self.excel_dir / "plot_explanations.xlsx", index=False)
        self._write_explanations(explanations)
        self._write_final_summary(ideal=None, hardware=None, data=data)
        self._write_artifact_index()

    def _dashboard(self, name: str, epochs: list, spec: dict) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(spec["plots"]), figsize=(9 * len(spec["plots"]), 5), constrained_layout=True)
        if len(spec["plots"]) == 1:
            axes = [axes]
        fig.suptitle(name.replace("_", " ").title(), fontsize=15, fontweight="bold")
        for idx, (title, series, ylabel) in enumerate(spec["plots"]):
            ax = axes[idx]
            x = spec.get("x_for_second", epochs) if idx == 1 and "x_for_second" in spec else epochs
            for label, values in series.items():
                plot_values, plot_ylabel = _plot_series(values, ylabel)
                legend_label = _legend_label(label, plot_values, plot_ylabel)
                if idx == 1 and "x_for_second" in spec:
                    ax.scatter(x, plot_values, label=legend_label)
                else:
                    ax.plot(x, plot_values, marker="o", markevery=_markevery(x), label=legend_label)
            ax.set_title(title)
            ax.set_xlabel("Learning rate" if idx == 1 and "x_for_second" in spec else "Epoch")
            ax.set_ylabel(_plot_ylabel(ylabel))
            ax.grid(True, linestyle="--", alpha=0.25)
            _style_legend(ax)
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _write_explanations(self, explanations: list[dict[str, str]]) -> None:
        if self.reports_dir is None:
            return
        lines = ["# Plot Explanations", ""]
        for item in explanations:
            lines.append(f"- `{item['graph']}`: {item['supports']}")
        (self.reports_dir / "plot_explanations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_final_summary(self, ideal: dict | None, hardware: dict | None, data: dict[str, list]) -> None:
        if self.reports_dir is None:
            return
        dataset = _first(data.get("dataset"), "")
        model = _first(data.get("model"), "")
        ideal_label = _first(data.get("ideal_label"), "Ideal")
        hardware_label = _first(data.get("hardware_label"), "Memristor")
        rows = [
            {"section": "run", "metric": "dataset", "value": dataset},
            {"section": "run", "metric": "model", "value": model},
            {"section": "run", "metric": "epochs_completed", "value": len(data.get("epoch", []))},
            {"section": "performance", "metric": f"{ideal_label} final accuracy", "value": _fmt_pct(_last(data.get("ideal_accuracy")))},
            {"section": "performance", "metric": f"{hardware_label} final accuracy", "value": _fmt_pct(_last(data.get("hardware_accuracy")))},
            {"section": "performance", "metric": f"{ideal_label} best accuracy", "value": _fmt_pct(_last(data.get("ideal_best_accuracy")))},
            {"section": "performance", "metric": f"{hardware_label} best accuracy", "value": _fmt_pct(_last(data.get("hardware_best_accuracy")))},
            {"section": "performance", "metric": "final accuracy drop", "value": _fmt_pct(_last(data.get("accuracy_drop")))},
            {"section": "loss", "metric": f"{ideal_label} final loss", "value": _fmt_float(_last(data.get("ideal_loss")))},
            {"section": "loss", "metric": f"{hardware_label} final loss", "value": _fmt_float(_last(data.get("hardware_loss")))},
            {"section": "loss", "metric": "final loss increase", "value": _fmt_float(_last(data.get("loss_increase")))},
            {"section": "optimization", "metric": "initial learning rate", "value": _fmt_float(_first(data.get("ideal_learning_rate")))},
            {"section": "optimization", "metric": "final learning rate", "value": _fmt_float(_last(data.get("ideal_learning_rate")))},
            {"section": "hardware", "metric": "final energy pJ", "value": _fmt_float(_last(data.get("energy_pj")))},
            {"section": "hardware", "metric": "final latency ns", "value": _fmt_float(_last(data.get("latency_ns")))},
            {"section": "artifacts", "metric": "main plot", "value": "outputs/plots/training_history.png"},
            {"section": "artifacts", "metric": "plot explanations", "value": "outputs/reports/plot_explanations.md"},
        ]
        pd.DataFrame(rows).to_excel(self.excel_dir / "final_summary.xlsx", index=False)

        lines = [
            "# Final Training Summary",
            "",
            "## Run",
            f"- Dataset: `{dataset}`",
            f"- Model: `{model}`",
            f"- Epochs completed: `{len(data.get('epoch', []))}`",
            "",
            "## Performance",
            f"- {ideal_label} final accuracy: {_fmt_pct(_last(data.get('ideal_accuracy')))}",
            f"- {hardware_label} final accuracy: {_fmt_pct(_last(data.get('hardware_accuracy')))}",
            f"- {ideal_label} best accuracy: {_fmt_pct(_last(data.get('ideal_best_accuracy')))}",
            f"- {hardware_label} best accuracy: {_fmt_pct(_last(data.get('hardware_best_accuracy')))}",
            f"- Final accuracy drop: {_fmt_pct(_last(data.get('accuracy_drop')))}",
            f"- {ideal_label} final loss: {_fmt_float(_last(data.get('ideal_loss')))}",
            f"- {hardware_label} final loss: {_fmt_float(_last(data.get('hardware_loss')))}",
            f"- Final loss increase: {_fmt_float(_last(data.get('loss_increase')))}",
            "",
            "## Optimization",
            f"- Initial LR: {_fmt_float(_first(data.get('ideal_learning_rate')))}",
            f"- Final LR: {_fmt_float(_last(data.get('ideal_learning_rate')))}",
            "- LR scheduler: ReduceLROnPlateau when validation loss stalls.",
            "",
            "## Hardware Metrics",
            f"- Final estimated energy: {_fmt_float(_last(data.get('energy_pj')))} pJ",
            f"- Final estimated latency: {_fmt_float(_last(data.get('latency_ns')))} ns",
            "",
            "## Quality Notes",
            "- Accuracy, loss, error-rate, generalization-gap, smoothed-curve, confusion-matrix, LR, parameter-norm, gradient-norm, energy, and latency plots are exported.",
            "- Every graph has a matching Excel file in `outputs/excel`.",
            "- Checkpoints include model, optimizer, scheduler, config, history, and extracted device model for restart.",
        ]
        (self.reports_dir / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_artifact_index(self) -> None:
        if self.reports_dir is None:
            return
        rows = []
        for folder, kind in [(self.plots_dir, "plot"), (self.excel_dir, "excel"), (self.reports_dir, "report")]:
            for path in sorted(folder.glob("*")):
                if path.is_file():
                    rows.append({"type": kind, "name": path.name, "path": str(path), "bytes": path.stat().st_size})
        pd.DataFrame(rows).to_excel(self.excel_dir / "artifact_index.xlsx", index=False)
        lines = ["# Artifact Index", ""]
        for row in rows:
            lines.append(f"- `{row['type']}` `{row['name']}`: {row['path']}")
        (self.reports_dir / "artifact_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _normalized_matrix(self, name: str, matrix: np.ndarray, xlabel: str, ylabel: str, class_names: list[str] | None = None, title_suffix: str = "") -> None:
        """Phase 6: Row-normalized confusion matrix (each row sums to 1)."""
        matrix = np.asarray(matrix, dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        normalized = matrix / row_sums
        n = matrix.shape[0]
        labels = _labels_for(class_names, n)

        with pd.ExcelWriter(self.excel_dir / f"{name}.xlsx") as writer:
            pd.DataFrame(normalized, index=labels, columns=labels).to_excel(writer, sheet_name="normalized")
            pd.DataFrame(matrix.astype(int), index=labels, columns=labels).to_excel(writer, sheet_name="raw_counts")

        import matplotlib.pyplot as plt
        size = max(7, min(16, n * 0.9))
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        title = name.replace("_", " ").title() + " (Row-Normalized)"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        tick_fs = _tick_fontsize(n)
        ax.set_xticks(range(n), labels=labels, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticks(range(n), labels=labels, fontsize=tick_fs)
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    val = normalized[i, j]
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if val > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion")
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=220)
        plt.close(fig)

    def _class_metrics(self, name: str, cm: np.ndarray, history: dict, class_names: list[str] | None = None, title_suffix: str = "") -> None:
        """Phase 6: Per-class precision, recall, F1, support from confusion matrix."""
        cm = np.asarray(cm, dtype=float)
        n = cm.shape[0]
        labels = _labels_for(class_names, n)
        rows = []
        for i in range(n):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()
            precision = tp / max(tp + fp, 1e-9)
            recall = tp / max(tp + fn, 1e-9)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9) if (precision + recall) > 0 else 0.0
            rows.append({
                "class_index": i,
                "class": labels[i],
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(support),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            })

        # Macro averages
        macro_p = np.mean([r["precision"] for r in rows])
        macro_r = np.mean([r["recall"] for r in rows])
        macro_f1 = np.mean([r["f1_score"] for r in rows])
        total_support = sum(r["support"] for r in rows)
        # Weighted averages
        weighted_p = sum(r["precision"] * r["support"] for r in rows) / max(total_support, 1)
        weighted_r = sum(r["recall"] * r["support"] for r in rows) / max(total_support, 1)
        weighted_f1 = sum(r["f1_score"] * r["support"] for r in rows) / max(total_support, 1)

        rows.append({"class_index": "", "class": "macro_avg", "precision": macro_p, "recall": macro_r, "f1_score": macro_f1, "support": total_support})
        rows.append({"class_index": "", "class": "weighted_avg", "precision": weighted_p, "recall": weighted_r, "f1_score": weighted_f1, "support": total_support})

        df = pd.DataFrame(rows)
        df.to_excel(self.excel_dir / f"{name}.xlsx", index=False)

        # Bar chart of per-class F1
        import matplotlib.pyplot as plt
        class_rows = [r for r in rows if isinstance(r["class_index"], int)]
        width = max(10, min(20, 0.45 * len(class_rows) + 4))
        fig, ax = plt.subplots(figsize=(width, 5))
        positions = [r["class_index"] for r in class_rows]
        class_labels = [r["class"] for r in class_rows]
        f1s = [r["f1_score"] for r in class_rows]
        colors = ["#2ca02c" if f > 0.9 else "#ff7f0e" if f > 0.8 else "#d62728" for f in f1s]
        ax.bar(positions, f1s, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=macro_f1, color="gray", linestyle="--", linewidth=1, label=f"Macro F1: {macro_f1:.4f}")
        ax.set_xlabel("Class")
        ax.set_ylabel("F1 Score")
        title = f"Per-Class F1 Score — {name.replace('_', ' ').title()}"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        ax.set_xticks(positions, labels=class_labels, rotation=45 if n > 10 else 0, ha="right" if n > 10 else "center", fontsize=_tick_fontsize(n))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        _style_legend(ax)
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _per_class_accuracy_bar(self, name: str, cm: np.ndarray, class_names: list[str] | None, title_suffix: str = "") -> None:
        """Phase 8: Per-class accuracy (recall) bar chart with macro line."""
        import matplotlib.pyplot as plt

        cm = np.asarray(cm, dtype=float)
        n = cm.shape[0]
        labels = _labels_for(class_names, n)
        supports = cm.sum(axis=1)
        accs = np.divide(np.diag(cm), np.where(supports == 0, 1, supports))
        macro_acc = float(np.mean(accs)) if n else 0.0

        rows = [{"class_index": i, "class": labels[i], "support": int(supports[i]), "accuracy": float(accs[i])} for i in range(n)]
        pd.DataFrame(rows).to_excel(self.excel_dir / f"{name}.xlsx", index=False)

        width = max(10, min(20, 0.45 * n + 4))
        fig, ax = plt.subplots(figsize=(width, 5))
        colors = ["#2ca02c" if a > 0.9 else "#ff7f0e" if a > 0.75 else "#d62728" for a in accs]
        ax.bar(range(n), accs * 100, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(macro_acc * 100, color="gray", linestyle="--", linewidth=1, label=f"Macro acc: {macro_acc * 100:.2f}%")
        ax.set_xticks(range(n), labels=labels, rotation=45 if n > 10 else 0, ha="right" if n > 10 else "center", fontsize=_tick_fontsize(n))
        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 105)
        title = "Per-Class Accuracy"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        _style_legend(ax)
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def _top_confused_pairs(self, name: str, cm: np.ndarray, class_names: list[str] | None, title_suffix: str = "", top_k: int = 15) -> None:
        """Phase 8: Horizontal bar of the top-K most-confused (true -> predicted) pairs."""
        import matplotlib.pyplot as plt

        cm = np.asarray(cm, dtype=float)
        n = cm.shape[0]
        if n < 2:
            return
        labels = _labels_for(class_names, n)
        off = cm.copy()
        np.fill_diagonal(off, 0)
        if off.sum() <= 0:
            return
        idxs = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0]
        pairs = []
        for i, j in idxs:
            if off[i, j] <= 0:
                break
            pairs.append({
                "true": labels[i],
                "predicted": labels[j],
                "count": int(off[i, j]),
                "share_of_class": float(off[i, j] / max(cm[i, :].sum(), 1)),
            })
            if len(pairs) >= top_k:
                break
        if not pairs:
            return
        pd.DataFrame(pairs).to_excel(self.excel_dir / f"{name}.xlsx", index=False)

        pair_labels = [f"{p['true']} -> {p['predicted']}" for p in pairs][::-1]
        counts = [p["count"] for p in pairs][::-1]
        height = max(4, 0.35 * len(pairs) + 2)
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(pair_labels, counts, color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Misclassified samples")
        title = f"Top {len(pairs)} Confused Class Pairs"
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close(fig)

    def class_wise_comparison(self, ideal: dict | None, hardware: dict | None) -> None:
        """Phase 6: Compare per-class performance between ideal and hardware."""
        if not ideal or not hardware:
            return
        ideal_cm = ideal.get("confusion_matrix")
        hw_cm = hardware.get("confusion_matrix")
        if ideal_cm is None or hw_cm is None:
            return

        ideal_cm = np.asarray(ideal_cm, dtype=float)
        hw_cm = np.asarray(hw_cm, dtype=float)
        n = min(ideal_cm.shape[0], hw_cm.shape[0])
        ideal_cm = ideal_cm[:n, :n]
        hw_cm = hw_cm[:n, :n]
        class_names = _class_names_from_history(ideal, n) or _class_names_from_history(hardware, n)
        labels = _labels_for(class_names, n)

        rows = []
        for i in range(n):
            # Ideal class accuracy = recall
            ideal_support = ideal_cm[i, :].sum()
            hw_support = hw_cm[i, :].sum()
            ideal_acc = ideal_cm[i, i] / max(ideal_support, 1e-9)
            hw_acc = hw_cm[i, i] / max(hw_support, 1e-9)
            # Per-class F1
            ideal_tp, hw_tp = ideal_cm[i, i], hw_cm[i, i]
            ideal_fp, hw_fp = ideal_cm[:, i].sum() - ideal_tp, hw_cm[:, i].sum() - hw_tp
            ideal_fn, hw_fn = ideal_cm[i, :].sum() - ideal_tp, hw_cm[i, :].sum() - hw_tp
            ideal_p = ideal_tp / max(ideal_tp + ideal_fp, 1e-9)
            hw_p = hw_tp / max(hw_tp + hw_fp, 1e-9)
            ideal_r = ideal_tp / max(ideal_tp + ideal_fn, 1e-9)
            hw_r = hw_tp / max(hw_tp + hw_fn, 1e-9)
            ideal_f1 = 2 * ideal_p * ideal_r / max(ideal_p + ideal_r, 1e-9) if (ideal_p + ideal_r) > 0 else 0
            hw_f1 = 2 * hw_p * hw_r / max(hw_p + hw_r, 1e-9) if (hw_p + hw_r) > 0 else 0

            rows.append({
                "class_index": i,
                "class": labels[i],
                "ideal_accuracy": ideal_acc,
                "hardware_accuracy": hw_acc,
                "accuracy_drop": ideal_acc - hw_acc,
                "ideal_f1": ideal_f1,
                "hardware_f1": hw_f1,
                "f1_drop": ideal_f1 - hw_f1,
                "ideal_precision": ideal_p,
                "hardware_precision": hw_p,
                "ideal_recall": ideal_r,
                "hardware_recall": hw_r,
            })

        df = pd.DataFrame(rows)
        df.to_excel(self.excel_dir / "class_wise_comparison.xlsx", index=False)

        # Grouped bar chart: ideal vs hardware per-class accuracy
        import matplotlib.pyplot as plt
        width_in = max(12, min(22, 0.55 * n + 6))
        fig, axes = plt.subplots(1, 2, figsize=(width_in, 6), constrained_layout=True)
        fig.suptitle("Class-Wise Ideal vs Hardware Comparison", fontsize=14, fontweight="bold")

        x = np.arange(n)
        width = 0.35
        tick_fs = _tick_fontsize(n)

        # Accuracy comparison
        ax = axes[0]
        ideal_accs = [r["ideal_accuracy"] * 100 for r in rows]
        hw_accs = [r["hardware_accuracy"] * 100 for r in rows]
        ax.bar(x - width / 2, ideal_accs, width, label="Ideal", color="#1f77b4", alpha=0.8)
        ax.bar(x + width / 2, hw_accs, width, label="Hardware", color="#d62728", alpha=0.8)
        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Per-Class Accuracy")
        ax.set_xticks(x, labels=labels, rotation=45 if n > 10 else 0, ha="right" if n > 10 else "center", fontsize=tick_fs)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        _style_legend(ax)

        # Accuracy drop
        ax = axes[1]
        drops = [r["accuracy_drop"] * 100 for r in rows]
        colors = ["#d62728" if d > 2 else "#ff7f0e" if d > 0 else "#2ca02c" for d in drops]
        ax.bar(x, drops, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.set_xlabel("Class")
        ax.set_ylabel("Accuracy Drop (%)")
        ax.set_title("Per-Class Accuracy Drop (Ideal - Hardware)")
        ax.set_xticks(x, labels=labels, rotation=45 if n > 10 else 0, ha="right" if n > 10 else "center", fontsize=tick_fs)
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")

        fig.savefig(self.plots_dir / "class_wise_comparison.png", dpi=200)
        plt.close(fig)

        # Phase 8: confusion matrix difference heatmap (hardware - ideal, row-normalized)
        ideal_rows = np.where(ideal_cm.sum(axis=1, keepdims=True) == 0, 1, ideal_cm.sum(axis=1, keepdims=True))
        hw_rows = np.where(hw_cm.sum(axis=1, keepdims=True) == 0, 1, hw_cm.sum(axis=1, keepdims=True))
        ideal_norm = ideal_cm / ideal_rows
        hw_norm = hw_cm / hw_rows
        diff = hw_norm - ideal_norm

        pd.DataFrame(diff, index=labels, columns=labels).to_excel(self.excel_dir / "confusion_matrix_diff_ideal_minus_hardware.xlsx")
        size = max(7, min(16, n * 0.9))
        fig, ax = plt.subplots(figsize=(size, size))
        vmax = float(np.abs(diff).max()) if diff.size else 1.0
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title("Confusion Matrix Shift (Hardware − Ideal, row-normalized)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n), labels=labels, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticks(range(n), labels=labels, fontsize=tick_fs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ proportion")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "confusion_matrix_diff_ideal_vs_hardware.png", dpi=220)
        plt.close(fig)

    def _hist(self, name: str, values, xlabel: str) -> None:
        export_excel(self.excel_dir / f"{name}.xlsx", {xlabel: list(values)})
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(values, bins=min(20, max(2, len(values))))
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"{name}.png", dpi=200)
        plt.close()

    def _matrix(self, name: str, matrix: np.ndarray, xlabel: str, ylabel: str, class_names: list[str] | None = None, title_suffix: str = "") -> None:
        matrix = np.asarray(matrix, dtype=float)
        total = matrix.sum()
        percent = matrix / total * 100.0 if total > 0 else np.zeros_like(matrix)
        n = matrix.shape[0]
        labels = _labels_for(class_names, n)
        with pd.ExcelWriter(self.excel_dir / f"{name}.xlsx") as writer:
            pd.DataFrame(matrix.astype(int), index=labels, columns=labels).to_excel(writer, sheet_name="counts")
            pd.DataFrame(percent, index=labels, columns=labels).to_excel(writer, sheet_name="percent")
        import matplotlib.pyplot as plt

        size = max(7, min(16, n * 0.9))
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(matrix, cmap="Blues")
        title = name.replace("_", " ").title()
        if title_suffix:
            title += f" — {title_suffix}"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        tick_fs = _tick_fontsize(n)
        ax.set_xticks(range(n), labels=labels, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticks(range(n), labels=labels, fontsize=tick_fs)
        threshold = matrix.max() * 0.55 if matrix.size else 0
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    value = int(matrix[i, j])
                    if value == 0:
                        text = "0"
                    else:
                        text = f"{value}\n{percent[i, j]:.1f}%"
                    ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if matrix[i, j] > threshold else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png", dpi=220)
        plt.close(fig)


def _param_rows(mode: str, history: dict, cfg: dict) -> list[dict]:
    epochs = list(history.get("epoch", []))
    metadata = history.get("metadata", {})
    rows = []
    for idx, epoch in enumerate(epochs):
        rows.append(
            {
                "mode": mode,
                "curve": metadata.get("curve", mode),
                "epoch": epoch,
                "dataset": metadata.get("dataset") or cfg.get("dataset", ""),
                "model": metadata.get("model") or cfg.get("model", ""),
                "training_mode": metadata.get("training_mode") or cfg.get("training_mode", ""),
                "optimizer": "AdamW",
                "batch_size": cfg.get("batch_size", ""),
                "weight_decay": cfg.get("weight_decay", ""),
                "trainable_parameters": _at(history.get("trainable_parameters"), idx, metadata.get("trainable_parameters", "")),
                "learning_rate": _at(history.get("learning_rate"), idx, ""),
                "parameter_norm": _at(history.get("parameter_norm"), idx, ""),
                "gradient_norm": _at(history.get("grad_norm"), idx, ""),
                "train_loss": _at(history.get("train_loss"), idx, ""),
                "train_accuracy": _at(history.get("train_accuracy"), idx, ""),
                "test_loss": _at(history.get("val_loss"), idx, ""),
                "test_accuracy": _at(history.get("val_accuracy"), idx, ""),
                "epoch_seconds": _at(history.get("seconds"), idx, ""),
                "energy_pj": _at(history.get("energy_pj"), idx, ""),
                "latency_ns": _at(history.get("latency_ns"), idx, ""),
                "adc_bits": _at(history.get("adc_bits"), idx, cfg.get("hardware", {}).get("adc_bits", "")),
                "weight_bits": _at(history.get("weight_bits"), idx, cfg.get("hardware", {}).get("weight_bits", "")),
                "status": history.get("status", ""),
            }
        )
    return rows


def _param_info_markdown(rows: list[dict], histories: dict[str, dict], cfg: dict, device: DeviceModel) -> str:
    epochs = [row["epoch"] for row in rows]
    lr_values = [row["learning_rate"] for row in rows if row["learning_rate"] != ""]
    lr_changes = _lr_change_rows(rows)
    scheduler = cfg.get("lr_scheduler") or {}
    dynamic_lr = bool(scheduler and scheduler.get("type", "none") not in {"none", None})
    hardware = cfg.get("hardware", {})
    device_info = device.to_dict()
    dataset = _cfg_history_value(cfg, histories, "dataset")
    model = _cfg_history_value(cfg, histories, "model")
    training_mode = _cfg_history_value(cfg, histories, "training_mode")
    requested_epochs = cfg.get("max_epochs") or (max(epochs) if epochs else "")
    initial_lr = cfg.get("learning_rate") if cfg.get("learning_rate") not in {None, ""} else _first(lr_values, "")

    lines = [
        "# Parameter Information",
        "",
        "## Run",
        f"- Dataset: `{dataset}`",
        f"- Model: `{model}`",
        f"- Training mode: `{training_mode}`",
        f"- Epoch range in this report: `{min(epochs) if epochs else 0}` to `{max(epochs) if epochs else 0}`",
        f"- Epochs requested: `{requested_epochs}`",
        f"- Batch size: `{cfg.get('batch_size', '')}`",
        f"- Optimizer: `AdamW`",
        f"- Weight decay: `{cfg.get('weight_decay', '')}`",
        f"- Initial learning rate: `{initial_lr}`",
        f"- LR scheduler: `{scheduler.get('type', 'none') if dynamic_lr else 'none'}`",
    ]
    if dynamic_lr:
        lines.extend(
            [
                f"- LR factor: `{scheduler.get('factor', '')}`",
                f"- LR patience: `{scheduler.get('patience', '')}`",
                f"- Minimum LR: `{scheduler.get('min_lr', '')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Learning Rate",
            f"- LR recorded per epoch: `{_fmt_float(min(lr_values))}` to `{_fmt_float(max(lr_values))}`" if lr_values else "- LR recorded per epoch: `n/a`",
        ]
    )
    if dynamic_lr:
        if lr_changes:
            lines.append("- Dynamic LR changes detected:")
            for row in lr_changes:
                lines.append(f"  - Epoch `{row['epoch']}` `{row['mode']}`: `{_fmt_float(row['previous_lr'])}` -> `{_fmt_float(row['learning_rate'])}`")
        else:
            lines.append("- Dynamic LR scheduler was enabled, but the recorded LR stayed constant during the completed epochs.")
    else:
        lines.append("- Dynamic LR scheduler was disabled; LR stayed under optimizer control only.")

    lines.extend(
        [
            "",
            "## Hardware And Device Settings",
            f"- Weight bits: `{hardware.get('weight_bits', '')}`",
            f"- Activation bits: `{hardware.get('activation_bits', '')}`",
            f"- ADC bits: `{hardware.get('adc_bits', '')}`",
            f"- DAC bits: `{hardware.get('dac_bits', '')}`",
            f"- Read noise: `{hardware.get('read_noise', '')}`",
            f"- Cycle variation scale: `{hardware.get('cycle_variation_scale', '')}`",
            f"- Stuck-at-zero rate: `{hardware.get('stuck_at_zero_rate', '')}`",
            f"- Stuck-at-one rate: `{hardware.get('stuck_at_one_rate', '')}`",
            f"- Device states: `{device_info.get('num_states', '')}`",
            f"- Device ON/OFF ratio: `{_fmt_float(device_info.get('on_off_ratio', ''))}`",
            f"- Cycle variation sigma: `{_fmt_float(device_info.get('cycle_variation_sigma', ''))}`",
            "",
            "## Epoch Parameters",
        ]
    )

    table_columns = [
        "mode",
        "epoch",
        "trainable_parameters",
        "learning_rate",
        "parameter_norm",
        "gradient_norm",
        "train_accuracy",
        "test_accuracy",
        "train_loss",
        "test_loss",
        "energy_pj",
        "latency_ns",
    ]
    lines.extend(_markdown_table(rows, table_columns))
    lines.extend(
        [
            "",
            "## Notes",
            "- `trainable_parameters` is the count of learnable model weights for the selected architecture.",
            "- `parameter_norm` and `gradient_norm` track model weight scale and optimization stability per epoch.",
            "- Hardware energy and latency are populated for memristor-aware evaluation rows.",
            "- Full source data is also saved in `outputs/excel/paramInfo.xlsx`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _lr_change_rows(rows: list[dict]) -> list[dict]:
    changes = []
    previous_by_mode = {}
    for row in rows:
        lr = row.get("learning_rate")
        if lr == "":
            continue
        mode = row.get("mode", "")
        previous = previous_by_mode.get(mode)
        if previous is not None and abs(float(lr) - float(previous)) > 1e-15:
            changed = dict(row)
            changed["previous_lr"] = previous
            changes.append(changed)
        previous_by_mode[mode] = lr
    return changes


def _cfg_history_value(cfg: dict, histories: dict[str, dict], key: str, default: str = ""):
    if cfg.get(key) not in {None, ""}:
        return cfg.get(key)
    for history in histories.values():
        value = history.get("metadata", {}).get(key)
        if value not in {None, ""}:
            return value
    return default


def _markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(column, "")) for column in columns) + " |")
    return lines


def _fmt_cell(value) -> str:
    if isinstance(value, float):
        return _fmt_float(value)
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _at(values, idx: int, default=""):
    if isinstance(values, (list, tuple)) and idx < len(values):
        return values[idx]
    return default


def _percent(values: list[float]) -> list[float]:
    return [float(v) * 100.0 for v in values]


def _markevery(values: list) -> int:
    return max(1, len(values) // 20)


def _style_legend(ax) -> None:
    legend = ax.legend(loc="best", frameon=True, fontsize=8, handlelength=2.6, borderpad=0.7, labelspacing=0.5)
    if legend is not None:
        legend.get_frame().set_alpha(0.92)


def _legend_label(label: str, values, ylabel: str) -> str:
    nums = _finite_values(values)
    if not nums:
        return _wrap_legend_text(label)
    lowered = f"{label} {ylabel}".lower()
    final = nums[-1]
    if _is_percent_label(ylabel):
        if "best" in lowered:
            suffix = f"Best: {_fmt_plot_value(max(nums), ylabel)}"
        elif "accuracy" in lowered and not any(token in lowered for token in ["error", "gap", "drop"]):
            suffix = f"Final: {_fmt_plot_value(final, ylabel)}, Best: {_fmt_plot_value(max(nums), ylabel)}"
        else:
            suffix = f"Final: {_fmt_plot_value(final, ylabel)}"
    else:
        suffix = f"Final: {_fmt_plot_value(final, ylabel)}"
    return _wrap_legend_text(f"{label} ({suffix})")


def _single_line_legend_label(name: str, x, y, xlabel: str, ylabel: str) -> str:
    base = name.replace("_", " ").title()
    parts = []
    x_values = _finite_values(x)
    y_values = _finite_values(y)
    if y_values:
        parts.append(f"Final {ylabel.lower()}: {_fmt_plot_value(y_values[-1], ylabel)}")
        lowered = f"{name} {ylabel}".lower()
        if _is_percent_label(ylabel) and "accuracy" in lowered and not any(token in lowered for token in ["error", "gap", "drop"]):
            parts.append(f"Best: {_fmt_plot_value(max(y_values), ylabel)}")
    plot_xlabel = _plot_ylabel(xlabel)
    if x_values and _is_percent_label(plot_xlabel):
        parts.insert(0, f"Final {plot_xlabel.lower()}: {_fmt_plot_value(x_values[-1] * 100.0 if max(abs(v) for v in x_values) <= 1.5 else x_values[-1], plot_xlabel)}")
    if not parts:
        return _wrap_legend_text(base)
    return _wrap_legend_text(f"{base} ({'; '.join(parts)})")


def _finite_values(values) -> list[float]:
    if values is None:
        return []
    out = []
    for value in list(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def _is_percent_label(ylabel: str) -> bool:
    return "%" in ylabel


def _fmt_plot_value(value: float, ylabel: str) -> str:
    if _is_percent_label(ylabel):
        return f"{float(value):.2f}%"
    return f"{float(value):.4g}"


def _wrap_legend_text(text: str, width: int = 48) -> str:
    return "\n".join(textwrap.wrap(text, width=width, subsequent_indent="  ")) if len(text) > width else text


def _plot_series(values: list, ylabel: str) -> tuple[list[float], str]:
    if _is_fraction_metric(ylabel, values):
        return _percent(values), _plot_ylabel(ylabel)
    return [float(v) for v in values], _plot_ylabel(ylabel)


def _plot_ylabel(ylabel: str) -> str:
    lowered = ylabel.lower()
    if any(token in lowered for token in ["accuracy", "error", "gap", "drop"]) and "%" not in ylabel:
        return f"{ylabel} (%)"
    return ylabel


def _is_fraction_metric(ylabel: str, values: list) -> bool:
    lowered = ylabel.lower()
    if not any(token in lowered for token in ["accuracy", "error", "gap", "drop"]):
        return False
    nums = [abs(float(v)) for v in values if v is not None]
    return bool(nums) and max(nums) <= 1.5


def _curve_label(history: dict, fallback: str) -> str:
    meta = history.get("metadata", {})
    curve = meta.get("curve", fallback)
    dataset = meta.get("dataset") or ""
    model = meta.get("model") or ""
    if dataset:
        base = " ".join(part for part in [dataset, model] if part)
        return f"{base} ({curve})" if curve else base
    return " ".join(part for part in [curve, model, dataset] if part)


def _history_label(history: dict, prefix: str) -> str:
    if history.get("metadata"):
        return _curve_label(history, prefix)
    return "Memristor" if prefix == "hardware_aware" else prefix.title()


def _first(values, default=None):
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return default


def _first_trace(device: DeviceModel, kind: str) -> dict | None:
    for trace in getattr(device, "source_traces", []) or []:
        if trace.get("kind") == kind and trace.get("conductance"):
            return trace
    return None


def _last(values, default=0.0):
    if isinstance(values, (list, tuple)) and values:
        return values[-1]
    return default


def _fmt_pct(value) -> str:
    try:
        return f"{float(value) * 100.0:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_float(value) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return "n/a"


def _aligned(values: list, n: int, default: float = 0.0) -> list:
    values = list(values or [])
    if len(values) >= n:
        return values[:n]
    return values + [default for _ in range(n - len(values))]


def _subtract(left: list, right: list) -> list:
    n = min(len(left), len(right))
    return [float(left[i]) - float(right[i]) for i in range(n)]


def _cummax(values: list) -> list:
    out = []
    best = float("-inf")
    for value in values:
        best = max(best, float(value))
        out.append(best)
    return out


def _safe_div(num: list, den: list) -> list:
    n = min(len(num), len(den))
    out = []
    for i in range(n):
        d = float(den[i])
        out.append(float(num[i]) / d if abs(d) > 1e-30 else 0.0)
    return out


def _comparison_metrics(data: dict[str, list], hardware: dict, n: int) -> dict[str, list]:
    ideal_acc = data["ideal_accuracy"]
    hw_acc = data["hardware_accuracy"]
    ideal_loss = data["ideal_loss"]
    hw_loss = data["hardware_loss"]
    energy = _aligned(hardware.get("energy_pj", []), n)
    latency = _aligned(hardware.get("latency_ns", []), n)
    return {
        "ideal_accuracy_gap": _subtract(data["ideal_train_accuracy"], ideal_acc),
        "hardware_accuracy_gap": _subtract(data["hardware_train_accuracy"], hw_acc),
        "ideal_loss_gap": _subtract(ideal_loss, data["ideal_train_loss"]),
        "hardware_loss_gap": _subtract(hw_loss, data["hardware_train_loss"]),
        "ideal_error_rate": [1.0 - float(v) for v in ideal_acc],
        "hardware_error_rate": [1.0 - float(v) for v in hw_acc],
        "accuracy_drop": _subtract(ideal_acc, hw_acc),
        "loss_increase": _subtract(hw_loss, ideal_loss),
        "ideal_best_accuracy": _cummax(ideal_acc),
        "hardware_best_accuracy": _cummax(hw_acc),
        "energy_pj": energy,
        "latency_ns": latency,
        "accuracy_per_energy": _safe_div(hw_acc, energy),
        "accuracy_per_latency": _safe_div(hw_acc, latency),
    }


def _smoothed_metrics(data: dict[str, list]) -> dict[str, list]:
    keys = [
        "ideal_accuracy",
        "hardware_accuracy",
        "ideal_train_accuracy",
        "hardware_train_accuracy",
        "ideal_loss",
        "hardware_loss",
        "ideal_train_loss",
        "hardware_train_loss",
        "accuracy_drop",
        "loss_increase",
    ]
    return {f"{key}_smooth": _moving_average(data.get(key, [])) for key in keys}


def _moving_average(values: list, window: int | None = None) -> list:
    values = [float(v) for v in values]
    if len(values) < 3:
        return values
    if window is None:
        window = min(7, max(3, len(values) // 12))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = []
    for idx in range(len(values)):
        start = max(0, idx - radius)
        end = min(len(values), idx + radius + 1)
        out.append(float(np.mean(values[start:end])))
    return out


def _dashboard_excel_data(epochs: list, spec: dict) -> dict[str, list]:
    data: dict[str, list] = {"epoch": list(epochs)}
    for idx, (_, series, _) in enumerate(spec["plots"]):
        if idx == 1 and "x_for_second" in spec:
            data["learning_rate"] = list(spec["x_for_second"])
        for label, values in series.items():
            key = label.lower().replace(" ", "_").replace("-", "_")
            data[key] = list(values)
    data["supports"] = [spec["why"]]
    return data


def _metric_quality(name: str, value: float) -> str:
    """Classify device metric quality for dashboard coloring."""
    name_lower = name.lower()
    if "on/off" in name_lower:
        return "good" if value >= 10 else ("fair" if value >= 5 else "poor")
    if "dynamic range" in name_lower:
        return "good" if value >= 20 else ("fair" if value >= 10 else "poor")
    if "nonlinearity" in name_lower:
        return "good" if abs(value) < 0.1 else ("fair" if abs(value) < 0.3 else "poor")
    if "symmetry" in name_lower:
        return "good" if value < 0.3 else ("fair" if value < 0.6 else "poor")
    if "variation" in name_lower or "stability" in name_lower:
        return "good" if abs(value) < 0.1 else ("fair" if abs(value) < 0.3 else "poor")
    if "states" in name_lower:
        return "good" if value >= 8 else ("fair" if value >= 4 else "poor")
    return "fair"


def _standard_explanations() -> list[dict[str, str]]:
    rows = [
        ("device_conductance_vs_pulse.png", "Shows extracted conductance states; supports the memristor state model used for hardware effects."),
        ("device_ltp_curve.png", "Shows potentiation direction; supports whether conductance increases with pulses."),
        ("device_ltd_curve.png", "Shows depression direction; supports whether conductance decreases with pulses."),
        ("device_histogram.png", "Shows conductance-state distribution; supports state-count and variability interpretation."),
        ("training_history.png", "Main combined graph; supports final ideal-vs-memristor training comparison."),
        ("comparison_accuracy.png", "Compares ideal and memristor test accuracy over epochs."),
        ("comparison_loss.png", "Compares ideal and memristor test loss over epochs."),
        ("comparison_accuracy_smoothed.png", "Shows raw and smoothed accuracy curves for cleaner trend reading."),
        ("comparison_loss_smoothed.png", "Shows raw and smoothed loss curves for cleaner convergence reading."),
        ("hardware_energy_vs_accuracy.png", "Relates accuracy to estimated energy cost."),
        ("hardware_latency_vs_accuracy.png", "Relates accuracy to estimated latency cost."),
        ("hardware_energy_delay_product.png", "Plots EDP (energy x latency) per epoch; supports joint energy/latency cost reading."),
        ("hardware_throughput_vs_epoch.png", "Plots MACs/sample / latency as inference throughput proxy."),
        ("confusion_matrix_diff_ideal_vs_hardware.png", "Row-normalized (hardware − ideal) confusion-matrix difference; highlights which transitions the device degrades."),
        ("class_wise_comparison.png", "Per-class accuracy + accuracy drop between ideal and hardware runs."),
    ]
    for prefix, label in [("ideal", "ideal"), ("hardware_aware", "memristor")]:
        rows.extend(
            [
                (f"{prefix}_train_test_accuracy_vs_epoch.png", f"Shows {label} train/test accuracy together; supports learning and generalization."),
                (f"{prefix}_train_test_loss_vs_epoch.png", f"Shows {label} train/test loss together; supports convergence behavior."),
                (f"{prefix}_generalization_gap_vs_epoch.png", f"Shows {label} train-test accuracy gap; supports overfitting/generalization analysis."),
                (f"{prefix}_loss_gap_vs_epoch.png", f"Shows {label} test-train loss gap; supports stability/generalization analysis."),
                (f"{prefix}_test_error_rate_vs_epoch.png", f"Shows {label} error rate; supports accuracy from the inverse perspective."),
                (f"{prefix}_best_accuracy_vs_epoch.png", f"Shows {label} best-so-far accuracy; supports model selection and convergence."),
                (f"{prefix}_learning_rate_vs_epoch.png", f"Documents learning-rate schedule used during {label} training/evaluation."),
                (f"{prefix}_test_accuracy_vs_learning_rate.png", f"Relates {label} test accuracy to learning rate; useful when LR schedules are enabled."),
                (f"{prefix}_test_loss_vs_learning_rate.png", f"Relates {label} test loss to learning rate; useful when LR schedules are enabled."),
                (f"{prefix}_epoch_time_vs_epoch.png", f"Shows {label} runtime per epoch; supports compute-cost reporting."),
                (f"{prefix}_parameter_norm_vs_epoch.png", f"Shows {label} model parameter norm over epochs; supports weight-dynamics monitoring."),
                (f"{prefix}_gradient_norm_vs_epoch.png", f"Shows {label} gradient norm over epochs; supports optimization stability monitoring."),
                (f"{prefix}_trainable_parameters_vs_epoch.png", f"Documents {label} trainable parameter count during the run."),
                (f"{prefix}_confusion_matrix.png", f"Shows {label} per-class prediction behavior; supports class-level error analysis."),
                (f"{prefix}_confusion_matrix_normalized.png", f"Row-normalized {label} confusion matrix (each true class sums to 1)."),
                (f"{prefix}_class_metrics.png", f"Per-class F1 bar chart for {label} run."),
                (f"{prefix}_per_class_accuracy.png", f"Per-class accuracy bar chart for {label} run."),
                (f"{prefix}_top_confused_pairs.png", f"Top confused (true -> predicted) pairs in the {label} run."),
            ]
        )
    return [{"graph": graph, "supports": supports} for graph, supports in rows]


# -------- Phase 8 helpers: dataset-aware labels & sizing --------
def _class_names_from_history(history: dict, n: int) -> list[str] | None:
    meta = history.get("metadata", {}) if isinstance(history, dict) else {}
    names = meta.get("class_names") if isinstance(meta, dict) else None
    if isinstance(names, (list, tuple)) and len(names) >= n:
        return [str(x) for x in names[:n]]
    return None


def _labels_for(class_names: list[str] | None, n: int) -> list[str]:
    if class_names and len(class_names) >= n:
        return [str(c) for c in class_names[:n]]
    return [str(i) for i in range(n)]


def _tick_fontsize(n: int) -> int:
    if n <= 10:
        return 10
    if n <= 20:
        return 8
    if n <= 50:
        return 6
    return 5


def _title_suffix(history: dict) -> str:
    meta = history.get("metadata", {}) if isinstance(history, dict) else {}
    dataset = meta.get("dataset") or ""
    model = meta.get("model") or ""
    curve = meta.get("curve") or ""
    parts = [p for p in [dataset, model, curve] if p]
    return " / ".join(parts)
