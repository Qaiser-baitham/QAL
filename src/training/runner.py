from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time

from src.data_loader.datasets import create_loaders
from src.device_model.extractor import DeviceModel
from src.hardware_sim.effects import HardwareAwareSimulator
from src.hardware_sim.dse import run_dse
from src.training.checkpoint import checkpoint_path, load_checkpoint, save_checkpoint
from src.utils.io import atomic_json
from src.visualization.exporter import Exporter


class ExperimentRunner:
    def __init__(self, cfg: dict, device_model: DeviceModel, paths: dict[str, Path]):
        self.cfg = cfg
        self.device_model = device_model
        self.paths = paths
        self.exporter = Exporter(paths["plots"], paths["excel"], paths.get("reports"))
        self.dashboard = ConsoleDashboard(cfg, paths)
        self.run_started_at = time.time()
        self.timing_rows: list[dict] = []

    def run(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("PyTorch is not installed. Install dependencies with `pip install -r requirements.txt`.") from exc

        from src.models.factory import create_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_started_at = time.time()
        logging.info("Compute device selected: %s; pin_memory=%s", device, device == "cuda")
        self.exporter.device(self.device_model)
        train_loader, test_loader, spec = create_loaders(
            self.cfg["dataset"],
            self.cfg.get("data_root", "datasets"),
            int(self.cfg.get("batch_size", 128)),
            int(self.cfg.get("num_workers", 0)),
            device,
        )
        self._print_run_banner(device, spec, len(train_loader.dataset), len(test_loader.dataset))
        histories = {}
        strategy = self.cfg.get("dual_strategy", "ideal_then_hardware")
        if self.cfg["training_mode"] == "dual" and strategy == "shared_model_eval":
            model = create_model(self.cfg["model"], self.cfg["dataset"], spec).to(device)
            histories = self._run_shared_dual(model, train_loader, test_loader, device, torch)
        else:
            modes = ["ideal", "hardware_aware"] if self.cfg["training_mode"] == "dual" else [self.cfg["training_mode"]]
            for mode in modes:
                model = create_model(self.cfg["model"], self.cfg["dataset"], spec).to(device)
                histories[mode] = self._run_one(mode, model, train_loader, test_loader, device, torch)
        if "ideal" in histories and "hardware_aware" in histories:
            self.dashboard.phase("FINAL COMPARISON")
            self.exporter.comparison(histories["ideal"], histories["hardware_aware"])
            self.exporter.class_wise_comparison(histories["ideal"], histories["hardware_aware"])
            self.dashboard.status("[COMPARE]", "Final ideal-vs-hardware plots, class metrics, and Excel tables updated.")
        # Phase 7: always run DSE for dual mode, optional for single mode
        if self.cfg.get("run_dse") or self.cfg.get("training_mode") == "dual":
            source = histories.get("hardware_aware") or histories.get("ideal")
            run_dse(source, self.cfg, self.paths["excel"])
        self.exporter.param_info(histories, self.cfg, self.device_model)
        self.exporter.runtime_timing(self.timing_rows)

    def _run_shared_dual(self, model, train_loader, test_loader, device: str, torch):
        from torch import nn, optim

        optimizer = optim.AdamW(model.parameters(), lr=float(self.cfg.get("learning_rate", 1e-3)), weight_decay=float(self.cfg.get("weight_decay", 1e-4)))
        scheduler = _make_scheduler(optimizer, self.cfg)
        criterion = nn.CrossEntropyLoss()
        simulator = HardwareAwareSimulator(self.device_model, self.cfg)
        ideal = self._empty_history()
        hardware = self._empty_history()
        ideal["metadata"] = self._metadata("Ideal")
        hardware["metadata"] = self._metadata("Memristor")
        ideal["metadata"]["trainable_parameters"] = _trainable_parameters(model)
        hardware["metadata"]["trainable_parameters"] = _trainable_parameters(model)
        best_ideal = -1.0
        best_hardware = -1.0
        max_epochs = int(self.cfg["max_epochs"])
        start_epoch = 1

        resume = self.cfg.get("resume", "fresh")
        if resume != "fresh":
            kind = "latest" if resume == "resume_latest" else "best"
            ckpt = load_checkpoint(checkpoint_path(self.paths["checkpoints"], "ideal", kind), torch)
            hw_ckpt = load_checkpoint(checkpoint_path(self.paths["checkpoints"], "hardware_aware", kind), torch)
            if ckpt:
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if scheduler is not None and ckpt.get("scheduler_state"):
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                ideal = ckpt.get("history", ideal)
                ideal.setdefault("metadata", self._metadata("Ideal"))
                best_ideal = ckpt.get("best_accuracy", best_ideal)
                start_epoch = int(ckpt["epoch"]) + 1
                logging.info("Resumed shared dual from %s checkpoint at epoch %d", kind, ckpt["epoch"])
            if hw_ckpt:
                hardware = hw_ckpt.get("history", hardware)
                hardware.setdefault("metadata", self._metadata("Memristor"))
                best_hardware = hw_ckpt.get("best_accuracy", best_hardware)

        try:
            self.dashboard.phase("PHASE 1: SHARED DUAL RUN")
            for epoch in range(start_epoch, max_epochs + 1):
                epoch_start = time.time()
                self.dashboard.section("IDEAL TRAIN")
                train_start = time.time()
                train_loss, train_acc, grad_norm = self._train_epoch(model, train_loader, optimizer, criterion, device, torch, None, f"[IDEAL][TRAIN] Epoch {epoch}/{max_epochs}")
                self.dashboard.metric("[IDEAL][TRAIN]", epoch, max_epochs, train_loss, train_acc, time.time() - train_start, time.time() - self.run_started_at, self._eta_from_history(ideal, epoch, max_epochs))
                self.dashboard.section("IDEAL TEST")
                val_loss, val_acc, cm, samples = self._evaluate(model, test_loader, criterion, device, torch, None, f"[IDEAL][TEST ] Epoch {epoch}/{max_epochs}")
                learning_rate = _current_lr(optimizer)
                param_norm = _model_parameter_norm(model)
                ideal_epoch_seconds = time.time() - epoch_start
                self._append_history(ideal, epoch, train_loss, train_acc, val_loss, val_acc, cm, None, ideal_epoch_seconds, learning_rate, param_norm, grad_norm)
                self._record_timing("ideal", ideal, epoch, max_epochs, ideal_epoch_seconds)
                self.dashboard.metric("[IDEAL][TEST ]", epoch, max_epochs, val_loss, val_acc, ideal_epoch_seconds, time.time() - self.run_started_at, _last(ideal.get("eta_seconds"), 0.0), _last(ideal.get("avg_epoch_seconds"), 0.0))

                hw_start = time.time()
                self.dashboard.section("HARDWARE TRAIN")
                hw_train_loss, hw_train_acc, _, _ = self._evaluate(model, train_loader, criterion, device, torch, simulator, f"[HARDWARE][TRAIN] Epoch {epoch}/{max_epochs}")
                self.dashboard.metric("[HARDWARE][TRAIN]", epoch, max_epochs, hw_train_loss, hw_train_acc, time.time() - hw_start, time.time() - self.run_started_at, self._eta_from_history(hardware, epoch, max_epochs))
                self.dashboard.section("HARDWARE TEST")
                hw_test_start = time.time()
                hw_val_loss, hw_val_acc, hw_cm, hw_samples = self._evaluate(model, test_loader, criterion, device, torch, simulator, f"[HARDWARE][TEST ] Epoch {epoch}/{max_epochs}")
                hw_metrics = simulator.estimate_metrics(model, hw_samples, hw_val_acc)
                hw_epoch_seconds = time.time() - hw_start
                self._append_history(hardware, epoch, hw_train_loss, hw_train_acc, hw_val_loss, hw_val_acc, hw_cm, hw_metrics, hw_epoch_seconds, learning_rate, param_norm, grad_norm)
                self._record_timing("hardware_aware", hardware, epoch, max_epochs, hw_epoch_seconds)
                self.dashboard.metric("[HARDWARE][TEST ]", epoch, max_epochs, hw_val_loss, hw_val_acc, time.time() - hw_test_start, time.time() - self.run_started_at, _last(hardware.get("eta_seconds"), 0.0), _last(hardware.get("avg_epoch_seconds"), 0.0))

                if scheduler is not None:
                    _step_scheduler(scheduler, val_loss)

                is_best_ideal = val_acc > best_ideal
                is_best_hardware = hw_val_acc > best_hardware
                best_ideal = max(best_ideal, val_acc)
                best_hardware = max(best_hardware, hw_val_acc)
                self._safe_epoch_save("ideal", model, optimizer, epoch, ideal, best_ideal, is_best_ideal, torch, "running", scheduler)
                self._safe_epoch_save("hardware_aware", model, optimizer, epoch, hardware, best_hardware, is_best_hardware, torch, "running", scheduler)
                logging.info(
                    "dual epoch %d/%d ideal_acc=%.4f memristor_acc=%.4f ideal_loss=%.4f memristor_loss=%.4f lr=%.6g param_norm=%.4f grad_norm=%.4f",
                    epoch,
                    max_epochs,
                    val_acc,
                    hw_val_acc,
                    val_loss,
                    hw_val_loss,
                    learning_rate,
                    param_norm,
                    grad_norm,
                )

                target = self.cfg.get("target_accuracy")
                if target is not None and max(val_acc, hw_val_acc) >= float(target):
                    winner = "memristor" if hw_val_acc >= val_acc else "ideal"
                    reached = hw_val_acc if winner == "memristor" else val_acc
                    ideal["status"] = f"target accuracy reached by {winner}: {reached:.4f} >= {float(target):.4f}"
                    hardware["status"] = ideal["status"]
                    logging.info(ideal["status"])
                    break
        except KeyboardInterrupt:
            ideal["status"] = "stopped early by user interrupt"
            hardware["status"] = ideal["status"]
            logging.warning(ideal["status"])
            self._safe_epoch_save("ideal", model, optimizer, max(start_epoch - 1, len(ideal["epoch"])), ideal, best_ideal, False, torch, "interrupted", scheduler)
            self._safe_epoch_save("hardware_aware", model, optimizer, max(start_epoch - 1, len(hardware["epoch"])), hardware, best_hardware, False, torch, "interrupted", scheduler)
        except Exception:
            ideal["status"] = "crashed; emergency checkpoint saved"
            hardware["status"] = ideal["status"]
            logging.exception("Training crashed; saving emergency checkpoints")
            self._safe_epoch_save("ideal", model, optimizer, max(start_epoch - 1, len(ideal["epoch"])), ideal, best_ideal, False, torch, "crashed", scheduler)
            self._safe_epoch_save("hardware_aware", model, optimizer, max(start_epoch - 1, len(hardware["epoch"])), hardware, best_hardware, False, torch, "crashed", scheduler)
            raise
        ideal.setdefault("status", "completed")
        hardware.setdefault("status", "completed")
        self.dashboard.complete("SHARED DUAL RUN COMPLETE")
        atomic_json(self.paths["history"] / "ideal_history.json", ideal)
        atomic_json(self.paths["history"] / "hardware_aware_history.json", hardware)
        self.exporter.training(ideal, "ideal")
        self.exporter.training(hardware, "hardware_aware")
        return {"ideal": ideal, "hardware_aware": hardware}

    def _run_one(self, mode: str, model, train_loader, test_loader, device: str, torch):
        from torch import nn, optim

        optimizer = optim.AdamW(model.parameters(), lr=float(self.cfg.get("learning_rate", 1e-3)), weight_decay=float(self.cfg.get("weight_decay", 1e-4)))
        scheduler = _make_scheduler(optimizer, self.cfg)
        criterion = nn.CrossEntropyLoss()
        simulator = HardwareAwareSimulator(self.device_model, self.cfg)
        history = self._empty_history()
        history["metadata"] = self._metadata("Memristor" if mode == "hardware_aware" else "Ideal")
        history["metadata"]["trainable_parameters"] = _trainable_parameters(model)
        start_epoch = 1
        best_acc = -1.0

        resume = self.cfg.get("resume", "fresh")
        if resume != "fresh":
            kind = "latest" if resume == "resume_latest" else "best"
            ckpt = load_checkpoint(checkpoint_path(self.paths["checkpoints"], mode, kind), torch)
            if ckpt:
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if scheduler is not None and ckpt.get("scheduler_state"):
                    scheduler.load_state_dict(ckpt["scheduler_state"])
                history = ckpt.get("history", history)
                history.setdefault("metadata", self._metadata("Memristor" if mode == "hardware_aware" else "Ideal"))
                best_acc = ckpt.get("best_accuracy", best_acc)
                start_epoch = int(ckpt["epoch"]) + 1
                logging.info("Resumed %s from %s checkpoint at epoch %d", mode, kind, ckpt["epoch"])

        patience = self.cfg.get("early_stopping_patience")
        stale = 0
        max_epochs = int(self.cfg["max_epochs"])
        stopped_reason = "completed"
        self.dashboard.phase(_phase_title(mode, self.cfg))
        try:
            for epoch in range(start_epoch, max_epochs + 1):
                epoch_start = time.time()
                prefix = _mode_prefix(mode)
                self.dashboard.section(f"{_mode_label(mode)} TRAIN")
                train_start = time.time()
                train_loss, train_acc, grad_norm = self._train_epoch(model, train_loader, optimizer, criterion, device, torch, simulator if mode == "hardware_aware" else None, f"{prefix}[TRAIN] Epoch {epoch}/{max_epochs}")
                self.dashboard.metric(f"{prefix}[TRAIN]", epoch, max_epochs, train_loss, train_acc, time.time() - train_start, time.time() - self.run_started_at, self._eta_from_history(history, epoch, max_epochs))
                self.dashboard.section(f"{_mode_label(mode)} TEST")
                val_loss, val_acc, cm, samples = self._evaluate(model, test_loader, criterion, device, torch, simulator if mode == "hardware_aware" else None, f"{prefix}[TEST ] Epoch {epoch}/{max_epochs}")
                hw_metrics = simulator.estimate_metrics(model, samples, val_acc) if mode == "hardware_aware" else None
                learning_rate = _current_lr(optimizer)
                param_norm = _model_parameter_norm(model)
                epoch_seconds = time.time() - epoch_start
                self._append_history(history, epoch, train_loss, train_acc, val_loss, val_acc, cm, hw_metrics, epoch_seconds, learning_rate, param_norm, grad_norm)
                self._record_timing(mode, history, epoch, max_epochs, epoch_seconds)
                self.dashboard.metric(f"{prefix}[TEST ]", epoch, max_epochs, val_loss, val_acc, epoch_seconds, time.time() - self.run_started_at, _last(history.get("eta_seconds"), 0.0), _last(history.get("avg_epoch_seconds"), 0.0))
                if scheduler is not None:
                    _step_scheduler(scheduler, val_loss)
                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc
                    stale = 0
                else:
                    stale += 1
                self._safe_epoch_save(mode, model, optimizer, epoch, history, best_acc, is_best, torch, "running", scheduler)
                logging.info("%s epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6g param_norm=%.4f grad_norm=%.4f", mode, epoch, max_epochs, train_loss, train_acc, val_loss, val_acc, learning_rate, param_norm, grad_norm)

                target = self.cfg.get("target_accuracy")
                if target is not None and val_acc >= float(target):
                    stopped_reason = f"target accuracy reached: {val_acc:.4f} >= {float(target):.4f}"
                    logging.info(stopped_reason)
                    break
                if patience is not None and stale >= int(patience):
                    stopped_reason = f"early stopping patience reached: {patience}"
                    logging.info(stopped_reason)
                    break
        except KeyboardInterrupt:
            stopped_reason = "stopped early by user interrupt"
            logging.warning(stopped_reason)
            self._safe_epoch_save(mode, model, optimizer, max(start_epoch - 1, len(history["epoch"])), history, best_acc, False, torch, "stopped early", scheduler)
        except Exception:
            stopped_reason = "crashed; emergency checkpoint saved"
            logging.exception("Training crashed; saving emergency checkpoint")
            self._safe_epoch_save(mode, model, optimizer, max(start_epoch - 1, len(history["epoch"])), history, best_acc, False, torch, "crashed", scheduler)
            raise
        history["status"] = stopped_reason
        self.dashboard.complete(f"{_mode_label(mode)} RUN COMPLETE")
        atomic_json(self.paths["history"] / f"{mode}_history.json", history)
        self.exporter.training(history, mode)
        return history

    def _train_epoch(self, model, loader, optimizer, criterion, device, torch, simulator, description: str = "train"):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_total = 0.0
        grad_steps = 0
        iterator = _progress(loader, description, enabled=bool(self.cfg.get("show_progress", True)))
        for x, y in iterator:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if simulator is not None:
                x = simulator.quantize_activation(x)
                with simulator.perturbed_weights(model):
                    out = simulator.adc_quantize_logits(model(x))
                    loss = criterion(out, y)
                    loss.backward()
            else:
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
            # Phase 4: gradient clipping for training stability
            clip_norm = self.cfg.get("gradient_clip_max_norm")
            if clip_norm is not None and float(clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_norm))
            grad_total += _model_grad_norm(model)
            grad_steps += 1
            optimizer.step()
            pred = out.argmax(dim=1)
            total_loss += float(loss.item()) * y.size(0)
            correct += int((pred == y).sum().item())
            total += y.size(0)
            _progress_update(iterator, loss=total_loss / max(total, 1), acc=correct / max(total, 1))
        return total_loss / max(total, 1), correct / max(total, 1), grad_total / max(grad_steps, 1)

    def _evaluate(self, model, loader, criterion, device, torch, simulator, description: str = "test"):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_pred = []
        all_true = []
        with torch.no_grad():
            iterator = _progress(loader, description, enabled=bool(self.cfg.get("show_progress", True)))
            for x, y in iterator:
                x, y = x.to(device), y.to(device)
                if simulator is not None:
                    x = simulator.quantize_activation(x)
                    with simulator.perturbed_weights(model):
                        out = simulator.adc_quantize_logits(model(x))
                else:
                    out = model(x)
                loss = criterion(out, y)
                pred = out.argmax(dim=1)
                total_loss += float(loss.item()) * y.size(0)
                correct += int((pred == y).sum().item())
                total += y.size(0)
                all_pred.extend(pred.cpu().tolist())
                all_true.extend(y.cpu().tolist())
                _progress_update(iterator, loss=total_loss / max(total, 1), acc=correct / max(total, 1))
        return total_loss / max(total, 1), correct / max(total, 1), _confusion(all_true, all_pred), total

    def _print_run_banner(self, device: str, spec: dict, train_samples: int, test_samples: int) -> None:
        self.dashboard.run_banner(device, spec, train_samples, test_samples, self.device_model)

    def _safe_epoch_save(self, mode, model, optimizer, epoch, history, best_acc, is_best, torch, status, scheduler=None):
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "config": self.cfg,
            "device_model": self.device_model.to_dict(),
            "history": copy.deepcopy(history),
            "best_accuracy": best_acc,
            "status": status,
        }
        save_checkpoint(checkpoint_path(self.paths["checkpoints"], mode, "latest"), torch, payload)
        self.dashboard.status("[CHECKPOINT]", f"{mode} latest saved at epoch {epoch}.")
        if is_best:
            save_checkpoint(checkpoint_path(self.paths["checkpoints"], mode, "best"), torch, payload)
            self.dashboard.status("[CHECKPOINT]", f"{mode} best updated at epoch {epoch} acc={best_acc * 100:.2f}%.")
        atomic_json(self.paths["history"] / f"{mode}_history.json", history)
        self.exporter.training(history, mode)
        self.dashboard.status("[SAVE]", f"{mode} history, plots, and tables updated.")

    def _append_history(self, history, epoch, train_loss, train_acc, val_loss, val_acc, cm, hw_metrics, seconds, learning_rate, parameter_norm, grad_norm):
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history.setdefault("train_accuracy", []).append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history.setdefault("learning_rate", []).append(learning_rate)
        history.setdefault("parameter_norm", []).append(parameter_norm)
        history.setdefault("grad_norm", []).append(grad_norm)
        history.setdefault("trainable_parameters", []).append(history.get("metadata", {}).get("trainable_parameters", 0))
        history["seconds"].append(seconds)
        history["confusion_matrix"] = cm
        if hw_metrics is not None:
            history["energy_pj"].append(hw_metrics.energy_pj)
            history["latency_ns"].append(hw_metrics.latency_ns)
            history["adc_bits"].append(hw_metrics.adc_bits)
            history["weight_bits"].append(hw_metrics.weight_bits)
            history.setdefault("macs_per_sample", []).append(hw_metrics.macs_per_sample)
            history.setdefault("energy_per_mac_pj", []).append(hw_metrics.energy_per_mac_pj)
            history.setdefault("tops_w_proxy", []).append(hw_metrics.tops_w_proxy)

    def _empty_history(self):
        return {"epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "learning_rate": [], "parameter_norm": [], "grad_norm": [], "trainable_parameters": [], "seconds": [], "elapsed_seconds": [], "avg_epoch_seconds": [], "eta_seconds": [], "energy_pj": [], "latency_ns": [], "adc_bits": [], "weight_bits": []}

    def _record_timing(self, mode: str, history: dict, epoch: int, max_epochs: int, epoch_seconds: float) -> None:
        elapsed = time.time() - self.run_started_at
        completed = len(history.get("epoch", []))
        avg_epoch = sum(float(v) for v in history.get("seconds", [])) / max(completed, 1)
        remaining = max(max_epochs - epoch, 0)
        eta = avg_epoch * remaining
        history.setdefault("elapsed_seconds", []).append(elapsed)
        history.setdefault("avg_epoch_seconds", []).append(avg_epoch)
        history.setdefault("eta_seconds", []).append(eta)
        row = {
            "mode": mode,
            "epoch": epoch,
            "elapsed_total_seconds": elapsed,
            "epoch_duration_seconds": epoch_seconds,
            "average_epoch_seconds": avg_epoch,
            "eta_seconds": eta,
            "train_accuracy": _last(history.get("train_accuracy"), 0.0),
            "test_accuracy": _last(history.get("val_accuracy"), 0.0),
            "train_loss": _last(history.get("train_loss"), 0.0),
            "test_loss": _last(history.get("val_loss"), 0.0),
        }
        self.timing_rows.append(row)
        self.exporter.runtime_timing(self.timing_rows)

    def _eta_from_history(self, history: dict, epoch: int, max_epochs: int) -> float:
        seconds = history.get("seconds", [])
        if not seconds:
            return 0.0
        avg_epoch = sum(float(v) for v in seconds) / len(seconds)
        return avg_epoch * max(max_epochs - epoch + 1, 0)

    def _metadata(self, curve: str) -> dict:
        from src.data_loader.datasets import class_names_for

        try:
            class_names = class_names_for(self.cfg.get("dataset", ""))
        except Exception:
            class_names = []
        return {
            "curve": curve,
            "dataset": self.cfg.get("dataset"),
            "model": self.cfg.get("model"),
            "training_mode": self.cfg.get("training_mode"),
            "target_accuracy": self.cfg.get("target_accuracy"),
            "class_names": class_names,
        }


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI SGR escape sequences so widths can be computed correctly."""
    return _ANSI_RE.sub("", text)


class ConsoleDashboard:
    """Pretty, expert-feel training console.

    Renders Unicode box-drawn banners, ANSI-coloured rule lines, per-epoch
    delta indicators (▲ / ▼), accuracy-tier colouring (red / yellow / green),
    an inline progress bar, and a final results frame with best accuracies.
    Falls back to ASCII + no-colour automatically when the terminal can't
    handle Unicode or ANSI (e.g. old Windows cmd, piped stdout, NO_COLOR).
    """

    _CODES = {
        "reset": "0",
        "bold": "1",
        "dim": "2",
        "italic": "3",
        "underline": "4",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
        "gray": "90",
        "bright_red": "91",
        "bright_green": "92",
        "bright_yellow": "93",
        "bright_blue": "94",
        "bright_magenta": "95",
        "bright_cyan": "96",
        "bright_white": "97",
    }

    def __init__(self, cfg: dict, paths: dict[str, Path]):
        self.cfg = cfg
        self.paths = paths
        self.enabled = bool(cfg.get("show_progress", True))
        term_cols = shutil.get_terminal_size((100, 24)).columns
        self.width = min(max(term_cols, 84), 120)
        self._enable_windows_vt()
        self._try_utf8_stdout()
        self.use_color = self._detect_color()
        self.use_unicode = self._detect_unicode()
        self._prev_acc: dict[str, float] = {}
        self._best_acc: dict[str, float] = {}

    # ---------- environment detection ----------
    @staticmethod
    def _enable_windows_vt() -> None:
        if os.name != "nt":
            return
        try:
            os.system("")  # side-effect: enables VT100 on Windows 10+
        except Exception:
            pass

    @staticmethod
    def _try_utf8_stdout() -> None:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    def _detect_color(self) -> bool:
        if not self.enabled:
            return False
        if os.environ.get("NO_COLOR"):
            return False
        try:
            if not sys.stdout.isatty():
                return False
        except Exception:
            return False
        return True

    def _detect_unicode(self) -> bool:
        try:
            enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        except Exception:
            enc = ""
        return "utf" in enc

    # ---------- public API ----------
    def run_banner(self, device: str, spec: dict, train_samples: int, test_samples: int, device_model: DeviceModel) -> None:
        if not self.enabled:
            return
        bolt = "⚡" if self.use_unicode else "*"
        diamond = "◆" if self.use_unicode else "-"
        mid_dot = "·" if self.use_unicode else "-"

        sched = self.cfg.get("lr_scheduler") or {}
        sched_name = (sched.get("type") or "none") if isinstance(sched, dict) else "none"
        tgt = self.cfg.get("target_accuracy")
        target_txt = f"{float(tgt) * 100:.1f}%" if tgt is not None else "not set"

        rows = [
            ("Dataset", f"{self.cfg.get('dataset','?')}  ({train_samples:,} train {mid_dot} {test_samples:,} test)"),
            ("Model", str(self.cfg.get("model", ""))),
            ("Training Mode", f"{str(self.cfg.get('training_mode','')).upper()} {mid_dot} {self.cfg.get('dual_strategy','n/a')}"),
            ("Epochs / Batch", f"{self.cfg.get('max_epochs')} epochs {mid_dot} batch {self.cfg.get('batch_size')}"),
            ("Learning Rate", f"{self.cfg.get('learning_rate')}  (scheduler: {sched_name})"),
            ("Target Accuracy", target_txt),
            ("Compute Device", device.upper()),
            ("Input Tensor", f"{spec['channels']}×{spec['size']}×{spec['size']} {mid_dot} {spec['classes']} classes"),
            ("Memristor", f"states={device_model.num_states} {mid_dot} on/off={device_model.on_off_ratio:.2f}x {mid_dot} σ={device_model.cycle_variation_sigma:.4f}"),
            ("Output Dir", str(self.paths.get("root", self.cfg.get("outputs_root", "outputs")))),
        ]

        title = f"  {bolt}  iMC  RESEARCH  RUN  {bolt}"
        subtitle = "In-Memory Compute · Memristor Hardware-Aware Training"

        print()
        self._box_top("double", "bright_cyan")
        self._box_row_centered(title, "double", "bright_cyan", bold=True, text_color="bright_white")
        self._box_row_centered(subtitle, "double", "bright_cyan", text_color="gray", dim=True)
        self._box_divider("double", "bright_cyan")
        for label, value in rows:
            inner = (
                f"  {self._color(diamond, 'bright_yellow')} "
                f"{self._color(f'{label:<16}', 'bright_white', bold=True)}"
                f"{self._color(str(value), 'white')}"
            )
            self._box_row_prerendered(inner, "double", "bright_cyan")
        self._box_bottom("double", "bright_cyan")

    def phase(self, title: str) -> None:
        if not self.enabled:
            return
        arrow = "▶" if self.use_unicode else ">"
        label = f"  {arrow}  {title}"
        print()
        self._box_top("heavy", "bright_blue")
        self._box_row_left(label, "heavy", "bright_blue", bold=True, text_color="bright_white")
        self._box_bottom("heavy", "bright_blue")

    def complete(self, title: str) -> None:
        if not self.enabled:
            return
        check = "✓" if self.use_unicode else "v"
        best = self._best_line()
        msg = f"  {check}  {title}"
        if best:
            msg += f"   ·   {best}"
        print()
        self._box_top("double", "bright_green")
        self._box_row_left(msg, "double", "bright_green", bold=True, text_color="bright_white")
        self._box_bottom("double", "bright_green")

    def section(self, title: str) -> None:
        if not self.enabled:
            return
        dash = "─" if self.use_unicode else "-"
        prefix = f"{dash * 3} {title} "
        tail = dash * max(self.width - len(prefix), 0)
        upper = title.upper()
        color = "bright_cyan" if "TRAIN" in upper else "bright_green" if "TEST" in upper else "cyan"
        print(self._color(prefix + tail, color, dim=True))

    def metric(
        self,
        prefix: str,
        epoch: int,
        max_epochs: int,
        loss: float,
        acc: float,
        step_seconds: float,
        elapsed: float,
        eta: float,
        avg_epoch: float | None = None,
    ) -> None:
        if not self.enabled:
            return

        upper = prefix.upper()
        is_hw = "HARDWARE" in upper
        is_test = "TEST" in upper

        mode_text = " HARDWARE " if is_hw else " IDEAL    "
        phase_text = " TEST  " if is_test else " TRAIN "
        mode_bg = "bright_magenta" if is_hw else "bright_cyan"
        phase_bg = "bright_green" if is_test else "bright_blue"
        tag = self._color(mode_text, mode_bg, bold=True) + self._color(phase_text, phase_bg, bold=True)

        # inline progress bar over the overall training horizon
        bar = self._bar(epoch / max(max_epochs, 1), width=14)

        # delta vs last same-phase call for this mode
        key = f"{'HW' if is_hw else 'ID'}:{'TEST' if is_test else 'TRAIN'}"
        delta_text = self._delta(acc, self._prev_acc.get(key))
        self._prev_acc[key] = acc
        self._best_acc[key] = max(self._best_acc.get(key, 0.0), acc)

        # colour-code accuracy by tier
        if acc < 0.5:
            acc_color = "bright_red"
        elif acc < 0.8:
            acc_color = "bright_yellow"
        else:
            acc_color = "bright_green"
        acc_str = self._color(f"{acc * 100:6.2f}%", acc_color, bold=True)
        loss_str = self._color(f"{loss:7.4f}", "white")
        epoch_str = self._color(f"{epoch:>3}/{max_epochs:<3}", "bright_white", bold=True)

        sep = self._color("│" if self.use_unicode else "|", "gray")
        avg_part = ""
        if avg_epoch:
            avg_part = (
                f" {sep} {self._color('avg', 'gray')} "
                f"{self._color(_fmt_duration(avg_epoch), 'white'):<8}"
            )

        line = (
            f"{tag} "
            f"{epoch_str} {bar} "
            f"{sep} {self._color('loss', 'gray')} {loss_str} "
            f"{sep} {self._color('acc', 'gray')}  {acc_str}{delta_text} "
            f"{sep} {self._color('t', 'gray')} {_fmt_duration(step_seconds):<7}"
            f"{avg_part} "
            f"{sep} {self._color('elapsed', 'gray')} {_fmt_duration(elapsed):<8} "
            f"{sep} {self._color('ETA', 'gray')} {_fmt_duration(eta):<8}"
        )
        print(line)

    def status(self, prefix: str, message: str) -> None:
        if not self.enabled:
            return
        dot = "●" if self.use_unicode else "*"
        tag = prefix.strip("[]").strip()
        palette = {
            "CHECKPOINT": "bright_yellow",
            "SAVE": "bright_blue",
            "COMPARE": "bright_magenta",
        }
        color = palette.get(tag, "cyan")
        print(
            f"  {self._color(dot, color)} "
            f"{self._color(f'{prefix:<12}', color, bold=True)} "
            f"{self._color(message, 'gray')}"
        )

    # ---------- rendering helpers ----------
    def _best_line(self) -> str:
        parts = []
        # Prefer test accuracies only
        order = [("ID:TEST", "Ideal"), ("HW:TEST", "Memristor")]
        for key, label in order:
            val = self._best_acc.get(key)
            if val is not None:
                parts.append(f"best {label} {val * 100:.2f}%")
        return "  ·  ".join(parts)

    def _bar(self, frac: float, width: int = 14) -> str:
        frac = max(0.0, min(1.0, frac))
        filled_char = "█" if self.use_unicode else "#"
        empty_char = "░" if self.use_unicode else "."
        filled = int(round(frac * width))
        body = filled_char * filled + empty_char * (width - filled)
        pct = f"{int(round(frac * 100)):3d}%"
        return (
            self._color("[", "gray")
            + self._color(body, "bright_cyan")
            + self._color("]", "gray")
            + " "
            + self._color(pct, "white")
        )

    def _delta(self, acc: float, prev: float | None) -> str:
        if prev is None:
            return ""
        diff = (acc - prev) * 100
        if abs(diff) < 0.01:
            arrow = "·" if self.use_unicode else "="
            return " " + self._color(arrow, "gray")
        up = diff > 0
        if up:
            arrow = "▲" if self.use_unicode else "^"
            color = "bright_green"
            sign = "+"
        else:
            arrow = "▼" if self.use_unicode else "v"
            color = "bright_red"
            sign = ""
        return " " + self._color(f"{arrow}{sign}{diff:.2f}", color, bold=True)

    def _color(self, text: str, *styles: str, bold: bool = False, dim: bool = False) -> str:
        if not self.use_color:
            return text
        codes: list[str] = []
        if bold:
            codes.append(self._CODES["bold"])
        if dim:
            codes.append(self._CODES["dim"])
        for s in styles:
            code = self._CODES.get(s)
            if code:
                codes.append(code)
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"

    # ---- box primitives ----
    _BOX_CHARS = {
        "double":  {"tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║", "ml": "╠", "mr": "╣"},
        "heavy":   {"tl": "┏", "tr": "┓", "bl": "┗", "br": "┛", "h": "━", "v": "┃", "ml": "┣", "mr": "┫"},
        "light":   {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│", "ml": "├", "mr": "┤"},
        "ascii":   {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "=", "v": "|", "ml": "+", "mr": "+"},
    }

    def _box(self, style: str) -> dict:
        if not self.use_unicode:
            return self._BOX_CHARS["ascii"]
        return self._BOX_CHARS.get(style, self._BOX_CHARS["light"])

    def _box_top(self, style: str, color: str) -> None:
        box = self._box(style)
        line = box["tl"] + box["h"] * (self.width - 2) + box["tr"]
        print(self._color(line, color))

    def _box_bottom(self, style: str, color: str) -> None:
        box = self._box(style)
        line = box["bl"] + box["h"] * (self.width - 2) + box["br"]
        print(self._color(line, color))

    def _box_divider(self, style: str, color: str) -> None:
        box = self._box(style)
        line = box["ml"] + box["h"] * (self.width - 2) + box["mr"]
        print(self._color(line, color))

    def _box_row_centered(self, text: str, style: str, frame_color: str, *, bold: bool = False, dim: bool = False, text_color: str = "white") -> None:
        box = self._box(style)
        inner_w = self.width - 2
        text = self._fit(text, inner_w)
        pad_left = (inner_w - len(text)) // 2
        pad_right = inner_w - len(text) - pad_left
        painted = self._color(text, text_color, bold=bold, dim=dim)
        frame = self._color(box["v"], frame_color)
        print(f"{frame}{' ' * pad_left}{painted}{' ' * pad_right}{frame}")

    def _box_row_left(self, text: str, style: str, frame_color: str, *, bold: bool = False, dim: bool = False, text_color: str = "white") -> None:
        box = self._box(style)
        inner_w = self.width - 2
        text = self._fit(text, inner_w)
        pad = inner_w - len(text)
        painted = self._color(text, text_color, bold=bold, dim=dim)
        frame = self._color(box["v"], frame_color)
        print(f"{frame}{painted}{' ' * pad}{frame}")

    def _box_row_prerendered(self, rendered: str, style: str, frame_color: str) -> None:
        """Row whose text already contains ANSI codes — pad based on *visible* length."""
        box = self._box(style)
        inner_w = self.width - 2
        visible = _strip_ansi(rendered)
        if len(visible) > inner_w:
            # truncate visible part while keeping ANSI intact is hard; strip ANSI and re-render plain.
            rendered = self._fit(visible, inner_w)
            pad = inner_w - len(_strip_ansi(rendered))
        else:
            pad = inner_w - len(visible)
        frame = self._color(box["v"], frame_color)
        print(f"{frame}{rendered}{' ' * pad}{frame}")

    @staticmethod
    def _fit(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"


def _centered_rule(title: str, width: int, char: str) -> str:
    """Legacy helper kept for backward compatibility."""
    text = f" {title} "
    if len(text) >= width:
        return text.strip()
    left = (width - len(text)) // 2
    right = width - len(text) - left
    return char * left + text + char * right


def _fmt_duration(seconds) -> str:
    try:
        seconds = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _mode_label(mode: str) -> str:
    return "HARDWARE" if mode == "hardware_aware" else "IDEAL"


def _mode_prefix(mode: str) -> str:
    return "[HARDWARE]" if mode == "hardware_aware" else "[IDEAL]"


def _phase_title(mode: str, cfg: dict) -> str:
    if cfg.get("training_mode") == "dual":
        number = "1" if mode == "ideal" else "2"
        return f"PHASE {number}: {_mode_label(mode)} RUN"
    return f"{_mode_label(mode)} RUN"


def _last(values, default=None):
    if isinstance(values, (list, tuple)) and values:
        return values[-1]
    return default


def _confusion(true: list[int], pred: list[int]) -> list[list[int]]:
    n = max(true + pred + [0]) + 1
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for t, p in zip(true, pred):
        matrix[t][p] += 1
    return matrix


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _make_scheduler(optimizer, cfg: dict):
    sched = cfg.get("lr_scheduler", {})
    if not sched or sched.get("type", "none") in {"none", None}:
        return None
    kind = str(sched.get("type", "reduce_on_plateau")).lower().replace(" ", "_")
    if kind == "reduce_on_plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched.get("factor", 0.5)),
            patience=int(sched.get("patience", 7)),
            min_lr=float(sched.get("min_lr", 1e-5)),
        )
    if kind in ("cosine", "cosine_annealing", "cosine_annealing_warm_restarts"):
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(sched.get("t_0", 10)),
            T_mult=int(sched.get("t_mult", 2)),
            eta_min=float(sched.get("min_lr", 1e-5)),
        )
    logging.warning("Unknown scheduler type '%s'; no scheduler will be used.", kind)
    return None


def _step_scheduler(scheduler, val_loss: float) -> None:
    """Step scheduler in a type-aware way. ReduceLROnPlateau needs a metric; others don't."""
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def _trainable_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _model_parameter_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += float(p.detach().norm().item()) ** 2
    return total ** 0.5


def _model_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm().item()) ** 2
    return total ** 0.5


def _progress(iterable, description: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print(f"{description} ...")
        return iterable
    return tqdm(iterable, desc=description, leave=False, dynamic_ncols=True, file=sys.stdout)


def _progress_update(iterator, loss: float, acc: float) -> None:
    if hasattr(iterator, "set_postfix"):
        iterator.set_postfix(loss=f"{loss:.4f}", acc=f"{acc * 100:.2f}%")
