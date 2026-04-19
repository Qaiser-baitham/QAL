from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.data_loader.raw_memristor import RawMemristorAnalyzer
from src.device_model.extractor import DeviceModelExtractor


def run_quality_check(root: Path | str = ".") -> dict:
    root = Path(root)
    plots = root / "outputs" / "plots"
    excel = root / "outputs" / "excel"
    reports = root / "outputs" / "reports"
    checkpoints = root / "outputs" / "checkpoints"
    reports.mkdir(parents=True, exist_ok=True)
    excel.mkdir(parents=True, exist_ok=True)

    checks: list[dict] = []

    raw = RawMemristorAnalyzer(root / "data" / "raw").analyze()
    device = DeviceModelExtractor().extract(raw)
    _add(checks, "raw traces detected", len(raw.traces) >= 2, f"{len(raw.traces)} trace(s)")
    _add(checks, "device on/off ratio valid", device.on_off_ratio > 1.0, f"{device.on_off_ratio:.4g}")
    _add(checks, "device states valid", device.num_states >= 2, str(device.num_states))

    ideal_history = _load_json(root / "outputs" / "history" / "ideal_history.json")
    hardware_history = _load_json(root / "outputs" / "history" / "hardware_aware_history.json")
    for name, history in [("ideal history", ideal_history), ("hardware history", hardware_history)]:
        _add(checks, f"{name} exists", bool(history), "")
        _add(checks, f"{name} has epochs", bool(history.get("epoch")), str(len(history.get("epoch", []))))
        _add(checks, f"{name} has train accuracy", bool(history.get("train_accuracy")), "")
        _add(checks, f"{name} has validation accuracy", bool(history.get("val_accuracy")), "")
        _add(checks, f"{name} has learning rate", bool(history.get("learning_rate")), "")
        _add(checks, f"{name} has parameter norm", bool(history.get("parameter_norm")), "")
        _add(checks, f"{name} has gradient norm", bool(history.get("grad_norm")), "")
        _add(checks, f"{name} accuracy has signal", _series_has_signal(history.get("val_accuracy")), _series_range(history.get("val_accuracy")))
        _add(checks, f"{name} gradients are nonzero", _series_has_positive(history.get("grad_norm")), _series_range(history.get("grad_norm")))

    for name, history in [("ideal history", ideal_history), ("hardware history", hardware_history)]:
        if history:
            chance = _chance_accuracy(history)
            final_acc = _last_float(history.get("val_accuracy"))
            _add(
                checks,
                f"{name} beats chance accuracy",
                final_acc is not None and final_acc > chance + 0.02,
                f"final={final_acc:.4g}, chance={chance:.4g}" if final_acc is not None else f"chance={chance:.4g}",
            )

    required_plots = [
        "training_history.png",
        "01_accuracy_loss_dashboard.png",
        "02_generalization_dashboard.png",
        "03_memristor_impact_dashboard.png",
        "04_learning_rate_dashboard.png",
        "06_curve_quality_dashboard.png",
        "ideal_confusion_matrix.png",
        "hardware_aware_confusion_matrix.png",
        # Phase 2: device characterization
        "device_characterization_dashboard.png",
        # Phase 6: class metrics and normalized CM
        "ideal_confusion_matrix_normalized.png",
        "hardware_aware_confusion_matrix_normalized.png",
        "ideal_class_metrics.png",
        "hardware_aware_class_metrics.png",
        "class_wise_comparison.png",
    ]
    for plot in required_plots:
        path = plots / plot
        _add(checks, f"plot exists: {plot}", path.exists() and path.stat().st_size > 0, str(path.stat().st_size if path.exists() else 0))
        xlsx = excel / f"{Path(plot).stem}.xlsx"
        _add(checks, f"matching excel exists: {xlsx.name}", xlsx.exists() and xlsx.stat().st_size > 0, str(xlsx.stat().st_size if xlsx.exists() else 0))

    for report in ["final_summary.md", "plot_explanations.md", "artifact_index.md"]:
        path = reports / report
        _add(checks, f"report exists: {report}", path.exists() and path.stat().st_size > 0, str(path.stat().st_size if path.exists() else 0))

    # Phase 7: DSE outputs
    dse_files = ["dse_precision_subarray.xlsx", "dse_noise_sensitivity.xlsx", "dse_heatmap.png", "dse_pareto.png"]
    for dse_file in dse_files:
        if dse_file.endswith(".png"):
            path = plots / dse_file if (plots / dse_file).exists() else excel / dse_file
        else:
            path = excel / dse_file
        _add(checks, f"DSE output: {dse_file}", path.exists() and path.stat().st_size > 0, str(path.stat().st_size if path.exists() else 0))

    for ckpt in ["ideal_latest.pt", "hardware_aware_latest.pt"]:
        path = checkpoints / ckpt
        _add(checks, f"checkpoint exists: {ckpt}", path.exists() and path.stat().st_size > 0, str(path.stat().st_size if path.exists() else 0))

    passed = sum(1 for check in checks if check["passed"])
    failed = len(checks) - passed
    status = "PASS" if failed == 0 else "CHECK"
    summary = {"status": status, "passed": passed, "failed": failed, "checks": checks}

    pd.DataFrame(checks).to_excel(excel / "quality_check.xlsx", index=False)
    lines = ["# Quality Check", "", f"- Status: `{status}`", f"- Passed: `{passed}`", f"- Failed: `{failed}`", ""]
    for check in checks:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{mark}` {check['name']}: {check['detail']}")
    (reports / "quality_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _add(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(values) -> list[float]:
    out = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        out.append(number)
    return out


def _series_has_signal(values) -> bool:
    nums = _numeric(values)
    return len(nums) >= 2 and max(nums) - min(nums) > 1e-6


def _series_has_positive(values) -> bool:
    nums = _numeric(values)
    return bool(nums) and max(nums) > 1e-12


def _series_range(values) -> str:
    nums = _numeric(values)
    if not nums:
        return "n/a"
    return f"min={min(nums):.4g}, max={max(nums):.4g}, n={len(nums)}"


def _last_float(values):
    nums = _numeric(values)
    return nums[-1] if nums else None


def _chance_accuracy(history: dict) -> float:
    matrix = history.get("confusion_matrix") or []
    classes = len(matrix) if isinstance(matrix, list) and matrix else 10
    return 1.0 / max(classes, 1)


if __name__ == "__main__":
    result = run_quality_check(Path.cwd())
    print(f"{result['status']}: {result['passed']} passed, {result['failed']} failed")
