from pathlib import Path


def ensure_project_dirs(outputs_root: Path) -> dict[str, Path]:
    names = ["checkpoints", "history", "plots", "excel", "reports", "ideal", "hardware", "comparison"]
    paths = {"root": outputs_root}
    for name in names:
        path = outputs_root / name
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    for folder in [
        Path("src/data_loader"),
        Path("src/device_model"),
        Path("src/models"),
        Path("src/training"),
        Path("src/hardware_sim"),
        Path("src/visualization"),
        Path("src/utils"),
        Path("configs"),
        Path("data/raw"),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    return paths
