from pathlib import Path


def ensure_project_dirs(outputs_root: Path, dataset: str | None = None) -> dict[str, Path]:
    """Set up the on-disk output layout.

    When ``dataset`` is provided the artefacts that change per dataset
    (``plots``, ``excel``, ``reports``, ``history``) live under
    ``{outputs_root}/{dataset_lower}_graphs/`` so repeated runs with different
    datasets accumulate side-by-side (e.g. ``fmnist_graphs/`` and
    ``cifar10_graphs/``) rather than overwriting each other.  Re-running the
    same dataset refreshes its own folder in place without touching the
    others.

    ``checkpoints`` and the legacy ``ideal``/``hardware``/``comparison``
    folders stay at the top of ``outputs_root`` — they are keyed by mode/run
    and are safe to share across datasets.
    """
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {"root": outputs_root}

    scoped_names = ["plots", "excel", "reports", "history"]
    if dataset:
        scope_root = outputs_root / f"{str(dataset).strip().lower()}_graphs"
        scope_root.mkdir(parents=True, exist_ok=True)
        paths["dataset_root"] = scope_root
        for name in scoped_names:
            path = scope_root / name
            path.mkdir(parents=True, exist_ok=True)
            paths[name] = path
    else:
        for name in scoped_names:
            path = outputs_root / name
            path.mkdir(parents=True, exist_ok=True)
            paths[name] = path

    for name in ["checkpoints", "ideal", "hardware", "comparison"]:
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
        Path("data/raw_data"),
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    return paths
