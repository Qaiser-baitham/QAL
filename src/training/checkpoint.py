from __future__ import annotations

from pathlib import Path


def checkpoint_path(root: Path, mode: str, kind: str) -> Path:
    return root / f"{mode}_{kind}.pt"


def save_checkpoint(path: Path, torch_module, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, torch_module) -> dict | None:
    if not path.exists():
        return None
    return torch_module.load(path, map_location="cpu")
