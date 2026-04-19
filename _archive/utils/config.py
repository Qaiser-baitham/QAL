"""Configuration loader - YAML -> dict with dot-access."""
from __future__ import annotations
import os, yaml, random
import numpy as np
import torch


class Config(dict):
    """Dict with attribute access and nested support."""
    def __getattr__(self, k):
        v = self[k]
        return Config(v) if isinstance(v, dict) else v
    def __setattr__(self, k, v): self[k] = v


def load_config(path: str = "configs/default.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if pref == "cpu": return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: Config):
    for k in ("processed_dir", "plots_dir", "excel_dir", "logs_dir"):
        os.makedirs(cfg.paths[k], exist_ok=True)


def mode_paths(cfg: Config, tag: str) -> dict:
    """Return per-mode output folder triplet under outputs/<tag>/.

    tag examples: 'ideal', 'hardware', 'comparison'
    """
    base = os.path.join(cfg.paths["outputs_dir"], tag)
    d = {
        "base":      base,
        "plots_dir": os.path.join(base, "plots"),
        "excel_dir": os.path.join(base, "excel"),
        "logs_dir":  os.path.join(base, "logs"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d
