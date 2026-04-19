"""Layer activation / weight trace capture for hardware evaluation."""
from __future__ import annotations
from typing import Dict, List
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TraceRecorder:
    """Register forward hooks and dump per-layer input & weight traces."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.inputs: Dict[str, List[np.ndarray]] = {}
        self.weights: Dict[str, np.ndarray] = {}
        self._handles = []

    def _hook(self, name):
        def _fn(module, inp, out):
            x = inp[0].detach().cpu().numpy()
            # store first sample only to keep traces compact
            self.inputs.setdefault(name, []).append(x[:1].reshape(-1))
            if hasattr(module, "weight") and module.weight is not None:
                self.weights[name] = module.weight.detach().cpu().numpy()
        return _fn

    def attach(self, layer_types=(nn.Linear, nn.Conv2d)):
        for name, m in self.model.named_modules():
            if isinstance(m, layer_types):
                self._handles.append(m.register_forward_hook(self._hook(name)))

    def detach(self):
        for h in self._handles: h.remove()
        self._handles.clear()

    def export(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # Inputs per layer (one row per observed batch)
        for name, lst in self.inputs.items():
            safe = name.replace(".", "_") or "layer"
            arr = np.stack(lst, axis=0)
            pd.DataFrame(arr).to_csv(os.path.join(out_dir, f"input_{safe}.csv"), index=False)
        # Weights per layer flattened
        for name, W in self.weights.items():
            safe = name.replace(".", "_") or "layer"
            pd.DataFrame(W.reshape(W.shape[0], -1)).to_csv(
                os.path.join(out_dir, f"weight_{safe}.csv"), index=False
            )
