"""
Device characterization — extract NeuroSim-style device parameters
from raw memristor traces parsed by MemristorLoader.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import pandas as pd
from ..data_loader.memristor_loader import MemristorTrace
from .device import DeviceParams


def _nonlinearity_factor(curve: np.ndarray) -> float:
    """Rough NeuroSim 'A' factor from deviation against linear fit (sign kept)."""
    if curve.size < 3: return 0.0
    x = np.linspace(0, 1, curve.size)
    y = (curve - curve.min()) / max(curve.max() - curve.min(), 1e-9)
    lin = x
    dev = float(np.mean(y - lin))  # +ve → concave, -ve → convex
    return float(np.clip(dev * 10.0, -5.0, 5.0))


def characterize(traces: List[MemristorTrace]) -> Dict:
    """Aggregate device parameters across all traces."""
    gons, goffs, stds, states, ltp_nl, ltd_nl = [], [], [], [], [], []
    for t in traces:
        if t.conductance is None: continue
        g = t.conductance[np.isfinite(t.conductance)]
        if g.size < 3: continue
        gons.append(np.max(g)); goffs.append(np.min(g[g > 0]) if np.any(g > 0) else np.min(g))
        stds.append(np.std(g)); states.append(min(len(np.unique(np.round(g, 8))), 256))
        if t.ltp is not None and t.ltp.size > 2: ltp_nl.append(_nonlinearity_factor(t.ltp))
        if t.ltd is not None and t.ltd.size > 2: ltd_nl.append(_nonlinearity_factor(t.ltd))
    if not gons:
        return {}
    result = {
        "g_on_uS": float(np.mean(gons) * 1e6) if np.mean(gons) < 1 else float(np.mean(gons)),
        "g_off_uS": float(np.mean(goffs) * 1e6) if np.mean(goffs) < 1 else float(np.mean(goffs)),
        "on_off_ratio": float(np.mean(gons) / max(np.mean(goffs), 1e-12)),
        "sigma_cycle": float(np.mean(stds) / max(np.mean(gons), 1e-12)),
        "num_states": int(np.median(states)),
        "nonlinearity_ltp": float(np.mean(ltp_nl)) if ltp_nl else 1.5,
        "nonlinearity_ltd": float(np.mean(ltd_nl)) if ltd_nl else -1.5,
    }
    return result


def update_device_params(base: DeviceParams, extracted: Dict) -> DeviceParams:
    for k, v in extracted.items():
        if hasattr(base, k) and v is not None and np.isfinite(v):
            setattr(base, k, v)
    return base


def characterization_summary(traces: List[MemristorTrace]) -> pd.DataFrame:
    rows = []
    for t in traces:
        rows.append({
            "trace": t.name,
            "g_on": t.meta.get("g_on", np.nan),
            "g_off": t.meta.get("g_off", np.nan),
            "on_off_ratio": t.meta.get("on_off_ratio", np.nan),
            "num_states_est": t.meta.get("num_states_est", 0),
            "std": t.meta.get("std", np.nan),
            "ltp_points": 0 if t.ltp is None else int(t.ltp.size),
            "ltd_points": 0 if t.ltd is None else int(t.ltd.size),
        })
    return pd.DataFrame(rows)
