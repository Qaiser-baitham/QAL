"""
Memristor device model (NeuroSim-style).
Maps normalized weights <-> multi-level conductance with realistic
non-idealities: limited states, device variation, cycle noise, drift,
and non-linear LTP/LTD update behavior.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass
class DeviceParams:
    num_states: int = 32
    g_on_uS: float = 50.0
    g_off_uS: float = 0.5
    sigma_device: float = 0.05
    sigma_cycle: float = 0.03
    drift_alpha: float = 0.01
    nonlinearity_ltp: float = 1.5
    nonlinearity_ltd: float = -1.5


class MemristorDevice:
    """
    Non-ideal memristor with:
        G(w) = G_off + (G_on - G_off) * f(w_norm)
    where w_norm ∈ [0,1] is mapped via NeuroSim-style non-linear curve,
    then perturbed by device and cycle variability.
    """
    def __init__(self, p: DeviceParams):
        self.p = p

    # ---------- weight <-> conductance ----------
    def normalize_weight(self, w: torch.Tensor) -> torch.Tensor:
        wmax = w.abs().max().clamp(min=1e-8)
        return (w / wmax * 0.5 + 0.5).clamp(0, 1), wmax  # type: ignore

    def quantize(self, w_norm: torch.Tensor) -> torch.Tensor:
        N = self.p.num_states
        return torch.round(w_norm * (N - 1)) / (N - 1)

    def _nonlinear(self, w_norm: torch.Tensor, A: float) -> torch.Tensor:
        """NeuroSim non-linearity: G = (1 - exp(-P/A)) / (1 - exp(-1/A))."""
        if abs(A) < 1e-6:
            return w_norm
        num = 1.0 - torch.exp(-w_norm / A)
        den = 1.0 - float(np.exp(-1.0 / A))
        return num / den

    def conductance(self, w_norm: torch.Tensor, phase: str = "ltp") -> torch.Tensor:
        A = self.p.nonlinearity_ltp if phase == "ltp" else self.p.nonlinearity_ltd
        f = self._nonlinear(w_norm, A)
        G = self.p.g_off_uS + (self.p.g_on_uS - self.p.g_off_uS) * f
        return G

    def add_variation(self, G: torch.Tensor, cycle: bool = True) -> torch.Tensor:
        if self.p.sigma_device > 0:
            G = G * (1 + self.p.sigma_device * torch.randn_like(G))
        if cycle and self.p.sigma_cycle > 0:
            G = G + self.p.sigma_cycle * G.abs() * torch.randn_like(G)
        return G.clamp(min=self.p.g_off_uS * 0.1)

    def drift(self, G: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        if self.p.drift_alpha <= 0: return G
        return G * (max(t, 1e-6) ** (-self.p.drift_alpha))

    # ---------- forward API ----------
    def apply(self, w: torch.Tensor, phase: str = "ltp",
              noise: bool = True, t: float = 1.0) -> torch.Tensor:
        """Return an effective weight tensor after device non-idealities."""
        w_norm, wmax = self.normalize_weight(w)
        w_q = self.quantize(w_norm)
        G = self.conductance(w_q, phase=phase)
        if noise: G = self.add_variation(G, cycle=True)
        G = self.drift(G, t=t)
        # map conductance back to the same scale as original weight
        G_norm = (G - self.p.g_off_uS) / max(self.p.g_on_uS - self.p.g_off_uS, 1e-9)
        w_eff = (G_norm - 0.5) * 2.0 * wmax
        return w_eff
