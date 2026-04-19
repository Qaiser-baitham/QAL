"""
Hardware-aware quantized layers for CIM simulation.
- QuantLinear / QuantConv2d replace weights with device-aware effective weights
- ADC quantization applied to MAC output (simulates column ADC resolution)
- Activation quantization with straight-through estimator
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..device_model.device import MemristorDevice


class _STEQuant(torch.autograd.Function):
    """Straight-through quantizer."""
    @staticmethod
    def forward(ctx, x, n_levels):
        if n_levels <= 0: return x
        xmax = x.abs().max().clamp(min=1e-8)
        x_n = (x / xmax).clamp(-1, 1)
        q = torch.round(x_n * (n_levels - 1) / 2) * (2 / (n_levels - 1))
        return q * xmax
    @staticmethod
    def backward(ctx, g): return g, None


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits is None or bits <= 0: return x
    return _STEQuant.apply(x, 2 ** bits)


class QuantLinear(nn.Linear):
    def __init__(self, *a, device_model: Optional[MemristorDevice] = None,
                 w_bits: int = 4, act_bits: int = 8, adc_bits: int = 6,
                 noise_std: float = 0.0, hw_aware: bool = True, **kw):
        super().__init__(*a, **kw)
        self.device_model = device_model
        self.w_bits = w_bits; self.act_bits = act_bits
        self.adc_bits = adc_bits; self.noise_std = noise_std
        self.hw_aware = hw_aware

    def forward(self, x):
        if not self.hw_aware:
            return F.linear(x, self.weight, self.bias)
        x_q = quantize(x, self.act_bits)
        w = self.weight
        if self.device_model is not None:
            w = self.device_model.apply(w, noise=self.training, t=1.0)
        else:
            w = quantize(w, self.w_bits)
        out = F.linear(x_q, w, self.bias)
        if self.noise_std > 0 and self.training:
            out = out + self.noise_std * out.abs().mean() * torch.randn_like(out)
        out = quantize(out, self.adc_bits)  # ADC
        return out


class QuantConv2d(nn.Conv2d):
    def __init__(self, *a, device_model: Optional[MemristorDevice] = None,
                 w_bits: int = 4, act_bits: int = 8, adc_bits: int = 6,
                 noise_std: float = 0.0, hw_aware: bool = True, **kw):
        super().__init__(*a, **kw)
        self.device_model = device_model
        self.w_bits = w_bits; self.act_bits = act_bits
        self.adc_bits = adc_bits; self.noise_std = noise_std
        self.hw_aware = hw_aware

    def forward(self, x):
        if not self.hw_aware:
            return self._conv_forward(x, self.weight, self.bias)
        x_q = quantize(x, self.act_bits)
        w = self.weight
        if self.device_model is not None:
            w = self.device_model.apply(w, noise=self.training, t=1.0)
        else:
            w = quantize(w, self.w_bits)
        out = self._conv_forward(x_q, w, self.bias)
        if self.noise_std > 0 and self.training:
            out = out + self.noise_std * out.abs().mean() * torch.randn_like(out)
        out = quantize(out, self.adc_bits)
        return out
