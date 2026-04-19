"""
Hardware metrics estimation (approximate, NeuroSim-inspired).

Model:
  Energy_MAC ~ C * V^2  per crossbar column access + ADC energy per readout
  Latency_MAC ~ R * C   per column + ADC settling (= clock period for simplicity)
  Throughput ~ total_MACs / total_latency
  Efficiency ~ accuracy / energy  (+ TOPS/W-style proxy)

All numbers are APPROXIMATIONS — absolute values should be interpreted
as relative/comparative indicators across design points, not as silicon specs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn


@dataclass
class HWParams:
    v_read: float = 0.2        # V
    c_line_fF: float = 1.0     # fF
    r_wire_ohm: float = 2.0    # ohm
    cell_area_um2: float = 0.01
    adc_energy_pJ: float = 2.0
    clock_period_ns: float = 5.0
    adc_bits: int = 6


def count_macs(model: nn.Module, input_shape) -> int:
    """Count MACs through a dummy forward pass."""
    macs = 0
    hooks = []
    def _hook(m, inp, out):
        nonlocal macs
        if isinstance(m, nn.Linear):
            macs += m.in_features * m.out_features
        elif isinstance(m, nn.Conv2d):
            _, _, H, W = out.shape
            macs += (m.in_channels // m.groups) * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * H * W
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            hooks.append(m.register_forward_hook(_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape, device=next(model.parameters()).device))
    for h in hooks: h.remove()
    return int(macs)


def count_weights(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate(model: nn.Module, input_shape, hw: HWParams,
             accuracy: float, latency_scale: float = 1.0) -> Dict:
    macs = count_macs(model, input_shape)
    nW = count_weights(model)

    C = hw.c_line_fF * 1e-15
    V = hw.v_read
    E_per_mac_J = C * V * V                       # Joules per MAC (crossbar column access)
    E_adc_J = hw.adc_energy_pJ * 1e-12            # per readout
    # assume 1 ADC readout per ~256 MACs as a lumped column model
    adc_ops = max(macs // 256, 1)
    E_total_J = E_per_mac_J * macs + E_adc_J * adc_ops

    RC_s = hw.r_wire_ohm * C
    t_mac_s = RC_s + hw.clock_period_ns * 1e-9
    total_lat_s = t_mac_s * macs * latency_scale

    throughput_ops = macs / max(total_lat_s, 1e-12)
    tops_w = (throughput_ops * 2) / max(E_total_J / total_lat_s, 1e-18) / 1e12  # OPs→TOPS/W proxy

    return {
        "params": nW,
        "macs": macs,
        "energy_J": E_total_J,
        "energy_pJ_per_mac": E_total_J / max(macs, 1) * 1e12,
        "latency_s": total_lat_s,
        "latency_ns_per_mac": t_mac_s * 1e9,
        "throughput_ops": throughput_ops,
        "tops_per_w_proxy": tops_w,
        "accuracy": accuracy,
        "acc_per_pJ": accuracy / max(E_total_J * 1e12, 1e-9),
    }
