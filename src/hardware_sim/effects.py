from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from src.device_model.extractor import DeviceModel


@dataclass
class HardwareMetrics:
    energy_pj: float
    latency_ns: float
    adc_bits: int
    weight_bits: int
    macs_per_sample: int = 0
    energy_per_mac_pj: float = 0.0
    tops_w_proxy: float = 0.0


class HardwareAwareSimulator:
    """NeuroSim-inspired behavioral simulator for training-time non-idealities."""

    def __init__(self, device: DeviceModel, cfg: dict):
        self.device = device
        self.cfg = cfg
        self.hw = cfg.get("hardware", {})

    @contextmanager
    def perturbed_weights(self, model):
        originals = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad and p.ndim > 1}
        perturbed = {}
        try:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in originals:
                        effective = self.apply_weight_effects(param)
                        perturbed[name] = effective.detach().clone()
                        param.copy_(effective)
            yield
        finally:
            with torch.no_grad():
                params = dict(model.named_parameters())
                keep_training_update = model.training and any(
                    params[name].grad is not None for name in originals
                )
                for name, value in originals.items():
                    if keep_training_update:
                        params[name].copy_(value + (params[name] - perturbed[name]))
                    else:
                        params[name].copy_(value)

    def apply_weight_effects(self, weight):
        import torch

        bits = int(self.hw.get("weight_bits", 8))
        levels = max(2, min(int(self.device.num_states), 2**bits))
        clipped = torch.clamp(weight, -1.0, 1.0)
        q = torch.round((clipped + 1.0) * (levels - 1) / 2.0)
        mapped = q / (levels - 1)
        g = self.device.g_off + mapped * (self.device.g_on - self.device.g_off)
        norm = 2.0 * (g - self.device.g_off) / max(self.device.g_on - self.device.g_off, 1e-30) - 1.0
        sigma = float(self.hw.get("read_noise", 0.0)) + float(self.device.cycle_variation_sigma) * float(self.hw.get("cycle_variation_scale", 1.0))
        if sigma > 0:
            norm = norm + torch.randn_like(norm) * sigma * torch.clamp(norm.abs(), min=0.05)
        norm = self._stuck_faults(norm)
        return torch.clamp(norm, -1.0, 1.0)

    def quantize_activation(self, x):
        import torch

        bits = int(self.hw.get("activation_bits", 8))
        levels = 2**bits - 1
        xmin, xmax = x.detach().amin(), x.detach().amax()
        if torch.isclose(xmin, xmax):
            return x
        q = torch.round((x - xmin) / (xmax - xmin) * levels) / levels
        quantized = q * (xmax - xmin) + xmin
        return x + (quantized - x).detach()

    def adc_quantize_logits(self, logits):
        import torch

        bits = int(self.hw.get("adc_bits", 6))
        levels = 2**bits - 1
        min_v, max_v = logits.detach().amin(), logits.detach().amax()
        if torch.isclose(min_v, max_v):
            return logits
        q = torch.round((logits - min_v) / (max_v - min_v) * levels) / levels
        quantized = q * (max_v - min_v) + min_v
        return logits + (quantized - logits).detach()

    def estimate_metrics(self, model, samples: int, accuracy: float) -> HardwareMetrics:
        """Phase 5: Physics-based hardware metrics (C*V^2 energy model, RC delay).

        Integrates the approach from metrics.py:
        - E_MAC = C_line * V_read^2 per crossbar column access
        - ADC energy: adc_energy_pJ per readout, 1 readout per ~256 MACs (column model)
        - Latency: RC_delay + clock_period per MAC operation
        - TOPS/W proxy for efficiency comparison

        All values are APPROXIMATE — intended for relative comparison across
        design points, not absolute silicon predictions.
        """
        from src.hardware_sim.metrics import count_macs, HWParams

        # Get input shape from model's first layer
        input_shape = self._infer_input_shape(model)
        try:
            macs_per_sample = count_macs(model, input_shape)
        except Exception:
            # Fallback to parameter count if hook-based counting fails
            macs_per_sample = sum(p.numel() for p in model.parameters() if p.ndim > 1)

        total_macs = macs_per_sample * samples

        # Physics parameters
        v_read = float(self.hw.get("v_read", 0.2))        # Volts
        c_line_fF = float(self.hw.get("c_line_fF", 1.0))  # femtofarads
        r_wire_ohm = float(self.hw.get("r_wire_ohm", 2.0))
        adc_energy_pJ = float(self.hw.get("adc_energy_pJ", 2.0))
        clock_ns = float(self.hw.get("base_latency_ns", 5.0))
        adc_bits = int(self.hw.get("adc_bits", 6))
        weight_bits = int(self.hw.get("weight_bits", 8))

        # E_MAC = C * V^2 (in Joules, then convert to pJ)
        C = c_line_fF * 1e-15  # convert fF to F
        e_mac_J = C * v_read * v_read
        e_mac_pJ = e_mac_J * 1e12

        # ADC: ~1 readout per 256 MACs (one per crossbar column)
        sub_array = self.hw.get("sub_array", [128, 128])
        col_size = sub_array[1] if isinstance(sub_array, (list, tuple)) and len(sub_array) > 1 else 128
        adc_ops = max(total_macs // col_size, 1)
        e_adc_pJ = adc_energy_pJ * adc_ops

        total_energy_pJ = e_mac_pJ * total_macs + e_adc_pJ

        # Latency: RC + clock per MAC, scaled by ADC resolution
        RC_s = r_wire_ohm * C
        t_mac_s = RC_s + clock_ns * 1e-9
        total_latency_s = t_mac_s * macs_per_sample  # per-sample latency
        total_latency_ns = total_latency_s * 1e9

        # TOPS/W proxy
        throughput = (2 * macs_per_sample) / max(total_latency_s, 1e-15)  # 2 ops per MAC (multiply + add)
        power_W = (total_energy_pJ * 1e-12) / max(total_latency_s * samples, 1e-15)
        tops_w = throughput / max(power_W, 1e-18) / 1e12

        return HardwareMetrics(
            energy_pj=float(total_energy_pJ),
            latency_ns=float(total_latency_ns),
            adc_bits=adc_bits,
            weight_bits=weight_bits,
            macs_per_sample=macs_per_sample,
            energy_per_mac_pj=e_mac_pJ,
            tops_w_proxy=tops_w,
        )

    def _infer_input_shape(self, model) -> tuple:
        """Infer model input shape from first layer."""
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                return (m.in_features,)
            if isinstance(m, torch.nn.Conv2d):
                in_c = m.in_channels
                # Assume 28x28 for 1-channel, 32x32 for 3-channel
                size = 28 if in_c == 1 else 32
                return (in_c, size, size)
        return (784,)  # fallback

    def _stuck_faults(self, weight):
        z = float(self.hw.get("stuck_at_zero_rate", 0.0))
        o = float(self.hw.get("stuck_at_one_rate", 0.0))
        if z <= 0 and o <= 0:
            return weight
        out = weight.clone()
        if z > 0:
            out[torch.rand_like(out) < z] = -1.0
        if o > 0:
            out[torch.rand_like(out) < o] = 1.0
        return out
