"""Neural network architectures for each dataset. Build with either
standard nn.Linear/Conv2d (ideal) or QuantLinear/QuantConv2d (hardware_aware)."""
from __future__ import annotations
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from .quant_layers import QuantLinear, QuantConv2d
from ..device_model.device import MemristorDevice


def _Linear(hw_aware, device_model, cfg):
    def make(*a, **kw):
        if hw_aware:
            return QuantLinear(*a, device_model=device_model,
                               w_bits=cfg["weight_bits"], act_bits=cfg["act_bits"],
                               adc_bits=cfg["adc_bits"], noise_std=cfg["noise_std"],
                               hw_aware=True, **kw)
        return nn.Linear(*a, **kw)
    return make

def _Conv(hw_aware, device_model, cfg):
    def make(*a, **kw):
        if hw_aware:
            return QuantConv2d(*a, device_model=device_model,
                               w_bits=cfg["weight_bits"], act_bits=cfg["act_bits"],
                               adc_bits=cfg["adc_bits"], noise_std=cfg["noise_std"],
                               hw_aware=True, **kw)
        return nn.Conv2d(*a, **kw)
    return make


class MLP(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=256, L=_Linear(False, None, {})):
        super().__init__()
        self.fc1 = L(in_dim, hidden); self.fc2 = L(hidden, hidden); self.fc3 = L(hidden, n_cls)
    def forward(self, x):
        x = x.flatten(1); x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return self.fc3(x)


class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_cls, img, C=_Conv(False, None, {}), L=_Linear(False, None, {})):
        super().__init__()
        self.c1 = C(in_ch, 32, 3, padding=1); self.c2 = C(32, 64, 3, padding=1)
        self.p  = nn.MaxPool2d(2)
        s = img // 4
        self.fc1 = L(64 * s * s, 128); self.fc2 = L(128, n_cls)
    def forward(self, x):
        x = self.p(F.relu(self.c1(x))); x = self.p(F.relu(self.c2(x)))
        x = x.flatten(1); x = F.relu(self.fc1(x)); return self.fc2(x)


class ResNetLite(nn.Module):
    """Lightweight residual CNN for CIFAR."""
    def __init__(self, in_ch, n_cls, C=_Conv(False, None, {}), L=_Linear(False, None, {})):
        super().__init__()
        self.stem = C(in_ch, 32, 3, padding=1); self.bn0 = nn.BatchNorm2d(32)
        self.b1a = C(32, 64, 3, padding=1); self.bn1a = nn.BatchNorm2d(64)
        self.b1b = C(64, 64, 3, padding=1); self.bn1b = nn.BatchNorm2d(64)
        self.sc1 = C(32, 64, 1)
        self.b2a = C(64, 128, 3, padding=1); self.bn2a = nn.BatchNorm2d(128)
        self.b2b = C(128, 128, 3, padding=1); self.bn2b = nn.BatchNorm2d(128)
        self.sc2 = C(64, 128, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc  = L(128, n_cls)
    def forward(self, x):
        x = F.relu(self.bn0(self.stem(x)))
        r = self.sc1(x); x = F.relu(self.bn1a(self.b1a(x))); x = self.bn1b(self.b1b(x))
        x = F.relu(x + r); x = F.max_pool2d(x, 2)
        r = self.sc2(x); x = F.relu(self.bn2a(self.b2a(x))); x = self.bn2b(self.b2b(x))
        x = F.relu(x + r); x = F.max_pool2d(x, 2)
        x = self.pool(x).flatten(1); return self.fc(x)


def build_model(info: dict, training_cfg: dict,
                device_model: Optional[MemristorDevice] = None) -> nn.Module:
    hw_aware = training_cfg["mode"] == "hardware_aware"
    qcfg = {"weight_bits": training_cfg["weight_bits"],
            "act_bits":    training_cfg["act_bits"],
            "adc_bits":    training_cfg["adc_bits"],
            "noise_std":   training_cfg["noise_std"]}
    C = _Conv(hw_aware, device_model, qcfg)
    L = _Linear(hw_aware, device_model, qcfg)
    name = info["name"]; n_cls = info["n_cls"]; ch = info["in_ch"]; img = info["img"]
    if name in ("MNIST",):
        return MLP(ch * img * img, n_cls, L=L)
    if name in ("FMNIST",):
        return SmallCNN(ch, n_cls, img, C=C, L=L)
    if name == "CIFAR10":
        return SmallCNN(ch, n_cls, img, C=C, L=L)
    if name == "CIFAR100":
        return ResNetLite(ch, n_cls, C=C, L=L)
    raise ValueError(f"No model for dataset {name}")
