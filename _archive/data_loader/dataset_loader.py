"""Torchvision dataset factory — MNIST / FMNIST / CIFAR10 / CIFAR100 / CUSTOM."""
from __future__ import annotations
import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

_NORM = {
    "MNIST":    ((0.1307,), (0.3081,)),
    "FMNIST":   ((0.2860,), (0.3530,)),
    "CIFAR10":  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}

def _tf(name: str, train: bool):
    mean, std = _NORM[name]
    if name in ("MNIST", "FMNIST"):
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose(aug + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def get_loaders(name: str, root: str, batch_size: int = 128,
                num_workers: int = 2) -> Tuple[DataLoader, DataLoader, dict]:
    name = name.upper()
    root = os.path.join(root, name.lower())
    os.makedirs(root, exist_ok=True)

    if name == "MNIST":
        tr = datasets.MNIST(root, train=True, download=True, transform=_tf(name, True))
        te = datasets.MNIST(root, train=False, download=True, transform=_tf(name, False))
        info = {"in_ch": 1, "img": 28, "n_cls": 10}
    elif name == "FMNIST":
        tr = datasets.FashionMNIST(root, train=True, download=True, transform=_tf(name, True))
        te = datasets.FashionMNIST(root, train=False, download=True, transform=_tf(name, False))
        info = {"in_ch": 1, "img": 28, "n_cls": 10}
    elif name == "CIFAR10":
        tr = datasets.CIFAR10(root, train=True, download=True, transform=_tf(name, True))
        te = datasets.CIFAR10(root, train=False, download=True, transform=_tf(name, False))
        info = {"in_ch": 3, "img": 32, "n_cls": 10}
    elif name == "CIFAR100":
        tr = datasets.CIFAR100(root, train=True, download=True, transform=_tf(name, True))
        te = datasets.CIFAR100(root, train=False, download=True, transform=_tf(name, False))
        info = {"in_ch": 3, "img": 32, "n_cls": 100}
    elif name == "CUSTOM":
        raise NotImplementedError("Place your custom dataset under datasets/custom and extend this loader.")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(te, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    info["name"] = name
    return train_loader, test_loader, info
