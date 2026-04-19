from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import resnet18, resnet34


COMPATIBILITY = {
    "MNIST": {"MLP", "ANN", "CNN", "LeNet", "SNN"},
    "FMNIST": {"MLP", "ANN", "CNN", "DeepCNN", "LeNet", "SNN"},
    "KMNIST": {"MLP", "ANN", "CNN", "LeNet", "SNN"},
    "EMNIST": {"MLP", "ANN", "CNN", "DeepCNN", "LeNet"},
    "CIFAR10": {"CNN", "DeepCNN", "VGGSmall", "VGG11", "VGG16", "ResNet18", "ResNet34"},
    "CIFAR100": {"CNN", "DeepCNN", "VGGSmall", "VGG11", "VGG16", "ResNet18", "ResNet34"},
    "SVHN": {"CNN", "DeepCNN", "VGGSmall", "VGG11", "ResNet18"},
    "TinyImageNet": {"DeepCNN", "VGGSmall", "VGG11", "ResNet18", "ResNet34"},
}


def create_model(model_type: str, dataset: str, spec: dict) -> nn.Module:
    if model_type == "CUSTOM":
        raise ValueError("CUSTOM model requires editing src/models/factory.py with a project-specific model.")
    allowed = COMPATIBILITY.get(dataset, set())
    if model_type not in allowed:
        raise ValueError(f"Model {model_type} is not compatible with {dataset}. Allowed: {sorted(allowed)}")
    c, size, classes = spec["channels"], spec["size"], spec["classes"]
    if model_type in {"MLP", "ANN", "SNN"}:
        model = MLP(c * size * size, classes)
    elif model_type == "CNN":
        model = SimpleCNN(c, classes)
    elif model_type == "DeepCNN":
        model = DeepCNN(c, classes)
    elif model_type == "LeNet":
        model = LeNet(c, classes, size)
    elif model_type == "VGGSmall":
        model = VGGSmall(c, classes)
    elif model_type == "VGG11":
        model = _vgg(c, classes, [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"])
    elif model_type == "VGG16":
        model = _vgg(c, classes, [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"])
    elif model_type == "ResNet18":
        model = _resnet(resnet18, c, classes)
    elif model_type == "ResNet34":
        model = _resnet(resnet34, c, classes)
    else:
        raise ValueError(f"Unsupported model: {model_type}")
    params = sum(p.numel() for p in model.parameters())
    logging.info("Selected dataset=%s model=%s input=%dx%dx%d classes=%d parameters=%d", dataset, model_type, c, size, size, classes, params)
    logging.info("Architecture:\n%s", model)
    return model


class MLP(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, classes))

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            _block(channels, 64),
            _block(64, 128),
            _block(128, 256),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256 * 2 * 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class LeNet(nn.Module):
    def __init__(self, channels: int, classes: int, size: int):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(channels, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2))
        with torch.no_grad():
            flat = self.features(torch.zeros(1, channels, size, size)).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class VGGSmall(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.features = nn.Sequential(_block(channels, 64), _block(64, 128), _block(128, 256), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def _block(cin: int, cout: int):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


def _vgg(channels: int, classes: int, cfg: list):
    layers = []
    in_c = channels
    for item in cfg:
        if item == "M":
            layers.append(nn.MaxPool2d(2))
        else:
            layers.extend([nn.Conv2d(in_c, item, 3, padding=1), nn.BatchNorm2d(item), nn.ReLU(inplace=True)])
            in_c = item
    return nn.Sequential(nn.Sequential(*layers), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(in_c, classes))


def _resnet(builder, channels: int, classes: int):
    model = builder(weights=None, num_classes=classes)
    if channels != 3:
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model
