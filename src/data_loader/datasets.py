from __future__ import annotations

from pathlib import Path


DATASET_ALIASES = {
    "mnist": "MNIST",
    "fmnist": "FMNIST",
    "fashionmnist": "FMNIST",
    "fashion_mnist": "FMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "svhn": "SVHN",
    "kmnist": "KMNIST",
    "emnist": "EMNIST",
    "tinyimagenet": "TinyImageNet",
    "tiny_image_net": "TinyImageNet",
    "custom": "CUSTOM",
}


# Human-readable class names per dataset. These are used by the confusion
# matrix, per-class metric plots and class-wise ideal-vs-hardware bars so the
# reader sees the real class (T-shirt / airplane / 3 / ...) instead of a
# numeric index.
CLASS_NAMES = {
    "MNIST": [str(i) for i in range(10)],
    "KMNIST": ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"],
    "FMNIST": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
    "EMNIST": [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z",
        "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
    ],
    "CIFAR10": [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
    "CIFAR100": [
        "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
        "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
        "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
        "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
        "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
        "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
        "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
        "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
        "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
        "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
        "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
        "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
        "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
        "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
        "worm",
    ],
    "SVHN": [str(i) for i in range(10)],
    "TinyImageNet": [f"class_{i:03d}" for i in range(200)],
}


def canonical_dataset(value: str) -> str:
    key = value.strip().replace("-", "").replace(" ", "").lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown dataset '{value}'. Valid choices: {sorted(set(DATASET_ALIASES.values()))}")
    return DATASET_ALIASES[key]


def class_names_for(name: str) -> list[str]:
    """Return human-readable class names for the dataset, or numeric fallback."""
    spec = dataset_spec(name)
    if name in CLASS_NAMES:
        names = CLASS_NAMES[name]
        if len(names) == spec["classes"]:
            return list(names)
    return [str(i) for i in range(spec["classes"])]


def dataset_spec(name: str) -> dict:
    specs = {
        "MNIST": {"channels": 1, "size": 28, "classes": 10, "mean": (0.1307,), "std": (0.3081,)},
        "FMNIST": {"channels": 1, "size": 28, "classes": 10, "mean": (0.2860,), "std": (0.3530,)},
        "KMNIST": {"channels": 1, "size": 28, "classes": 10, "mean": (0.1918,), "std": (0.3483,)},
        "EMNIST": {"channels": 1, "size": 28, "classes": 47, "mean": (0.1736,), "std": (0.3248,)},
        "CIFAR10": {"channels": 3, "size": 32, "classes": 10, "mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
        "CIFAR100": {"channels": 3, "size": 32, "classes": 100, "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761)},
        "SVHN": {"channels": 3, "size": 32, "classes": 10, "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970)},
        "TinyImageNet": {"channels": 3, "size": 64, "classes": 200, "mean": (0.4802, 0.4481, 0.3975), "std": (0.2302, 0.2265, 0.2262)},
    }
    if name not in specs:
        raise ValueError("CUSTOM dataset requires user-provided Dataset integration in src/data_loader/datasets.py.")
    return specs[name]


def _train_transform(name: str, spec: dict):
    """Dataset-appropriate training-time augmentation."""
    from torchvision import transforms

    resize = transforms.Resize((spec["size"], spec["size"]))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(spec["mean"], spec["std"])

    # Heavy augmentation helps CIFAR/SVHN/TinyImageNet significantly.
    if name in {"CIFAR10", "CIFAR100"}:
        return transforms.Compose(
            [
                resize,
                transforms.RandomCrop(spec["size"], padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                to_tensor,
                normalize,
            ]
        )
    if name == "SVHN":
        # SVHN digits must NOT be horizontally flipped (6 vs 9) but mild crop helps.
        return transforms.Compose(
            [
                resize,
                transforms.RandomCrop(spec["size"], padding=4, padding_mode="reflect"),
                to_tensor,
                normalize,
            ]
        )
    if name == "TinyImageNet":
        return transforms.Compose(
            [
                transforms.Resize((spec["size"] + 8, spec["size"] + 8)),
                transforms.RandomCrop(spec["size"], padding=0),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                to_tensor,
                normalize,
            ]
        )
    if name in {"FMNIST", "KMNIST"}:
        # Very mild augmentation for grayscale garment/character data.
        return transforms.Compose(
            [
                resize,
                transforms.RandomCrop(spec["size"], padding=2, padding_mode="reflect"),
                to_tensor,
                normalize,
            ]
        )
    # MNIST / EMNIST: no flip, no crop — preserves digit identity.
    return transforms.Compose([resize, to_tensor, normalize])


def _eval_transform(spec: dict):
    from torchvision import transforms

    return transforms.Compose(
        [transforms.Resize((spec["size"], spec["size"])), transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    )


def create_loaders(name: str, data_root: str, batch_size: int, num_workers: int, device: str):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets

    spec = dataset_spec(name)
    pin = device == "cuda"
    root = Path(data_root)
    train_tf = _train_transform(name, spec)
    eval_tf = _eval_transform(spec)

    if name == "MNIST":
        train = datasets.MNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.MNIST(root, train=False, download=True, transform=eval_tf)
    elif name == "FMNIST":
        train = datasets.FashionMNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=eval_tf)
    elif name == "KMNIST":
        train = datasets.KMNIST(root, train=True, download=True, transform=train_tf)
        test = datasets.KMNIST(root, train=False, download=True, transform=eval_tf)
    elif name == "EMNIST":
        train = datasets.EMNIST(root, split="balanced", train=True, download=True, transform=train_tf)
        test = datasets.EMNIST(root, split="balanced", train=False, download=True, transform=eval_tf)
    elif name == "CIFAR10":
        train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    elif name == "CIFAR100":
        train = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(root, train=False, download=True, transform=eval_tf)
    elif name == "SVHN":
        train = datasets.SVHN(root, split="train", download=True, transform=train_tf)
        test = datasets.SVHN(root, split="test", download=True, transform=eval_tf)
    elif name == "TinyImageNet":
        train = datasets.ImageFolder(root / "tiny-imagenet-200" / "train", transform=train_tf)
        test = datasets.ImageFolder(root / "tiny-imagenet-200" / "val", transform=eval_tf)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # Attach class_names to spec for downstream consumers (confusion matrix,
    # per-class metrics, UI banners).
    spec = dict(spec)
    spec["class_names"] = class_names_for(name)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        spec,
    )
