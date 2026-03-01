"""Config-driven dataset loading for federated learning."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_datasets_config(config_path: str | Path) -> Dict[str, Any]:
    """Load dataset configurations from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def get_input_shape(dataset_name: str, config_path: str | Path) -> list[int]:
    """Get input shape for a dataset by name."""
    configs = load_datasets_config(config_path)
    if dataset_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return configs[dataset_name]["input_shape"]


def get_num_classes(dataset_name: str, config_path: str | Path) -> int:
    """Get number of classes for a dataset by name."""
    configs = load_datasets_config(config_path)
    if dataset_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return configs[dataset_name]["num_classes"]


def load_dataset(
    name: str,
    config_path: str | Path,
    train: bool,
    root_override: str | None = None,
) -> Any:
    """
    Load a dataset by name using configuration from JSON file.

    Args:
        name: Dataset name as defined in config file.
        config_path: Path to datasets.json configuration file.
        train: Whether to load train or test split.
        root_override: Override the root path from config (useful for local vs Docker).

    Returns:
        PyTorch Dataset instance.

    Raises:
        ValueError: If dataset name or type is unknown.
    """
    configs = load_datasets_config(config_path)

    if name not in configs:
        available = list(configs.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    config = configs[name]
    dataset_type = config.get("type", "torchvision")
    root = root_override or config.get("root", "data")

    if dataset_type == "torchvision":
        return _load_torchvision_dataset(config, root, train)
    elif dataset_type == "csv":
        return _load_csv_dataset(config, train)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def _load_torchvision_dataset(config: Dict[str, Any], root: str, train: bool) -> Any:
    """Load a torchvision built-in dataset."""
    # Imported here so non-torchvision code paths (e.g. CSV-only, TTP) do not pay the
    # torch/torchvision import cost.
    from torchvision import datasets, transforms

    class_name = config["class"]
    input_shape = config["input_shape"]

    if input_shape[0] == 1:
        # Grayscale (MNIST / Fashion-MNIST): no spatial augmentation needed.
        tform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    elif input_shape[0] == 3 and train:
        # RGB training split: random crop + flip reduce overfitting on small non-IID partitions.
        tform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    else:
        tform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    tform = transforms.Compose(tform_list)

    ds_class = getattr(datasets, class_name, None)
    if ds_class is None:
        raise ValueError(f"Unknown torchvision dataset class: {class_name}")

    return ds_class(root=root, train=train, download=True, transform=tform)


def _load_csv_dataset(config: Dict[str, Any], train: bool) -> Any:
    """Load a CSV-based dataset."""
    import pandas as pd
    import torch
    from torch.utils.data import Dataset

    class _CSVDataset(Dataset):
        def __init__(self, df: Any, label_column: str, input_shape: list[int]) -> None:
            self.labels = df[label_column].values.tolist()
            self.features = df.drop(columns=[label_column]).values
            self.input_shape = input_shape

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[Any, int]:
            x = torch.tensor(self.features[idx], dtype=torch.float32)
            x = x.view(*self.input_shape)
            return x, self.labels[idx]

    path = config["train_path"] if train else config["test_path"]
    label_column = config.get("label_column", "label")
    input_shape = config["input_shape"]

    df = pd.read_csv(path)
    return _CSVDataset(df, label_column, input_shape)


def get_labels(dataset: Any) -> Dict[int, int]:
    """
    Extract label mapping from a dataset for partitioning.

    Args:
        dataset: PyTorch Dataset instance.

    Returns:
        Dictionary mapping sample index to label.
    """
    labels = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels[i] = int(label)
    return labels
