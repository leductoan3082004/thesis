"""Dataset loading helpers used by node and TTP services."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def load_dataset(
    dataset_name: str,
    config_path: str | Path,
    train: bool = True,
    root_override: str | Path | None = None,
) -> Any:
    """Load a dataset instance from config/datasets.json metadata."""
    configs = _load_datasets_config(config_path)
    key = dataset_name.strip().lower()
    if key not in configs:
        raise KeyError(f"Unknown dataset '{dataset_name}' in {config_path}")

    cfg = configs[key]
    ds_type = str(cfg.get("type", "torchvision")).lower()
    if ds_type != "torchvision":
        raise ValueError(f"Unsupported dataset type '{ds_type}' for '{dataset_name}'")

    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("torchvision is required to load configured datasets") from exc

    class_name = cfg.get("class")
    if not class_name:
        raise ValueError(f"Dataset '{dataset_name}' is missing required 'class' field")

    ds_class = getattr(datasets, class_name, None)
    if ds_class is None:
        raise ValueError(f"Unknown torchvision dataset class '{class_name}'")

    root = str(root_override or cfg.get("root", "data"))
    transform = transforms.ToTensor()

    try:
        return ds_class(root=root, train=train, download=True, transform=transform)
    except TypeError:
        split = "train" if train else "test"
        return ds_class(root=root, split=split, download=True, transform=transform)


def get_labels(dataset: Any) -> Dict[int, int]:
    """Extract {index -> class_label} from a dataset object."""
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            raw = getattr(dataset, attr)
            return {i: int(raw[i]) for i in range(len(raw))}

    if hasattr(dataset, "y"):
        raw = getattr(dataset, "y")
        return {i: int(raw[i]) for i in range(len(raw))}

    labels: Dict[int, int] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if not isinstance(sample, Sequence) or len(sample) < 2:
            raise ValueError("Unable to infer labels from dataset samples")
        labels[idx] = int(sample[1])
    return labels


def _load_datasets_config(config_path: str | Path) -> Mapping[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid dataset config format at {path}")
    return data
