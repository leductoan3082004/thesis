"""Download and prepare datasets for Docker containers."""

import argparse
import json
import ssl
from pathlib import Path

from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context

CONFIG_PATH = Path(__file__).parent.parent / "config" / "datasets.json"


def load_datasets_config() -> dict:
    """Load available datasets from config file."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def download_torchvision_dataset(class_name: str, data_dir: Path) -> None:
    """Download a torchvision dataset by class name."""
    ds_class = getattr(datasets, class_name, None)
    if ds_class is None:
        raise ValueError(f"Unknown torchvision dataset: {class_name}")

    tform = transforms.ToTensor()

    print(f"Downloading {class_name} training set...")
    train_ds = ds_class(root=str(data_dir), train=True, download=True, transform=tform)
    print(f"Downloaded {len(train_ds)} training samples")

    print(f"Downloading {class_name} test set...")
    test_ds = ds_class(root=str(data_dir), train=False, download=True, transform=tform)
    print(f"Downloaded {len(test_ds)} test samples")


def main():
    """Download dataset to data directory."""
    parser = argparse.ArgumentParser(description="Download datasets for federated learning")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset name from config/datasets.json (default: mnist)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    args = parser.parse_args()

    configs = load_datasets_config()

    if args.list:
        print("Available datasets:")
        for name, cfg in configs.items():
            ds_type = cfg.get("type", "torchvision")
            print(f"  - {name} (type: {ds_type})")
        return

    if args.dataset not in configs:
        print(f"Error: Unknown dataset '{args.dataset}'")
        print(f"Available: {list(configs.keys())}")
        return

    config = configs[args.dataset]
    ds_type = config.get("type", "torchvision")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if ds_type == "torchvision":
        class_name = config["class"]
        download_torchvision_dataset(class_name, data_dir)
        print(f"\n{args.dataset} dataset prepared successfully!")
        print(f"Data location: {data_dir.absolute()}")
    elif ds_type == "csv":
        print(f"CSV dataset '{args.dataset}' does not require download.")
        print(f"Ensure files exist at:")
        print(f"  - {config.get('train_path')}")
        print(f"  - {config.get('test_path')}")
    else:
        print(f"Unknown dataset type: {ds_type}")


if __name__ == "__main__":
    main()
