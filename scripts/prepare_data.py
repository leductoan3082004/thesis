"""Download and prepare MNIST dataset for Docker containers."""

import ssl
from pathlib import Path

from torchvision import datasets, transforms

# Disable SSL verification for MNIST download
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    """Download MNIST dataset to data directory."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    print("Downloading MNIST training set...")
    tform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tform)
    print(f"Downloaded {len(train_ds)} training samples")

    print("Downloading MNIST test set...")
    test_ds = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=tform)
    print(f"Downloaded {len(test_ds)} test samples")

    print("\nMNIST dataset prepared successfully!")
    print(f"Data location: {data_dir.absolute()}")


if __name__ == "__main__":
    main()
