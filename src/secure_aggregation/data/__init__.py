from .dataset import get_labels, load_dataset
from .partition import dirichlet_partition, iid_partition

__all__ = [
    "dirichlet_partition",
    "iid_partition",
    "get_labels",
    "load_dataset",
]
