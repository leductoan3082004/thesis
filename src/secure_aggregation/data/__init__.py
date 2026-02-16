"""Dataset loading and non-IID partition helpers."""

from secure_aggregation.data.dataset import get_labels, load_dataset
from secure_aggregation.data.partition import dirichlet_partition

__all__ = ["dirichlet_partition", "get_labels", "load_dataset"]
