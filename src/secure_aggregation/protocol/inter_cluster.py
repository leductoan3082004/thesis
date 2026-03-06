"""
Inter-cluster aggregation via decentralized SGD averaging.

This module implements the cluster-level merge algorithm for combining
intra-cluster models with neighbor cluster models from IPFS.

Reference:
- McMahan et al., "Adaptive Clipping for Private Federated Learning", NeurIPS 2021
- Lian et al., "Can Decentralized Algorithms Outperform Centralized Algorithms?", NeurIPS 2017
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MergeConfig:
    """Configuration for inter-cluster merge algorithm."""

    max_neighbors: Optional[int] = 4
    neighbor_history: Dict[str, int] = field(default_factory=dict)


class InterClusterMerger:
    """
    Merges intra-cluster model with verified neighbor cluster models using D-SGD.
    """

    def __init__(self, config: Optional[MergeConfig] = None) -> None:
        self.config = config or MergeConfig()

    def merge(
        self,
        theta_local: np.ndarray,
        neighbor_models: List[np.ndarray],
    ) -> np.ndarray:
        """
        Perform simple decentralized averaging with neighbor models.

        Args:
            theta_local: Intra-cluster model from SAP.
            neighbor_models: List of verified neighbor cluster models.

        Returns:
            Merged model theta_cluster^(t+1).
        """
        if not neighbor_models:
            return theta_local.copy()

        stacked_models = [theta_local] + neighbor_models
        theta_final = np.mean(np.stack(stacked_models, axis=0), axis=0)
        return theta_final
