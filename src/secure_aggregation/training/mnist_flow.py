"""
MNIST secure aggregation demo with local training and quantized secure-agg rounds.

Requires the optional `mnist` extra (torch + torchvision):
pip install -e .[mnist]
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from secure_aggregation.config.models import NodeRole
from secure_aggregation.data import dirichlet_partition
from secure_aggregation.node import NodeEngine, NodeRuntimeConfig, ReliabilityScore


class MnistLinear(nn.Module):
    """Simple linear classifier for MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()


def load_params(model: nn.Module, flat: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(flat, model.parameters())


def quantize_vector(vec: torch.Tensor, scale: float) -> List[int]:
    return [int(round(v.item() * scale)) for v in vec]


def load_quantized_into_model(model: nn.Module, ints: Sequence[int], scale: float) -> None:
    tensor = torch.tensor([i / scale for i in ints], dtype=torch.float32)
    load_params(model, tensor)


@dataclass
class TrainingConfig:
    num_clients: int = 5
    alpha: float = 0.5
    local_epochs: int = 1
    batch_size: int = 64
    rounds: int = 2
    scale: float = 1e6
    threshold: int | None = None
    seed: int = 42
    reliability_noise: float = 0.05


def _client_reliability(idx: int, cfg: TrainingConfig) -> ReliabilityScore:
    base = 1.0 - idx * 0.05
    jitter = random.Random(cfg.seed + idx).uniform(-cfg.reliability_noise, cfg.reliability_noise)
    uptime = max(0.1, base + jitter)
    bandwidth = uptime
    latency = 0.1 + idx * 0.01
    return ReliabilityScore(uptime=uptime, bandwidth=bandwidth, latency=latency)


def _local_train(model: nn.Module, loader: DataLoader, epochs: int, device: torch.device) -> nn.Module:
    model = copy.deepcopy(model).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.train()
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            opt.zero_grad()
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            opt.step()
    return model.cpu()


def run_mnist_secure_agg_demo(cfg: TrainingConfig) -> Dict[str, float]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load MNIST
    tform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tform)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tform)
    labels = {i: int(train_ds[i][1]) for i in range(len(train_ds))}
    parts = dirichlet_partition(
        list(range(len(train_ds))), labels, num_clients=cfg.num_clients, alpha=cfg.alpha, seed=cfg.seed
    )

    # Initial model
    global_model = MnistLinear()
    global_vec = flatten_params(global_model)
    threshold = cfg.threshold or max(2, cfg.num_clients // 2 + 1)

    # Build node engines for rotation (reliability scores)
    engines = [
        NodeEngine(
            NodeRuntimeConfig(
                node_id=f"client_{i}",
                role=NodeRole.HYBRID,
                reliability=_client_reliability(i, cfg),
            )
        )
        for i in range(cfg.num_clients)
    ]

    stats = {"round_acc": {}}
    for rnd in range(cfg.rounds):
        model_vectors: Dict[str, List[int]] = {}
        # Train locally
        for client_id, idxs in parts.items():
            client_model = copy.deepcopy(global_model)
            loader = DataLoader(Subset(train_ds, idxs), batch_size=cfg.batch_size, shuffle=True)
            trained = _local_train(client_model, loader, cfg.local_epochs, device)
            vec = flatten_params(trained)
            model_vectors[client_id] = quantize_vector(vec, cfg.scale)
        # Secure aggregation
        dropouts = []  # could simulate by adding client ids here
        agg_id, result = NodeEngine.orchestrate_window(
            engines, model_vectors, threshold=threshold, window_index=rnd, dropouts=dropouts
        )
        # Update global model
        load_quantized_into_model(global_model, result.aggregate_mean, cfg.scale)
        global_vec = flatten_params(global_model)
        # Eval
        acc = _evaluate(global_model, test_ds, device)
        stats["round_acc"][rnd] = acc
        stats["aggregator"] = agg_id
    stats["final_acc"] = stats["round_acc"][cfg.rounds - 1]
    return stats


def _evaluate(model: nn.Module, test_ds: datasets.MNIST, device: torch.device) -> float:
    model = copy.deepcopy(model).to(device)
    model.eval()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total else 0.0
