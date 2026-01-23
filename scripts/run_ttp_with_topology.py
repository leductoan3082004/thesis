#!/usr/bin/env python
"""TTP service startup script with D-Cliques topology configuration."""

import argparse
import json
import os
from pathlib import Path

from torchvision import datasets, transforms

from secure_aggregation.communication.ttp_service import TopologyConfig, serve
from secure_aggregation.data import dirichlet_partition
from secure_aggregation.topology import build_full_topology, compute_node_labels_from_partition
from secure_aggregation.utils import configure_logging, get_logger

logger = get_logger("ttp_startup")


def load_mnist_labels(data_dir: str = "/app/data") -> dict[int, int]:
    """Load MNIST dataset labels."""
    logger.info(f"Loading MNIST labels from {data_dir}")
    tform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tform)
    labels = {i: int(train_ds[i][1]) for i in range(len(train_ds))}
    logger.info(f"Loaded {len(labels)} MNIST labels")
    return labels


def write_topology_file(
    labels: dict[int, int],
    num_clients: int,
    clique_size: int,
    alpha: float,
    seed: int,
    edge_mode: str,
    iterations: int,
    output_path: str,
) -> None:
    """Build topology and write to shared config file for nodes to read."""
    dataset_indices = list(labels.keys())

    partition = dirichlet_partition(
        dataset=dataset_indices,
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )
    partition = {f"node_{key.split('_')[-1]}": value for key, value in partition.items()}
    node_labels = compute_node_labels_from_partition(partition, labels)

    cliques, intra_edges, inter_edges, edge_counts = build_full_topology(
        node_labels=node_labels,
        clique_size=clique_size,
        iterations=iterations,
        edge_mode=edge_mode,
        seed=seed,
    )

    topology_data = {
        "num_cliques": len(cliques),
        "cliques": [sorted(list(c)) for c in cliques],
        "inter_edges": [[e[0], e[1]] for e in inter_edges],
        "edge_counts": edge_counts,
        "topology_type": "d_cliques",
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(topology_data, f, indent=2)

    logger.info(
        f"Wrote topology to {output_path}: {len(cliques)} cliques, {len(inter_edges)} inter-edges"
    )


def main():
    configure_logging()

    parser = argparse.ArgumentParser(description="TTP Service with D-Cliques Topology")
    parser.add_argument("--port", type=int, default=50051, help="TTP service port")
    parser.add_argument("--num-clients", type=int, default=4, help="Number of clients")
    parser.add_argument("--clique-size", type=int, default=4, help="Clique size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--edge-mode", type=str, default="small_world", help="Inter-clique edge mode")
    parser.add_argument("--iterations", type=int, default=1000, help="Topology iterations")
    parser.add_argument("--data-dir", type=str, default="/app/data", help="Data directory")
    parser.add_argument("--config", type=str, help="JSON config file (overrides CLI args)")
    parser.add_argument("--topology-output", type=str, default="/app/config/topology.json",
                        help="Path to write topology config for nodes")
    args = parser.parse_args()

    # Load config from file if provided
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_data = json.load(f)
        num_clients = config_data.get("num_clients", args.num_clients)
        clique_size = config_data.get("clique_size", args.clique_size)
        alpha = config_data.get("alpha", args.alpha)
        seed = config_data.get("seed", args.seed)
        edge_mode = config_data.get("inter_clique_edges", args.edge_mode)
        iterations = config_data.get("topology_iterations", args.iterations)
        logger.info(f"Loaded config from {args.config}")
    else:
        num_clients = args.num_clients
        clique_size = args.clique_size
        alpha = args.alpha
        seed = args.seed
        edge_mode = args.edge_mode
        iterations = args.iterations

    # Environment variable overrides
    num_clients = int(os.environ.get("NUM_CLIENTS", num_clients))
    clique_size = int(os.environ.get("CLIQUE_SIZE", clique_size))
    alpha = float(os.environ.get("ALPHA", alpha))
    seed = int(os.environ.get("SEED", seed))

    logger.info(f"Starting TTP with topology: num_clients={num_clients}, clique_size={clique_size}, alpha={alpha}")

    # Load MNIST labels
    labels = load_mnist_labels(args.data_dir)

    # Write topology file for nodes to read inter_edges
    write_topology_file(
        labels=labels,
        num_clients=num_clients,
        clique_size=clique_size,
        alpha=alpha,
        seed=seed,
        edge_mode=edge_mode,
        iterations=iterations,
        output_path=args.topology_output,
    )

    # Create topology config
    topology_config = TopologyConfig(
        num_clients=num_clients,
        clique_size=clique_size,
        alpha=alpha,
        seed=seed,
        inter_clique_edges=edge_mode,
        topology_iterations=iterations,
    )

    # Start TTP service
    serve(port=args.port, topology_config=topology_config, labels=labels)


if __name__ == "__main__":
    main()
