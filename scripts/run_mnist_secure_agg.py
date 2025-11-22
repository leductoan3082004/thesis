"""
MNIST secure aggregation demo.

Usage:
  pip install -e .[mnist]
  python scripts/run_mnist_secure_agg.py
"""

from secure_aggregation.training.mnist_flow import TrainingConfig, run_mnist_secure_agg_demo
from secure_aggregation.utils import configure_logging, get_logger


def main() -> None:
    configure_logging()
    logger = get_logger("mnist_secure_agg")
    cfg = TrainingConfig()
    stats = run_mnist_secure_agg_demo(cfg)
    logger.info(f"Aggregators per round: {stats['aggregator']}")
    logger.info(f"Round accuracies: {stats['round_acc']}")
    logger.info(f"Final accuracy: {stats['final_acc']:.4f}")


if __name__ == "__main__":
    main()
