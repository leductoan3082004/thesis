from secure_aggregation.node import NodeEngine, NodeRuntimeConfig, ReliabilityScore
from secure_aggregation.config.models import NodeRole


def test_small_scale_secure_aggregation_cycles() -> None:
    engines = [
        NodeEngine(
            NodeRuntimeConfig(
                node_id=f"u{i}",
                role=NodeRole.HYBRID,
                reliability=ReliabilityScore(uptime=1.0, bandwidth=1.0 - i * 0.01, latency=0.1 + i * 0.01),
            )
        )
        for i in range(10)
    ]
    model_vectors = {e.config.node_id: [float(i), float(i + 1)] for i, e in enumerate(engines)}
    for window in range(3):
        agg_id, result = NodeEngine.orchestrate_window(
            engines, model_vectors, threshold=6, window_index=window, dropouts=["u9"] if window == 1 else []
        )
        assert agg_id in model_vectors
        assert len(result.survivors) >= 6
