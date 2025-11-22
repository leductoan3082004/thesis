import pytest

torch = pytest.importorskip("torch")

from secure_aggregation.training.mnist_flow import (
    MnistLinear,
    flatten_params,
    load_quantized_into_model,
    quantize_vector,
)


def test_quantization_roundtrip_tolerance() -> None:
    model = MnistLinear()
    vec = flatten_params(model)
    scale = 1e4
    ints = quantize_vector(vec, scale)
    load_quantized_into_model(model, ints, scale)
    vec2 = flatten_params(model)
    assert torch.allclose(vec, vec2, atol=1e-3)
