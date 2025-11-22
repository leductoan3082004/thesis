import pytest

from secure_aggregation.models import ModelRegistry, VectorModel


def test_vector_model_roundtrip_and_updates() -> None:
    model = VectorModel(weights=[1.0, 2.0, 3.0])
    state = model.state_dict()
    assert state == [1.0, 2.0, 3.0]
    model.apply_update([1.0, -2.0, 0.5])
    assert model.weights == [2.0, 0.0, 3.5]
    update = model.compute_update([3.0, 1.0, 4.0])
    assert update == [1.0, 1.0, 0.5]
    model.load_state_dict([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        model.apply_update([1.0, 2.0])  # length mismatch


def test_model_registry_register_and_create() -> None:
    registry = ModelRegistry()
    created = registry.create("vector", weights=[1.0])
    assert isinstance(created, VectorModel)
    with pytest.raises(ValueError):
        registry.register("vector", lambda: None)
    registry.register("custom", lambda value: {"value": value})
    assert registry.create("custom", value=5) == {"value": 5}
