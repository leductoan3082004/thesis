"""Simple model registry."""

from __future__ import annotations

from typing import Callable, Dict

from secure_aggregation.models.vector import VectorModel


class ModelRegistry:
    """Pluggable registry for model factories."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., object]] = {}
        # Default vector model
        self.register("vector", lambda **kwargs: VectorModel(**kwargs))

    def register(self, name: str, factory: Callable[..., object]) -> None:
        if name in self._registry:
            raise ValueError(f"Model '{name}' already registered")
        self._registry[name] = factory

    def create(self, name: str, **kwargs) -> object:
        if name not in self._registry:
            raise ValueError(f"Unknown model '{name}'")
        return self._registry[name](**kwargs)
