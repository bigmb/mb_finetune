"""Model registry – maps short names to adapter classes.

Usage:
    from mb.finetune.models.registry import ModelRegistry

    adapter_cls = ModelRegistry.get("qwen")
    adapter = adapter_cls(config)
"""

from __future__ import annotations

from typing import Dict, Type

from mb.finetune.models.base import BaseModelAdapter

__all__ = ["ModelRegistry"]


class ModelRegistry:
    """Central registry that maps model names → adapter classes."""

    _registry: Dict[str, Type[BaseModelAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a new model adapter.

        Example::

            @ModelRegistry.register("qwen")
            class QwenAdapter(BaseModelAdapter):
                ...
        """

        def decorator(adapter_cls: Type[BaseModelAdapter]):
            cls._registry[name.lower()] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseModelAdapter]:
        """Return the adapter class for *name*.

        Raises ``KeyError`` if the name has not been registered.
        """
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown model '{name}'. Available adapters: {available}"
            )
        return cls._registry[key]

    @classmethod
    def list(cls):
        """Return sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def register_defaults(cls):
        """Import all built-in adapters so they register themselves."""
        # Importing the modules triggers the @ModelRegistry.register decorators.
        from mb.finetune.models import qwen as _  # noqa: F401
        from mb.finetune.models import blip as _  # noqa: F401
        from mb.finetune.models import clip as _  # noqa: F401


# Auto-register built-in adapters on import.
ModelRegistry.register_defaults()
