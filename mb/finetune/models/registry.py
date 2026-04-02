"""
Model registry maps short names to adapter classes.

Usage:
    from mb.finetune.models.registry import ModelRegistry

    adapter_cls = ModelRegistry.get("qwen")
    adapter = adapter_cls(config)
"""

from __future__ import annotations
from typing import Dict, Type
from mb.finetune.models.base import ModelBaseAdapter

__all__ = ["ModelRegistry"]


class ModelRegistry:
    """
    Central registry that maps model names -> adapter classes.
    Adapters must be registered with the '@ModelRegistry.register(name)' decorator.

    Example:

        @ModelRegistry.register("qwen")
        class QwenAdapter(ModelBaseAdapter):
            ...
    """

    _registry: Dict[str, Type[ModelBaseAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a new model adapter.

        Example::

            @ModelRegistry.register("qwen")
            class QwenAdapter(ModelBaseAdapter):
                ...
        """

        def decorator(adapter_cls: Type[ModelBaseAdapter]):
            cls._registry[name.lower()] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[ModelBaseAdapter]:
        """
        Return the adapter class for *name*.

        Raises 'KeyError' if the name has not been registered.
        """
        key = name.lower()
        if key not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown model '{name}'. Available adapters: {available}"
            )
        return cls._registry[key]

    @classmethod
    def list(cls) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(cls._registry.keys())

    @classmethod
    def register_defaults(cls):
        """
        Import all built-in adapters so they register themselves.
        This is called automatically when the module is imported, but can be called manually if needed (e.g. after dynamically adding new adapters).
        """
        # Importing the modules triggers the @ModelRegistry.register decorators.
        from mb.finetune.models import qwen as _
        from mb.finetune.models import blip as _
        from mb.finetune.models import clip as _
        from mb.finetune.models import gemini as _
 

# Auto-register built-in adapters on import.
ModelRegistry.register_defaults()
