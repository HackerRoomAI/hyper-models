"""hyper-models: A model zoo for non-Euclidean embedding models."""

from hyper_models.registry import list_models
from hyper_models.loader import load

__all__ = ["load", "list_models"]
__version__ = "0.1.0"
