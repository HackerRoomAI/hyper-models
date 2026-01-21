"""Model registry and metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Metadata for a registered model."""

    name: str
    geometry: str  # 'hyperboloid', 'poincare', 'sphere'
    dim: int
    hub_id: str
    license: str
    description: str = ""


# Model registry
_MODELS: dict[str, ModelInfo] = {
    "hycoclip-vit-s": ModelInfo(
        name="hycoclip-vit-s",
        geometry="hyperboloid",
        dim=512,
        hub_id="hyperview-org/hyperbolic-clip",
        license="CC-BY-NC",
        description="HyCoCLIP ViT-Small image encoder",
    ),
    # Future models:
    # "hycoclip-vit-b": ModelInfo(...),
    # "meru-vit-s": ModelInfo(...),
}


def list_models(geometry: str | None = None) -> list[str]:
    """List available model names.

    Args:
        geometry: Filter by geometry type ('hyperboloid', 'poincare', 'sphere')

    Returns:
        List of model names
    """
    if geometry is None:
        return list(_MODELS.keys())
    return [name for name, info in _MODELS.items() if info.geometry == geometry]


def get_model_info(name: str) -> ModelInfo:
    """Get metadata for a model.

    Args:
        name: Model name

    Returns:
        ModelInfo dataclass

    Raises:
        KeyError: If model not found
    """
    if name not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _MODELS[name]
