"""Model loading."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hyper_models.registry import get_model_info
from hyper_models.hub import download_model

if TYPE_CHECKING:
    from hyper_models.models.base import Model


def load(name: str, *, local_path: str | Path | None = None) -> "Model":
    """Load a model by name.

    Args:
        name: Model name (e.g., 'hycoclip-vit-s')
        local_path: Optional local ONNX path (skips Hub download)

    Returns:
        Model instance ready for inference
    """
    from hyper_models.models.onnx import ONNXModel

    info = get_model_info(name)

    if local_path is not None:
        onnx_path = Path(local_path)
    else:
        onnx_path = download_model(info.hub_id)

    return ONNXModel(
        path=onnx_path,
        geometry=info.geometry,
        dim=info.dim,
    )
