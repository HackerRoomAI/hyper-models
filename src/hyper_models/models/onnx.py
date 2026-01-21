"""ONNX model wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hyper_models.models import Model


class ONNXModel(Model):
    """ONNX Runtime model wrapper."""

    def __init__(self, path: Path, geometry: str, dim: int) -> None:
        self._path = path
        self._geometry = geometry
        self._dim = dim
        self._session = None

    @property
    def geometry(self) -> str:
        return self._geometry

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_session(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self._path),
            providers=["CPUExecutionProvider"],
        )

    def encode(self, images: np.ndarray) -> np.ndarray:
        """Encode images to embeddings.

        Args:
            images: (B, 3, H, W) float32 in [0, 1]

        Returns:
            (B, D) embeddings
        """
        self._ensure_session()

        # Run inference
        outputs = self._session.run(None, {"image": images})

        # First output is the embedding
        return outputs[0]
