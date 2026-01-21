"""Base model class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Base class for all hyper-models."""

    @property
    @abstractmethod
    def geometry(self) -> str:
        """Geometry type: 'hyperboloid', 'poincare', 'sphere'."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...

    @abstractmethod
    def encode(self, images: np.ndarray) -> np.ndarray:
        """Encode images to embeddings.

        Args:
            images: (B, H, W, 3) uint8 or (B, 3, H, W) float32 in [0, 1]

        Returns:
            (B, D) embeddings
        """
        ...
