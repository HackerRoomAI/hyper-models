"""Hugging Face Hub integration."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(hub_id: str, filename: str = "onnx/model.onnx") -> Path:
    """Download model from Hugging Face Hub.

    Args:
        hub_id: Repository ID (e.g., 'hyperview-org/hyperbolic-clip')
        filename: Path to ONNX file within the repo

    Returns:
        Path to the downloaded ONNX file
    """
    # Use snapshot_download to get both .onnx and .onnx.data files
    local_dir = snapshot_download(
        hub_id,
        allow_patterns=[f"{filename}*"],  # Gets .onnx and .onnx.data
    )
    return Path(local_dir) / filename
