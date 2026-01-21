# hyper-models

<p align="center">
  <strong>A model zoo for non-Euclidean embedding models</strong>
  <br>
  <em>Hyperbolic · Spherical · Product Manifolds</em>
</p>

<p align="center">
  <a href="https://huggingface.co/collections/hyperview-org/hyper-models-67900e48542fa2ea29a26684">
    <img src="https://img.shields.io/badge/🤗_Models-Hugging_Face-orange" alt="Hugging Face">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue" alt="License: MIT">
  </a>
</p>

---

## Why?

- **Standardized access** to non-Euclidean embedding models
- **Torch-free runtime** via ONNX (models published to Hugging Face Hub)
- **Simple API** — `load()` and `encode()`

## Installation

```bash
pip install hyper-models
```

## Usage

```python
import hyper_models

# List available models
hyper_models.list_models()
# ['hycoclip-vit-s', 'hycoclip-vit-b', 'meru-vit-s', ...]

# Load model (auto-downloads from Hugging Face Hub)
model = hyper_models.load("hycoclip-vit-s")

# Encode images
embeddings = model.encode(images)  # (B, D) ndarray

# Metadata
model.geometry   # 'hyperboloid'
model.dim        # 512
```

## Models

### Hyperbolic

| Model | Available | Paper | License | Code |
|-------|:---------:|-------|---------|------|
| `hycoclip-vit-s` | ✓ | [ICLR 2025](https://arxiv.org/abs/2410.06912) | CC-BY-NC | [PalAvik/hycoclip](https://github.com/PalAvik/hycoclip) |
| `hycoclip-vit-b` | | [ICLR 2025](https://arxiv.org/abs/2410.06912) | CC-BY-NC | [PalAvik/hycoclip](https://github.com/PalAvik/hycoclip) |
| `meru-vit-s` | | [ICML 2023](https://arxiv.org/abs/2304.09172) | CC-BY-NC | [facebookresearch/meru](https://github.com/facebookresearch/meru) |
| `meru-vit-b` | | [ICML 2023](https://arxiv.org/abs/2304.09172) | CC-BY-NC | [facebookresearch/meru](https://github.com/facebookresearch/meru) |
| `hyp-vit` | | [CVPR 2022](https://arxiv.org/abs/2203.10833) | MIT | [htdt/hyp_metric](https://github.com/htdt/hyp_metric) |
| `hie` | | [CVPR 2020](https://arxiv.org/abs/1904.02239) | MIT | [leymir/hyperbolic-image-embeddings](https://github.com/leymir/hyperbolic-image-embeddings) |
| `hcnn` | | [ICLR 2024](https://openreview.net/forum?id=ekz1hN5QNh) | MIT | [kschwethelm/HyperbolicCV](https://github.com/kschwethelm/HyperbolicCV) |

### Spherical

| Model | Available | Paper | License | Code |
|-------|:---------:|-------|---------|------|
| `sphereface` | | [CVPR 2017](https://arxiv.org/abs/1704.08063) | MIT | [wy1iu/sphereface](https://github.com/wy1iu/sphereface) |
| `arcface` | | [CVPR 2019](https://arxiv.org/abs/1801.07698) | MIT | [deepinsight/insightface](https://github.com/deepinsight/insightface) |

### Product Manifolds

| Model | Available | Paper | License | Code |
|-------|:---------:|-------|---------|------|
| `hyperbolics` | | [ICLR 2019](https://openreview.net/forum?id=HJxeWnCcF7) | MIT | [HazyResearch/hyperbolics](https://github.com/HazyResearch/hyperbolics) |

## Export Tooling

This repo also contains tooling to export PyTorch models to ONNX:

```bash
cd export/hycoclip
uv run python export_onnx.py --checkpoint model.pth --onnx model.onnx
```

See [export/hycoclip/README.md](export/hycoclip/README.md) for details.

## References

- [HyCoCLIP](https://github.com/PalAvik/hycoclip)
- [MERU](https://github.com/facebookresearch/meru)
- [geoopt](https://github.com/geoopt/geoopt)
