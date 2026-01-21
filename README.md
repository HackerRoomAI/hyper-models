# hyper_models

A model zoo for non-Euclidean embedding models (hyperbolic, spherical, mixed-curvature), designed to plug into HyperView while keeping **runtime inference torch-free**.

This directory currently lives inside the HyperView monorepo. The long-term goal is to evolve this into a standalone `hyper-models` package.

## Why this exists

- HyperView needs non-Euclidean embeddings to visualize hierarchical structure (e.g., Poincare disk).
- We do not want to ship PyTorch as a default dependency for HyperView.
- The practical compromise: export models to ONNX, publish artifacts to Hugging Face Hub, and run inference with `onnxruntime`.

## Repository layout

```
hyper_models/
  model_zoo.py         # ModelSpec constructors for HyperView
  hycoclip_onnx/        # HyCoCLIP/MERU -> ONNX export harness + validation
```

## Using with HyperView

The public integration point is a HyperView `ModelSpec`. This repo provides convenience constructors in `model_zoo.py`.

Example (torch-free runtime):

```python
import hyperview as hv
from hyper_models import model_zoo as mz

ds = hv.Dataset("my_dataset")
ds.add_images_dir("/path/to/images")

spec = mz.hycoclip_onnx(
    model_id="hycoclip_vit_s",
    onnx_path="/path/to/hycoclip_vit_s_image_encoder.onnx",
)

ds.compute_embeddings(spec)
ds.compute_visualization(geometry="poincare")
hv.launch(ds)
```

Important: ONNX exports may use external weights, so the `.onnx` and `.onnx.data` files must sit next to each other.

## Exporting HyCoCLIP/MERU to ONNX

See `hycoclip_onnx/README.md` for the full workflow. The short version:

```bash
cd hycoclip_onnx

uv sync

git clone https://github.com/PalAvik/hycoclip hycoclip_repo
huggingface-cli download avik-pal/hycoclip hycoclip_vit_s.pth --local-dir ./checkpoints

uv run python export_onnx.py \
  --hycoclip-repo ./hycoclip_repo \
  --checkpoint ./checkpoints/hycoclip_vit_s.pth \
  --variant vit_s \
  --onnx ./outputs/hycoclip_vit_s_image_encoder.onnx
```

The exported model has:
- input: `image` (float32, BCHW, in `[0, 1]`, shape `(B, 3, 224, 224)`)
- outputs: `embedding_hyperboloid` (float32, `(B, D+1)`), `curvature` (float32, `(1,)`)

## Publishing ONNX artifacts to Hugging Face

The intent is to host ONNX artifacts on the Hub so HyperView can stay torch-free.

Recommended HF repo structure:

```
README.md
onnx/
  image_encoder.onnx
  image_encoder.onnx.data
```

Consumption tip: for models with external weights, use `huggingface_hub.snapshot_download(...)` (not `hf_hub_download`) so you reliably fetch *both* files.

```python
import os
from huggingface_hub import snapshot_download

snapshot_dir = snapshot_download(
    "your-org/hycoclip-vit-s-image-encoder-onnx",
    allow_patterns=["onnx/image_encoder.onnx", "onnx/image_encoder.onnx.data"],
)
onnx_path = os.path.join(snapshot_dir, "onnx", "image_encoder.onnx")
```

See `hycoclip_onnx/hf/README.md` and `hycoclip_onnx/hf/upload_to_hf.py`.

## Best way to do this in HyperView

HyperView already ships a torch-free provider `hycoclip_onnx`, but it currently expects a local `.onnx` path (and enforces batch size 1).

The best UX for Hub-hosted ONNX is:
- add `hf://repo_id#onnx/image_encoder.onnx` support
- implement it via `huggingface_hub.snapshot_download(...)` so the `.onnx.data` file is present
- then load from the downloaded snapshot directory

Until then, the recommended workflow is to `snapshot_download` in user code and pass the resulting local `.onnx` path to `ModelSpec.checkpoint`.

## Candidate Non-Euclidean Models

Curated list of models suitable for embedding visualization. Priority given to models with pretrained weights and permissive licenses.

### Hyperbolic Vision-Language (Image + Text)

| Model | Venue | Geometry | License | Weights | Links |
|-------|-------|----------|---------|---------|-------|
| **HyCoCLIP** | ICLR 2025 | Lorentz | CC-BY-NC | ✅ ViT-S/B | [paper](https://arxiv.org/abs/2410.06912) · [code](https://github.com/PalAvik/hycoclip) · [weights](https://huggingface.co/avik-pal/hycoclip) |
| **MERU** | ICML 2023 | Lorentz | CC-BY-NC | ✅ ViT-S/B/L | [paper](https://arxiv.org/abs/2304.09172) · [code](https://github.com/facebookresearch/meru) |

### Hyperbolic Vision Encoders / Metric Learning

| Model | Venue | Geometry | License | Weights | Links |
|-------|-------|----------|---------|---------|-------|
| **Hyp-ViT** | CVPR 2022 | Poincaré | **MIT** ✅ | ❌ (uses timm backbones) | [paper](https://arxiv.org/abs/2203.10833) · [code](https://github.com/htdt/hyp_metric) |
| **HIE** | CVPR 2020 | Poincaré | **MIT** ✅ | ❌ | [paper](https://arxiv.org/abs/1904.02239) · [code](https://github.com/leymir/hyperbolic-image-embeddings) |
| **HCNN** | ICLR 2024 | Lorentz | **MIT** ✅ | ❌ | [paper](https://openreview.net/forum?id=ekz1hN5QNh) · [code](https://github.com/kschwethelm/HyperbolicCV) |
| **Hyp-ZSL** | CVPR 2020 | Hyperbolic | Unknown | ❌ | [code](https://github.com/ShaoTengLiu/Hyperbolic_ZSL) |

### Hyperbolic Graph / Hierarchy Embeddings

| Model | Venue | Geometry | License | Links |
|-------|-------|----------|---------|-------|
| **HGCN** | NeurIPS 2019 | Poincaré/Lorentz | Unknown | [paper](http://snap.stanford.edu/hgcn/) · [code](https://github.com/HazyResearch/hgcn) |
| **Poincaré Embeddings** | NeurIPS 2017 | Poincaré | Unknown | [paper](https://arxiv.org/abs/1705.08039) · [code](https://github.com/facebookresearch/poincare-embeddings) (archived) |
| **Entailment Cones** | ICML 2018 | Poincaré | Apache-2.0 | [paper](https://arxiv.org/abs/1804.01882) · [code](https://github.com/dalab/hyperbolic_cones) |
| **Poincaré GloVe** | ICLR-W 2019 | Poincaré | LGPL-2.1 | [code](https://github.com/alex-tifrea/poincare_glove) |

### Data Curation (Hyperbolic)

| Model | Venue | Use Case | License | Links |
|-------|-------|----------|---------|-------|
| **HYPE** | ECCV 2024 | Filtering underspecified data | Other | [paper](https://arxiv.org/abs/2404.17507) · [code](https://github.com/naver-ai/hype) |

### Spherical Embeddings

| Model | Venue | Geometry | License | Links |
|-------|-------|----------|---------|-------|
| **Spherical Text** | NeurIPS 2019 | Spherical | Apache-2.0 | [code](https://github.com/yumeng5/Spherical-Text-Embedding) |
| **SphereFace** | CVPR 2017 | Hypersphere | MIT | [paper](https://arxiv.org/abs/1704.08063) · [code](https://github.com/wy1iu/sphereface) |
| **ArcFace** | CVPR 2019 | Hypersphere | MIT | [paper](https://arxiv.org/abs/1801.07698) · [code](https://github.com/deepinsight/insightface) |

### Product Manifolds (Mixed Curvature)

| Model | Venue | Geometry | License | Links |
|-------|-------|----------|---------|-------|
| **Hyperbolics** | ICLR 2019 | H×S×E products | MIT | [paper](https://openreview.net/forum?id=HJxeWnCcF7) · [code](https://github.com/HazyResearch/hyperbolics) |
| **CurvLearn** | ICDE 2022 | Mixed-curvature | Apache-2.0 | [paper](https://arxiv.org/abs/2203.14683) · [code](https://github.com/alibaba/Curvature-Learning-Framework) |

### Priority for ONNX Export

Based on usefulness for HyperView and license compatibility:

| Priority | Model | Why |
|----------|-------|-----|
| 🥇 | HyCoCLIP/MERU | Best hyperbolic CLIP models, already working |
| 🥈 | Hyp-ViT | MIT license, standard ViT backbone, easy export |
| 🥈 | HIE | MIT license, foundational work, includes hyptorch |
| 🥉 | HCNN | MIT license, but harder export (custom ops) |

### Related Libraries

| Library | Purpose | PyPI |
|---------|---------|------|
| [geoopt](https://github.com/geoopt/geoopt) | Riemannian optimization in PyTorch | `pip install geoopt` |
| [HypLL](https://github.com/maxvanspengler/hyperbolic_learning_library) | Hyperbolic layers for PyTorch | `pip install hypll` |
| [hyptorch](https://github.com/leymir/hyperbolic-image-embeddings) | Poincaré ball operations | (in HIE repo) |

## License and attribution

The tooling in this directory follows the HyperView project license (MIT).

Model checkpoints and derived ONNX artifacts may have more restrictive licenses.
- HyCoCLIP and MERU are CC BY-NC in their upstream repos (non-commercial).
- Always verify upstream licensing before publishing or redistributing derived artifacts.
