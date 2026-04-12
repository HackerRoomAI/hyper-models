## 0.2.0 - 2026-04-12

### Features
- Add UNCHA catalog entries for `uncha-vit-s` and `uncha-vit-b`
- Introduce an internal loader abstraction so catalog entries can route to ONNX or optional torch-backed runtimes behind one public `hyper_models.load(...)` API
- Add the optional `ml` extra for torch-backed catalog entries

### Documentation
- Clarify that hyper-models is a timm-like catalog for non-Euclidean models
- Document the torch-free default install path and how HyperView uses hyper-models catalog entries