# apply_seg_onnx

ONNX Runtime segmentation inference package, see README.md (module index, config keys).

- Nothing in the package may import `torch`.
- Required deps: minc2_simple, numpy, scipy, onnx, ONNX Runtime >= 1.18 (CUDA option `use_tf32`). ONNX Runtime is a pip extra (`[cpu]` onnxruntime / `[gpu]` onnxruntime-gpu), not a plain dependency, because the two distributions clash; the conda recipe requires `onnxruntime >=1.18`. Keep `pyproject.toml` and `conda-recipe/meta.yaml` in sync. nibabel and tqdm are optional and imported lazily / guarded; `.mnc` input must never need nibabel (MINC `reorient` uses the fixed standard-order layout, `volume.reorient_to(..., minc=True)`).
- New behaviour goes behind config keys (default off) so existing configs keep their results.
- Never install packages; tell the user what is missing.
- Tests: `python -m pytest` (synthetic ONNX models built in `tests/conftest.py`; markers `gpu`, `reference`).
