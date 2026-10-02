# apply_seg_onnx

ONNX Runtime segmentation inference package, see README.md (module index, config keys).

- Nothing in the package may import `torch`.
- Required deps: minc2_simple, numpy, scipy, onnx, onnxruntime. nibabel and tqdm are optional and imported lazily / guarded.
- New behaviour goes behind config keys (default off) so existing configs keep their results.
- Never install packages; tell the user what is missing.
- Tests: `python tests/test_package.py` (pytest-compatible).
