# apply_seg_onnx

ONNX Runtime segmentation inference package, see README.md (module index, config keys).

- Generic tool: no application-specific names (MindGlide, MONAI, ...) in code, comments, tests or docs; such pipelines are JSON settings only. The exceptions are the setting value `"resample": "mindglide"` and `examples/` (configs for published models, described in the README section "Example configs"; user decision).
- numpy/ONNX only: nothing may import `torch` or carry torch-tensor code paths.
- Defaults reproduce the original `apply_multi_model_onnx.py` except `window_layout` (default `"dense"`; `"legacy"` = original); every difference is listed in `CHANGES_FROM_ORIGINAL.md` (update it with any change of behaviour).
- Required deps: minc2_simple, numpy, scipy, onnx, ONNX Runtime >= 1.18 (CUDA option `use_tf32`). ONNX Runtime is a pip extra (`[cpu]` onnxruntime / `[gpu]` onnxruntime-gpu), not a plain dependency, because the two distributions clash; the conda recipe requires `onnxruntime >=1.18`. Keep `pyproject.toml` and `conda-recipe/meta.yaml` in sync. nibabel and tqdm are optional and imported lazily / guarded; `.mnc` input must never need nibabel (MINC `reorient` uses the fixed standard-order layout, `volume.reorient_to(..., minc=True)`).
- Python interface: `apply_seg_onnx.segment` / `segment_batch` / `load_config` (`api.py`): file paths + config dict; keep its keyword arguments in step with the command line.
- New behaviour goes behind config keys (default off) so existing configs keep their results.
- Never install packages; tell the user what is missing.
- Tests: `python -m pytest` (synthetic ONNX models built in `tests/conftest.py`; markers `gpu`, `examples`). `examples/models/` and `examples/data/` (ONNX models, `subject43_1_t2w.mnc`, reference outputs) are git-ignored and never committed; they ship as `apply_seg_onnx_example_data.tar.gz` (`examples/make_data_archive.sh`, `examples/DATA.md`); the `examples` tests skip without them.
