# Example data archive

`apply_seg_onnx_example_data.tar.gz` holds what the example configs need and git does not: the ONNX models and one
test scan. Unpack it at the root of the repository:

```bash
tar xzf apply_seg_onnx_example_data.tar.gz
sha256sum -c examples/data/SHA256SUMS
```

| file | content |
|---|---|
| `examples/models/_20240404_conjurer_trained_dice_7733.onnx` | model of `mindglide.json` |
| `examples/models/WMH-SynthSeg_v10_231110_new.onnx` + `.onnx.data` | model of `wmh_synthseg.json` (weights in the `.data` file, keep both together) |
| `examples/models/synthsr_v20_230130_batch.onnx` | model of `synthsr.json` and `synthsr_no_tta.json` (made by `synthsr_make_batch_dynamic.py` from the fixed-batch export) |
| `examples/data/subject43_1_t2w.mnc` | test scan: T2-weighted, 2 mm slices |
| `examples/data/reference/subject43_1_t2w_<config>.mnc` | output of each example config on the test scan (this package, GPU, fp32) |
| `examples/data/reference/subject43_1_t2w_<config>.csv` | label volumes in mm³ (`--measure`) of the two segmentation configs |

Run an example:

```bash
apply_seg_onnx --config examples/mindglide.json --model_prefix examples/models/ \
    examples/data/subject43_1_t2w.mnc seg.mnc --measure volumes.csv
```

Tests (`python -m pytest -m examples`, part of the plain `python -m pytest` run) run every example config on the scan and
compare the outputs and the measured label volumes with the references; they are skipped when the files are absent. The files can live elsewhere: set
`APPLY_SEG_ONNX_EXAMPLES` to the directory that contains `models/` and `data/`. `APPLY_SEG_ONNX_TEST_CPU=1` runs
them on the CPU when a GPU is available.

The archive is built with `examples/make_data_archive.sh`.
