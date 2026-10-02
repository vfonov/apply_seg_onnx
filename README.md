# apply_seg_onnx

ONNX Runtime inference of 3D MRI segmentation models on MINC (and NIfTI) volumes, without PyTorch.
Moved out of `py_deep_seg` (`apply_multi_model_onnx.py` and the inference part of `seg_common/`, `minc/`, `nifti/`).

Pipelines are selected by a JSON config file. They include:
- whole-volume or MONAI-style sliding-window inference;
- majority voting or averaging over several models;
- flip test-time augmentation;
- MindGlide-compatible preprocessing and postprocessing, which reproduces `mindglide` voxel for voxel in fp32;
- exact whole-volume inference of GroupNorm networks, such as WMH-SynthSeg, with GPU memory bounded by a tile size (`TiledGroupNormSession`).

## Installation

```bash
pip install .              # minc2_simple, numpy, scipy, onnx, onnxruntime
pip install .[nifti]       # + nibabel: .nii.gz I/O, "reorient"
pip install .[mindglide]   # + nibabel, scikit-image: MindGlide pipeline ("reorient", "resample": "mindglide")
pip install .[all]         # + tqdm (--progress)
```

`minc2_simple` is not on PyPI. Install it from https://github.com/vfonov/minc2-simple first.

For the GPU, use an onnxruntime build that has the CUDA execution provider: `onnxruntime-gpu` from PyPI, or `onnxruntime` from conda-forge. Do not install the CPU `onnxruntime` wheel next to `onnxruntime-gpu`. If `onnxruntime-gpu` is already present, install this package with `pip install --no-deps .`.

You can also run the package without installing it:

```bash
PYTHONPATH=/path/to/apply_seg_onnx python -m apply_seg_onnx ...
```

## Usage

```bash
apply_seg_onnx --config cfg.json [--model_prefix DIR/] in.mnc out.mnc [--measure volumes.csv]
apply_seg_onnx --config cfg.json --li inputs.txt --lo outputs.txt --progress     # batch of scans
apply_seg_onnx --help                                                            # all options
```

GPU is used by default (`--cpu` to disable, `--device_id`, `--use_tf32`). `--measure` writes per-label volumes; label names come from the config key `labels_desc`.

### Example configs (in the MindGlide/WMH-SynthSeg replication workspace)

| config | pipeline |
|---|---|
| `mindglide_config_conjurer_patch_mk2.json` | The steps below run in this order:<br>1. reorient to RAS<br>2. crop to the nonzero bounding box<br>3. MindGlide resample to 1 mm<br>4. nonzero z-score<br>5. sliding window 128×128×64 with 50% overlap and Gaussian weights<br>6. recover the labels<br>7. keep the largest component |
| `synthseg_wmh_tiled_tta.json` | WMH-SynthSeg as in `minc_wmh_synthseg.py --trim`:<br>• native grid<br>• x/max normalization<br>• whole volume with exact GroupNorm statistics, run in 128³ tiles (`tiled_groupnorm`)<br>• flip-X TTA<br>• FreeSurfer label values (`label_values`) |

### Config keys added for these pipelines

All of these keys are off by default.

| key | effect |
|---|---|
| `reorient` | Reorient to these axis codes, e.g. `"RAS"`, like MONAI `Orientationd`, and reorient back afterwards (needs nibabel). |
| `crop_foreground` | Crop to the bounding box of voxels > 0, then un-crop the result. |
| `resample: "mindglide"` | Resample on the voxel grid the way MindGlide does (needs scikit-image). |
| `spacing_float32` | Round the affine to float32 before MindGlide's exact spacing test. |
| `normalize_mean_std_nonzero` | Z-score of the nonzero voxels only (MONAI `NormalizeIntensity(nonzero=True)`). |
| `largest`, `largest_connectivity` | Keep the largest connected component. Connectivity 1 means 6-connectivity. |
| `sigma_scale`, `sw_batch_size` | Gaussian window weights; windows per ONNX call. |
| `tiled_groupnorm` | Tile size for `TiledGroupNormSession`, used with `whole`. 128 needs about 9.5 GB of GPU memory, 96 about 8 GB, 64 about 4.5 GB. |
| `trim_center` | With `whole` + `trim`, also centre the trimmed box along Z. |
| `label_values` | Map class index to the saved label value. A list `labels_desc` then refers to `label_values[1:]`. |

## Modules

| module | content |
|---|---|
| `inference` | Command line (`main`), `make_onnx_sessions`, `segment_whole`, `segment_with_patches_overlap` (MONAI window layout), `segment_with_onnx[_batched]`, MindGlide pre/post-processing |
| `onnx_tiled` | `TiledGroupNormSession`: drop-in for `InferenceSession.run()` that cuts the graph at every GroupNorm, runs the local stages tile by tile with a halo, and computes exact statistics from per-tile Σx, Σx² |
| `volume` | Normalizations, crop/pad, reorientation, foreground bbox, MindGlide resample/recover, MONAI window starts and Gaussian map (numpy; nibabel/skimage imported lazily) |
| `postprocess` | `find_largest_component`, `measure_volumes`, `save_measurements` |
| `io` | `load_volume_np` / `save_volume`, which dispatch on `.mnc` / `.nii.gz` |
| `minc_io` | MINC2 I/O through `minc2_simple`, `resample_volume`, `uniformize_volume` |
| `nifti_io` | NIfTI I/O through nibabel (optional; raises `ImportError` when it is missing) |
| `geo` | Affine `decompose` / `compose` |

Rule: nothing in this package imports `torch`.

## Tests

```bash
python tests/test_package.py      # or: pytest tests
```

The tests cover:
- `TiledGroupNormSession` against plain ORT on a synthetic GroupNorm U-Net;
- MINC and NIfTI round-trips, including a clear error when nibabel is missing;
- the volume helpers;
- that importing the package does not import torch.
