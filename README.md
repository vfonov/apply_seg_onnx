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

First install `minc2_simple`, which is not on PyPI: `conda install -c minc-forge minc2-simple-sa`, or build it from https://github.com/vfonov/minc2-simple.

Then choose the ONNX Runtime build with an extra. `onnxruntime` and `onnxruntime-gpu` provide the same module, so only one of them may be installed:

```bash
pip install '.[gpu]'          # onnxruntime-gpu >= 1.18 (CUDA)
pip install '.[cpu]'          # onnxruntime >= 1.18 (CPU only)
pip install .                 # ONNX Runtime already installed (e.g. conda-forge onnxruntime, also its CUDA builds)
pip install '.[gpu,all]'      # + nibabel (.nii.gz I/O) and tqdm (--progress)
```

The extras `nifti` (nibabel) and `progress` (tqdm) can also be chosen separately. MINC input needs neither: all config keys, `reorient` included, work on `.mnc` without nibabel.

ONNX Runtime 1.18 or newer is required: the CUDA execution provider is configured with `use_tf32`, an option added in 1.18. Results were verified bit-identical with 1.30; other versions may differ by floating-point noise. To check which providers you have: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`.

### conda

The recipe in `conda-recipe/` builds a noarch package. `minc2_simple` comes from minc-forge as either `minc2-simple-sa` (depends only on libminc) or `minc2-simple` (full minc-toolkit-v2). Conda can't require one of two packages, so two builds are made, `minc2_simple_sa_0` and `minc2_simple_0`. The solver takes the one matching what is installed; in a fresh environment it prefers the `-sa` build. Everything else comes from conda-forge:

```bash
conda build -c minc-forge -c conda-forge conda-recipe
conda install -c minc-forge -c conda-forge --use-local apply_seg_onnx nibabel tqdm   # nibabel, tqdm optional
```

The build tests run the pytest suite against the installed package, without the `gpu` tests. minc-forge has `minc2-simple-sa` for Python 3.9–3.12, and `minc2-simple` up to 3.13.

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
| `reorient` | Reorient to these axis codes, e.g. `"RAS"`, like MONAI `Orientationd`, and reorient back afterwards. For NIfTI, the orientation comes from the affine through nibabel. MINC is read in standard order (positive steps, i,j,k = x,y,z = RAS), so the flips and permutation follow from the axis codes alone, without nibabel. For `"RAS"` this is the identity. |
| `crop_foreground` | Crop to the bounding box of voxels > 0, then un-crop the result. |
| `resample: "mindglide"` | Resample on the voxel grid the way MindGlide does (`scipy.ndimage.zoom`, bit-identical to MindGlide). |
| `spacing_float32` | Round the affine to float32 before MindGlide's exact spacing test. |
| `normalize_mean_std_nonzero` | Z-score of the nonzero voxels only (MONAI `NormalizeIntensity(nonzero=True)`). |
| `largest`, `largest_connectivity` | Keep the largest connected component. Connectivity 1 means 6-connectivity. |
| `sigma_scale`, `sw_batch_size` | Gaussian window weights; windows per ONNX call. |
| `tiled_groupnorm` | Tile size for `TiledGroupNormSession`, used with `whole`. 128 needs about 9.5 GB of GPU memory, 96 about 8 GB, 64 about 4.5 GB. |
| `trim_center` | With `whole` + `trim`, also centre the trimmed box along Z. |
| `label_values` | Map class index to the saved label value. A list `labels_desc` then refers to `label_values[1:]`. The output uses the smallest type that holds the values (uint8/uint16/uint32, int8/16/32 if negative). |

## Modules

| module | content |
|---|---|
| `inference` | Command line (`main`), `make_onnx_sessions`, `segment_whole`, `segment_with_patches_overlap` (MONAI window layout), `segment_with_onnx[_batched]`, MindGlide pre/post-processing |
| `onnx_tiled` | `TiledGroupNormSession`: drop-in for `InferenceSession.run()` that cuts the graph at every GroupNorm, runs the local stages tile by tile with a halo, and computes exact statistics from per-tile Σx, Σx² |
| `volume` | Normalizations, crop/pad, reorientation, foreground bbox, MindGlide resample/recover (`_resize`, on `scipy.ndimage.zoom`), MONAI window starts and Gaussian map (numpy/scipy; nibabel imported lazily, only for NIfTI reorientation) |
| `postprocess` | `find_largest_component`, `measure_volumes`, `save_measurements` |
| `io` | `load_volume_np` / `save_volume`, which dispatch on `.mnc` / `.nii.gz` |
| `minc_io` | MINC2 I/O through `minc2_simple`, `resample_volume`, `uniformize_volume` |
| `nifti_io` | NIfTI I/O through nibabel (optional; raises `ImportError` when it is missing) |
| `geo` | Affine `decompose` / `compose` |

Rule: nothing in this package imports `torch`.

## Tests

```bash
python -m pytest            # from the repository root; needs pytest
python -m pytest -m "not gpu and not reference"   # skip the CUDA and MONAI cross-checks
```

The tests need no data files or PyTorch: they build small synthetic ONNX models on the fly. One is a pointwise
1×1×1 conv whose labels are a known function of intensity; the other is a 2-level GroupNorm U-Net.

| file | covers |
|---|---|
| `test_volume.py` | normalisation, crop/pad, bbox, reorient (NIfTI with nibabel; MINC without it, checked against nibabel), MindGlide resample/recovery, window layout and Gaussian weights (`reference`: against MONAI) |
| `test_resize.py` | `_resize` (scipy port of skimage resize) |
| `test_io.py` | MINC/NIfTI round-trips and affines, metadata/history, missing nibabel, world-space resampling |
| `test_postprocess.py` | largest component, volume measurements, CSV |
| `test_onnx_tiled.py` | `TiledGroupNormSession` against plain ORT (`gpu`: on CUDA) |
| `test_inference.py` | sliding window, whole volume, MindGlide pre/post-processing |
| `test_pipeline.py` | `segment_with_onnx[_batched]` on files: minibatches, measure, recover, fuzzy, label_values, flip TTA, majority, tiled config, MindGlide pipeline on MINC without nibabel, CLI |
| `test_package.py` | the package never imports torch |

Markers:
- `gpu` tests are skipped without `CUDAExecutionProvider`.
- `reference` tests are skipped without monai.

One test is a strict `xfail` documenting a known issue: the precision of tiled GroupNorm depends on the tile size.
