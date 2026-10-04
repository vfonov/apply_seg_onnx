# apply_seg_onnx

ONNX Runtime inference of 3D MRI segmentation models on MINC (and NIfTI) volumes, without PyTorch.
Moved out of `py_deep_seg` (`apply_multi_model_onnx.py` and the inference part of `seg_common/`, `minc/`, `nifti/`).

Pipelines are selected by a JSON config file. They include:
- whole-volume or sliding-window inference;
- majority voting or averaging over several models;
- flip test-time augmentation;
- optional geometry preprocessing (reorientation, foreground crop, voxel-grid resampling) undone on the output;
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

### From Python

```python
import apply_seg_onnx

config = {"models": ["model.onnx"], "n_classes": 3, "patch_sz": 64, "stride": 32}   # keys: see the reference below
apply_seg_onnx.segment("in.mnc", "seg.mnc", config)                                 # one scan, returns the output path
apply_seg_onnx.segment(["t1.mnc", "t2.mnc"], "seg.mnc", config, measure="vol.csv")  # channels of a multi-channel model
apply_seg_onnx.segment_batch(["a.mnc", "b.mnc"], ["a_seg.mnc", "b_seg.mnc"], config, minibatch_size=2)

config = apply_seg_onnx.load_config("cfg.json")                                     # a JSON config file as a dict
apply_seg_onnx.segment("in.nii.gz", "seg.nii.gz", config, model_prefix="/models/", cpu=True)
```

Inputs and outputs are file paths (`str` or `pathlib.Path`); the config dict is not modified. Keyword arguments mirror
the command line: `model_prefix`, `cpu` (default false: GPU), `threads`, `device_id`, `use_tf32`, `measure`, `fuzzy`,
`history`; `segment_batch` adds `minibatch_size`, `progress`, `recover`, `skip_errors`. Errors are raised
(`skip_errors=True` reports a failed minibatch and continues, as the command line does without `--crash`).
`segment_batch` loads the models once for all scans.

### Example configs

Configs for three published models are in `examples/`. `models` holds bare file names, so point `--model_prefix`
(or `model_prefix=` in Python) at the directory with the ONNX files. The models and a test scan are not in git: they
come in a separate archive that unpacks into `examples/models/` and `examples/data/` (see `examples/DATA.md`).

```bash
tar xzf apply_seg_onnx_example_data.tar.gz          # at the root of the repository
apply_seg_onnx --config examples/mindglide.json --model_prefix examples/models/ \
    examples/data/subject43_1_t2w.mnc seg.mnc --measure volumes.csv
```

| config | model | pipeline |
|---|---|---|
| `mindglide.json` | MindGlide brain and lesion segmentation, 20 classes (`_20240404_conjurer_trained_dice_7733.onnx`, exported from the PyTorch checkpoint) | The steps run in this order:<br>1. reorient to RAS<br>2. crop to the nonzero bounding box<br>3. `resample: "mindglide"` to 1 mm<br>4. nonzero z-score<br>5. sliding window 128×128×64 with 50% overlap and Gaussian weights (`gaussian_map: "separable"`)<br>6. recover the labels<br>7. keep the largest component (6-neighbourhood)<br>The original tool runs its convolutions in TF32 on the GPU: add `--use_tf32` to follow it. |
| `wmh_synthseg.json` | WMH-SynthSeg, 33 classes (`WMH-SynthSeg_v10_231110_new.onnx`) | • native grid, axes in (x, y, z) order<br>• x/max normalization<br>• zero padding to multiples of 32 voxels, centred (`pad_center`); nothing is cut off<br>• whole volume with exact GroupNorm statistics, run in 128³ tiles (`tiled_groupnorm`; about 9.5 GB GPU, 64 needs about 4.5 GB)<br>• flip-X TTA with left ↔ right classes swapped<br>• FreeSurfer label values (`label_values`) |
| `synthsr.json` | SynthSR, synthetic 1 mm T1 from any contrast (`synthsr_v20_230130_batch.onnx`) | Regression model on the whole volume with flip-X TTA:<br>1. float64 input, `uniformize_method: "grid"` to 1 mm<br>2. centred padding to multiples of 32 (`pad_center`)<br>3. `normalize_min_max`<br>4. output × 255 clipped to [0, 128] per pass, the two passes averaged<br>5. unsharp mask (σ 1.5)<br>6. saved on the 1 mm grid<br>MINC input only (`reorient` cannot be combined with `uniformize`). |
| `synthsr_no_tta.json` | SynthSR, the same model | As `synthsr.json` without the flipped pass (no `augment_tta`): half the memory and time. |

`synthsr.json` needs a model that accepts a batch of two scans (the scan and its flipped copy). The one in the
example data archive was made with `examples/synthsr_make_batch_dynamic.py` from the fixed-batch export of the network
(input `scan` float32, output `scan_out`); same weights, same result for one scan:

```bash
python examples/synthsr_make_batch_dynamic.py fixed_batch.onnx synthsr_v20_230130_batch.onnx
```

## Config file reference

The config is one JSON object. Every key is optional except `models`; the default applies when a key is absent.
With `--config`, the pipeline flags of the command line are ignored (only `--model` overrides `models`).
Without `--config`, the settings are built from the flags; keys marked "config only" have no flag.
Differences from the original `apply_multi_model_onnx.py` are listed in `CHANGES_FROM_ORIGINAL.md`.

### Model

| key | default | flag | meaning |
|---|---|---|---|
| `models` | – | `--model` | ONNX file, or list of files. `--model_prefix` is prepended to each name as a string. Several models: outputs are averaged, or voted on with `majority`. The model input must be named `scan`. |
| `n_classes` | 2 | `-n` | Number of output classes. Sets the channel count in patch mode and the length of the `flip_x` map. |
| `use_classes` | none | `-u` | Keep only the first N output channels (for models with extra channels). |
| `bck` | 0 | `--bck` | Background class: used where nothing is predicted and outside `mask`. |
| `continuous` | false | `--continuous` | Regression model: output `scan_out` is saved as float32, no argmax. |
| `dist` | false | `--distance` | Distance model: output `dist`; label = argmin (one channel: `< 1.0`). Otherwise the output is `seg` and the label is the argmax. |
| `channel_last` | false | `--channel_last` | Model takes and returns `(N, X, Y, Z, C)`. |
| `majority` | false | `--majority` | With several models: per-voxel majority vote of the labels instead of the argmax of the mean. Not applied together with `augment_tta`. |
| `tiled_groupnorm` | none | config only | Tile size (voxels) for `TiledGroupNormSession`, used with `whole`: whole-volume result of GroupNorm networks with bounded GPU memory. 128 needs about 9.5 GB, 96 about 8 GB, 64 about 4.5 GB. |

### Input geometry

| key | default | flag | meaning |
|---|---|---|---|
| `reference` | none | `-R` | Volume file: the input is resampled onto its grid (linear) before the model. `--model_prefix` is prepended. |
| `uniformize` | none | `-U` | Voxel size in mm: the input is resampled to isotropic voxels (linear) before the model. |
| `uniformize_method` | `"affine"` | config only | How `uniformize` resamples. `"affine"`: world-space linear resampling, zero outside the volume, nothing done when the voxels already have that size. `"grid"`: along the array axes, always applied: Gaussian blur of 0.25/factor voxels on the axes that are not upsampled (factor = voxel size / `uniformize`), then linear interpolation at voxel-edge aligned positions clamped to the volume, `ceil(size · factor)` samples per axis; computed in float64. Any other value is an error. |
| `input_dtype` | `"float32"` | config only | Precision the scans are loaded and resampled in: `"float32"` or `"float64"`. With `"float64"`, `normalize_min_max` in whole-volume mode is also computed in float64. The model is always fed float32. |
| `save_uniformized` | false | `-S` | Save the output on the `reference` / `uniformize` grid. Otherwise labels are resampled back to the input grid (nearest neighbour). |
| `resample` | `"legacy"` | config only | `"legacy"`: the only resampling is `uniformize` / `reference`. `"mindglide"`: resample to 1 mm on the voxel grid before the model (cubic spline through `scipy.ndimage.zoom`, truncated target shape; nearest along the slice axis when the spacing ratio is ≥ 3) and bring the labels back class by class. Any other value is an error. |
| `spacing_float32` | false | config only | With `resample: "mindglide"`: round the affine to float32 before the exact spacing = 1 test, so MINC and NIfTI inputs decide alike. |
| `reorient` | none | config only | Axis codes, e.g. `"RAS"`: reorient before the model and back afterwards. NIfTI: orientation from the affine, through nibabel. MINC is read in standard order (RAS), so no nibabel is needed and `"RAS"` is the identity. |
| `crop_foreground` | false | config only | Crop to the bounding box of voxels > 0; the result is put back, background outside. |
| `cropvol` | 0 | `--cropvol` | Remove N voxels at every border before the model; the result is put back with label 0 outside. |
| `padvol`, `padfill` | 0, 0.0 | `--padvol`, `--padfill` | Pad N voxels at every border with `padfill` before the model, removed afterwards. Ignored when `cropvol` > 0. |
| `nibabel` | false | `--nibabel` | Feed the model with axes in (x, y, z) order instead of the loaded (z, y, x) order. |
| `freesurfer` | false | `--freesurfer` | As `nibabel`, with the y axis flipped. Takes precedence over `nibabel`. |

### Intensity normalization

At most one of the first three is applied, in this order of precedence; `normalize_min_max` is used only when none of them is set.

| key | default | flag | meaning |
|---|---|---|---|
| `normalize` | false | `--normalize` | Subtract the minimum, divide by the 99th percentile, clip to [0, 1]. |
| `normalize_max` | false | `--max_normalize` | Divide by the maximum, clip to [0, 1]. |
| `normalize_mean_std` | false | `--mean_std_normalize` | Subtract the mean and divide by the standard deviation of the voxels > 0 (all voxels are transformed). |
| `normalize_min_max` | false | config only | Subtract the minimum, divide by the maximum of the result; no clipping. In whole-volume mode it is computed after the padding, so the zeros of the padding count for the minimum. |
| `normalize_mean_std_nonzero` | false | config only | Z-score of the voxels ≠ 0 only; zeros stay zero. Applied per channel, before the three above. |

### Whole-volume mode

| key | default | flag | meaning |
|---|---|---|---|
| `whole` | false | `--whole` | Run the model once on the whole volume instead of patches. |
| `quant_size` | 64 | `--quant` | The volume size is made a multiple of this. |
| `trim` | false | `--trim` | false: zero-pad at the end of each axis and cut the result back. true: cut a box out instead; outside it the result is class 0. The box is centred along the first two array axes and starts at 0 along the last one. |
| `trim_center` | false | config only | With `trim`: centre the box along the last array axis too. |
| `pad_center` | false | config only | Without `trim`: put the zero padding on both sides of each axis, `floor(d/2)` voxels before and the rest after, instead of at the end. With `augment_tta` the padded volume is flipped, so both passes see the same padding. |

### Sliding-window mode (when `whole` is false)

| key | default | flag | meaning |
|---|---|---|---|
| `patch_sz` | 64 | `--patch_sz` | Patch size: one number or three. Axes shorter than the patch are zero-padded and cut back. |
| `stride` | 32 | `--stride` | Step between windows: one number or three. The flag defaults to `patch_sz − 2·crop`. |
| `crop` | 0 | `--crop` | Discard N voxels at every edge of each predicted patch. The outer N voxels of the volume are then never predicted and become background. |
| `window_layout` | `"dense"` | config only | `"dense"`: fewest windows spaced by `stride`, the last one clamped to the end, no duplicates. `"legacy"`: layout of the original script, `ceil(size/stride)` windows per axis with those past the end clamped onto the last position, which is then counted more than once. The two coincide when no window is clamped twice. |
| `use_gaussian_weights` | false | `--use_gaussian_weights` | Weight each patch by a Gaussian centred on it when overlapping patches are averaged. |
| `sigma_scale` | 0.25 | config only | Gaussian sigma as a fraction of the used patch size. |
| `gaussian_map` | `"normalized"` | config only | `"normalized"`: the original map, maximum 1, floored at 0.001. `"separable"`: product of 1D float32 Gaussians, not normalised (maximum below 1 for even sizes), floored at 0.001. Same Gaussian up to scale; they differ in float rounding and in where the floor applies. |
| `sw_batch_size` | 1 | config only | Windows per ONNX call. Does not change the result. |

### Test-time augmentation

| key | default | flag | meaning |
|---|---|---|---|
| `augment_tta` | none | config only | `{"flip_x": [...]}`: also run the scan flipped along x and average the two softmax outputs. The list is a permutation of the `n_classes` class indices that maps each class to its mirror (left ↔ right). For `continuous` models the two outputs are averaged and the list is not used. Only `flip_x` is supported. |

### Output

| key | default | flag | meaning |
|---|---|---|---|
| `largest` | false | `--largest` | Keep only the largest connected component of the non-background voxels. |
| `largest_connectivity` | 3 | config only | Neighbourhood for `largest`: 3 = 26 neighbours, 1 = 6 neighbours. |
| `mask` | none | `--mask` | Volume file: voxels where it is < 1 are set to `bck`. Resampled (nearest neighbour) when its grid differs from the output. |
| `label_values` | none | config only | List: class index → label value written to the file. The output uses the smallest integer type that holds the values. |
| `labels_desc` | none | config only | Label names for `--measure`: a list (item *i* names class *i*+1, or `label_values[i+1]` when that key is set), an object `{"value": "name"}`, or the name of a JSON file holding such an object. Without it `--measure` writes nothing. |
| `fuzzy` | none | `-F` | Prefix for per-class probability maps: `<prefix>_<class>.<ext>`, or `<prefix>_<scan>_<class>.<ext>` for several scans, in the format of the output. |
| `output_scale` | none | config only | `continuous` models: multiply the output of each model (and of each TTA pass) by this number, before averaging. |
| `output_clip` | none | config only | `continuous` models: `[low, high]`, clip the output of each model and TTA pass after `output_scale`, before averaging. |
| `unsharp_sigma`, `unsharp_amount` | none, 1.0 | config only | `continuous` models: unsharp mask on the final volume, `v + amount · (v − blur(v))` with a Gaussian blur of `unsharp_sigma` voxels. Applied on the grid the model ran on, before any resampling back to the input grid. |
| `history` | command line | – | History string stored in the output; set by the command line. |

### Order of the steps

1. Load the scan (all channels on the same grid).
2. `reorient` → `crop_foreground` → `resample: "mindglide"` → `normalize_mean_std_nonzero`.
3. `reference` / `uniformize` resampling.
4. `pad_center` padding; flip copy for `augment_tta`; `cropvol` or `padvol`.
5. Model: `whole` (pad or `trim`) or sliding window; axis convention and `normalize*` are applied here.
6. `output_scale` / `output_clip`; average or vote over models; average the flipped pass; undo `pad_center`; argmax.
7. `unsharp_sigma`. Undo `cropvol` / `padvol`, then step 2, then step 3 (unless `save_uniformized`).
8. `largest` → `fuzzy` maps → `mask` → `label_values` → save → `--measure`.

## Modules

| module | content |
|---|---|
| `api` | Python interface: `segment`, `segment_batch`, `load_config` (exported by the package) |
| `inference` | Command line (`main`), `make_onnx_sessions`, `segment_whole`, `segment_with_patches_overlap`, `segment_with_onnx[_batched]`, geometry pre/post-processing (`preprocess_volume`, `postprocess_labels`, `postprocess_fuzzy`) |
| `onnx_tiled` | `TiledGroupNormSession`: drop-in for `InferenceSession.run()` that cuts the graph at every GroupNorm, runs the local stages tile by tile with a halo, and computes exact statistics from per-tile Σx, Σx² |
| `volume` | Normalizations, crop/pad, reorientation, foreground bbox, voxel-grid resample/recover (`_resize`, on `scipy.ndimage.zoom`), window starts and Gaussian map (numpy/scipy; nibabel imported lazily, only for NIfTI reorientation) |
| `postprocess` | `find_largest_component`, `measure_volumes`, `save_measurements` |
| `io` | `load_volume_np` / `save_volume`, which dispatch on `.mnc` / `.nii.gz` |
| `minc_io` | MINC2 I/O through `minc2_simple`, `resample_volume`, `uniformize_volume`, `uniformize_volume_grid` |
| `nifti_io` | NIfTI I/O through nibabel (optional; raises `ImportError` when it is missing) |
| `geo` | Affine `decompose` / `compose` |

Rule: the package is numpy/ONNX only; nothing in it imports `torch` or handles torch tensors.

## Tests

```bash
python -m pytest            # from the repository root; needs pytest
python -m pytest -m "not gpu"   # skip the CUDA tests
python -m pytest -m "not examples"   # skip the example pipelines on the real scan (slow on a CPU)
```

The tests need no data files or PyTorch (the `examples` ones run only when the example data archive is unpacked,
see `examples/DATA.md`): they build small synthetic ONNX models on the fly. One is a pointwise
1×1×1 conv whose labels are a known function of intensity; the other is a 2-level GroupNorm U-Net.

| file | covers |
|---|---|
| `test_volume.py` | normalisation, crop/pad, bbox, reorient (NIfTI with nibabel; MINC without it, checked against nibabel), voxel-grid resample/recovery, window layout and Gaussian weights |
| `test_resize.py` | `_resize` (scipy port of skimage resize) |
| `test_io.py` | MINC/NIfTI round-trips and affines, metadata/history, missing nibabel, world-space resampling |
| `test_postprocess.py` | largest component, volume measurements, CSV |
| `test_onnx_tiled.py` | `TiledGroupNormSession` against plain ORT (`gpu`: on CUDA) |
| `test_inference.py` | sliding window (`legacy` layout against a copy of the original loop, `dense` default, `gaussian_map`), whole volume, geometry pre/post-processing, `resample` modes |
| `test_pipeline.py` | `segment_with_onnx[_batched]` on files: minibatches, measure, recover, fuzzy, label_values, flip TTA, majority, tiled config, geometry pipeline on MINC without nibabel, missing input, CLI |
| `test_continuous.py` | regression models on files: `pad_center` (with flip TTA and `trim`), `normalize_min_max`, `input_dtype`, `uniformize_method: "grid"`, `output_scale` / `output_clip`, unsharp mask |
| `test_api.py` | `segment` / `segment_batch` with paths and a config dict: `model_prefix`, measure, errors, config left unmodified |
| `test_examples.py` | `examples/*.json`: valid JSON, bare model names, every key documented in this file. Marker `examples`: every example config on the test scan, output and `--measure` label volumes compared with the references; skipped without the example data archive |
| `test_package.py` | the package never imports torch; the MINC writer takes numpy arrays only |

Markers:
- `gpu` tests are skipped without `CUDAExecutionProvider`.

One test is a strict `xfail` documenting a known issue: the precision of tiled GroupNorm depends on the tile size.
