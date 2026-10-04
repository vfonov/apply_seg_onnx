# Changes from the original code

Original = `py_deep_seg` at the commit before `13102a4`: `apply_multi_model_onnx.py`, `seg_common/{io,volume,postprocess}.py`,
`minc/{io,geo}.py`, `nifti/io.py`.

One numerical default differs from the original: the sliding-window layout (`window_layout`, section 1, row 0).
With `"window_layout": "legacy"` the results are those of the original script. Checked on real scans:
`conjurer_patch_mk2_legacy.json` and a stride-[100,100,48] config with that key give 0 voxels different from the
outputs of the original code (T1/T2/PD); without the key the stride-100 config differs by 299 voxels on the T1.

## 1. Behaviour of existing options that differs

| # | Option | Original | Now |
|---|---|---|---|
| 0 | sliding-window positions | `ceil(size/stride)` windows per axis, those past the end clamped onto the last position and counted 2–4 times | no duplicate windows (`window_layout: "dense"`, default); `"legacy"` gives the original |
| 1 | `largest` / `--largest` | read, never applied | keeps the largest connected component (26-neighbourhood; `largest_connectivity`) |
| 2 | `augment_tta` in single-scan mode | ignored (only the batch function had it) | applied |
| 3 | `augment_tta` with `--minibatch_size` > 1 | flipped copies appended inside the scan loop: wrong batches | fixed |
| 4 | `majority` in batch mode | vote overwritten by the argmax of the mean | vote is used (not combined with TTA) |
| 5 | `-F/--fuzzy` | command-line value never used; only `settings['fuzzy']`, single mode, always `.mnc` | works from the command line and in batch mode (`<prefix>_<scan>_<class>`), in the output's format |
| 6 | `cropvol` | labels not un-cropped (batch: result discarded; single: error with fuzzy output) | un-cropped |
| 7 | `--measure` in batch mode | counted the whole minibatch, named after the last scan | one row per scan |
| 8 | `whole` + `quant_size` | `quant_size` not passed to `segment_whole` (always 64) | passed |
| 9 | `mask` | single-scan mode only | both modes |
| 10 | `use_classes` with patches | channel-count mismatch error unless equal to `n_classes` | first `use_classes` channels |
| 11 | history in single-scan mode | taken from `settings['history']`: none when `--config` is used | command line recorded |
| 12 | axis shorter than the patch | error | zero-padded, cropped back |
| 13 | scans of different shape in a minibatch | assertion | run one by one |
| 14 | errors in single-scan mode | exception | traceback printed, then the same exception |

Not changed: the command line (`parse_options`), `segment_whole` (apart from `trim_center`), the Gaussian
weights (`gaussian_map: "normalized"`, default), softmax/argmax results. Still broken as in the
original: `--channels` > 1 uses `params.fill`, which does not exist.

## 2. New config keys (all off by default)

| Key | Effect |
|---|---|
| `window_layout` | `"dense"` (default) = no duplicate windows; `"legacy"` = original layout |
| `gaussian_map` | `"normalized"` (default) = original map; `"separable"` = product of 1D float32 Gaussians, not normalised |
| `sigma_scale` | Gaussian sigma as a fraction of the patch; default 0.25 = the value hard-coded originally |
| `sw_batch_size` | windows per model call (default 1) |
| `resample` | `"legacy"` (default) = only `uniformize` / `reference`; `"mindglide"` = voxel-grid resampling to 1 mm, labels recovered per class |
| `spacing_float32` | with `resample: "mindglide"`: float32 affine for the spacing test |
| `reorient` | reorient to axis codes before the model and back afterwards |
| `crop_foreground` | crop to the bounding box of voxels > 0 |
| `normalize_mean_std_nonzero` | z-score of the nonzero voxels |
| `largest_connectivity` | neighbourhood for `largest` (default 3 = 26, as the original helper) |
| `trim_center` | with `whole` + `trim`: centre the box along Z too |
| `tiled_groupnorm` | tile size for exact tiled whole-volume inference of GroupNorm networks |
| `label_values` | class index → saved label value (`labels_desc` list then refers to `label_values[1:]`) |

## 3. Code structure

- `segment_with_onnx` is `segment_with_onnx_batched` with one scan (was a second implementation); a missing input
  still raises.
- `segment_with_onnx_batched`: scans may be lists of channels (file names or constants); argument `fuzzy_output`
  (unused flag) replaced by `fuzzy` (prefix).
- New functions in `inference.py`: `legacy_window_starts`, `resample_mode`, `preprocess_volume`, `postprocess_labels`,
  `postprocess_fuzzy`, `keep_largest`, `make_onnx_sessions`, `load_scan`, `main`.
  `get_gaussian_weights` is the original.
- New module `onnx_tiled.py` (`TiledGroupNormSession`).
- `volume.py` additions: `smallest_int_dtype`, `reorient_to` / `reorient_back`, `affine_spacing`, `foreground_bbox`,
  `grid_resample_shape` / `grid_resample_image` / `grid_recover_labels` / `grid_recover_prob` (`_resize`, on
  `scipy.ndimage.zoom`), `nonzero_mean_std_normalize`, `window_starts`, `window_starts_overlap`, `pad_to_size`,
  `separable_gaussian_weights`. The original functions are unchanged.

## 4. Packaging and I/O

- Standalone package: `pyproject.toml`, conda recipe, `python -m apply_seg_onnx` / `apply_seg_onnx`, pytest suite.
  `minc.io`, `minc.geo`, `nifti.io` copied in as `minc_io.py`, `geo.py`, `nifti_io.py`.
- numpy/ONNX only. Removed with the copy: `minc_io.load_minc_volume` (torch tensor), `load_nl_xfm`, `load_lin_xfm`,
  the torch-tensor branch of `save_minc_volume` (non-array input is now a `TypeError`), and from `geo.py`
  `create_v2p_matrix` and the augmentation matrix builders (`create_rotation_matrix`, `create_scale_matrix`,
  `create_translation_matrix`, `create_shear_matrix`, `create_transform`). `geo.py` keeps `decompose` / `compose`.
- `minc_io`: affine returned as `ndarray` (was `np.matrix`); `dtype='native'`; the writer also takes bool, float16,
  (u)int32 and 64-bit integers (was an assertion).
- `nifti_io`: nibabel is optional (clear `ImportError`), `as_byte` honoured, `dtype='native'`.
- `postprocess`: `find_largest_component(input, connectivity=3)` (same default result, faster count);
  `measure_volumes` reads an existing output in its own integer type (was int16).
