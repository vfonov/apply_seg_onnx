"""ONNX Runtime inference of 3D MRI segmentation models (MINC / NIfTI), without PyTorch.

Command line: ``apply_seg_onnx`` (or ``python -m apply_seg_onnx``), see ``apply_seg_onnx --help``.

Python: ``segment(input, output, config)`` / ``segment_batch(inputs, outputs, config)`` with file paths and the
configuration as a dict (module ``api``); ``load_config`` reads a JSON config file.

Modules:
    api          -- segment, segment_batch, load_config
    inference    -- command line, sessions, whole-volume / sliding-window segmentation pipelines
    onnx_tiled   -- TiledGroupNormSession: exact whole-volume inference of GroupNorm networks in tiles
    volume       -- intensity normalization, crop/pad, reorientation, voxel-grid resampling and sliding-window helpers
    postprocess  -- largest connected component, label volume measurements
    io           -- load_volume_np / save_volume: MINC extensions through minc_io, every other file through nibabel
    minc_io      -- MINC2 I/O (minc2_simple), world-space resampling
    nifti_io     -- NIfTI, Analyze and other nibabel formats (optional nibabel)
    geo          -- affine decomposition / composition
"""
__version__ = "0.1.0"

from .api import load_config, segment, segment_batch  # noqa: E402

__all__ = ["segment", "segment_batch", "load_config", "__version__"]
