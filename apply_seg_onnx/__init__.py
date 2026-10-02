"""ONNX Runtime inference of 3D MRI segmentation models (MINC / NIfTI), without PyTorch.

Command line: ``apply_seg_onnx`` (or ``python -m apply_seg_onnx``), see ``apply_seg_onnx --help``.

Modules:
    inference    -- command line, sessions, whole-volume / sliding-window segmentation pipelines
    onnx_tiled   -- TiledGroupNormSession: exact whole-volume inference of GroupNorm networks in tiles
    volume       -- intensity normalization, crop/pad, MindGlide/MONAI-compatible pre/post-processing helpers
    postprocess  -- largest connected component, label volume measurements
    io           -- load_volume_np / save_volume dispatching on file extension (.mnc, .nii.gz)
    minc_io      -- MINC2 I/O (minc2_simple), world-space resampling
    nifti_io     -- NIfTI I/O (optional nibabel)
    geo          -- affine decomposition / composition
"""
__version__ = "0.1.0"
