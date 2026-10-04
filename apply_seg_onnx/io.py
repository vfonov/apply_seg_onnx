#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I/O utilities for 3D MRI brain segmentation inference.
Dispatches on the file extension: MINC (.mnc, .minc, also gzipped) through minc2_simple,
everything else (NIfTI, Analyze, MGH, ...) through nibabel.
"""
import gzip
import os
import shutil
import tempfile

from .minc_io import load_minc_volume_np, save_minc_volume, format_history
from .nifti_io import load_nifti_volume_np, save_nifti_volume

MINC_EXTENSIONS = ('.mnc', '.minc', '.mnc.gz', '.minc.gz')


def is_minc(fname):
    """True for .mnc, .minc, .mnc.gz, .minc.gz (any case); every other file goes through nibabel."""
    return isinstance(fname, str) and fname.lower().endswith(MINC_EXTENSIONS)


def volume_extension(fname):
    """Extension of a volume file including a trailing .gz: 'a/b.nii.gz' -> '.nii.gz', 'x.img' -> '.img'."""
    base = os.path.basename(fname)
    stem, ext = os.path.splitext(base)
    if ext.lower() == '.gz':
        ext = os.path.splitext(stem)[1] + ext
    return ext


def load_volume_np(fname, dtype=None, as_byte=False):
    """
    Load a volume: MINC (see is_minc) through minc2_simple, any other file through nibabel.
    
    Args:
        fname: Path to volume file (.mnc, .mnc.gz, .nii, .nii.gz, .img/.hdr, .mgz, ...)
        dtype: Data type for numpy array: dtype name, None (float64) or 'native' (the file's own type)
        as_byte: Load as uint8 (for labels/masks)
    
    Returns:
        tuple: (volume_array, affine_matrix)
    """
    if is_minc(fname):
        return load_minc_volume_np(fname, as_byte=as_byte, dtype=dtype)
    return load_nifti_volume_np(fname, as_byte=as_byte, dtype=dtype)


def save_volume(fname, data, aff, ref_fname=None, history=None):
    """
    Save a volume: MINC (see is_minc) through minc2_simple, any other file through nibabel,
    in the format of its extension.
    
    Args:
        fname: Output path
        data: numpy array with volume data
        aff: 4x4 affine matrix
        ref_fname: Optional reference file for metadata (used for MINC output when it is a MINC file)
        history: Optional history string to embed
    """
    if not is_minc(fname):
        save_nifti_volume(fname, data, aff, ref_fname=ref_fname, history=history)
        return
    if not is_minc(ref_fname):
        ref_fname = None  # MINC metadata can only be copied from a MINC file
    if not fname.lower().endswith('.gz'):
        save_minc_volume(fname, data, aff, ref_fname=ref_fname, history=history)
        return
    # the MINC library does not compress by file name: write next to the target, then gzip
    fd, tmp = tempfile.mkstemp(suffix='.mnc', dir=os.path.dirname(os.path.abspath(fname)))
    os.close(fd)
    try:
        os.remove(tmp)  # the writer creates the file itself
        save_minc_volume(tmp, data, aff, ref_fname=ref_fname, history=history)
        with open(tmp, 'rb') as src, gzip.open(fname, 'wb') as dst:
            shutil.copyfileobj(src, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
