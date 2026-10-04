#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Volume I/O (numpy) through nibabel, an optional dependency (pip install apply_seg_onnx[nifti]):
NIfTI and every other format nibabel reads (Analyze .img/.hdr, MGH .mgz, ...), chosen by the file extension."""
import numpy as np

try:
    import nibabel as nib
    have_nibabel=True
except ImportError:
    # nibabel not available :(
    have_nibabel=False


def _require_nibabel():
    if not have_nibabel:
        raise ImportError("Files other than MINC (NIfTI, Analyze, ...) need nibabel: pip install nibabel "
                          "(or apply_seg_onnx[nifti])")



""" 
    Load nifti volume into numpy volume and return voxel2world matrix too
"""
def load_nifti_volume_np(fname, as_byte=False, dtype=None):

    _require_nibabel()

    """dtype: numpy dtype name, None (float64) or 'native' (the file's own type, float if scaled)"""
    try:
        x = nib.load(fname)
    except nib.filebasedimages.ImageFileError as e:
        raise ValueError(f"Unsupported file format: {fname} ({e})") from e
    aff = x.affine

    if as_byte:
        dtype='uint8'
    elif dtype is None:
        dtype='float64'

    if dtype == 'native':
        return np.asanyarray(x.dataobj).squeeze().transpose([2,1,0]).copy(), aff
    volume = x.get_fdata().squeeze().transpose([2,1,0])
    return volume.astype(dtype), aff


def save_nifti_volume(fn, data, aff, ref_fname=None, history=None):
    _require_nibabel()

    #if ref_fname is not None:
    #    x = nib.load(ref_fname)
    #    header = x.header
    #else:
    #    header = None
    header = None

    out = nib.Nifti1Image(data.transpose([2,1,0]).copy(), aff, header=header)
    if history is not None:
        out.header['descrip'] = history

    # nibabel picks the format from the extension (.nii, .nii.gz, .img/.hdr as a NIfTI pair, .mgz, ...)
    try:
        nib.save(out, fn)
    except nib.filebasedimages.ImageFileError as e:
        raise ValueError(f"Unsupported file format: {fn} ({e})") from e

