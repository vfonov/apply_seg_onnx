#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Volume preparation utilities for 3D MRI brain segmentation inference.
All functions use NumPy only (no PyTorch dependency).
"""

import warnings

import numpy as np

_INT_DTYPES_UNSIGNED = (np.uint8, np.uint16, np.uint32)
_INT_DTYPES_SIGNED = (np.int8, np.int16, np.int32)


def smallest_int_dtype(values):
    """
    Smallest integer dtype (at most 32 bits, as supported by MINC) that holds all values:
    uint8/uint16/uint32 for non-negative values, int8/int16/int32 otherwise.
    Falls back to float64 with a warning when 32 bits are not enough.
    """
    values = np.asarray(values)
    lo, hi = (int(values.min()), int(values.max())) if values.size else (0, 0)
    for dt in (_INT_DTYPES_UNSIGNED if lo >= 0 else _INT_DTYPES_SIGNED):
        if np.iinfo(dt).min <= lo and hi <= np.iinfo(dt).max:
            return np.dtype(dt)
    warnings.warn(f"integer values in [{lo}, {hi}] do not fit in 32 bits, using float64")
    return np.dtype(np.float64)


def autonorm_np(arr):
    """
    Normalize array using quantile-based scaling (0-1 range).
    
    Args:
        arr: Input numpy array
    
    Returns:
        Normalized array with values in [0, 1]
    """
    arr = arr - np.min(arr)
    p99 = np.percentile(arr, 99)
    if p99 > 0:
        arr = np.clip(arr / p99, 0.0, 1.0)
    return arr


def maxnorm_np(arr):
    """
    Normalize array using maximum value (0-1 range).
    
    Args:
        arr: Input numpy array
    
    Returns:
        Normalized array with values in [0, 1]
    """
    max_val = np.max(arr)
    if max_val > 0:
        arr = arr / max_val
    return arr


def mean_std_normalize_np(arr):
    """
    Normalize array using mean and standard deviation.
    
    Args:
        arr: Input numpy array
    
    Returns:
        Normalized array (subtract mean, divide by std)
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        arr = (arr - mean) / std
    return arr


def apply_cropvol(dset, cropvol):
    """
    Crop volume by removing border voxels.
    
    Args:
        dset: Input numpy array (B, C, X, Y, Z)
        cropvol: Number of voxels to crop from each border
    
    Returns:
        tuple: (cropped_dset, orig_size) where orig_size is the original shape
    """
    orig_size = dset.shape
    cropped = dset[:, :, cropvol: orig_size[2]-cropvol,
                        cropvol: orig_size[3]-cropvol,
                        cropvol: orig_size[4]-cropvol]
    return cropped, orig_size


def apply_padvol(dset, padvol, padfill=0.0):
    """
    Pad volume with constant border.
    
    Args:
        dset: Input numpy array (B, C, X, Y, Z)
        padvol: Number of voxels to pad on each border
        padfill: Value to fill padding with
    
    Returns:
        tuple: (padded_dset, orig_size) where orig_size is the original shape
    """
    orig_size = dset.shape
    padded = np.pad(dset, pad_width=((0,0), (0,0), 
                                    (padvol, padvol), 
                                    (padvol, padvol), 
                                    (padvol, padvol)), 
                    mode='constant', constant_values=padfill)
    return padded, orig_size


def undo_cropvol(dset_out, orig_size, cropvol, bck=0):
    """
    Restore cropped volume to original size (with background fill).
    
    Args:
        dset_out: Cropped output array
        orig_size: Original size before cropping
        cropvol: Crop amount used
        bck: Background value to fill
    
    Returns:
        Restored array at original size
    """
    restored = np.full(orig_size, bck, dtype=dset_out.dtype)
    restored[:, :, cropvol: orig_size[2]-cropvol,
                   cropvol: orig_size[3]-cropvol,
                   cropvol: orig_size[4]-cropvol] = dset_out
    return restored


def undo_padvol(dset_out, orig_size, padvol):
    """
    Remove padding from volume.
    
    Args:
        dset_out: Padded output array
        orig_size: Original size before padding
        padvol: Padding amount used
    
    Returns:
        Array with padding removed
    """
    return dset_out[:, :, padvol: orig_size[2]+padvol, 
                      padvol: orig_size[3]+padvol, 
                      padvol: orig_size[4]+padvol]


def parse_bracket_input(spec):
    """
    Parse bracket-format input spec [a,b,...] into list of filenames/floats.
    
    Args:
        spec: String like "[a,b,c]" where a,b,c are either filenames or constant numbers
    
    Returns:
        tuple: (inputs_list, ref_file) where inputs_list contains filenames/floats
    """
    import re
    m = re.match(r"\[(.*)\]", spec)
    if m is None:
        return None, None
    
    inp = m[1].split(",")
    inputs = []
    ref_file = None
    
    for i in inp:
        q = re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$", i)
        if q is not None:
            inputs.append(float(q[0]))
        else:
            inputs.append(i)
            if ref_file is None:
                ref_file = i
    
    return inputs, ref_file


# ---------------------------------------------------------------------------
# Optional geometry preprocessing: reorientation, foreground crop, voxel-grid resampling
# Arrays here are in voxel index order (i,j,k) matching the affine, i.e. the
# reverse of the (k,j,i) order returned by load_volume_np.
# ---------------------------------------------------------------------------

def _import_nibabel():
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError("reorienting NIfTI volumes needs nibabel: pip install nibabel (or apply_seg_onnx[nifti])") from e
    return nib


def reorient_to(arr, aff, axcodes='RAS', minc=False):
    """
    Reorient a voxel-ordered (i,j,k) array to the given axis codes.

    minc: the array comes from a MINC file read in standard order (positive steps, i,j,k = x,y,z),
          i.e. it is RAS: the transform follows from `axcodes` alone, without nibabel.
          Otherwise (NIfTI) the orientation is derived from the affine with nibabel.

    Returns:
        tuple: (reoriented_array, new_affine, transform) where transform undoes
               the operation via reorient_back()
    """
    aff = np.asarray(aff, dtype=np.float64)
    if minc:
        tr = _ras_to_axcodes(axcodes)
        # voxel index in the new array -> voxel index in the old one
        new_to_old = np.zeros((4, 4))
        new_to_old[3, 3] = 1.0
        for i, (j, flip) in enumerate(tr):
            new_to_old[i, int(j)] = flip
            new_to_old[i, 3] = arr.shape[i] - 1 if flip < 0 else 0
        return np.ascontiguousarray(_apply_ornt(arr, tr)), aff @ new_to_old, tr
    nib = _import_nibabel()
    ornt = nib.orientations.io_orientation(aff)
    tr = nib.orientations.ornt_transform(ornt, nib.orientations.axcodes2ornt(axcodes))
    new_arr = nib.orientations.apply_orientation(arr, tr)
    new_aff = aff @ nib.orientations.inv_ornt_aff(tr, arr.shape)
    return np.ascontiguousarray(new_arr), new_aff, tr


def reorient_back(arr, tr, minc=False):
    """Undo reorient_to() using the transform it returned (same `minc` as there)."""
    if minc:
        return np.ascontiguousarray(_apply_ornt(arr, _inverse_ornt(tr)))
    nib = _import_nibabel()
    return np.ascontiguousarray(nib.orientations.apply_orientation(arr, _inverse_ornt(tr)))


def _ras_to_axcodes(axcodes):
    """Transform (nibabel format: row i = [new position of axis i, +1/-1 flip]) from RAS to e.g. 'LPS'."""
    axes = [next((i for i, pair in enumerate(('LR', 'PA', 'IS')) if code in pair), None) for code in axcodes]
    if sorted(axes, key=str) != [0, 1, 2]:
        raise ValueError(f"axis codes {axcodes!r} must name each of L/R, P/A, I/S once")
    tr = np.zeros((3, 2))
    for j, (i, code) in enumerate(zip(axes, axcodes)):
        tr[i] = [j, 1.0 if code in 'RAS' else -1.0]
    return tr


def _apply_ornt(arr, tr):
    """Flip axes with -1, then move axis i to position tr[i, 0] (nibabel apply_orientation)."""
    for ax, flip in enumerate(tr[:, 1]):
        if flip < 0:
            arr = np.flip(arr, axis=ax)
    return arr.transpose(np.argsort(tr[:, 0]))


def _inverse_ornt(tr):
    """Inverse of a nibabel orientation transform."""
    inv = np.zeros_like(tr)
    for src, (dst, flip) in enumerate(tr):
        inv[int(dst)] = [src, flip]
    return inv


def affine_spacing(aff):
    """Voxel spacing: column norms of the affine."""
    aff = np.asarray(aff, dtype=np.float64)
    return np.sqrt(np.sum(aff[:3, :3] ** 2, axis=0))


def foreground_bbox(arr):
    """
    Bounding box of voxels > 0.

    Returns:
        tuple: (start, end) lists, end exclusive. Whole volume if nothing is positive.
    """
    nz = np.nonzero(arr > 0)
    if len(nz[0]) == 0:
        return [0] * arr.ndim, list(arr.shape)
    return [int(i.min()) for i in nz], [int(i.max()) + 1 for i in nz]


def grid_resample_shape(spacing, shape, target_spacing=(1.0, 1.0, 1.0)):
    """
    Voxel-grid resampling decision and target shape: no resampling when the spacing equals the target
    exactly (float comparison); target shape truncated; anisotropic when the spacing ratio is >= 3.

    Returns:
        tuple: (resample_flag, new_shape, anisotropy_flag)
    """
    spacing = [float(s) for s in spacing]
    target_spacing = [float(s) for s in target_spacing]
    if spacing == target_spacing:
        return False, list(shape), False
    new_shape = (np.array(spacing) / np.array(target_spacing) * np.array(shape)).astype(int).tolist()
    anis = (np.max(spacing) / np.min(spacing) >= 3) or (np.max(target_spacing) / np.min(target_spacing) >= 3)
    return True, new_shape, bool(anis)


def _resize(img, shape, order, mode='edge', cval=0.0):
    """
    Resize `img` to `shape` (same number of dimensions) on the voxel grid: spline interpolation
    of `order` with grid_mode=True (voxel corners aligned), `mode` 'edge' (replicate) or 'constant' (cval),
    no anti-aliasing, output clipped to the input value range (widened to cval when cval is used).
    """
    import scipy.ndimage as ndi
    img = np.asarray(img)
    if img.dtype == np.float16:
        img = img.astype(np.float32)
    if order > 0 and img.dtype.char not in 'fd':
        img = img.astype(np.float64)
    ndi_mode = {'edge': 'nearest', 'constant': 'grid-constant'}[mode]
    factors = np.divide(img.shape, shape)
    out = ndi.zoom(img, [1 / f for f in factors], order=order, mode=ndi_mode, cval=cval, grid_mode=True)
    lo, hi = np.min(img), np.max(img)
    if mode == 'constant' and not lo <= cval <= hi and np.min(out) <= cval <= np.max(out):
        cval = img.dtype.type(cval)  # cval used by the interpolation widens the clip range
        lo, hi = min(lo, cval), max(hi, cval)
    np.clip(out, np.asarray(lo), np.asarray(hi), out=out)
    return out


def grid_resample_image(img, shape, anisotropy_flag):
    """Resample one 3D channel to `shape`: cubic spline; when anisotropic, cubic in-plane and nearest
    along the last axis."""
    if anisotropy_flag:
        slices = [_resize(img[:, :, i], shape[:-1], order=3, mode='edge', cval=0) for i in range(img.shape[-1])]
        out = np.stack(slices, axis=-1)
        return _resize(out, shape, order=0, mode='constant', cval=0)
    return _resize(img, shape, order=3, mode='edge', cval=0)


def grid_recover_labels(labels, n_classes, shape, anisotropy_flag):
    """
    Inverse of grid_resample_image for labels: bring a label map
    back to `shape` by resizing every class mask (1..n_classes-1) and thresholding at 0.5.
    Ties resolve to the lowest label; voxels claimed by no class become 0.
    """
    out = np.zeros(shape, dtype=np.uint8)
    claimed = np.zeros(shape, dtype=bool)
    for c in range(1, n_classes):
        mask = (labels == c)
        if anisotropy_flag:
            h, w = mask.shape[:2]
            d = shape[-1]
            m_d = _resize(mask.astype(float), (h, w, d), order=0, mode='constant', cval=0) >= 0.5
            m = np.zeros(shape, dtype=bool)
            for k in range(d):
                m[:, :, k] = _resize(m_d[:, :, k].astype(float), shape[:-1], order=1, mode='edge',
                                    cval=0) >= 0.5
        else:
            m = _resize(mask.astype(float), shape, order=1, mode='edge', cval=0) >= 0.5
        # argmax over a one-hot volume picks the lowest class that is set
        new = m & ~claimed
        out[new] = c
        claimed |= m
    return out


def grid_recover_prob(prob, shape, anisotropy_flag):
    """
    Bring a probability map (one channel) back to `shape` with the same interpolation
    grid_recover_labels uses for the class masks: linear, or nearest along the last axis then linear in-plane.
    """
    if anisotropy_flag:
        h, w = prob.shape[:2]
        p_d = _resize(prob, (h, w, shape[-1]), order=0, mode='constant', cval=0)
        return np.stack([_resize(p_d[:, :, k], shape[:-1], order=1, mode='edge', cval=0)
                         for k in range(shape[-1])], axis=-1).astype(np.float32)
    return _resize(prob, shape, order=1, mode='edge', cval=0).astype(np.float32)


def nonzero_mean_std_normalize(arr):
    """Z-score of the voxels != 0, zeros untouched (float32)."""
    arr = arr.astype(np.float32, copy=True)
    nz = arr != 0
    if not nz.any():
        return arr
    vals = arr[nz]
    mean, std = vals.mean(dtype=np.float64), vals.std(dtype=np.float64)
    if std == 0.0:
        std = 1.0
    arr[nz] = ((vals - np.float32(mean)) / np.float32(std))
    return arr


def window_starts(image_size, roi_size, interval):
    """
    Window start positions per axis for a given step:
    the fewest windows (spaced by `interval`) that cover the axis, the last one clamped to the end.
    No duplicate windows. Requires image_size >= roi_size (pad first).
    """
    starts = []
    for n, r, step in zip(image_size, roi_size, interval):
        step = r if r == n else max(int(step), 1)
        if r == n:
            num = 1
        else:
            num = int(np.ceil(float(n) / step))
            num = next((d for d in range(num) if d * step + r >= n), None)
            num = num + 1 if num is not None else 1
        s = []
        for idx in range(num):
            st = idx * step
            st -= max(st + r - n, 0)
            s.append(st)
        starts.append(s)
    return starts


def window_starts_overlap(image_size, roi_size, overlap):
    """window_starts() with the step given as an overlap fraction of the window: int(roi * (1 - overlap))."""
    return window_starts(image_size, roi_size, [int(r * (1 - o)) for r, o in zip(roi_size, overlap)])


def pad_to_size(arr, min_size, axes, value=0):
    """
    Constant-pad `axes` of `arr` up to `min_size` (diff//2 before, rest after).
    Returns (padded array, list of (before, after) per axis in `axes`).
    """
    pad = [(0, 0)] * arr.ndim
    pads = []
    for ax, r in zip(axes, min_size):
        diff = max(r - arr.shape[ax], 0)
        pad[ax] = (diff // 2, diff - diff // 2)
        pads.append(pad[ax])
    if any(p for pp in pads for p in pp):
        arr = np.pad(arr, pad, mode='constant', constant_values=value)
    return arr, pads


def separable_gaussian_weights(roi_size, sigma_scale=0.125):
    """Gaussian window weights as a product of 1D float32 Gaussians (sigma = n * sigma_scale, centred),
    not normalised, floored at max(min, 1e-3)."""
    m = None
    for i, n in enumerate(roi_size):
        x = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1, dtype=np.float32)
        x = np.exp(x ** 2 / (-2 * (n * sigma_scale) ** 2)).astype(np.float32)
        m = x if m is None else m[..., None] * x[(None,) * i]
    return np.maximum(m, max(float(m.min()), 1e-3)).astype(np.float32)
