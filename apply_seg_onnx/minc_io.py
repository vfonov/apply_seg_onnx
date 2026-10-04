#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""MINC2 volume I/O (numpy) through minc2_simple."""

from minc2_simple import minc2_file
from minc2_simple import minc2_dim
from minc2_simple import minc2_error

from time import gmtime, strftime

import numpy as np

from .geo import decompose,compose
from .volume import smallest_int_dtype

# numpy dtype -> (MINC storage type, MINC representation type); integers are stored unscaled,
# floating point data as scaled 16-bit integers
_MINC_TYPES = {
    np.dtype(np.uint8):   (minc2_file.MINC2_UBYTE,  minc2_file.MINC2_UBYTE),
    np.dtype(np.int8):    (minc2_file.MINC2_BYTE,   minc2_file.MINC2_BYTE),
    np.dtype(np.uint16):  (minc2_file.MINC2_USHORT, minc2_file.MINC2_USHORT),
    np.dtype(np.int16):   (minc2_file.MINC2_SHORT,  minc2_file.MINC2_SHORT),
    np.dtype(np.uint32):  (minc2_file.MINC2_UINT,   minc2_file.MINC2_UINT),
    np.dtype(np.int32):   (minc2_file.MINC2_INT,    minc2_file.MINC2_INT),
    np.dtype(np.float32): (minc2_file.MINC2_SHORT,  minc2_file.MINC2_FLOAT),
    np.dtype(np.float64): (minc2_file.MINC2_SHORT,  minc2_file.MINC2_DOUBLE),
}

""" 
    Create minc-style history entry
"""
def format_history(argv):
    stamp=strftime("%a %b %d %T %Y>>>", gmtime())
    return stamp+(' '.join(argv))

""" 
    Convert minc file header int voxel to world affine matrix
"""
def hdr_to_affine(hdr):
    rot=np.zeros((3,3))
    scales=np.zeros((3,3))
    start=np.zeros(3)

    ax = np.array([h.id for h in hdr])

    for i in range(3):
        aa=np.where(ax == (i+1))[0][0] # HACK, assumes DIM_X=1,DIM_Y=2 etc
        if hdr[aa].have_dir_cos:
            rot[:,i] = hdr[aa].dir_cos
        else:
            rot[i,i] = 1

        scales[i,i] = hdr[aa].step
        start[i] = hdr[aa].start
    
    origin = rot@start
    out = np.eye(4)

    out[0:3,0:3] = rot@scales
    out[0:3,3]   = origin
    return out


"""
    Convert affine matrix into minc file dimension description
"""
def affine_to_dims(aff, shape):
    # convert to minc2 sampling format
    start, step, dir_cos = decompose(aff)
    if len(shape) == 3: # this is a 3D volume
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=np.ascontiguousarray(dir_cos[0:3,i])) for i in range(3)
            ]
    elif len(shape) == 4: # this is a 3D grid volume, vector space is the last one
        dims=[
                minc2_dim(id=i+1, length=shape[2-i], start=start[i], step=step[i], have_dir_cos=True, dir_cos=np.ascontiguousarray(dir_cos[0:3,i])) for i in range(3)
             ] + [ 
                minc2_dim(id=minc2_file.MINC2_DIM_VEC, length=shape[3], start=0, step=1, have_dir_cos=False, dir_cos=[0,0,0])
             ]
    else:
        assert False, f"Unsupported number of dimensions: {len(shape)}"
    return dims


""" 
    Load minc volume into numpy volume and return voxel2world matrix too
"""
def load_minc_volume_np(fname, as_byte=False, dtype=None):
    """dtype: numpy dtype name, None (float64) or 'native' (the file's own type, e.g. uint16 for label files)"""
    mm=minc2_file(fname)
    mm.setup_standard_order()

    if as_byte:
        dtype='uint8'
    elif dtype is None:
        dtype='float64'
    elif dtype == 'native':
        dtype=mm.representation_dtype()

    d = mm.load_complete_volume(dtype)
    aff=hdr_to_affine(mm.representation_dims())

    mm.close()
    return d, aff


"""
    Save numpy volume into minc file
"""
def save_minc_volume(fn, data, aff, ref_fname=None, history=None):
    """
    Integer data (8, 16, 32 bit, signed or unsigned) is stored as is, floating point data as scaled short.
    bool -> uint8, float16 -> float32, 64-bit integers -> smallest integer type that holds the values
    (float64 stored as double, with a warning, if 32 bits are not enough).
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(f"save_minc_volume needs a numpy array, got {type(data).__name__}")
    dims=affine_to_dims(aff, data.shape)
    out=minc2_file()
    store_type = None
    if data.dtype == np.bool_:
        data = data.astype(np.uint8)
    elif data.dtype == np.float16:
        data = data.astype(np.float32)
    elif data.dtype in (np.int64, np.uint64):
        dt = smallest_int_dtype(data)
        if dt == np.float64:
            store_type = minc2_file.MINC2_DOUBLE  # keep large integer values exact
        data = data.astype(dt)
    if data.dtype not in _MINC_TYPES:
        raise ValueError(f"unsupported dtype {data.dtype} for MINC")
    default_store, representation = _MINC_TYPES[data.dtype]
    out.define(dims, store_type or default_store, representation)

    out.create(fn)
    
    if ref_fname is not None:
        ref=minc2_file(ref_fname)
        out.copy_metadata(ref)

    if history is not None:
        try:
            old_history=out.read_attribute("","history")
            old_history=old_history+"\n"
        except minc2_error: # assume no history available
            old_history=""
            
        new_history=old_history+history
        out.write_attribute("","history",new_history)

    out.setup_standard_order()
    out.save_complete_volume(np.ascontiguousarray(data))
    out.close()


"""
    Resample volume to different sampling
"""
def resample_volume(in_data, in_v2w, out_shape, out_v2w, order=1, fill=0.0):
    import scipy.ndimage
    
    # voxel storage matrix
    xyz_to_zyx = np.array([[0,0,1,0],
                        [0,1,0,0],
                        [1,0,0,0],
                        [0,0,0,1]])
    
    # have to account for the shift of the voxel center

    full_xfm = xyz_to_zyx @ np.linalg.inv(in_v2w) @ out_v2w @ xyz_to_zyx

    new_data = scipy.ndimage.affine_transform(in_data, full_xfm, output_shape=out_shape, order=order, mode='constant',cval=fill)
    
    return new_data, out_v2w


"""
    Resample volume to the uniform sampling, if needed
"""
def uniformize_volume(data, v2w, tolerance=0.1, order=1, step=1.0):

    start, step_, dir_cos = decompose(v2w)

    # check if we need to resample
    if np.any(np.abs(step_ - step) > tolerance):
        # voxel storage matrix
        xyz_to_zyx = np.array([[0,0,1,0],
                            [0,1,0,0],
                            [1,0,0,0],
                            [0,0,0,1]])
        # need to account for the different order of dimensions
        new_shape = np.ceil(np.array(data.shape) * step_[[2,1,0]]).astype(int)
        # have to account for the shift of the voxel center
        new_start = start - step_*0.5 + np.ones(3)*step*0.5
        new_v2w = compose(new_start, np.ones(3)*step, dir_cos)

        return resample_volume(data, v2w, new_shape, new_v2w, order=order, fill=0.0)
    else:
        return data, v2w


"""
    Resample volume to the uniform sampling on its own voxel grid
"""
def uniformize_volume_grid(data, v2w, step=1.0):
    """
    Voxel-grid resampling to `step` mm (always applied, the axes keep their directions):
    Gaussian blur with sigma = 0.25/factor voxels along the axes that are not upsampled
    (factor = voxel size / step, so sigma = 0.25 at factor 1), then linear interpolation at
    voxel-edge aligned positions clamped to the volume, ceil(shape * factor) samples per axis.
    The computation runs in float64 on the (x, y, z) ordered array; the result has the dtype of `data`.

    Args:
        data: volume in (z, y, x) order
        v2w:  4x4 voxel-to-world affine
    Returns:
        (resampled volume in (z, y, x) order, its affine)
    """
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    v2w = np.asarray(v2w, dtype=np.float64)
    volume = np.ascontiguousarray(np.asarray(data, dtype=np.float64).transpose([2, 1, 0]))

    pixdim = np.sqrt(np.sum(v2w * v2w, axis=0))[:-1]
    factor = pixdim / np.array([step, step, step], dtype=np.float64)
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # no blur when upsampling

    volume = gaussian_filter(volume, sigmas)

    grid = tuple(np.arange(0, n) for n in volume.shape)
    interpolator = RegularGridInterpolator(grid, volume, method='linear')

    start = - (factor - 1) / (2 * factor)
    step_ = 1.0 / factor
    stop = start + step_ * np.ceil(volume.shape * factor)

    coords = []
    for c in range(3):
        x = np.arange(start=start[c], stop=stop[c], step=step_[c])
        x[x < 0] = 0
        x[x > (volume.shape[c] - 1)] = volume.shape[c] - 1
        coords.append(x)

    new_volume = interpolator(tuple(np.meshgrid(*coords, indexing='ij', sparse=True)))

    new_v2w = v2w.copy()
    for c in range(3):
        new_v2w[:-1, c] = new_v2w[:-1, c] / factor[c]
    new_v2w[:-1, -1] = new_v2w[:-1, -1] - np.matmul(new_v2w[:-1, :-1], 0.5 * (factor - 1))

    return np.ascontiguousarray(new_volume.transpose([2, 1, 0])).astype(np.asarray(data).dtype, copy=False), new_v2w
