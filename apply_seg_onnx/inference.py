#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#
# @author Vladimir S. FONOV
# @date 29/01/2018
"""Apply pre-trained segmentation model(s) with ONNX Runtime (command line: apply_seg_onnx)."""

import argparse
import re
import sys
import math
import json
import os
import traceback

import numpy as np

from .io import load_volume_np, save_volume, format_history
from .volume import (smallest_int_dtype, autonorm_np, maxnorm_np, mean_std_normalize_np,
                     apply_cropvol, apply_padvol, undo_cropvol, undo_padvol,
                     parse_bracket_input,
                     reorient_to, reorient_back, affine_spacing, foreground_bbox,
                     grid_resample_shape, grid_resample_image,
                     grid_recover_labels, grid_recover_prob, nonzero_mean_std_normalize,
                     separable_gaussian_weights,
                     window_starts, pad_to_size)
from .postprocess import find_largest_component, measure_volumes, save_measurements
from .onnx_tiled import TiledGroupNormSession
from .minc_io import resample_volume, uniformize_volume

import onnxruntime

from scipy.special import softmax


def parse_options():
    parser = argparse.ArgumentParser(description='Apply pre-trained model using ONNX runtime',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", 
                        type=str, 
                        help="Path to JSON configuration file")
    
    parser.add_argument("--model", 
                        type=str, 
                        nargs='+',
                        help="pretrained model(s) (ONNX), can specify multiple for majority voting. If not specified, will use models from config file.")

    parser.add_argument("--model_prefix", 
                        type=str, 
                        help="Add prefix to model names from config file")

    parser.add_argument("input", type=str, nargs='?',
                        help="Input minc file, or input spec in [a,b,...] where a,b is ether const number of file name")
    
    parser.add_argument("output", type=str, nargs='?',
                        help="Output minc file")

    parser.add_argument("--bi", type=str, nargs='+',
                        help="Batch inputs")

    parser.add_argument("--bo", type=str, nargs='+',
                        help="Batch outputs")

    parser.add_argument("--li", type=str, 
                        help="Batch inputs in file")

    parser.add_argument("--lo", type=str, 
                        help="Batch outputs in file")
    
    parser.add_argument("--minibatch_size", type=int, default=1,
                        help="Batch size for processing multiple inputs at once")
    
    parser.add_argument('--progress', action="store_true",
                        default=False,
                        help='Show progress bar' )

    parser.add_argument("--add", 
                        type=str,
                        nargs='+',
                        help="Input minc file")

    parser.add_argument("--patch_sz",
                        nargs='+',
                        type=int, default=[64, 64, 64],
                        help="Patch size")
    
    parser.add_argument("--quant", type=int, default=64,
                        help="Spatial quantization factor for whole volume processing")
                        
    parser.add_argument("--stride", type=int, default=None,
                        nargs='+',
                        help="Stride, default patch_sz-crop*2")

    parser.add_argument("--channels", type=int, default=1,
                        help="add more input channels, fill them with 38.81240207 for now")

    parser.add_argument("--crop", type=int, default=0,
                        help="Crop edges of patch (segment with overlapping patches)")

    parser.add_argument("--cropvol", type=int, default=0,
                        help="Crop edges of the input whole volume before applying the model")

    parser.add_argument("--padvol", type=int, default=0,
                        help="pad the input volume before applying the model")

    parser.add_argument("--padfill", type=float, default=0,
                        help="pad with this value")

    parser.add_argument("--mask", type=str,
                        help="Apply mask to result")

    parser.add_argument("--bck", type=int, default=0,
                        help="Background label")

    parser.add_argument('--cpu', action="store_true",
                        dest="cpu",
                        default=False,
                        help='Do everything in cpu' )
    
    parser.add_argument("--device_id", type=int, default=0,
                        help="Devide ID")

    parser.add_argument('-q','--quiet', action="store_true",
                        default=False,
                        help='Suppress warnings' )

    parser.add_argument('-F','--fuzzy',
                        help='Output fuzzy volume(s)' )

    parser.add_argument('-T','--threads',type=int, default=0,
                        help='Number of threads to use' )

    parser.add_argument('-n','--n_classes',type=int, default=0,
                        help='Number of segmentation classes, needed for overlap only' )
    
    parser.add_argument('-u','--use_classes',type=int, default=None,
                        help='Use only these classes (for models that produce something else in additional channels)' )
    
    parser.add_argument('-U','--uniformize',type=float,
                        help='Uniformize image resolution befor applying CNN' )

    parser.add_argument('-R','--reference',type=str,
                        help='Resample input as reference, usefull for small ROIs in stx space' )

    parser.add_argument('-S','--saveuniform', action="store_true",
                        default=False,
                        help='Output segmentation at uniform resolution or reference space' )
    
    parser.add_argument('--measure',
                        default=None,
                        help='Perform volumetric measurements, store in file' )
    
    parser.add_argument('--recover', action="store_true",
                        default=False,
                        help='Recover from incomplete batch mode' )
    
    parser.add_argument('--crash', action="store_true",
                        default=False,
                        help='Crash on error' )
    
    parser.add_argument('--distance', action="store_true",
                        default=False,
                        help='Uses Distance model' )

    parser.add_argument('--whole', action="store_true",
                        default=False,
                        help='Apply model to the whole image, without overlapping patches' )
    
    parser.add_argument('--trim',default=False, action='store_true',help='Trim instead of expand')

    parser.add_argument("--nibabel", action="store_true", default=False,
                        help="Use nibabel coordinate conversion in model")
    
    parser.add_argument('--freesurfer', action="store_true",
                        default=False,
                        help='Apply model using freesurfer coordinate convention' )
    
    parser.add_argument('--normalize', action="store_true",
                        default=False,
                        help='Apply intensity normalization between 0 and 1 using quantiles' )

    parser.add_argument('--max_normalize', action="store_true",
                        default=False,
                        help='Apply intensity normalization between 0 and 1 using maximum' )
    
    parser.add_argument('--mean_std_normalize', action="store_true",
                        default=False,
                        help='Subtract mean and devide by std, for nonzero voxels' )
    
    parser.add_argument('--largest', action="store_true",
                        default=False,
                        help='Apply largest component filtering' )

    parser.add_argument('--use_tf32', action="store_true",
                        default=False,
                        help='Use TF32 precision in CUDA execution provider' )

    parser.add_argument('--use_gaussian_weights', action="store_true",
                        default=False,
                        help='Use Gaussian weights for overlapping regions in sliding window inference' )

    parser.add_argument('--continuous', action="store_true",
                        default=False,
                        help='Model produces continuous single channel output' )

    parser.add_argument('--majority', action="store_true",
                        default=False,
                        help='Majority voting for multiple models' )
    
    parser.add_argument('--channel_last', action="store_true",
                        default=False,
                        help='Use channel last format for input and output' )

    params = parser.parse_args()
    
    # Handle stride default value
    if params.stride is None:
        if isinstance(params.patch_sz, list):
            params.stride = min(params.patch_sz[0]-params.crop*2,params.patch_sz[1]-params.crop*2,params.patch_sz[2]-params.crop*2)
        else:
            params.stride = params.patch_sz-params.crop*2

    return params

# def log_softmax(x, axis=1):
#     e_x = np.exp(x - np.max(x,axis=axis))
#     return np.log(e_x / e_x.sum(axis=axis))


# def softmax(x,axis=1):
#     e_x = np.exp(x - np.max(x,axis=axis))
#     return e_x / e_x.sum(axis=axis)



def segment_whole(
    dataset, model, 
    quant_size=64,
    normalize=False,
    normalize_max=False,
    normalize_mean_std=False,
    freesurfer=False,
    nibabel=False,
    largest_component=False,
    dist=False,
    continuous=False,
    trim=False,
    trim_center=False,
    use_classes=None,
    channel_last=False
    ):
    """
    Apply model to dataset of arbitrary size 
    Args:
        dataset: Input data array
        model: ONNX model
        crop: Number of voxels to crop from patch edges
        n_classes: Number of output classes
        freesurfer: Whether to use FreeSurfer coordinate convention
        nibabel: Whether to use NIBabel coordinate convention
        normalize: Whether to normalize using quantiles
        normalize_max: Whether to normalize using max value
        normalize_mean_std: Whether to normalize using mean and std
        dist: Whether to use distance-based segmentation
        continuous: model should produce continuous output
        trim: Whether to trim the dataset to a multiple of quant_size
        trim_center: centre the trimmed box along Z too (as minc_wmh_synthseg.py); by default Z is not shifted
        use_classes: Use only these classes (for models that produce something else in additional channels)
        channel_last: Whether to use channel last format for input and output
    Returns:
        output_fuzzy: Fuzzy output 
    """

    if continuous:
        out_name = "scan_out"
    elif dist:
        out_name = "dist"
    else:
        out_name = "seg" 


##
    batch_size = dataset.shape[0]

    if trim:
        target_shape = (np.floor(np.array(dataset.shape[2:]) / quant_size) * quant_size).astype(int)

        if np.any(target_shape != dataset.shape[2:]):
            _trim = ((dataset.shape[2:] - target_shape) // 2).astype(int)
            ### HACK: do not shift Z axis, to avoid cutting cerebellum
            if not trim_center:
                _trim[2] = 0

            conformed = np.ascontiguousarray(
                            dataset[:,:,
                                    _trim[0]:_trim[0]+target_shape[0], 
                                    _trim[1]:_trim[1]+target_shape[1], 
                                    _trim[2]:_trim[2]+target_shape[2]]).astype('float32')
        else:
            conformed = dataset.astype('float32')
    else:
        target_shape = np.ceil(np.array(dataset.shape[2:]) / quant_size).astype(int) * quant_size

        if np.any(target_shape != dataset.shape[2:]):
            conformed = np.zeros( (batch_size,1, *target_shape), dtype='float32')
            conformed[:,:, :dataset.shape[2], :dataset.shape[3], :dataset.shape[4]] = dataset
        else:
            conformed = dataset.astype('float32') # to be compatible with spatial expectation of the model

    print(f"{dataset.shape=} {conformed.shape=}")

    if freesurfer:
        conformed=np.ascontiguousarray(conformed.transpose([0,1,4,3,2])[:,:,:,::-1,:]).copy()
    elif nibabel:
        conformed=np.ascontiguousarray(conformed.transpose([0,1,4,3,2])).copy()

    # intensity normalization
    if normalize:
        # Quantile-based normalization (0-1 range)
        conformed = conformed - conformed.min()
        conformed = np.clip(conformed / np.percentile(conformed,99), 0.0, 1.0)
    elif normalize_max:
        # Max normalization (0-1 range)
        conformed = np.clip(conformed / np.max(conformed), 0.0, 1.0)
    elif normalize_mean_std:
        # Mean-std normalization for nonzero voxels
        mean = np.mean(conformed[conformed>0])
        std = np.std(conformed[conformed>0])
        conformed = (conformed - mean) / std

    # Run inference
    if channel_last:
        conformed = np.ascontiguousarray(conformed.transpose([0, 2, 3, 4, 1]))

    out = model.run([out_name],{'scan':conformed})[0]

    if channel_last:
        out = out.transpose([0, 4, 1, 2, 3])

    if use_classes is not None:
        out = out[:, 0:use_classes, :, :, :]

    if freesurfer:
        out=np.ascontiguousarray(out[:,:,:,::-1,:].transpose([0,1,4,3,2]))
    elif nibabel:
        out=np.ascontiguousarray(out.transpose([0,1,4,3,2]))
    
    print(f"{dataset.shape=} {out.shape=}")

    # unpad
    if np.any(target_shape != dataset.shape[2:]):
        if trim:
            # pad with zeros
            _out=np.zeros((out.shape[0], out.shape[1], dataset.shape[2], dataset.shape[3], dataset.shape[4]))
            _out[:,0,:,:,:] = 1.0 # set BG
            _out[:,:,
                _trim[0]:_trim[0]+target_shape[0], 
                _trim[1]:_trim[1]+target_shape[1], 
                _trim[2]:_trim[2]+target_shape[2]]=out
            out = _out
        else:
            out=out[:,:,:dataset.shape[2], :dataset.shape[3], :dataset.shape[4]]

    return out

def get_gaussian_weights(patch_size, sigma_scale=1.0/8):
    """
    Generate Gaussian weights for overlapping regions in sliding window inference.
    Args:
        patch_size: Size of the patch
        sigma_scale: Scale factor for sigma
    Returns:
        Gaussian weights array (1, 1, *patch_size), maximum 1
    """
    if not isinstance(patch_size, (list, tuple)):
        patch_size = [patch_size] * 3
    
    sigma = [patch_size[i] * sigma_scale for i in range(3)]
    coords = [np.arange(patch_size[i]) for i in range(3)]
    mesh = np.meshgrid(*coords, indexing='ij')
    
    # Calculate distances from center
    center = [(patch_size[i] - 1) / 2 for i in range(3)]
    dist =sum(((mesh[i] - center[i]) / sigma[i]) ** 2 for i in range(3))
    
    # Calculate Gaussian weights
    weights = np.exp(-0.5 * dist)
    weights = weights / np.max(weights)
    # handle non-positive weights
    min_non_zero = max(np.min(weights), 1e-3)
    weights = np.clip(weights, min=min_non_zero)

    # add batch and channel dimensions
    weights = np.expand_dims(np.expand_dims(weights, 0), 0)

    return weights.astype(np.float32)

def legacy_window_starts(image_size, roi_size, interval):
    """
    Original window layout: ceil(image_size/interval) windows per axis, those that would run past the end
    are clamped onto the last position, so that position can occur (and be counted) more than once.
    """
    return [[max(min(k * step, n - r), 0) for k in range(math.ceil(n / step))]
            for n, r, step in zip(image_size, roi_size, interval)]

def segment_with_patches_overlap(
        dataset, model, 
        crop=0,
        patch_sz = None, 
        stride = None,
        n_classes=2,
        use_classes=None,
        bck = 0, 
        freesurfer=False,
        nibabel=False,
        normalize=False,
        normalize_max=False,
        normalize_mean_std=False,
        dist=False,
        use_gaussian_weights=False,
        sigma_scale=0.25,
        continuous=False,
        orig_aff=None,
        channel_last=False,
        sw_batch_size=1,
        window_layout='dense',
        gaussian_map='normalized'):
    """
    Apply model to dataset of arbitrary size using sliding window inference
    Args:
        dataset: Input data array
        model: ONNX model
        crop: Number of voxels to crop from patch edges
        patch_sz: Size of patches to process
        stride: Step size between patches
        n_classes: Number of output classes
        bck: Background value
        out_fuzzy: Whether to output fuzzy results
        freesurfer: Whether to use FreeSurfer coordinate convention
        nibabel: Whether to use NIBabel coordinate convention
        normalize: Whether to normalize using quantiles
        normalize_max: Whether to normalize using max value
        normalize_mean_std: Whether to normalize using mean and std
        dist: Whether to use distance-based segmentation
        use_gaussian_weights: Whether to use Gaussian weights for overlapping regions
        sigma_scale: Gaussian sigma as a fraction of the (cropped) patch size
        sw_batch_size: number of windows per model call
        window_layout: 'dense' (default) - fewest windows spaced by `stride`, the last one clamped to the end,
                           no duplicates (window_starts());
                       'legacy' - original layout: ceil(size/stride) windows per axis, the ones past the end
                           clamped onto the last position, which is then counted more than once
        gaussian_map: 'normalized' (default) - original map, maximum 1 (get_gaussian_weights());
                      'separable' - product of 1D float32 Gaussians, not normalised (separable_gaussian_weights())
    Axes shorter than the patch are zero-padded and cropped back.
    """
    if continuous:
        out_name = "scan_out"
    elif dist:
        out_name = "dist"
    else:
        out_name = "seg" 

    out_classes = 1 if continuous or (dist and n_classes is None) else n_classes
    if not continuous and not dist and use_classes is not None:
        out_classes = use_classes  # only the first use_classes output channels are accumulated

    if not isinstance(patch_sz, list):
        patch_sz = [patch_sz, patch_sz, patch_sz]

    if not isinstance(stride, list):
        stride = [stride, stride, stride]

    if freesurfer:
        dataset=np.ascontiguousarray(dataset.transpose([0,1,4,3,2])[:,:,:,::-1,:]).copy()
    elif nibabel:
        dataset=np.ascontiguousarray(dataset.transpose([0,1,4,3,2])).copy()

    # intensity normalization
    if normalize:
        dataset = dataset - dataset.min()
        dataset = np.clip(dataset / np.percentile(dataset,99), min=0.0, max=1.0)
    elif normalize_max:
        dataset = np.clip(dataset / np.max(dataset), min=0.0, max=1.0)
    elif normalize_mean_std:
        mean = np.mean(dataset[dataset>0])
        std = np.std(dataset[dataset>0])
        dataset = (dataset - mean) / std

    # Axes shorter than the patch are zero-padded (diff//2 before, rest after), cropped back at the end
    dataset, pads = pad_to_size(dataset, patch_sz, axes=(2, 3, 4))
    dsize = dataset.shape
    output_size = list(dsize)
    output_size[1] = 1
    output_size_fuzzy = list(dsize)
    output_size_fuzzy[1] = out_classes

    output_fuzzy  = np.zeros(output_size_fuzzy, dtype=np.float32)
    output_weight = np.zeros(output_size, dtype=np.float32)

    # only the central patch_sz_ part of every prediction is used
    patch_sz_ = [patch_sz[0] - crop*2, patch_sz[1] - crop*2, patch_sz[2] - crop*2]
    out_roi = [dsize[2]-crop*2, dsize[3]-crop*2, dsize[4]-crop*2]

    if window_layout not in ('legacy', 'dense'):
        raise ValueError(f"window_layout: expected 'dense' or 'legacy', got {window_layout!r}")
    if gaussian_map not in ('normalized', 'separable'):
        raise ValueError(f"gaussian_map: expected 'normalized' or 'separable', got {gaussian_map!r}")

    # Generate Gaussian weights if requested
    if not use_gaussian_weights:
        gaussian_weights = np.ones((1, 1, *patch_sz_), dtype=np.float32)
    elif gaussian_map == 'separable':
        gaussian_weights = separable_gaussian_weights(patch_sz_, sigma_scale)[None, None]
    else:
        gaussian_weights = get_gaussian_weights(patch_sz_, sigma_scale=sigma_scale)

    # Window positions of the used (cropped) part of the patches
    if window_layout == 'dense':
        starts = window_starts(out_roi, patch_sz_, stride)
    else:
        starts = legacy_window_starts(out_roi, patch_sz_, stride)
    windows = np.array(np.meshgrid(*starts, indexing='ij')).reshape(3, -1).T + crop

    nb = dsize[0]
    for w0 in range(0, len(windows), sw_batch_size):
        batch_w = windows[w0:w0 + sw_batch_size]
        # Extract patches (input patch starts `crop` voxels before the used part)
        in_data = np.ascontiguousarray(np.concatenate(
            [dataset[:, :, c[0]-crop: c[0]-crop+patch_sz[0],
                           c[1]-crop: c[1]-crop+patch_sz[1],
                           c[2]-crop: c[2]-crop+patch_sz[2]] for c in batch_w], axis=0))

        if channel_last:
            in_data = np.ascontiguousarray(in_data.transpose([0, 2, 3, 4, 1]))

        # Run inference
        out = model.run([out_name],{'scan':in_data})[0]

        if channel_last:
            out = out.transpose([0, 4, 1, 2, 3])

        if not continuous and not dist and use_classes is not None:
            out = out[:, 0:use_classes, :, :, :]

        for i, c in enumerate(batch_w):
            patch_output = out[i*nb:(i+1)*nb]
            # Apply Gaussian weights to the used part of the patch output and accumulate
            output_fuzzy [:, :, c[0]: c[0]+patch_sz_[0], c[1]: c[1]+patch_sz_[1], c[2]: c[2]+patch_sz_[2]] += \
                gaussian_weights * patch_output[:, :, crop: crop+patch_sz_[0], crop: crop+patch_sz_[1], crop: crop+patch_sz_[2]]
            output_weight[:, :, c[0]: c[0]+patch_sz_[0], c[1]: c[1]+patch_sz_[1], c[2]: c[2]+patch_sz_[2]] += gaussian_weights

    # Normalize accumulated results (with crop>0 the outer `crop` voxels are never predicted)
    invalid = output_weight < 1e-3
    output_weight[invalid] = 1.0
    output_fuzzy =  output_fuzzy/output_weight

    # Handle invalid regions
    if not continuous:
        for q in range(output_fuzzy.shape[1]):
            output_fuzzy[:,(q):(q+1),:,:,:][invalid] = 0.0
        output_fuzzy[:,bck:bck+1,:,:,:][invalid] = 1.0

    if any(p for pp in pads for p in pp):
        output_fuzzy = output_fuzzy[:, :, pads[0][0]:dsize[2]-pads[0][1],
                                          pads[1][0]:dsize[3]-pads[1][1],
                                          pads[2][0]:dsize[4]-pads[2][1]]

    if freesurfer:
        output_fuzzy = output_fuzzy[:,:,:,::-1,:].transpose([0,1,4,3,2])

    elif nibabel:
        output_fuzzy=output_fuzzy.transpose([0,1,4,3,2]).copy()
    return output_fuzzy 

def resample_mode(settings):
    """
    Config `resample`:
        "legacy" (default, also None) - original behaviour: the only resampling is `uniformize` / `reference`
        "mindglide" - voxel-grid resampling to 1 mm before the model (cubic spline, truncated target shape),
                      labels brought back per class (volume.grid_resample_* / grid_recover_*)
    """
    mode = settings.get('resample', None) or 'legacy'
    if mode not in ('legacy', 'mindglide'):
        raise ValueError(f"resample: expected 'legacy' or 'mindglide', got {mode!r}")
    return mode


def preprocess_volume(data, aff, settings, ctx=None, minc=False):
    """
    Optional geometry preprocessing, each step enabled by a config key:
        reorient (axis codes, e.g. "RAS"/None), crop_foreground (bounding box of voxels > 0),
        resample ("legacy": none here, see resample_mode() / voxel-grid resampling to 1 mm) + spacing_float32,
        normalize_mean_std_nonzero
    Args:
        data: volume as returned by load_volume_np (reversed voxel order, k,j,i)
        aff:  4x4 voxel-to-world affine
        ctx:  geometry from a previous call, to apply the same reorient/crop/resample to another channel
        minc: data comes from a MINC file (standard order, RAS): `reorient` needs no nibabel
    Returns:
        (array to feed to the model, ctx for postprocess_labels / postprocess_fuzzy); with none of the keys set
        `data` is returned unchanged and ctx is None
    """
    reorient = settings.get('reorient', None)
    crop_fg = settings.get('crop_foreground', False)
    resample = resample_mode(settings) != 'legacy'
    nz_norm = settings.get('normalize_mean_std_nonzero', False)
    if not (reorient or crop_fg or resample or nz_norm):
        return data, None

    # voxel order (i,j,k) matching the affine
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32).transpose([2, 1, 0]))
    aff = np.asarray(aff, dtype=np.float64)
    if reorient:
        arr, aff_r, tr = reorient_to(arr, aff, reorient, minc=minc)
    else:
        aff_r, tr = aff, None

    if ctx is None:
        ctx = {'tr': tr, 'tr_minc': minc, 'full_shape': arr.shape}
        if crop_fg:
            ctx['bb_start'], ctx['bb_end'] = foreground_bbox(arr)
        else:
            ctx['bb_start'], ctx['bb_end'] = [0, 0, 0], list(arr.shape)
        bb_start, bb_end = ctx['bb_start'], ctx['bb_end']
        ctx['crop_shape'] = [e - s_ for s_, e in zip(bb_start, bb_end)]
        ctx['resample_flag'], ctx['new_shape'], ctx['anisotropy_flag'] = False, None, False
        if resample:
            # NIfTI stores the affine as float32; MINC gives float64 with different rounding noise,
            # which flips the exact spacing==1 test. Rounding to float32 makes MINC match NIfTI.
            sp_aff = aff_r.astype(np.float32) if settings.get('spacing_float32', False) else aff_r
            ctx['resample_flag'], ctx['new_shape'], ctx['anisotropy_flag'] = \
                grid_resample_shape(affine_spacing(sp_aff), ctx['crop_shape'])

    bb_start, bb_end = ctx['bb_start'], ctx['bb_end']
    arr = arr[bb_start[0]:bb_end[0], bb_start[1]:bb_end[1], bb_start[2]:bb_end[2]]
    if ctx['resample_flag']:
        arr = grid_resample_image(arr, ctx['new_shape'], ctx['anisotropy_flag'])
    if nz_norm:
        arr = nonzero_mean_std_normalize(arr)
    return np.ascontiguousarray(arr, dtype=np.float32), ctx


def postprocess_labels(labels, ctx, n_classes, bck=0):
    """Undo preprocess_volume for a label map: recover resampling, un-crop, reorient back, (k,j,i) order."""
    if ctx is None:
        return labels
    if ctx['resample_flag']:
        labels = grid_recover_labels(labels, n_classes, ctx['crop_shape'], ctx['anisotropy_flag'])
    seg = np.full(ctx['full_shape'], bck, dtype=labels.dtype)
    bb_start, bb_end = ctx['bb_start'], ctx['bb_end']
    seg[bb_start[0]:bb_end[0], bb_start[1]:bb_end[1], bb_start[2]:bb_end[2]] = labels
    if ctx['tr'] is not None:
        seg = reorient_back(seg, ctx['tr'], minc=ctx['tr_minc'])
    return np.ascontiguousarray(seg.transpose([2, 1, 0]))


def postprocess_fuzzy(prob, ctx, bck=0):
    """Undo preprocess_volume for per-class maps (C,...): linear resampling back, background
    channel = 1 / others = 0 outside the crop, reorient back, (k,j,i) order."""
    if ctx is None:
        return prob
    out = []
    bb_start, bb_end = ctx['bb_start'], ctx['bb_end']
    for c in range(prob.shape[0]):
        p = prob[c]
        if ctx['resample_flag']:
            p = grid_recover_prob(p, ctx['crop_shape'], ctx['anisotropy_flag'])
        full = np.full(ctx['full_shape'], 1.0 if c == bck else 0.0, dtype=np.float32)
        full[bb_start[0]:bb_end[0], bb_start[1]:bb_end[1], bb_start[2]:bb_end[2]] = p
        if ctx['tr'] is not None:
            full = reorient_back(full, ctx['tr'], minc=ctx['tr_minc'])
        out.append(full.transpose([2, 1, 0]))
    return np.ascontiguousarray(np.stack(out))


def keep_largest(seg, settings, bck=0):
    """Config `largest`: keep the largest connected component of the foreground (`largest_connectivity`,
    3 = 26-neighbourhood, the default, 1 = 6-neighbourhood)."""
    if settings.get('largest', False):
        seg = seg.copy()
        seg[~find_largest_component(seg != bck, connectivity=settings.get('largest_connectivity', 3))] = bck
    return seg


def make_onnx_sessions(models, cpu=True, threads=0, device_id=None, use_tf32=False, tiled=None):
    """Create onnxruntime sessions for a list of model files.
    tiled: tile size (voxels) -> TiledGroupNormSession: whole-volume result of GroupNorm networks with
           GPU memory bounded by the tile size (onnx_tiled.py)"""
    sess_options = onnxruntime.SessionOptions()
    if threads > 0:
        sess_options.intra_op_num_threads = threads
    cuda_opts = {"use_tf32": 1 if use_tf32 else 0}
    if device_id is not None:
        cuda_opts["device_id"] = device_id
    providers = ['CPUExecutionProvider'] if cpu else [("CUDAExecutionProvider", cuda_opts)]
    if not isinstance(models, list):
        models = [models]
    if tiled:
        return [TiledGroupNormSession(m, sess_options, providers=providers, tile=tiled) for m in models]
    return [onnxruntime.InferenceSession(m, sess_options, providers=providers) for m in models]


def load_scan(channels, settings, ref_data=None, ref_aff=None):
    """
    Load one scan for segmentation.
    Args:
        channels: file name, or list of input channels (file names, or numbers for constant-filled channels)
    Returns:
        (array (1, C, ...) to feed to the model, dict with the geometry needed to save the output)
    """
    uniformize = settings.get('uniformize', None)
    if not isinstance(channels, (list, tuple)):
        channels = [channels]
    data_ch = []
    info = {'ref_file': None, 'aff': None, 'shape': None, 'new_aff': None, 'prep_ctx': None}
    for ch in channels:
        if not isinstance(ch, str):
            data_ch.append(float(ch))  # constant channel, filled once the shape is known
            continue
        data, aff = load_volume_np(ch, dtype='float32')
        # make sure all files have the same shape and orientation
        if info['shape'] is None:
            info['ref_file'], info['shape'], info['aff'] = ch, np.array(data.shape), aff
        else:
            assert np.all(info['shape'] == np.array(data.shape)), f"{ch}: shape differs from {info['ref_file']}"
            assert np.all(np.abs(np.asarray(info['aff']) - np.asarray(aff)) < 1e-3), f"{ch}: affine differs from {info['ref_file']}"

        # optional reorient/crop/resample/normalize (config keys), same geometry for all channels
        data, info['prep_ctx'] = preprocess_volume(data, aff, settings, info['prep_ctx'],
                                                      minc=ch.endswith('.mnc'))

        if ref_aff is not None:
            data, info['new_aff'] = resample_volume(data, aff, ref_data.shape, ref_aff)
        if uniformize is not None:
            data, info['new_aff'] = uniformize_volume(data, aff, step=uniformize)
        data_ch.append(data)

    shape = next(d.shape for d in data_ch if isinstance(d, np.ndarray))
    data_ch = [d if isinstance(d, np.ndarray) else np.full(shape, d, dtype=np.float32) for d in data_ch]
    return np.stack(data_ch, axis=0)[None], info


def segment_with_onnx(in_scans, out_seg, settings,
    cpu=True, threads=0,
    history=None,device_id=None,use_tf32=False,
    measure=None, fuzzy=None):
    """
    Segment one scan: in_scans is the list of its input channels. Runs segment_with_onnx_batched with batch size 1.
    """
    missing = [f for f in in_scans if isinstance(f, str) and not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Input file(s) do not exist: {missing}")
    segment_with_onnx_batched([in_scans], [out_seg], settings,
        cpu=cpu, threads=threads, history=history, device_id=device_id, use_tf32=use_tf32,
        minibatch_size=1, measure=measure, fuzzy=fuzzy, crash=True)
    return out_seg


def segment_with_onnx_batched(in_scans, out_segs, 
    settings,
    cpu=True, 
    threads=0,
    history=None,
    device_id=None,
    use_tf32=False,
    minibatch_size=1,
    fuzzy=None,
    progress=False,
    measure=None,
    recover=False,
    crash=False):
    """
    Segment a list of scans, minibatch_size scans per model call.
    Args:
        in_scans: list of scans; each is a file name or a list of input channels
                  (file names, or numbers for constant-filled channels)
        out_segs: output segmentation file per scan
        fuzzy: prefix for per-class probability maps (<prefix>_<class>.<ext>; <prefix>_<scan>_<class>.<ext>
               when several scans are given; same format as the output segmentation); defaults to settings['fuzzy']
        measure: csv file for label volumes (needs labels_desc in settings)
        recover: skip scans whose output already exists (volumes are measured from the existing output)
        crash: re-raise errors instead of skipping the failed batch
    Scans in a minibatch are stacked into one array when their (preprocessed) shapes agree,
    otherwise they are run through the models one at a time.
    """
    # Extract parameters from settings dictionary
    n_classes = settings.get('n_classes', 2)
    use_classes = settings.get('use_classes', None)
    mask = settings.get('mask', None)
    models = settings.get('models', None)
    patch_sz = settings.get('patch_sz', 64)
    stride = settings.get('stride', 32)
    crop = settings.get('crop', 0)
    cropvol = settings.get('cropvol', 0)
    padvol = settings.get('padvol', 0)
    padfill = settings.get('padfill', 0.0)
    bck = settings.get('bck', 0)
    if history is None:
        history = settings.get('history', None)
    if fuzzy is None:
        fuzzy = settings.get('fuzzy', None)
    whole = settings.get('whole', False)
    quant_size = settings.get('quant_size', 64)
    freesurfer = settings.get('freesurfer', False)
    nibabel = settings.get('nibabel', False)
    normalize = settings.get('normalize', False)
    normalize_max = settings.get('normalize_max', False)
    normalize_mean_std = settings.get('normalize_mean_std', False)
    dist = settings.get('dist', False)
    reference= settings.get('reference', None)
    uniformize = settings.get('uniformize', None)
    use_gaussian_weights = settings.get('use_gaussian_weights', False)
    sigma_scale = settings.get('sigma_scale', 0.25)
    sw_batch_size = settings.get('sw_batch_size', 1)
    window_layout = settings.get('window_layout', 'dense')
    gaussian_map = settings.get('gaussian_map', 'normalized')
    resample_mode(settings)  # validate
    continuous = settings.get('continuous', False)
    trim = settings.get('trim', False)
    channel_last = settings.get('channel_last', False)
    save_uniformized = settings.get('save_uniformized', False)
    labels_desc = settings.get('labels_desc', None)
    majority = settings.get('majority', False)
    augment_tta = settings.get('augment_tta', None)
    trim_center = settings.get('trim_center', False)
    tiled = settings.get('tiled_groupnorm', None)
    label_values = settings.get('label_values', None)
    if label_values is not None:
        label_values = np.array(label_values)
        if isinstance(labels_desc, list):  # labels_desc[i] describes class i+1 -> output value label_values[i+1]
            labels_desc = {int(label_values[i + 1]): d for i, d in enumerate(labels_desc)}

    assert len(out_segs) == len(in_scans), "Number of output segments must match number of input scans"

    ### for now only flip map augmentation is supported
    if augment_tta is not None:
        assert "flip_x" in augment_tta, "Only flip augmentation is supported for TTA now"
        if not continuous:
            flip_map = augment_tta["flip_x"]
            assert isinstance(flip_map, list),"Flip map should be a list of indexes which are remapped after flipping"
            flip_map = np.array(flip_map,dtype=int)
            assert np.all(np.sort(flip_map) == np.arange(len(flip_map))), "Flip map should be a permutation of [0, 1, ..., n_classes-1]"
            assert len(flip_map) == n_classes, "Flip map length should match number of classes"

    models_onnx = make_onnx_sessions(models, cpu=cpu, threads=threads, device_id=device_id, use_tf32=use_tf32,
                                     tiled=tiled)
    all_measurements=[]

    if reference is not None:
        ref_data, ref_aff = load_volume_np(reference, dtype='uint8', as_byte=True)
    else:
        ref_data = None
        ref_aff = None

    if mask is not None:
        mask_data, mask_aff = load_volume_np(mask, dtype='uint8', as_byte=True)

    def scan_name(scan):
        return scan if isinstance(scan, str) else next((c for c in scan if isinstance(c, str)), None)

    def scan_files(scan):
        return [scan] if isinstance(scan, str) else [c for c in scan if isinstance(c, str)]

    if progress:
        from tqdm import tqdm
        prog = tqdm(total=len(in_scans), desc="Processing scans", unit="scan")
    
    for b in range(0, len(in_scans), minibatch_size):
        # inputs
        batch_scans = in_scans[b:b+minibatch_size]
        # outputs
        out_batch_segs = out_segs[b:b+minibatch_size]

        try:
            # check if inputs and output exists
            input_exists= all([os.path.exists(f) for scan in batch_scans for f in scan_files(scan)])
            output_exists= all([os.path.exists(i) for i in out_batch_segs])

            if not input_exists:
                print(f"Skipping batch {batch_scans}: some input files do not exist",file=sys.stderr)
                if labels_desc is not None and not continuous and measure is not None:
                    all_measurements += [measure_volumes(None, None, labels_desc, out_seg_f=out_seg, in_scan=scan_name(in_scan)) for in_scan, out_seg in zip(batch_scans, out_batch_segs)]
                if progress:
                    prog.update(len(batch_scans))
                continue

            if output_exists and recover:
                if labels_desc is not None and not continuous and measure is not None:
                    all_measurements += [measure_volumes(None, None, labels_desc, out_seg_f=out_seg, in_scan=scan_name(in_scan), load_output=True) for in_scan, out_seg in zip(batch_scans, out_batch_segs)]
                if progress:
                    prog.update(len(batch_scans))
                continue

            loaded = [load_scan(scan, settings, ref_data, ref_aff) for scan in batch_scans]
            # scans are stacked when their shapes agree (foreground cropping/resampling makes them differ)
            if len(set(d.shape for d, _ in loaded)) == 1:
                groups = [list(range(len(loaded)))]
            else:
                groups = [[i] for i in range(len(loaded))]

            for grp in groups:
                infos = [loaded[i][1] for i in grp]
                batch_inputs = [loaded[i][0] for i in grp]
                if augment_tta is not None:
                    # flip along X: last array axis as loaded, first after the geometry preprocessing
                    flip_axis = 2 if infos[0]['prep_ctx'] is not None else 4
                    batch_inputs += [np.flip(i, axis=flip_axis) for i in batch_inputs]
                dset = np.concatenate(batch_inputs, axis=0)

                if whole:
                    patch_sz_g = np.clip(np.ceil((np.array(dset.shape[2:]) - cropvol*2 + padvol*2) / quant_size).astype(int) * quant_size, quant_size*2, quant_size*5).tolist()
                elif not isinstance(patch_sz, list):
                    patch_sz_g = [patch_sz, patch_sz, patch_sz]
                else:
                    patch_sz_g = patch_sz

                if cropvol>0:
                    orig_size = dset.shape
                    dset = dset[:, :, cropvol: orig_size[2]-cropvol, cropvol: orig_size[3]-cropvol, cropvol: orig_size[4]-cropvol]
                elif padvol>0:
                    orig_size = dset.shape
                    dset = np.ascontiguousarray( np.pad(dset, pad_width=((0,0),(0,0),(padvol,padvol),(padvol,padvol),(padvol,padvol)),
                        mode='constant', constant_values = padfill))

                # Apply models and collect results
                all_fuzzy_outputs = []
                for model in models_onnx:
                    if whole:
                        dset_out_fuzzy = segment_whole(
                            dset, model,
                            quant_size=quant_size,
                            freesurfer=freesurfer,
                            nibabel=nibabel,
                            normalize=normalize,
                            normalize_max=normalize_max,
                            normalize_mean_std=normalize_mean_std,
                            dist=dist,
                            continuous=continuous,
                            trim=trim,
                            trim_center=trim_center,
                            use_classes=use_classes,
                            channel_last=channel_last) 
                    else:
                        dset_out_fuzzy = segment_with_patches_overlap(
                            dset, model, 
                            n_classes=n_classes,use_classes=use_classes,
                            patch_sz=patch_sz_g, crop=crop, 
                            bck=bck, stride=stride, 
                            freesurfer=freesurfer,
                            nibabel=nibabel,
                            normalize=normalize,
                            normalize_max=normalize_max,
                            normalize_mean_std=normalize_mean_std,
                            dist=dist,
                            use_gaussian_weights=use_gaussian_weights,
                            sigma_scale=sigma_scale,
                            sw_batch_size=sw_batch_size,
                            window_layout=window_layout,
                            gaussian_map=gaussian_map,
                            continuous=continuous,
                            channel_last=channel_last)
                    all_fuzzy_outputs.append(dset_out_fuzzy)

                dset_out = None
                if len(models_onnx) > 1 and majority and not continuous and augment_tta is None:
                    if dist and all_fuzzy_outputs[0].shape[1]==1:
                        stacked_outputs = np.stack([(i<1.0).astype(np.uint8) for i in all_fuzzy_outputs],axis=0)
                    elif dist:
                        stacked_outputs = np.stack([np.argmin(i, axis=1,keepdims=True).astype(np.uint8) for i in all_fuzzy_outputs],axis=0)
                    else:
                        stacked_outputs = np.stack([np.argmax(i, axis=1,keepdims=True).astype(np.uint8) for i in all_fuzzy_outputs],axis=0)
                    # majority vote over models, per voxel
                    dset_out = np.apply_along_axis(lambda x: np.bincount(x.astype(np.int32)).argmax(), 0, stacked_outputs).astype(np.uint8)
                dset_out_fuzzy = np.mean(all_fuzzy_outputs, axis=0) if len(all_fuzzy_outputs) > 1 else all_fuzzy_outputs[0]

                if augment_tta is not None:
                    # average original and flipped outputs
                    half = dset_out_fuzzy.shape[0] // 2
                    dset_out_fuzzy_orig = dset_out_fuzzy[:half]
                    dset_out_fuzzy_flip = np.flip(dset_out_fuzzy[half:], axis=flip_axis) # flip back
                    if continuous:
                        dset_out_fuzzy = (dset_out_fuzzy_orig + dset_out_fuzzy_flip) / 2.0
                    else:
                        dset_out_fuzzy = softmax(dset_out_fuzzy_orig,axis=1)*0.5 + \
                                        softmax(dset_out_fuzzy_flip[:,flip_map,:,:,:],axis=1) * 0.5 # remap classes
                elif fuzzy is not None and not continuous and not dist:
                    dset_out_fuzzy = softmax(dset_out_fuzzy, axis=1)

                if continuous:
                    dset_out = dset_out_fuzzy
                elif dset_out is not None:
                    pass # majority vote
                elif dist and dset_out_fuzzy.shape[1]==1:
                    dset_out = (dset_out_fuzzy < 1.0)
                elif dist:
                    dset_out = np.argmin(dset_out_fuzzy, axis=1,keepdims=True).astype(np.uint8)
                else:
                    # argmax of logits (or of probabilities after TTA); softmax would not change it
                    dset_out = np.argmax(dset_out_fuzzy, axis=1,keepdims=True).astype(np.uint8)

                if cropvol>0:
                    dset_out_ = np.zeros((dset_out.shape[0], dset_out.shape[1], *orig_size[2:]), dtype=dset_out.dtype)
                    dset_out_[:, :, cropvol: orig_size[2]-cropvol, cropvol: orig_size[3]-cropvol, cropvol: orig_size[4]-cropvol]=\
                        dset_out
                    dset_out = dset_out_
                    if fuzzy is not None:
                        dset_out_fuzzy_ = np.zeros((dset_out_fuzzy.shape[0], dset_out_fuzzy.shape[1], *orig_size[2:]), dtype=np.float32)
                        dset_out_fuzzy_[:, :, cropvol: orig_size[2]-cropvol, cropvol: orig_size[3]-cropvol, cropvol: orig_size[4]-cropvol]=\
                            dset_out_fuzzy
                        dset_out_fuzzy = dset_out_fuzzy_
                elif padvol>0:
                    dset_out = dset_out[:, :, padvol: orig_size[2]+padvol, padvol: orig_size[3]+padvol, padvol: orig_size[4]+padvol]
                    if fuzzy is not None:
                        dset_out_fuzzy = dset_out_fuzzy[:, :, padvol: orig_size[2]+padvol, padvol: orig_size[3]+padvol, padvol: orig_size[4]+padvol]

                for i, info in zip(grp, infos):
                    k = grp.index(i)
                    in_scan, out_seg = batch_scans[i], out_batch_segs[i]
                    orig_aff, orig_shape, new_aff = info['aff'], info['shape'], info['new_aff']

                    dst_out_ = np.ascontiguousarray(dset_out[k].squeeze(), dtype=np.float32 if continuous else np.uint8)
                    # undo the geometry preprocessing
                    dst_out_ = postprocess_labels(dst_out_, info['prep_ctx'], use_classes or n_classes, bck)

                    if not save_uniformized and (uniformize is not None or ref_aff is not None) and np.any(np.array(dst_out_.shape) != orig_shape):
                        dst_out_ = resample_volume(dst_out_, new_aff, orig_shape, orig_aff, order=0, fill=bck)[0]
                    if save_uniformized and (uniformize is not None or ref_aff is not None):
                        # adjust  for output
                        out_aff, out_shape = new_aff, dst_out_.shape
                    else:
                        out_aff, out_shape = orig_aff, orig_shape

                    if not continuous:
                        dst_out_ = keep_largest(dst_out_, settings, bck)

                    if fuzzy is not None:
                        fuzzy_prefix = fuzzy if len(in_scans) == 1 else f"{fuzzy}_{b + i}"
                        fuzzy_ext = '.nii.gz' if out_seg.endswith('.nii.gz') else '.mnc'  # same format as the segmentation
                        prob = postprocess_fuzzy(dset_out_fuzzy[k], info['prep_ctx'], bck)
                        for f in range(prob.shape[0]):
                            dset_out_f = prob[f]
                            if not save_uniformized and (uniformize is not None or ref_aff is not None) and np.any(np.array(dset_out_f.shape) != orig_shape):
                                dset_out_f = resample_volume(dset_out_f, new_aff, orig_shape, orig_aff, order=1, fill=bck)[0]
                            save_volume(fuzzy_prefix+'_{}{}'.format(f, fuzzy_ext),
                                        np.ascontiguousarray(dset_out_f), out_aff, ref_fname=info['ref_file'], history=history)

                    if mask is not None:
                        mask_ = mask_data
                        if np.any(np.array(mask_.shape) != np.array(out_shape)) or np.any(np.abs(np.asarray(mask_aff) - np.asarray(out_aff)) > 1e-3): # resample mask
                            mask_ = resample_volume(mask_data, mask_aff, out_shape, out_aff, order=0, fill=0)[0]
                        dst_out_[mask_<1] = bck

                    if label_values is not None and not continuous:
                        # class index -> output label value
                        dst_out_ = label_values.astype(smallest_int_dtype(label_values))[dst_out_]

                    save_volume(out_seg, dst_out_, out_aff, ref_fname=info['ref_file'], history=history)

                    if labels_desc is not None and not continuous and measure is not None:
                        all_measurements+=[measure_volumes(dst_out_, out_aff, labels_desc, out_seg_f=out_seg, in_scan=scan_name(in_scan))]

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print(f"Error processing batch scans {batch_scans}: {e}",file=sys.stderr)
            print(traceback.format_exc(),file=sys.stderr)
            if crash:
                raise e # if crash flag is set, otherwise just skip to next batch
            if labels_desc is not None and not continuous and measure is not None:
                all_measurements+=[measure_volumes(None, None, labels_desc, out_seg_f=out_seg, in_scan=scan_name(in_scan))  for in_scan, out_seg in zip(batch_scans, out_batch_segs)]

        
        if progress:
            prog.update(len(batch_scans))

    if progress:
        prog.close()

    if measure is not None and len(all_measurements)>0:
        save_measurements(measure, all_measurements)

def main():
    _history = format_history(sys.argv)
    params = parse_options()
    # Create settings dictionary from parameters

    if params.config is not None:
        with open(params.config, 'r') as f:
            settings = json.load(f)
        # allow overrrides from command line
        if params.model is not None:
            settings['models'] = params.model
    else:
        settings = {
            'models': params.model,
            'n_classes': params.n_classes,
            'use_classes': params.use_classes,
            'patch_sz': params.patch_sz,
            'crop': params.crop,
            'bck': params.bck,
            'stride': params.stride,
            'padvol': params.padvol,
            'cropvol': params.cropvol,
            'mask': params.mask,
            'uniformize': params.uniformize,
            'reference': params.reference,
            'save_uniformized': params.saveuniform,
            'history': _history,
            'whole': params.whole,
            'freesurfer': params.freesurfer,
            'nibabel': params.nibabel,
            'normalize': params.normalize,
            'normalize_max': params.max_normalize,
            'normalize_mean_std': params.mean_std_normalize,
            'largest': params.largest,
            'quant_size': params.quant,
            'dist': params.distance,
            'use_gaussian_weights': params.use_gaussian_weights,
            'continuous': params.continuous,
            'trim': params.trim,
            'channel_last': params.channel_last,
            'majority': params.majority
        }
    
    if params.model_prefix is not None:
        if not isinstance(settings['models'], list):
            settings['models'] = [settings['models']]
        settings['models'] = [params.model_prefix + m for m in settings['models']]
        if settings.get('reference', None) is not None:
            settings['reference'] = params.model_prefix + settings['reference']

    if params.input is not None and \
       params.output is not None:
        
        m = re.match(r"\[(.*)\]", params.input)
        if m is not None:
            inp = m[1].split(",")
            inputs=[]
            for i in inp:
                q=re.match(r"^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$",i)
                if q is not None:
                    inputs.append(float(q[0]))
                else:
                    inputs.append(i)
        else:
            inputs=[params.input]

            # attach additional channels
            if params.add is not None:
                for a in params.add:
                    inputs.append(a)
            if params.channels>1:
                for i in range(params.channels-1):
                    inputs.append(params.fill)

        segment_with_onnx(inputs, params.output, settings,
            cpu=params.cpu,
            threads=params.threads,
            device_id=params.device_id,
            use_tf32=params.use_tf32,
            measure=params.measure,
            fuzzy=params.fuzzy,
            history=_history)
            
    elif params.bi is not None and params.bo is not None:
            segment_with_onnx_batched(params.bi, params.bo, settings,
                cpu=params.cpu,
                threads=params.threads,
                device_id=params.device_id,
                use_tf32=params.use_tf32,
                progress=params.progress)
    elif params.li is not None and params.lo is not None:
            # read lists of input and output files
            with open(params.li, 'r') as f:
                li = [line.strip() for line in f if line.strip()]
            with open(params.lo, 'r') as f:
                lo = [line.strip() for line in f if line.strip()]

            import time

            start_time = time.time()
            segment_with_onnx_batched(li, lo, settings,
                cpu=params.cpu,
                threads=params.threads,
                device_id=params.device_id,
                use_tf32=params.use_tf32,
                minibatch_size=params.minibatch_size,
                progress=params.progress,
                measure=params.measure,
                recover=params.recover,
                crash=params.crash)
            elapsed_time = time.time() - start_time

            if not params.progress:
                print(f"Processed {len(li)} scans in {elapsed_time:.2f} seconds")
                if len(li) > 0:
                    print(f"Average time per scan: {elapsed_time/len(li):.4f} seconds")


    else:
      print("Run with --help")
   


if __name__ == '__main__':
    main()

# kate: space-indent on; indent-width 4; indent-mode python;replace-tabs on;word-wrap-column 80
