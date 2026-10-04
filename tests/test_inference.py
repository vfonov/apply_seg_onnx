"""apply_seg_onnx.inference building blocks: sliding window, whole volume, geometry pre/post-processing.

The pointwise model (conftest) makes every voxel's logits a known function of its intensity, so any
window layout / padding / weighting must reproduce them exactly (up to float32 rounding of the weighted mean).
"""
import numpy as np
import onnxruntime
import pytest

from apply_seg_onnx import inference as I
from apply_seg_onnx import volume as V
from apply_seg_onnx.onnx_tiled import TiledGroupNormSession
from conftest import AFF_1MM, phantom, pointwise_logits


@pytest.fixture(scope='module')
def pw(pointwise_model):
    return onnxruntime.InferenceSession(pointwise_model, providers=['CPUExecutionProvider'])


def vol(shape=(1, 1, 30, 26, 21), seed=0):
    return np.random.default_rng(seed).uniform(0, 3, shape).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# sliding window
@pytest.mark.parametrize('patch, stride, gaussian, sw_batch', [
    ([16, 16, 16], [8, 8, 8], False, 1),
    ([16, 16, 16], [8, 8, 8], True, 1),
    ([16, 12, 8], [16, 5, 3], True, 4),     # odd strides, last windows clamped to the end
    ([32, 32, 32], [16, 16, 16], True, 2),  # patch larger than the volume: zero padded, cropped back
])
def test_overlap_reproduces_pointwise_logits(pw, patch, stride, gaussian, sw_batch):
    x = vol()
    out = I.segment_with_patches_overlap(x, pw, patch_sz=patch, stride=stride, n_classes=3,
                                         use_gaussian_weights=gaussian, sigma_scale=0.125, sw_batch_size=sw_batch)
    assert out.shape == (1, 3, 30, 26, 21)
    assert np.allclose(out, pointwise_logits(x[0, 0])[None], rtol=1e-5, atol=1e-5)


def test_overlap_batch_of_two(pw):
    x = np.concatenate([vol(seed=1), vol(seed=2)])
    out = I.segment_with_patches_overlap(x, pw, patch_sz=16, stride=8, n_classes=3)
    for b in range(2):
        assert np.allclose(out[b], pointwise_logits(x[b, 0]), rtol=1e-5, atol=1e-5)


def test_overlap_sw_batch_size_does_not_change_result(pw):
    x = vol()
    kw = dict(patch_sz=[16, 16, 8], stride=[6, 7, 5], n_classes=3, use_gaussian_weights=True)
    a = I.segment_with_patches_overlap(x, pw, sw_batch_size=1, **kw)
    assert np.array_equal(a, I.segment_with_patches_overlap(x, pw, sw_batch_size=5, **kw))


def test_overlap_crop_border(pw):
    x = vol()
    out = I.segment_with_patches_overlap(x, pw, patch_sz=16, stride=6, crop=2, n_classes=3, bck=0)
    inner = (slice(None), slice(None), slice(2, -2), slice(2, -2), slice(2, -2))
    assert np.allclose(out[inner], pointwise_logits(x[0, 0])[None][inner], rtol=1e-5, atol=1e-5)
    # the outer `crop` voxels are never predicted: background probability 1
    assert np.all(out[0, 0, :2] == 1) and np.all(out[0, 1:, :2] == 0)


@pytest.mark.parametrize('convention', ['freesurfer', 'nibabel'])
def test_overlap_axis_conventions_round_trip(pw, convention):
    x = vol()
    out = I.segment_with_patches_overlap(x, pw, patch_sz=16, stride=8, n_classes=3, **{convention: True})
    assert out.shape == (1, 3, 30, 26, 21)
    assert np.allclose(out, pointwise_logits(x[0, 0])[None], rtol=1e-5, atol=1e-5)


def test_overlap_use_classes(pw):
    out = I.segment_with_patches_overlap(vol(), pw, patch_sz=16, stride=8, n_classes=3, use_classes=2)
    assert out.shape[1] == 2


def test_overlap_normalize_max(pw):
    x = vol()
    out = I.segment_with_patches_overlap(x, pw, patch_sz=16, stride=8, n_classes=3, normalize_max=True)
    assert np.allclose(out, pointwise_logits(x[0, 0] / x.max())[None], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------------------------------------
# whole volume
def test_whole_pads_to_quant_and_crops_back(pw):
    x = vol()
    out = I.segment_whole(x, pw, quant_size=16)
    assert out.shape == (1, 3, 30, 26, 21)
    assert np.allclose(out, pointwise_logits(x[0, 0])[None], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('trim_center', [False, True])
def test_whole_trim(pw, trim_center):
    x = vol((1, 1, 37, 21, 26))
    out = I.segment_whole(x, pw, quant_size=16, trim=True, trim_center=trim_center)
    assert out.shape == (1, 3, 37, 21, 26)
    # trimmed to (32, 16, 16), centred along the first two axes; the last axis only with trim_center
    z0 = 5 if trim_center else 0
    box = (slice(None), slice(None), slice(2, 34), slice(2, 18), slice(z0, z0 + 16))
    assert np.allclose(out[box], pointwise_logits(x[0, 0])[None][box], rtol=1e-6, atol=1e-6)
    outside = np.ones(out.shape[2:], bool)
    outside[box[2:]] = False
    assert np.all(out[0, 0][outside] == 1) and np.all(out[0, 1:][:, outside] == 0)  # background


def test_whole_with_tiled_groupnorm_session(gn_unet_model):
    x = vol((1, 1, 30, 26, 21))
    plain = I.segment_whole(x, onnxruntime.InferenceSession(gn_unet_model, providers=['CPUExecutionProvider']),
                            quant_size=8)
    tiled = I.segment_whole(x, TiledGroupNormSession(gn_unet_model, providers=['CPUExecutionProvider'], tile=16),
                            quant_size=8)
    assert np.abs(tiled - plain).max() / np.abs(plain).max() < 1e-4


def test_make_onnx_sessions(pointwise_model, gn_unet_model):
    s = I.make_onnx_sessions(pointwise_model, cpu=True)
    assert len(s) == 1 and isinstance(s[0], onnxruntime.InferenceSession)
    assert s[0].get_providers() == ['CPUExecutionProvider']
    t = I.make_onnx_sessions([gn_unet_model, gn_unet_model], cpu=True, threads=1, tiled=32)
    assert len(t) == 2 and all(isinstance(m, TiledGroupNormSession) for m in t)


# ---------------------------------------------------------------------------------------------------------
# geometry pre/post-processing (reorient, crop, resample)
MG = {'reorient': 'RAS', 'crop_foreground': True, 'resample': 'mindglide', 'normalize_mean_std_nonzero': True}
AFF_RPI_2MM = np.array([[-1.0, 0, 0, 90], [0, 1.0, 0, -120], [0, 0, -2.0, 70], [0, 0, 0, 1]])


def test_preprocess_without_keys_is_identity():
    d = phantom()
    out, ctx = I.preprocess_volume(d, AFF_1MM, {})
    assert out is d and ctx is None
    assert I.postprocess_labels(d, None, 3) is d
    assert I.postprocess_fuzzy(d[None], None) is not None


def test_reorient_crop_round_trip():
    pytest.importorskip('nibabel')
    d = phantom()  # (k, j, i)
    settings = {'reorient': 'RAS', 'crop_foreground': True}
    pre, ctx = I.preprocess_volume(d, AFF_RPI_2MM, settings)
    bb = I.foreground_bbox(d.transpose(2, 1, 0))
    assert pre.shape == tuple(e - s for s, e in zip(*bb))  # cropped to the bounding box (axes kept for RPI->RAS)
    assert pre.dtype == np.float32
    labels = (pre > 0).astype(np.uint8)
    back = I.postprocess_labels(labels, ctx, 2)
    assert back.shape == d.shape and np.array_equal(back, (d > 0).astype(np.uint8))


def test_resample_round_trip_shapes():
    pytest.importorskip('nibabel')
    d = phantom((12, 24, 28))  # 2 mm along k
    pre, ctx = I.preprocess_volume(d, AFF_RPI_2MM, MG)
    assert ctx['resample_flag'] and not ctx['anisotropy_flag']
    assert ctx['new_shape'][2] == 2 * ctx['crop_shape'][2]  # 2 mm -> 1 mm
    nz = pre != 0
    assert abs(pre[nz].mean()) < 1e-4 and abs(pre[nz].std() - 1) < 1e-3  # nonzero z-score
    labels = (pre > 0).astype(np.uint8) + (pre > 1).astype(np.uint8)
    back = I.postprocess_labels(labels, ctx, 3)
    assert back.shape == d.shape and set(np.unique(back)) <= {0, 1, 2}
    assert not back[d == 0].any() or (back[d == 0] > 0).mean() < 0.05  # labels stay (almost) inside the head
    prob = np.stack([1 - (pre > 0), (pre > 0)]).astype(np.float32)
    fz = I.postprocess_fuzzy(prob, ctx)
    assert fz.shape == (2, *d.shape)
    assert np.allclose(fz.sum(0), 1, atol=1e-6)  # outside the crop: background channel 1, others 0


def test_second_channel_uses_first_channel_geometry():
    pytest.importorskip('nibabel')
    d = phantom()
    pre, ctx = I.preprocess_volume(d, AFF_RPI_2MM, MG)
    other = np.zeros_like(d)  # the second channel would have an empty bbox on its own
    pre2, ctx2 = I.preprocess_volume(other, AFF_RPI_2MM, MG, ctx)
    assert ctx2 is ctx and pre2.shape == pre.shape


def test_spacing_float32():
    pytest.importorskip('nibabel')
    aff = AFF_1MM.copy()
    aff[:3, :3] = np.diag([1.0, 1.0, 1.0 + 1e-12])  # MINC-like rounding noise
    d = phantom()
    _, ctx = I.preprocess_volume(d, aff, {'resample': 'mindglide'})
    assert ctx['resample_flag']  # exact comparison: not 1 mm
    _, ctx = I.preprocess_volume(d, aff, {'resample': 'mindglide', 'spacing_float32': True})
    assert not ctx['resample_flag']


def test_keep_largest():
    seg = np.zeros((10, 10, 10), np.uint8)
    seg[1:5, 1:5, 1:5] = 1
    seg[7:9, 7:9, 7:9] = 2
    assert np.array_equal(I.keep_largest(seg, {}), seg)
    kept = I.keep_largest(seg, {'largest': True})
    assert kept[2, 2, 2] == 1 and not kept[8, 8, 8]
    assert seg[8, 8, 8] == 2  # input not modified


def test_keep_largest_connectivity_default_is_26():
    seg = np.zeros((8, 8, 8), np.uint8)
    seg[1:4, 1:4, 1:4] = 1
    seg[4, 4, 4] = 1  # touches the cube only through a corner
    assert I.keep_largest(seg, {'largest': True})[4, 4, 4] == 1
    assert I.keep_largest(seg, {'largest': True, 'largest_connectivity': 1})[4, 4, 4] == 0


# ---------------------------------------------------------------------------------------------------------
# window_layout 'legacy' + the default gaussian_map are the original sliding window, kept here as the reference
def original_gaussian_weights(patch_size, sigma_scale):
    sigma = [patch_size[i] * sigma_scale for i in range(3)]
    mesh = np.meshgrid(*[np.arange(patch_size[i]) for i in range(3)], indexing='ij')
    center = [(patch_size[i] - 1) / 2 for i in range(3)]
    dist = sum(((mesh[i] - center[i]) / sigma[i]) ** 2 for i in range(3))
    weights = np.exp(-0.5 * dist)
    weights = weights / np.max(weights)
    weights = np.clip(weights, max(np.min(weights), 1e-3), None)
    return weights[None, None].astype(np.float32)


def original_patches_overlap(dataset, model, crop, patch_sz, stride, n_classes, bck=0, use_gaussian_weights=False):
    """sliding window of the original apply_multi_model_onnx.py"""
    import math
    dsize = dataset.shape
    output_fuzzy = np.zeros((dsize[0], n_classes, *dsize[2:]), dtype=np.float32)
    output_weight = np.zeros((dsize[0], 1, *dsize[2:]), dtype=np.float32)
    patch_sz_ = [p - crop * 2 for p in patch_sz]
    out_roi = [d - crop * 2 for d in dsize[2:]]
    if use_gaussian_weights:
        gaussian_weights = original_gaussian_weights(patch_sz_, 0.25)
    else:
        gaussian_weights = np.ones((1, 1, *patch_sz_), dtype=np.float32)
    n_windows = 0
    for k in range(math.ceil(out_roi[0] / stride[0])):
        for l in range(math.ceil(out_roi[1] / stride[1])):
            for m in range(math.ceil(out_roi[2] / stride[2])):
                c = [k * stride[0] + crop, l * stride[1] + crop, m * stride[2] + crop]
                for i in range(3):
                    c[i] = max(min(c[i], dsize[i + 2] - patch_sz[i] + crop), crop)
                in_data = np.ascontiguousarray(dataset[:, :, c[0]-crop: c[0]-crop+patch_sz[0],
                                                       c[1]-crop: c[1]-crop+patch_sz[1],
                                                       c[2]-crop: c[2]-crop+patch_sz[2]])
                out = model.run(['seg'], {'scan': in_data})[0]
                weighted_output = out[:, :, crop: crop+patch_sz_[0], crop: crop+patch_sz_[1],
                                      crop: crop+patch_sz_[2]] * gaussian_weights
                sl = (slice(None), slice(None), slice(c[0], c[0]+patch_sz_[0]), slice(c[1], c[1]+patch_sz_[1]),
                      slice(c[2], c[2]+patch_sz_[2]))
                output_fuzzy[sl] += weighted_output
                output_weight[sl] += gaussian_weights
                n_windows += 1
    invalid = output_weight < 1e-3
    output_weight[invalid] = 1.0
    output_fuzzy = output_fuzzy / output_weight
    for q in range(output_fuzzy.shape[1]):
        output_fuzzy[:, q:q+1][invalid] = 0.0
    output_fuzzy[:, bck:bck+1][invalid] = 1.0
    return output_fuzzy, n_windows


@pytest.mark.parametrize('crop, stride, gaussian', [
    (0, [10, 10, 6], False),   # stride < patch and not aligned: the original loop repeats the last window
    (0, [10, 10, 6], True),
    (2, [12, 12, 4], True),    # stride = used core: no duplicates
    (2, [7, 9, 5], True),
])
def test_legacy_layout_is_the_original_sliding_window(gn_unet_model, crop, stride, gaussian):
    model = onnxruntime.InferenceSession(gn_unet_model, providers=['CPUExecutionProvider'])
    x = vol((1, 1, 30, 26, 22), seed=3)  # the GroupNorm U-Net output depends on the window, unlike the pointwise model
    patch = [16, 16, 8]
    ref, n_windows = original_patches_overlap(x, model, crop, patch, stride, 3, use_gaussian_weights=gaussian)
    kw = dict(patch_sz=patch, stride=stride, crop=crop, n_classes=3, use_gaussian_weights=gaussian)
    # default gaussian_map ('normalized') and sigma_scale 0.25
    assert np.array_equal(I.segment_with_patches_overlap(x, model, window_layout='legacy', **kw), ref)
    assert np.array_equal(I.segment_with_patches_overlap(x, model, window_layout='legacy',
                                                         gaussian_map='normalized', **kw), ref)
    starts = I.legacy_window_starts([s - 2 * crop for s in x.shape[2:]], [p - 2 * crop for p in patch], stride)
    assert np.prod([len(s) for s in starts]) == n_windows


def test_dense_layout_has_no_duplicate_windows_and_differs(gn_unet_model):
    model = onnxruntime.InferenceSession(gn_unet_model, providers=['CPUExecutionProvider'])
    x = vol((1, 1, 30, 26, 22), seed=3)
    kw = dict(patch_sz=[16, 16, 8], stride=[10, 10, 6], n_classes=3)
    legacy = I.legacy_window_starts(x.shape[2:], [16, 16, 8], [10, 10, 6])
    assert any(len(set(s)) < len(s) for s in legacy)  # this layout has duplicates
    a = I.segment_with_patches_overlap(x, model, window_layout='legacy', **kw)
    b = I.segment_with_patches_overlap(x, model, window_layout='dense', **kw)
    assert a.shape == b.shape and not np.array_equal(a, b)
    assert np.array_equal(b, I.segment_with_patches_overlap(x, model, **kw))  # dense is the default
    dense = I.window_starts(x.shape[2:], [16, 16, 8], [10, 10, 6])
    assert all(len(set(s)) == len(s) for s in dense)


def test_gaussian_map_is_independent_of_window_layout(gn_unet_model):
    model = onnxruntime.InferenceSession(gn_unet_model, providers=['CPUExecutionProvider'])
    x = vol((1, 1, 30, 26, 22), seed=3)
    kw = dict(patch_sz=[16, 16, 8], stride=[8, 8, 4], n_classes=3, use_gaussian_weights=True, sigma_scale=0.125)
    out = {(l, g): I.segment_with_patches_overlap(x, model, window_layout=l, gaussian_map=g, **kw)
           for l in ('dense', 'legacy') for g in ('normalized', 'separable')}
    assert np.array_equal(out[('dense', 'normalized')], I.segment_with_patches_overlap(x, model, **kw))  # defaults
    for l in ('dense', 'legacy'):  # same Gaussian up to scale; the 1e-3 floor sits at a different relative level
        assert not np.array_equal(out[(l, 'normalized')], out[(l, 'separable')])
    for g in ('normalized', 'separable'):  # stride divides the volume here: both layouts have the same windows
        assert np.array_equal(out[('dense', g)], out[('legacy', g)]) == \
            (I.legacy_window_starts(x.shape[2:], [16, 16, 8], [8, 8, 4]) == V.window_starts(x.shape[2:], [16, 16, 8], [8, 8, 4]))
    with pytest.raises(ValueError):
        I.segment_with_patches_overlap(x, model, gaussian_map='other', **kw)


def test_unknown_window_layout_raises(pw):
    with pytest.raises(ValueError):
        I.segment_with_patches_overlap(vol(), pw, patch_sz=16, stride=8, n_classes=3, window_layout='other')


def test_legacy_gaussian_weights():
    w = I.get_gaussian_weights([16, 16, 8], sigma_scale=0.25)
    assert w.shape == (1, 1, 16, 16, 8) and w.dtype == np.float32 and w.max() == 1.0
    assert np.array_equal(w, original_gaussian_weights([16, 16, 8], 0.25))


def test_resample_modes():
    d = vol((12, 10, 8))
    assert I.resample_mode({}) == I.resample_mode({'resample': None}) == I.resample_mode({'resample': 'legacy'}) == 'legacy'
    out, ctx = I.preprocess_volume(d, AFF_RPI_2MM, {'resample': 'legacy'})
    assert out is d and ctx is None  # nothing happens: resampling is left to uniformize / reference
    _, ctx = I.preprocess_volume(d, AFF_RPI_2MM, {'resample': 'mindglide'})
    assert ctx['resample_flag']
    with pytest.raises(ValueError):
        I.preprocess_volume(d, AFF_RPI_2MM, {'resample': 'grid'})
