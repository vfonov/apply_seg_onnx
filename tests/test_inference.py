"""apply_seg_onnx.inference building blocks: sliding window, whole volume, MindGlide pre/post-processing.

The pointwise model (conftest) makes every voxel's logits a known function of its intensity, so any
window layout / padding / weighting must reproduce them exactly (up to float32 rounding of the weighted mean).
"""
import numpy as np
import onnxruntime
import pytest

from apply_seg_onnx import inference as I
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
# MindGlide-style pre/post-processing
MG = {'reorient': 'RAS', 'crop_foreground': True, 'resample': 'mindglide', 'normalize_mean_std_nonzero': True}
AFF_RPI_2MM = np.array([[-1.0, 0, 0, 90], [0, 1.0, 0, -120], [0, 0, -2.0, 70], [0, 0, 0, 1]])


def test_preprocess_without_keys_is_identity():
    d = phantom()
    out, ctx = I.mindglide_preprocess(d, AFF_1MM, {})
    assert out is d and ctx is None
    assert I.mindglide_postprocess(d, None, 3) is d
    assert I.mindglide_postprocess_fuzzy(d[None], None) is not None


def test_reorient_crop_round_trip():
    pytest.importorskip('nibabel')
    d = phantom()  # (k, j, i)
    settings = {'reorient': 'RAS', 'crop_foreground': True}
    pre, ctx = I.mindglide_preprocess(d, AFF_RPI_2MM, settings)
    bb = I.foreground_bbox(d.transpose(2, 1, 0))
    assert pre.shape == tuple(e - s for s, e in zip(*bb))  # cropped to the bounding box (axes kept for RPI->RAS)
    assert pre.dtype == np.float32
    labels = (pre > 0).astype(np.uint8)
    back = I.mindglide_postprocess(labels, ctx, 2)
    assert back.shape == d.shape and np.array_equal(back, (d > 0).astype(np.uint8))


def test_resample_round_trip_shapes():
    pytest.importorskip('nibabel')
    d = phantom((12, 24, 28))  # 2 mm along k
    pre, ctx = I.mindglide_preprocess(d, AFF_RPI_2MM, MG)
    assert ctx['resample_flag'] and not ctx['anisotropy_flag']
    assert ctx['new_shape'][2] == 2 * ctx['crop_shape'][2]  # 2 mm -> 1 mm
    nz = pre != 0
    assert abs(pre[nz].mean()) < 1e-4 and abs(pre[nz].std() - 1) < 1e-3  # nonzero z-score
    labels = (pre > 0).astype(np.uint8) + (pre > 1).astype(np.uint8)
    back = I.mindglide_postprocess(labels, ctx, 3)
    assert back.shape == d.shape and set(np.unique(back)) <= {0, 1, 2}
    assert not back[d == 0].any() or (back[d == 0] > 0).mean() < 0.05  # labels stay (almost) inside the head
    prob = np.stack([1 - (pre > 0), (pre > 0)]).astype(np.float32)
    fz = I.mindglide_postprocess_fuzzy(prob, ctx)
    assert fz.shape == (2, *d.shape)
    assert np.allclose(fz.sum(0), 1, atol=1e-6)  # outside the crop: background channel 1, others 0


def test_second_channel_uses_first_channel_geometry():
    pytest.importorskip('nibabel')
    d = phantom()
    pre, ctx = I.mindglide_preprocess(d, AFF_RPI_2MM, MG)
    other = np.zeros_like(d)  # the second channel would have an empty bbox on its own
    pre2, ctx2 = I.mindglide_preprocess(other, AFF_RPI_2MM, MG, ctx)
    assert ctx2 is ctx and pre2.shape == pre.shape


def test_spacing_float32():
    pytest.importorskip('nibabel')
    aff = AFF_1MM.copy()
    aff[:3, :3] = np.diag([1.0, 1.0, 1.0 + 1e-12])  # MINC-like rounding noise
    d = phantom()
    _, ctx = I.mindglide_preprocess(d, aff, {'resample': 'mindglide'})
    assert ctx['resample_flag']  # exact comparison: not 1 mm
    _, ctx = I.mindglide_preprocess(d, aff, {'resample': 'mindglide', 'spacing_float32': True})
    assert not ctx['resample_flag']


def test_keep_largest():
    seg = np.zeros((10, 10, 10), np.uint8)
    seg[1:5, 1:5, 1:5] = 1
    seg[7:9, 7:9, 7:9] = 2
    assert np.array_equal(I.keep_largest(seg, {}), seg)
    kept = I.keep_largest(seg, {'largest': True})
    assert kept[2, 2, 2] == 1 and not kept[8, 8, 8]
    assert seg[8, 8, 8] == 2  # input not modified
