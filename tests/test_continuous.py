"""Regression (`continuous`) pipeline keys: pad_center, normalize_min_max, input_dtype, uniformize_method,
output_scale / output_clip, unsharp_sigma / unsharp_amount.

The block model replaces every 2x2x2 block by its mean, so its output depends on where the zero padding
is put: an odd shift of the volume changes the blocks.
"""
import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import helper, numpy_helper, TensorProto
from scipy.ndimage import gaussian_filter

from apply_seg_onnx import inference as I
from apply_seg_onnx.io import load_volume_np, save_volume
from apply_seg_onnx.minc_io import uniformize_volume_grid
from apply_seg_onnx.postprocess import unsharp_mask
from conftest import AFF_1MM, phantom, pointwise_logits

AFF_ANISO = np.array([[1.0, 0, 0, -10], [0, 1.5, 0, -20], [0, 0, 2.0, -30], [0, 0, 0, 1]])
SHAPE = (18, 22, 27)  # padding to multiples of 16: 14, 10 and 5 voxels -> centred (7, 7), (5, 5), (2, 3)


@pytest.fixture(scope='module')
def block_model(tmp_path_factory):
    """scan (N, 1, X, Y, Z) -> scan_out: mean of every 2x2x2 block, repeated over the block"""
    path = str(tmp_path_factory.mktemp('models') / 'block.onnx')
    nodes = [helper.make_node('AveragePool', ['scan'], ['pooled'], kernel_shape=[2, 2, 2], strides=[2, 2, 2]),
             helper.make_node('ConvTranspose', ['pooled', 'ones'], ['scan_out'], kernel_shape=[2, 2, 2], strides=[2, 2, 2])]
    graph = helper.make_graph(
        nodes, 'block',
        [helper.make_tensor_value_info('scan', TensorProto.FLOAT, ['N', 1, 'X', 'Y', 'Z'])],
        [helper.make_tensor_value_info('scan_out', TensorProto.FLOAT, ['N', 1, 'X', 'Y', 'Z'])],
        [numpy_helper.from_array(np.ones((1, 1, 2, 2, 2), np.float32), 'ones')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture(scope='module')
def pw(pointwise_model):
    return onnxruntime.InferenceSession(pointwise_model, providers=['CPUExecutionProvider'])


def block_mean(x):
    """expected output of the block model for an even-sized 3D array"""
    s = x.shape
    m = x.reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5))
    return np.repeat(np.repeat(np.repeat(m, 2, axis=0), 2, axis=1), 2, axis=2)


def block_padded(d, widths):
    """block model on d zero padded by widths [(before, after)] * 3, padding removed"""
    out = block_mean(np.pad(d, widths))
    return out[tuple(slice(b, out.shape[i] - a) for i, (b, a) in enumerate(widths))]


def end_pad(shape, quant=16):
    return [(0, -n % quant) for n in shape]


def run(tmp_path, model, shape=SHAPE, aff=AFF_1MM, **kw):
    """write a phantom, run the continuous pipeline; returns (input as loaded, output, output affine)"""
    fn, out = str(tmp_path / 'in.mnc'), str(tmp_path / 'out.mnc')
    save_volume(fn, phantom(shape, 0), aff)
    d = load_volume_np(fn, dtype='float64')[0]
    settings = {'models': [model], 'continuous': True, 'whole': True, 'quant_size': 16, **kw}
    assert I.segment_with_onnx([fn], out, settings) == out
    o, oa = load_volume_np(out, dtype='float32')
    return d, o, oa


def close(a, b, tol=2e-4):
    """the output is stored with a limited number of levels: compare relative to the range"""
    return a.shape == b.shape and np.abs(a - b).max() <= tol * max(np.ptp(b), 1e-6)


# ---------------------------------------------------------------------------------------------------------
# helpers
def test_center_pad_widths():
    assert I.center_pad_widths((18, 22, 27), 16) == [(7, 7), (5, 5), (2, 3)]  # odd difference: the rest goes after
    assert I.center_pad_widths((32, 16, 48), 16) == [(0, 0), (0, 0), (0, 0)]
    x = np.random.default_rng(0).random((2, 1, 18, 22, 27))
    widths = I.center_pad_widths(x.shape[2:], 16)
    padded = np.pad(x, [(0, 0), (0, 0)] + widths)
    assert padded.shape == (2, 1, 32, 32, 32)
    assert np.array_equal(I.undo_center_pad(padded, widths), x)


def test_scale_clip_output():
    x = np.array([-0.2, 0.1, 0.4, 0.9], np.float32)
    assert I.scale_clip_output(x) is x
    assert np.allclose(I.scale_clip_output(x, scale=10), [-2, 1, 4, 9])
    assert np.allclose(I.scale_clip_output(x, clip=[0, 0.5]), [0, 0.1, 0.4, 0.5])
    assert np.allclose(I.scale_clip_output(x, scale=10, clip=[0, 5]), [0, 1, 4, 5])  # scaled first, then clipped


def test_unsharp_mask():
    v = np.random.default_rng(1).random((9, 10, 11)).astype(np.float32)
    s = unsharp_mask(v, 1.5, 0.5)
    assert s.dtype == v.dtype and s.shape == v.shape
    assert np.allclose(s, v + 0.5 * (v - gaussian_filter(v, 1.5)), atol=1e-6)
    assert np.array_equal(unsharp_mask(v, 1.5, 0.0), v)
    assert np.allclose(unsharp_mask(np.full((6, 6, 6), 3.0, np.float32), 2.0), 3.0)  # nothing to sharpen


def test_uniformize_volume_grid_isotropic():
    d = np.random.default_rng(2).random((5, 6, 7)).astype(np.float32)
    u, ua = uniformize_volume_grid(d, AFF_1MM, step=1.0)
    assert u.dtype == d.dtype and u.shape == d.shape and np.allclose(ua, AFF_1MM)
    # same grid: only the blur of 0.25 voxels is left
    assert np.allclose(u, gaussian_filter(d.astype(np.float64), 0.25), atol=1e-6)


def test_uniformize_volume_grid_anisotropic():
    z = np.arange(5, dtype=np.float64)
    d = np.broadcast_to(z[:, None, None], (5, 6, 7)).copy()  # ramp along z (2 mm), constant along y (1.5 mm) and x
    u, ua = uniformize_volume_grid(d, AFF_ANISO, step=1.0)
    assert u.shape == (10, 9, 7)  # (z, y, x): 5 x 2 mm, 6 x 1.5 mm, 7 x 1 mm
    assert np.allclose(np.sqrt((ua[:3, :3] ** 2).sum(axis=0)), 1.0)
    # voxel edges stay in place: the first sample sits a quarter of an old voxel before the first centre
    assert np.allclose(ua[:3, 3], AFF_ANISO[:3, 3] + [0, -0.25, -0.5])
    # no blur along the upsampled axis: linear interpolation, clamped at both ends
    expected = np.clip(-0.25 + 0.5 * np.arange(10), 0, 4)
    assert np.allclose(u, np.broadcast_to(expected[:, None, None], u.shape))


@pytest.mark.parametrize('key, value', [('uniformize_method', 'cubic'), ('input_dtype', 'float16')])
def test_invalid_values(tmp_path, block_model, key, value):
    with pytest.raises(ValueError, match=key):
        run(tmp_path, block_model, **{key: value})


# ---------------------------------------------------------------------------------------------------------
# normalize_min_max
def test_whole_normalize_min_max(pw):
    x = np.random.default_rng(3).uniform(1, 3, (1, 1, 30, 26, 21))  # float64
    out = I.segment_whole(x, pw, quant_size=16, normalize_min_max=True)
    # the padding is part of the volume: its zeros are the minimum
    assert np.allclose(out, pointwise_logits((x[0, 0] / x.max()).astype(np.float32))[None], rtol=1e-6, atol=1e-6)
    x32 = x.astype(np.float32)
    out = I.segment_whole(x32, pw, quant_size=2, normalize_min_max=True)  # (30, 26, 21) -> 22: still padded
    assert np.allclose(out, pointwise_logits(x32[0, 0] / x32.max())[None], rtol=1e-6, atol=1e-6)
    out = I.segment_whole(x32[..., :20], pw, quant_size=2, normalize_min_max=True)  # no padding: minimum removed
    v = x32[0, 0, ..., :20] - x32[..., :20].min()
    assert np.allclose(out, pointwise_logits(v / v.max())[None], rtol=1e-6, atol=1e-6)


def test_normalize_min_max_has_lowest_precedence(pw):
    x = np.random.default_rng(4).uniform(1, 3, (1, 1, 16, 16, 16)).astype(np.float32)
    a = I.segment_whole(x, pw, quant_size=16, normalize_max=True, normalize_min_max=True)
    assert np.array_equal(a, I.segment_whole(x, pw, quant_size=16, normalize_max=True))


def test_overlap_normalize_min_max(pw):
    x = np.random.default_rng(5).uniform(1, 3, (1, 1, 30, 26, 21)).astype(np.float32)
    out = I.segment_with_patches_overlap(x, pw, patch_sz=16, stride=8, n_classes=3, normalize_min_max=True)
    v = x[0, 0] - x.min()
    assert np.allclose(out, pointwise_logits(v / v.max())[None], rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------------------------------------
# pipeline
def test_pad_center(tmp_path, block_model):
    d, o, aff = run(tmp_path, block_model)
    assert close(o, block_padded(d, end_pad(SHAPE)))  # default: padded at the end
    assert np.allclose(aff, AFF_1MM, atol=1e-5)
    d, c, _ = run(tmp_path, block_model, pad_center=True)
    expected = block_padded(d, I.center_pad_widths(SHAPE, 16))
    assert close(c, expected)
    assert not close(expected, block_padded(d, end_pad(SHAPE)), tol=0.01)  # the two really differ


def test_pad_center_tta(tmp_path, block_model):
    # both passes see the same padded volume and the block model commutes with flipping it:
    # the average is the single pass (padding the flipped copy instead would move the blocks along x)
    d, o, _ = run(tmp_path, block_model, pad_center=True, augment_tta={'flip_x': []})
    assert close(o, block_padded(d, I.center_pad_widths(SHAPE, 16)))


def test_pad_center_ignored_with_trim(tmp_path, block_model):
    d, o, _ = run(tmp_path, block_model, shape=(18, 22, 20), trim=True, pad_center=True)
    box = (slice(1, 17), slice(3, 19), slice(0, 16))  # the trimmed box, nothing is padded
    assert close(o[box], block_mean(d[box]))


def test_normalize_min_max_float64(tmp_path, block_model):
    d, o, _ = run(tmp_path, block_model, pad_center=True, normalize_min_max=True, input_dtype='float64')
    w = I.center_pad_widths(SHAPE, 16)
    assert close(o, block_padded(d, w) / d.max())  # padded with zeros: the minimum is 0
    _, o32, _ = run(tmp_path, block_model, pad_center=True, normalize_min_max=True)
    assert close(o32, o)


def test_output_scale_clip(tmp_path, block_model):
    d, o, _ = run(tmp_path, block_model, output_scale=3.0, output_clip=[0.5, 1.0])
    assert close(o, np.clip(3.0 * block_padded(d, end_pad(SHAPE)), 0.5, 1.0))
    d, o, _ = run(tmp_path, block_model, output_scale=3.0, output_clip=[0.5, 1.0], augment_tta={'flip_x': []},
                  pad_center=True)
    assert close(o, np.clip(3.0 * block_padded(d, I.center_pad_widths(SHAPE, 16)), 0.5, 1.0))


def test_unsharp(tmp_path, block_model):
    d, o, _ = run(tmp_path, block_model, unsharp_sigma=1.5, unsharp_amount=0.5)
    e = block_padded(d, end_pad(SHAPE)).astype(np.float32)
    assert close(o, unsharp_mask(e, 1.5, 0.5))
    assert not close(o, e, tol=0.01)
    _, o1, _ = run(tmp_path, block_model, unsharp_sigma=1.5)  # amount defaults to 1
    assert close(o1, unsharp_mask(e, 1.5, 1.0))


def test_uniformize_grid(tmp_path, block_model):
    shape = (9, 11, 16)
    d, o, aff = run(tmp_path, block_model, shape=shape, aff=AFF_ANISO, uniformize=1.0, uniformize_method='grid',
                    save_uniformized=True, input_dtype='float64')
    u, ua = uniformize_volume_grid(d, AFF_ANISO, step=1.0)
    assert u.shape == (18, 17, 16)
    assert close(o, block_padded(u, end_pad(u.shape)))
    assert np.allclose(aff, ua, atol=1e-5)
    # 1 mm input: still resampled (blurred), unlike the default method
    d, o, _ = run(tmp_path, block_model, uniformize=1.0, uniformize_method='grid', save_uniformized=True)
    u = uniformize_volume_grid(d, AFF_1MM, step=1.0)[0]
    assert close(o, block_padded(u, end_pad(SHAPE)))
    assert not np.allclose(u, d, atol=1e-5)
