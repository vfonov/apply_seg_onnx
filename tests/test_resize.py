"""apply_seg_onnx.volume._resize: voxel-grid resampling on scipy.ndimage.zoom (grid_mode)."""
import numpy as np
import pytest

from apply_seg_onnx.volume import _resize


@pytest.mark.parametrize('order', [0, 1, 3])
@pytest.mark.parametrize('mode', ['edge', 'constant'])
def test_identity_shape_dtype_clip(order, mode):
    img = np.random.default_rng(3).normal(5, 1, (12, 13, 7))
    assert np.array_equal(_resize(img, img.shape, order, mode), img)  # same grid: identity
    out = _resize(img, (25, 9, 14), order, mode)
    assert out.shape == (25, 9, 14) and out.dtype == img.dtype
    lo = 0 if mode == 'constant' else img.min()  # clipped to the input range (+ cval when it was used)
    assert out.min() >= lo and out.max() <= img.max()


def test_float32_preserved():
    img = np.random.default_rng(0).random((6, 7, 8)).astype(np.float32)
    assert _resize(img, (12, 7, 4), 3).dtype == np.float32
    assert _resize(img.astype(np.float16), (12, 7, 4), 1).dtype == np.float32


def test_nearest_upsampling_repeats_voxels():
    img = np.random.default_rng(1).random((12, 13, 7))
    # with grid_mode (voxel corners aligned) 2x nearest upsampling repeats every voxel twice
    assert np.array_equal(_resize(img, (24, 13, 7), 0, 'edge'), np.repeat(img, 2, axis=0))
    assert np.array_equal(_resize(img, (12, 13, 21), 0, 'constant'), np.repeat(img, 3, axis=2))


def test_nearest_downsampling():
    img = np.arange(8.0).reshape(8, 1, 1)
    # 8 -> 4: output voxel centres at input coordinates 0.5, 2.5, 4.5, 6.5 -> round half to even voxel
    out = _resize(img, (4, 1, 1), 0, 'edge').ravel()
    assert out.shape == (4,) and np.all(np.diff(out) == 2)


def test_linear_preserves_ramp_in_interior():
    n, m = 10, 25
    img = np.arange(n, dtype=np.float64)[:, None] * 2.0 + 1.0  # 2D ramp along axis 0
    out = _resize(img, (m, 1), 1, 'edge')[:, 0]
    # output voxel i is sampled at input coordinate (i + 0.5) * n / m - 0.5 (grid_mode)
    x = (np.arange(m) + 0.5) * n / m - 0.5
    interior = (x >= 0) & (x <= n - 1)
    assert np.allclose(out[interior], 2.0 * x[interior] + 1.0)
    assert np.allclose(out[~interior], np.clip(2.0 * x[~interior] + 1.0, 1.0, 2.0 * (n - 1) + 1.0))  # edge


def test_cubic_reproduces_constant_and_mask_range():
    assert np.allclose(_resize(np.full((5, 6, 7), 3.5), (9, 4, 11), 3), 3.5)
    mask = np.zeros((10, 10, 10))
    mask[3:7, 3:7, 3:7] = 1
    out = _resize(mask, (20, 20, 20), 1, 'edge')
    assert out.min() == 0 and out.max() == 1
    # 1D profile through the centre: voxels 3..6 -> 6..13 (sample points (i + 0.5) / 2 - 0.5 in [2.5, 6.5])
    assert np.array_equal(np.nonzero(out[10, 10] >= 0.5)[0], np.arange(6, 14))


def test_constant_mode_cval_widens_clip_range():
    img = np.ones((8, 1, 1))
    img[0] = img[7] = 100.0  # cubic undershoot next to the bright border voxels goes below cval = 0
    out = _resize(img, (16, 1, 1), 3, 'constant', cval=0.0)
    assert out.min() == 0.0 and (out < 1).any()  # clip range [1, 100] widened to [0, 100], not [1, 100]
    assert _resize(img, (16, 1, 1), 3, 'edge').min() == 1.0  # edge mode: plain input range
    # cval never reached by the interpolation: plain input range
    assert np.allclose(_resize(np.full((6, 6, 6), 5.0), (12, 12, 12), 3, 'constant'), 5.0)
