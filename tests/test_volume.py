"""apply_seg_onnx.volume: normalizations, crop/pad, MindGlide/MONAI-compatible helpers."""
import sys

import numpy as np
import pytest

from apply_seg_onnx import volume as V


# ---------------------------------------------------------------------------------------------------------
# intensity normalization
def test_autonorm_np():
    a = np.arange(1000, dtype=np.float64) + 5
    n = V.autonorm_np(a)
    assert n.min() == 0 and n.max() == 1
    assert np.isclose(n[500], 500 / np.percentile(a - 5, 99))
    assert np.array_equal(V.autonorm_np(np.full(10, 3.0)), np.zeros(10))  # constant: no division by 0


def test_maxnorm_np():
    a = np.array([0.0, 2.0, 4.0])
    assert np.array_equal(V.maxnorm_np(a), [0, 0.5, 1])
    assert np.array_equal(V.maxnorm_np(np.zeros(3)), np.zeros(3))


def test_mean_std_normalize_np():
    a = np.random.default_rng(0).normal(7, 3, 1000)
    n = V.mean_std_normalize_np(a)
    assert abs(n.mean()) < 1e-12 and abs(n.std() - 1) < 1e-12
    assert np.array_equal(V.mean_std_normalize_np(np.ones(4)), np.ones(4))


def test_nonzero_mean_std_normalize():
    a = np.zeros((10, 12, 14), np.float32)
    a[2:5, 3:9, 4:6] = np.arange(36, dtype=np.float32).reshape(3, 6, 2) + 1
    a[0, 0, 0] = -4  # negative values count as nonzero (MONAI nonzero=True selects != 0)
    n = V.nonzero_mean_std_normalize(a)
    assert n.dtype == np.float32
    assert np.all(n[a == 0] == 0)
    assert abs(n[a != 0].mean()) < 1e-5 and abs(n[a != 0].std() - 1) < 1e-5
    c = np.zeros((4, 4, 4), np.float32)
    c[1, 1, 1] = c[2, 2, 2] = 5  # constant nonzero part: std 0 -> divided by 1
    assert np.array_equal(V.nonzero_mean_std_normalize(c)[c != 0], [0, 0])
    assert np.array_equal(V.nonzero_mean_std_normalize(np.zeros((3, 3, 3))), np.zeros((3, 3, 3)))


# ---------------------------------------------------------------------------------------------------------
# crop / pad of (B, C, X, Y, Z) arrays
def test_cropvol_roundtrip():
    d = np.random.default_rng(1).random((2, 1, 10, 11, 12))
    c, orig = V.apply_cropvol(d, 2)
    assert c.shape == (2, 1, 6, 7, 8)
    r = V.undo_cropvol(c, orig, 2, bck=-1)
    assert np.array_equal(r[:, :, 2:-2, 2:-2, 2:-2], d[:, :, 2:-2, 2:-2, 2:-2])
    assert np.all(r[:, :, :2] == -1) and np.all(r[:, :, :, :, -2:] == -1)


def test_padvol_roundtrip():
    d = np.random.default_rng(2).random((1, 2, 5, 6, 7))
    p, orig = V.apply_padvol(d, 3, padfill=9.0)
    assert p.shape == (1, 2, 11, 12, 13) and p[0, 0, 0, 0, 0] == 9.0
    assert np.array_equal(V.undo_padvol(p, orig, 3), d)


def test_pad_to_size():
    a = np.ones((1, 1, 5, 8, 3))
    p, pads = V.pad_to_size(a, (8, 6, 6), axes=(2, 3, 4), value=0)
    assert p.shape == (1, 1, 8, 8, 6)
    assert pads == [(1, 2), (0, 0), (1, 2)]  # MONAI split: diff//2 before, rest after
    assert p.sum() == a.sum()
    assert np.array_equal(p[:, :, 1:6, :, 1:4], a)
    same, pads = V.pad_to_size(a, (5, 8, 3), axes=(2, 3, 4))
    assert same is a and pads == [(0, 0)] * 3


def test_parse_bracket_input():
    inputs, ref = V.parse_bracket_input('[t1.mnc,0.5,t2.mnc,-1e-2]')
    assert inputs == ['t1.mnc', 0.5, 't2.mnc', -0.01] and ref == 't1.mnc'
    assert V.parse_bracket_input('t1.mnc') == (None, None)


# ---------------------------------------------------------------------------------------------------------
# geometry
def test_affine_spacing():
    aff = np.array([[0, 0, -2.0, 1], [0, 1.5, 0, 2], [1.0, 0, 0, 3], [0, 0, 0, 1]])
    assert np.allclose(V.affine_spacing(aff), [1.0, 1.5, 2.0])


def test_foreground_bbox():
    a = np.zeros((10, 11, 12))
    a[2, 3:5, 7] = 1
    a[6, 4, 9] = 2
    a[8, 8, 8] = -1  # only > 0 is foreground
    assert V.foreground_bbox(a) == ([2, 3, 7], [7, 5, 10])
    assert V.foreground_bbox(np.zeros((3, 4, 5))) == ([0, 0, 0], [3, 4, 5])


def test_reorient_roundtrip():
    pytest.importorskip('nibabel')
    # RPI voxel axes (the samples' orientation): i -> R... x flipped, k -> inferior
    aff = np.array([[-1.0, 0, 0, 90], [0, 1.0, 0, -120], [0, 0, -2.0, 70], [0, 0, 0, 1]])
    a = np.random.default_rng(3).random((5, 6, 7))
    r, new_aff, tr = V.reorient_to(a, aff, 'RAS')
    assert r.shape == (5, 6, 7)
    assert np.all(np.diag(new_aff)[:3] > 0)  # RAS: positive diagonal
    # the same voxel keeps its world coordinate
    ijk = np.array([1, 2, 3, 1.0])
    rijk = np.array([5 - 1 - 1, 2, 7 - 1 - 3, 1.0])
    assert np.allclose(aff @ ijk, new_aff @ rijk)
    assert r[3, 2, 3] == a[1, 2, 3]
    assert np.array_equal(V.reorient_back(r, tr), a)


def test_reorient_permuted_axes():
    pytest.importorskip('nibabel')
    aff = np.array([[0, 0, 1.0, 0], [1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1]])  # axes order A I R
    a = np.random.default_rng(4).random((4, 5, 6))
    r, new_aff, tr = V.reorient_to(a, aff, 'RAS')
    assert r.shape == (6, 4, 5)
    assert np.allclose(np.abs(new_aff[:3, :3]), np.eye(3)) and np.all(np.diag(new_aff)[:3] > 0)
    assert np.array_equal(V.reorient_back(r, tr), a)


# MINC (standard order: positive steps, i,j,k = x,y,z, i.e. RAS): the transform follows from the axis codes alone
AXCODES = ['RAS', 'LPS', 'LPI', 'RPI', 'SAR', 'PIL', 'ASL']
AFF_MINC = np.array([[2.0, 0, 0, -10], [0, 1.5, 0, -20], [0, 0, 1.0, -30], [0, 0, 0, 1]])


@pytest.fixture
def no_nibabel(monkeypatch):
    monkeypatch.setitem(sys.modules, 'nibabel', None)  # any "import nibabel" now raises ImportError


@pytest.mark.parametrize('axcodes', AXCODES)
def test_reorient_minc_without_nibabel(no_nibabel, axcodes):
    a = np.random.default_rng(5).random((5, 6, 7))
    r, new_aff, tr = V.reorient_to(a, AFF_MINC, axcodes, minc=True)
    assert new_aff.dtype == np.float64
    # every voxel keeps its world coordinate
    for ijk in np.ndindex(*r.shape):
        old = np.linalg.solve(AFF_MINC, new_aff @ np.array([*ijk, 1.0]))
        assert r[ijk] == a[tuple(np.rint(old[:3]).astype(int))]
    # axis k of the result points along axcodes[k]
    for k, code in enumerate(axcodes):
        w = 'RAS'.index(code) if code in 'RAS' else 'LPI'.index(code)
        assert np.sign(new_aff[w, k]) == (1 if code in 'RAS' else -1)
    assert np.array_equal(V.reorient_back(r, tr, minc=True), a)


def test_reorient_minc_ras_is_identity(no_nibabel):
    a = np.random.default_rng(6).random((4, 5, 6))
    r, new_aff, tr = V.reorient_to(a, AFF_MINC, 'RAS', minc=True)
    assert np.array_equal(r, a) and np.array_equal(new_aff, AFF_MINC)
    assert np.array_equal(tr, [[0, 1], [1, 1], [2, 1]])


@pytest.mark.parametrize('axcodes', AXCODES)
def test_reorient_minc_matches_nibabel(axcodes):
    pytest.importorskip('nibabel')
    a = np.random.default_rng(7).random((5, 6, 7))
    r1, aff1, tr1 = V.reorient_to(a, AFF_MINC, axcodes, minc=True)
    r2, aff2, tr2 = V.reorient_to(a, AFF_MINC, axcodes)
    assert np.array_equal(r1, r2) and np.array_equal(tr1, tr2) and np.allclose(aff1, aff2, rtol=0, atol=1e-12)


@pytest.mark.parametrize('axcodes', ['RA', 'RAX', 'RRS', 'LAR', 'RASI', 'ras'])
def test_reorient_minc_invalid_axcodes(axcodes):
    with pytest.raises(ValueError):
        V.reorient_to(np.zeros((2, 3, 4)), AFF_MINC, axcodes, minc=True)


def test_reorient_nifti_without_nibabel(no_nibabel):
    with pytest.raises(ImportError, match='nibabel'):
        V.reorient_to(np.zeros((2, 3, 4)), AFF_MINC, 'RAS')
    with pytest.raises(ImportError, match='nibabel'):
        V.reorient_back(np.zeros((2, 3, 4)), np.array([[0, 1], [1, 1], [2, 1.0]]))

# ---------------------------------------------------------------------------------------------------------
# MindGlide resampling
@pytest.mark.parametrize('spacing, shape, expected', [
    ([1.0, 1.0, 1.0], [100, 120, 90], (False, [100, 120, 90], False)),
    ([2.0, 1.0, 1.0], [45, 120, 90], (True, [90, 120, 90], False)),        # ratio 2: isotropic path
    ([1.0, 1.0, 3.0], [100, 120, 30], (True, [100, 120, 90], True)),       # ratio 3: anisotropic path
    ([0.9, 0.9, 0.9], [101, 101, 101], (True, [90, 90, 90], False)),       # truncation of 90.9
    ([1.0000001, 1.0, 1.0], [10, 10, 10], (True, [10, 10, 10], False)),   # exact float comparison
])
def test_mindglide_resample_shape(spacing, shape, expected):
    assert V.mindglide_resample_shape(spacing, shape) == expected


@pytest.mark.parametrize('anis', [False, True])
def test_mindglide_resample_image(anis):
    img = np.random.default_rng(5).random((20, 18, 10)).astype(np.float32) * 10
    out = V.mindglide_resample_image(img, [30, 18, 25], anis)
    assert out.shape == (30, 18, 25)
    assert out.min() >= img.min() and out.max() <= img.max()  # clipped like skimage resize(clip=True)
    assert np.array_equal(V.mindglide_resample_image(img, list(img.shape), anis), img)


@pytest.mark.parametrize('anis', [False, True])
def test_mindglide_recover_labels(anis):
    lab = np.zeros((20, 18, 10), np.uint8)
    lab[4:12, 3:9, 2:6] = 1
    lab[12:18, 10:16, 5:9] = 2
    # same grid: unchanged
    assert np.array_equal(V.mindglide_recover_labels(lab, 3, list(lab.shape), anis), lab)
    # 2x along the last axis and back
    up = V.mindglide_recover_labels(lab, 3, [20, 18, 20], anis)
    assert up.shape == (20, 18, 20) and set(np.unique(up)) == {0, 1, 2}
    assert np.array_equal(V.mindglide_recover_labels(up, 3, [20, 18, 10], anis)[:, :, 1:-1], lab[:, :, 1:-1])


def test_mindglide_recover_labels_ties_lowest_class():
    lab = np.zeros((2, 1, 1), np.uint8)
    lab[0], lab[1] = 1, 2
    # downsampling 2 -> 1 voxel: both classes cover exactly 0.5 -> lowest label wins
    assert V.mindglide_recover_labels(lab, 3, [1, 1, 1], False)[0, 0, 0] == 1


def test_mindglide_recover_prob():
    p = np.random.default_rng(6).random((8, 9, 5)).astype(np.float32)
    assert np.array_equal(V.mindglide_recover_prob(p, [8, 9, 5], False), p)
    for anis in (False, True):
        out = V.mindglide_recover_prob(p, [16, 9, 10], anis)
        assert out.dtype == np.float32 and out.shape == (16, 9, 10)
        assert out.min() >= p.min() and out.max() <= p.max()


# ---------------------------------------------------------------------------------------------------------
# sliding window layout and weights
@pytest.mark.parametrize('n, r, step', [(176, 128, 64), (128, 128, 64), (129, 128, 64), (300, 128, 100),
                                        (65, 64, 32), (100, 64, 7), (64, 32, 32), (200, 64, 64)])
def test_window_starts_properties(n, r, step):
    s = V.window_starts([n], [r], [step])[0]
    assert s[0] == 0 and s[-1] == n - r                         # first at 0, last flush with the end
    assert len(set(s)) == len(s) and s == sorted(s)             # no duplicates
    assert all(b - a <= step for a, b in zip(s, s[1:]))         # gaps at most one step -> full coverage
    covered = np.zeros(n, bool)
    for st in s:
        covered[st:st + r] = True
    assert covered.all()


def test_window_starts_known():
    assert V.window_starts([176, 128, 97], [128, 128, 64], [64, 64, 32]) == [[0, 48], [0], [0, 32, 33]]
    assert V.monai_window_starts([176, 128, 97], [128, 128, 64], [0.5, 0.5, 0.5]) == [[0, 48], [0], [0, 32, 33]]


def test_monai_gaussian_importance():
    w = V.monai_gaussian_importance([16, 16, 8], 0.125)
    assert w.shape == (16, 16, 8) and w.dtype == np.float32
    assert abs(w.min() - 1e-3) < 1e-7  # floor
    assert w.argmax() == np.ravel_multi_index((7, 7, 3), w.shape)
    assert np.array_equal(w, w[::-1, ::-1, ::-1])  # symmetric
    odd = V.monai_gaussian_importance([5, 5, 5], 0.125)
    assert odd[2, 2, 2] == 1.0  # odd size: exact centre


@pytest.mark.reference
def test_window_starts_vs_monai():
    utils = pytest.importorskip('monai.data.utils')
    from monai.inferers.utils import _get_scan_interval
    rng = np.random.default_rng(7)
    for _ in range(50):
        roi = [int(x) for x in rng.integers(8, 64, 3)]
        size = [r + int(x) for r, x in zip(roi, rng.integers(0, 150, 3))]
        overlap = float(rng.choice([0.25, 0.5, 0.6, 0.75]))
        interval = _get_scan_interval(size, roi, 3, [overlap] * 3)
        slices = utils.dense_patch_slices(size, roi, interval)
        monai = sorted({tuple(s.start for s in sl) for sl in slices})
        ours = V.monai_window_starts(size, roi, [overlap] * 3)
        assert sorted(tuple(int(v) for v in t) for t in np.array(np.meshgrid(*ours, indexing='ij')).reshape(3, -1).T) \
            == monai, (size, roi, overlap)


@pytest.mark.reference
def test_gaussian_importance_vs_monai():
    utils = pytest.importorskip('monai.data.utils')
    for roi in ([128, 128, 64], [16, 15, 9]):
        m = utils.compute_importance_map(roi, mode='gaussian', sigma_scale=0.125, device='cpu').numpy()
        assert np.allclose(V.monai_gaussian_importance(roi, 0.125), m, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('values, expected', [
    ([0, 255], 'uint8'), ([0, 256], 'uint16'), ([65535], 'uint16'), ([0, 65536], 'uint32'),
    ([2**32 - 1], 'uint32'), ([-128, 127], 'int8'), ([-129, 0], 'int16'), ([-1, 40000], 'int32'), ([], 'uint8')])
def test_smallest_int_dtype(values, expected):
    assert V.smallest_int_dtype(np.array(values, np.int64)) == expected


@pytest.mark.parametrize('values', [[0, 2**32], [-2**31 - 1, 0]])
def test_smallest_int_dtype_beyond_32_bits(values):
    with pytest.warns(UserWarning, match='32 bits'):
        assert V.smallest_int_dtype(np.array(values, np.int64)) == np.float64
