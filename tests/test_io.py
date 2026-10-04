"""apply_seg_onnx.io / minc_io / nifti_io / geo: file round-trips, affines, world-space resampling."""
import math
import os
import re

import numpy as np
import pytest

from apply_seg_onnx.geo import compose, decompose
from apply_seg_onnx.io import format_history, load_volume_np, save_volume
from apply_seg_onnx.io import is_minc, volume_extension
from apply_seg_onnx.minc_io import resample_volume, uniformize_volume
from apply_seg_onnx.nifti_io import have_nibabel


def rotation(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


AFFINES = {
    'axial_1mm': np.array([[1.0, 0, 0, -10], [0, 1.0, 0, -20], [0, 0, 1.0, -30], [0, 0, 0, 1]]),
    'anisotropic': np.array([[1.0, 0, 0, -10], [0, 1.5, 0, -20], [0, 0, 2.0, -30], [0, 0, 0, 1]]),
    'flipped_rpi': np.array([[-1.0, 0, 0, 90], [0, 1.0, 0, -120], [0, 0, -2.0, 70], [0, 0, 0, 1]]),
    'oblique': np.vstack([np.hstack([rotation(0.1, -0.2, 0.3) @ np.diag([0.9, 1.1, 1.3]), [[5], [-6], [7]]]),
                          [[0, 0, 0, 1]]]),
}


# ---------------------------------------------------------------------------------------------------------
# geo
@pytest.mark.parametrize('name', AFFINES)
def test_decompose_compose_roundtrip(name):
    aff = AFFINES[name]
    start, step, dir_cos = decompose(aff)
    assert np.allclose(dir_cos @ dir_cos.T, np.eye(3))  # orthonormal
    assert np.allclose(compose(start, step, dir_cos), aff)


# ---------------------------------------------------------------------------------------------------------
# MINC
INT_DTYPES = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32]


def minc_types(fn):
    from minc2_simple import minc2_file
    m = minc2_file(fn)
    t = (m.store_dtype(), m.representation_dtype())
    m.close()
    return t


@pytest.mark.parametrize('dtype', INT_DTYPES)
def test_minc_integer_roundtrip(tmp_path, dtype):
    info = np.iinfo(dtype)
    d = np.random.default_rng(1).integers(info.min, info.max, (6, 7, 8), endpoint=True).astype(dtype)
    d[0, 0, :2] = info.min, info.max  # full range
    fn = str(tmp_path / 'v.mnc')
    save_volume(fn, d, AFFINES['anisotropic'])
    name = np.dtype(dtype).name
    assert minc_types(fn) == (name, name)  # stored as is, without scaling
    for load_as in (name, 'native'):
        d2, aff2 = load_volume_np(fn, dtype=load_as)
        assert d2.dtype == dtype and d2.shape == (6, 7, 8) and np.array_equal(d, d2)
        assert np.allclose(aff2, AFFINES['anisotropic'])


def test_minc_bool_and_float16(tmp_path):
    import warnings
    fn = str(tmp_path / 'v.mnc')
    save_volume(fn, np.eye(4, dtype=bool)[None].repeat(3, 0), AFFINES['axial_1mm'])
    assert minc_types(fn) == ('uint8', 'uint8')
    d = np.random.default_rng(7).normal(0, 10, (3, 4, 5)).astype(np.float16)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # float16 -> float32 silently
        save_volume(fn, d, AFFINES['axial_1mm'])
    assert minc_types(fn) == ('int16', 'float32')
    d2, _ = load_volume_np(fn, dtype='native')
    d32 = d.astype(np.float32)  # tolerance in float32: 65535 overflows float16
    assert d2.dtype == np.float32 and np.abs(d2 - d32).max() <= (d32.max() - d32.min()) / 65535 * 1.01


@pytest.mark.parametrize('values, stored', [([0, 5, 300], 'uint16'), ([-3, 5], 'int8'), ([0, 2**32 - 1], 'uint32')])
def test_minc_int64_downcast(tmp_path, values, stored):
    fn = str(tmp_path / 'v.mnc')
    d = np.resize(np.array(values, np.int64), (3, 4, 5))
    save_volume(fn, d, AFFINES['axial_1mm'])
    assert minc_types(fn)[0] == stored
    assert np.array_equal(load_volume_np(fn, dtype='native')[0], d)


def test_minc_int64_beyond_32_bits_stored_as_double(tmp_path):
    fn = str(tmp_path / 'v.mnc')
    d = np.resize(np.array([0, 1, 2**33 + 1], np.int64), (3, 4, 5))
    with pytest.warns(UserWarning, match='32 bits'):
        save_volume(fn, d, AFFINES['axial_1mm'])
    assert minc_types(fn) == ('float64', 'float64')  # exact, not scaled short
    assert np.array_equal(load_volume_np(fn, dtype='native')[0], d)


def test_minc_unsupported_dtype(tmp_path):
    with pytest.raises(ValueError, match='unsupported dtype'):
        save_volume(str(tmp_path / 'v.mnc'), np.zeros((2, 2, 2), np.complex64), np.eye(4))


def test_minc_float_native_is_representation(tmp_path):
    fn = str(tmp_path / 'v.mnc')
    save_volume(fn, np.random.default_rng(8).random((3, 4, 5)).astype(np.float32), AFFINES['axial_1mm'])
    assert load_volume_np(fn, dtype='native')[0].dtype == np.float32
    assert load_volume_np(fn)[0].dtype == np.float64  # default unchanged


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_minc_float_roundtrip(tmp_path, dtype):
    d = np.random.default_rng(2).normal(100, 30, (5, 6, 7)).astype(dtype)
    fn = str(tmp_path / 'v.mnc')
    save_volume(fn, d, AFFINES['axial_1mm'])
    d2, _ = load_volume_np(fn)  # float64 by default
    assert d2.dtype == np.float64
    # stored as scaled 16-bit integers: error below one quantisation step of the value range
    assert np.abs(d2 - d).max() <= (d.max() - d.min()) / 65535 * 1.01


@pytest.mark.parametrize('name', AFFINES)
def test_minc_affine_roundtrip(tmp_path, name):
    fn = str(tmp_path / 'v.mnc')
    save_volume(fn, np.zeros((4, 5, 6), np.uint8), AFFINES[name])
    _, aff2 = load_volume_np(fn, as_byte=True)
    assert np.allclose(aff2, AFFINES[name], atol=1e-6)


def test_minc_voxel_order_matches_affine(tmp_path):
    """arrays are (k, j, i): the last array axis is the first voxel index of the affine (MINC x)"""
    d = np.zeros((4, 5, 6), np.uint8)
    d[3, 1, 2] = 7
    fn = str(tmp_path / 'v.mnc')
    save_volume(fn, d, AFFINES['anisotropic'])
    d2, aff = load_volume_np(fn, as_byte=True)
    k, j, i = np.argwhere(d2 == 7)[0]
    world = np.asarray(aff) @ np.array([i, j, k, 1.0])
    assert np.allclose(world[:3], [2 * 1.0 - 10, 1 * 1.5 - 20, 3 * 2.0 - 30])


def test_minc_history_and_metadata(tmp_path):
    from minc2_simple import minc2_file
    from apply_seg_onnx.minc_io import affine_to_dims
    ref = str(tmp_path / 'ref.mnc')
    m = minc2_file()
    m.define(affine_to_dims(np.eye(4), (3, 3, 3)), minc2_file.MINC2_UBYTE, minc2_file.MINC2_UBYTE)
    m.create(ref)
    m.write_attribute('patient', 'name', 'test subject')
    m.setup_standard_order()
    m.save_complete_volume(np.zeros((3, 3, 3), np.uint8))
    m.close()
    out = str(tmp_path / 'out.mnc')
    save_volume(out, np.ones((3, 3, 3), np.uint8), AFFINES['axial_1mm'], ref_fname=ref, history='second command')
    m = minc2_file(out)
    meta = m.metadata()
    m.close()
    assert meta['patient']['name'] == 'test subject'  # attribute groups copied from the reference
    assert meta['']['history'].rstrip('\x00').endswith('second command')


def test_format_history():
    h = format_history(['apply_seg_onnx', 'in.mnc', 'out.mnc'])
    assert re.match(r'^\w{3} \w{3} [ \d]\d \d\d:\d\d:\d\d \d{4}>>>apply_seg_onnx in.mnc out.mnc$', h), h


@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
def test_unsupported_extension(tmp_path):
    """anything that is not MINC goes to nibabel, which does not know every extension"""
    (tmp_path / 'v.nrrd').write_bytes(b'NRRD0004')
    with pytest.raises(ValueError, match='Unsupported file format'):
        load_volume_np(str(tmp_path / 'v.nrrd'))
    with pytest.raises(ValueError, match='Unsupported file format'):
        save_volume(str(tmp_path / 'w.nrrd'), np.zeros((2, 2, 2), np.float32), np.eye(4))


@pytest.mark.parametrize('name, minc, ext', [
    ('a.mnc', True, '.mnc'), ('a.minc', True, '.minc'), ('dir.x/a.mnc.gz', True, '.mnc.gz'),
    ('a.MINC.GZ', True, '.MINC.GZ'), ('a.nii', False, '.nii'), ('a.b.nii.gz', False, '.nii.gz'),
    ('a.img', False, '.img'), ('a.hdr', False, '.hdr'), ('a.mgz', False, '.mgz'), ('a.mnc.bak', False, '.bak'),
    ('noext', False, '')])
def test_format_by_extension(name, minc, ext):
    assert is_minc(name) is minc
    assert volume_extension(name) == ext


@pytest.mark.parametrize('ext', ['minc', 'mnc.gz', 'minc.gz'])
def test_minc_other_extensions(tmp_path, ext):
    fn = str(tmp_path / f'v.{ext}')
    d = np.random.default_rng(11).integers(0, 200, (5, 6, 7)).astype(np.uint8)
    save_volume(fn, d, AFFINES['anisotropic'], history='made by a test')
    with open(fn, 'rb') as f:
        assert (f.read(2) == b'\x1f\x8b') == ext.endswith('.gz')  # really gzipped
    d2, aff2 = load_volume_np(fn, dtype='uint8')
    assert np.array_equal(d, d2) and np.allclose(aff2, AFFINES['anisotropic'])
    assert sorted(os.listdir(tmp_path)) == [f'v.{ext}']  # no temporary file left


@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
def test_minc_output_with_other_reference(tmp_path):
    """metadata is copied from MINC references only: another format as reference is ignored"""
    d = np.random.default_rng(12).integers(0, 200, (5, 6, 7)).astype(np.uint8)
    save_volume(str(tmp_path / 'ref.nii.gz'), d, AFFINES['axial_1mm'])
    save_volume(str(tmp_path / 'v.mnc'), d, AFFINES['axial_1mm'], ref_fname=str(tmp_path / 'ref.nii.gz'))
    assert np.array_equal(load_volume_np(str(tmp_path / 'v.mnc'), dtype='uint8')[0], d)


@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
@pytest.mark.parametrize('ext', ['nii', 'img', 'hdr', 'mgz'])
def test_other_formats_roundtrip(tmp_path, ext):
    fn = str(tmp_path / f'v.{ext}')
    d = np.random.default_rng(13).random((6, 7, 8)).astype(np.float32)
    save_volume(fn, d, AFFINES['anisotropic'], history='made by a test')
    d2, aff2 = load_volume_np(fn, dtype='float32')
    assert np.array_equal(d, d2) and np.allclose(aff2, AFFINES['anisotropic'], atol=1e-5)


@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
def test_analyze_input(tmp_path):
    """Analyze 7.5 pair written by nibabel: read through either file of the pair, as the same data in NIfTI"""
    import nibabel as nib
    d = np.random.default_rng(14).integers(0, 200, (5, 6, 7)).astype(np.int16)  # (k, j, i)
    aff = np.diag([1.0, 1.5, 2.0, 1.0])
    nib.save(nib.AnalyzeImage(d.transpose(2, 1, 0).copy(), aff), str(tmp_path / 'a.img'))
    assert 'Analyze' in type(nib.load(str(tmp_path / 'a.img'))).__name__  # not a NIfTI pair
    for name in ('a.img', 'a.hdr'):
        a, aa = load_volume_np(str(tmp_path / name), dtype='int16')
        assert np.array_equal(a, d)
        assert np.allclose(np.abs(np.diag(aa)[:3]), [1.0, 1.5, 2.0])  # voxel sizes; Analyze has no full affine


# ---------------------------------------------------------------------------------------------------------
# NIfTI (optional nibabel)
@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
@pytest.mark.parametrize('name', AFFINES)
def test_nifti_roundtrip(tmp_path, name):
    fn = str(tmp_path / 'v.nii.gz')
    d = np.random.default_rng(3).random((6, 7, 8)).astype(np.float32)
    save_volume(fn, d, AFFINES[name], history='made by a test')
    d2, aff2 = load_volume_np(fn, dtype='float32')
    assert np.array_equal(d, d2)
    assert np.allclose(aff2, AFFINES[name], atol=1e-5)  # NIfTI stores the affine in float32


@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.int32])
def test_nifti_native_and_as_byte(tmp_path, dtype):
    fn = str(tmp_path / 'v.nii.gz')
    d = np.random.default_rng(9).integers(0, min(np.iinfo(dtype).max, 100000), (5, 6, 7)).astype(dtype)
    save_volume(fn, d, AFFINES['axial_1mm'])
    d2, _ = load_volume_np(fn, dtype='native')
    assert d2.dtype == dtype and np.array_equal(d2, d)
    assert load_volume_np(fn, as_byte=True)[0].dtype == np.uint8


@pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')
def test_minc_and_nifti_agree(tmp_path):
    """same array + affine written as MINC and NIfTI loads back identically (same (k,j,i) convention)"""
    d = np.random.default_rng(4).integers(0, 200, (5, 6, 7)).astype(np.uint8)
    for ext in ('mnc', 'nii.gz'):
        save_volume(str(tmp_path / f'v.{ext}'), d, AFFINES['oblique'])
    a, aa = load_volume_np(str(tmp_path / 'v.mnc'), dtype='uint8')
    b, ba = load_volume_np(str(tmp_path / 'v.nii.gz'), dtype='uint8')
    assert np.array_equal(a, b) and np.allclose(aa, ba, atol=1e-5)


def test_nifti_without_nibabel(tmp_path, monkeypatch):
    import apply_seg_onnx.nifti_io as nio
    monkeypatch.setattr(nio, 'have_nibabel', False)
    with pytest.raises(ImportError, match='nibabel'):
        save_volume(str(tmp_path / 'v.nii.gz'), np.zeros((2, 2, 2), np.float32), np.eye(4))
    with pytest.raises(ImportError, match='nibabel'):
        load_volume_np(str(tmp_path / 'v.nii.gz'))


# ---------------------------------------------------------------------------------------------------------
# world-space resampling
def test_resample_volume_identity_and_shift():
    d = np.random.default_rng(5).random((6, 7, 8))
    aff = AFFINES['anisotropic']
    same, out_aff = resample_volume(d, aff, d.shape, aff, order=1)
    assert np.allclose(same, d) and out_aff is aff
    shifted = aff.copy()
    shifted[0, 3] += 1.0  # output grid moved by one voxel along x (= last array axis)
    s, _ = resample_volume(d, aff, d.shape, shifted, order=0, fill=-1)
    assert np.array_equal(s[:, :, :-1], d[:, :, 1:]) and np.all(s[:, :, -1] == -1)


def test_uniformize_volume():
    d = np.random.default_rng(6).random((5, 6, 7))
    aff = AFFINES['axial_1mm']
    same, a = uniformize_volume(d, aff, step=1.0)
    assert same is d and a is aff  # already 1 mm: untouched
    u, ua = uniformize_volume(d, AFFINES['anisotropic'], step=1.0, order=0)
    start, step, _ = decompose(ua)
    assert np.allclose(step, 1.0)
    assert u.shape == (10, 9, 7)  # (k, j, i): z 5 x 2 mm, y 6 x 1.5 mm, x 7 x 1 mm
