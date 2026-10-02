"""Tests of apply_seg_onnx without PyTorch (run with pytest, or directly: python tests/test_package.py)."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper, TensorProto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apply_seg_onnx import volume as V  # noqa: E402
from apply_seg_onnx.io import load_volume_np, save_volume  # noqa: E402
from apply_seg_onnx.nifti_io import have_nibabel  # noqa: E402
from apply_seg_onnx.onnx_tiled import TiledGroupNormSession  # noqa: E402


def _groupnorm(nodes, inits, x, c, groups, name):
    """GroupNorm in the torch.onnx.export form: Reshape -> InstanceNormalization -> Reshape(Shape) -> Mul -> Add"""
    rng = np.random.default_rng(len(inits))
    inits += [numpy_helper.from_array(np.array([0, groups, -1], np.int64), f'{name}_shape'),
              numpy_helper.from_array(np.ones(groups, np.float32), f'{name}_s'),
              numpy_helper.from_array(np.zeros(groups, np.float32), f'{name}_b'),
              numpy_helper.from_array(rng.uniform(0.5, 1.5, (c, 1, 1, 1)).astype(np.float32), f'{name}_gamma'),
              numpy_helper.from_array(rng.normal(0, 0.1, (c, 1, 1, 1)).astype(np.float32), f'{name}_beta')]
    nodes += [helper.make_node('Reshape', [x, f'{name}_shape'], [f'{name}_r1']),
              helper.make_node('InstanceNormalization', [f'{name}_r1', f'{name}_s', f'{name}_b'], [f'{name}_in'],
                               epsilon=1e-5),
              helper.make_node('Shape', [x], [f'{name}_xs']),
              helper.make_node('Reshape', [f'{name}_in', f'{name}_xs'], [f'{name}_r2']),
              helper.make_node('Mul', [f'{name}_r2', f'{name}_gamma'], [f'{name}_m']),
              helper.make_node('Add', [f'{name}_m', f'{name}_beta'], [f'{name}_y'])]
    return f'{name}_y'


def _conv(nodes, inits, x, cin, cout, name, act=True):
    rng = np.random.default_rng(len(inits) + 100)
    inits += [numpy_helper.from_array(rng.normal(0, 0.3, (cout, cin, 3, 3, 3)).astype(np.float32), f'{name}_w'),
              numpy_helper.from_array(rng.normal(0, 0.1, cout).astype(np.float32), f'{name}_bias')]
    nodes.append(helper.make_node('Conv', [x, f'{name}_w', f'{name}_bias'], [f'{name}_c'], pads=[1] * 6))
    if not act:
        return f'{name}_c'
    nodes.append(helper.make_node('LeakyRelu', [f'{name}_c'], [f'{name}_a'], alpha=0.01))
    return f'{name}_a'


def make_gn_unet(path):
    """two-level U-Net with GroupNorm before every Conv (layer order 'gcl', as WMH-SynthSeg)"""
    nodes, inits = [], []
    h = _conv(nodes, inits, _groupnorm(nodes, inits, 'scan', 1, 1, 'gn0'), 1, 4, 'c0')
    skip = _conv(nodes, inits, _groupnorm(nodes, inits, h, 4, 2, 'gn1'), 4, 4, 'c1')
    nodes.append(helper.make_node('MaxPool', [skip], ['pool'], kernel_shape=[2, 2, 2], strides=[2, 2, 2]))
    h = _conv(nodes, inits, _groupnorm(nodes, inits, 'pool', 4, 2, 'gn2'), 4, 8, 'c2')
    inits.append(numpy_helper.from_array(np.array([1, 1, 2, 2, 2], np.float32), 'up_scales'))
    nodes += [helper.make_node('Resize', [h, '', 'up_scales'], ['up'], mode='nearest'),
              helper.make_node('Concat', [skip, 'up'], ['cat'], axis=1)]
    h = _conv(nodes, inits, _groupnorm(nodes, inits, 'cat', 12, 4, 'gn3'), 12, 4, 'c3')
    _conv(nodes, inits, h, 4, 3, 'c4', act=False)
    nodes.append(helper.make_node('Identity', ['c4_c'], ['seg']))
    dims = ['N', 1, 'X', 'Y', 'Z']
    graph = helper.make_graph(nodes, 'gn_unet', [helper.make_tensor_value_info('scan', TensorProto.FLOAT, dims)],
                              [helper.make_tensor_value_info('seg', TensorProto.FLOAT, ['N', 3, 'X', 'Y', 'Z'])],
                              inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def test_tiled_groupnorm_matches_whole_volume():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'gn_unet.onnx')
        make_gn_unet(path)
        x = np.random.default_rng(0).normal(1.0, 1.0, (2, 1, 32, 48, 24)).astype(np.float32)
        ref = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider']).run(['seg'], {'scan': x})[0]
        tiled = TiledGroupNormSession(path, providers=['CPUExecutionProvider'], tile=16)
        assert len(tiled.gns) == 4
        out = tiled.run(['seg'], {'scan': x})[0]
        assert out.shape == ref.shape
        err = np.abs(out - ref).max() / np.abs(ref).max()
        assert err < 1e-5, err  # float32 summation order only


def test_volume_helpers():
    # MONAI dense_patch_slices layout: no duplicate windows, last window flush with the end
    assert list(V.window_starts([176], [128], [64])[0]) == [0, 48]
    w = V.monai_gaussian_importance([16, 16, 8], 0.125)
    assert w.shape == (16, 16, 8) and abs(w.min() - 1e-3) < 1e-7 and w.argmax() == np.ravel_multi_index((7, 7, 3), w.shape)
    assert np.array_equal(w, w[::-1, ::-1, ::-1])  # symmetric
    a = np.zeros((10, 12, 14), np.float32)
    a[2:5, 3:9, 4:6] = np.arange(36, dtype=np.float32).reshape(3, 6, 2) + 1
    n = V.nonzero_mean_std_normalize(a)
    assert np.all(n[a == 0] == 0) and abs(n[a != 0].mean()) < 1e-5


def test_resize():
    rng = np.random.default_rng(3)
    img = rng.normal(5, 1, (12, 13, 7))
    for order in (0, 1, 3):
        for mode in ('edge', 'constant'):
            assert np.array_equal(V._resize(img, img.shape, order, mode), img)  # same grid: identity
            out = V._resize(img, (25, 9, 14), order, mode)
            assert out.shape == (25, 9, 14) and out.dtype == img.dtype
            lo = 0 if mode == 'constant' else img.min()  # clipped to the input range (+ cval when used)
            assert out.min() >= lo and out.max() <= img.max()
    # 2x upsampling along an axis with grid_mode: nearest neighbour repeats every voxel twice
    assert np.array_equal(V._resize(img, (24, 13, 7), 0, 'edge'), np.repeat(img, 2, axis=0))


def test_minc_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'v.mnc')
        aff = np.array([[1.0, 0, 0, -10], [0, 1.5, 0, -20], [0, 0, 2.0, -30], [0, 0, 0, 1]])
        d = np.random.default_rng(1).integers(0, 30, (6, 7, 8)).astype(np.uint8)
        save_volume(fn, d, aff)
        d2, aff2 = load_volume_np(fn, dtype='uint8')
        assert np.array_equal(d, d2) and np.allclose(aff, aff2)


def test_nifti_roundtrip_or_clear_error():
    with tempfile.TemporaryDirectory() as tmp:
        fn = os.path.join(tmp, 'v.nii.gz')
        aff = np.diag([1.0, 1.0, 2.0, 1.0])
        d = np.random.default_rng(2).random((6, 7, 8)).astype(np.float32)
        if not have_nibabel:
            try:
                save_volume(fn, d, aff)
            except ImportError:
                return
            raise AssertionError('expected ImportError without nibabel')
        save_volume(fn, d, aff)
        d2, aff2 = load_volume_np(fn, dtype='float32')
        assert np.allclose(d, d2) and np.allclose(aff, aff2)


def test_no_torch_import():
    code = ('import sys; import apply_seg_onnx.inference, apply_seg_onnx.onnx_tiled; '
            'sys.exit(1 if "torch" in sys.modules else 0)')
    subprocess.run([sys.executable, '-c', code], check=True, env={**os.environ, 'PYTHONPATH': ROOT})


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
            print('PASS', k)
    print('ALL OK')
