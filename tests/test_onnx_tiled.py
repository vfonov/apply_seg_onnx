"""apply_seg_onnx.onnx_tiled.TiledGroupNormSession: exact whole-volume GroupNorm inference in tiles."""
import numpy as np
import onnxruntime
import pytest

from apply_seg_onnx.onnx_tiled import TiledGroupNormSession
from conftest import has_cuda, make_gn_unet


def _rel_err(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.fixture(scope='module')
def volume():
    return np.random.default_rng(0).normal(1.0, 1.0, (1, 1, 32, 48, 24)).astype(np.float32)


@pytest.fixture(scope='module')
def reference(gn_unet_model, volume):
    return onnxruntime.InferenceSession(gn_unet_model, providers=['CPUExecutionProvider']).run(
        ['seg'], {'scan': volume})[0]


def test_graph_analysis(gn_unet_model):
    s = TiledGroupNormSession(gn_unet_model, providers=['CPUExecutionProvider'], tile=16)
    assert len(s.gns) == 4
    assert sorted(g['groups'] for g in s.gns.values()) == [1, 2, 2, 4]
    assert all(g['gamma'] is not None and g['beta'] is not None for g in s.gns.values())
    # the first GroupNorm (on the input) is applied inside stage 0; each stage ends at a GroupNorm input
    assert [[n.op_type for n in st['nodes']] for st in s.stages] == [
        ['Conv', 'LeakyRelu'], ['Conv', 'LeakyRelu', 'MaxPool'], ['Conv', 'LeakyRelu', 'Resize', 'Concat'],
        ['Conv', 'LeakyRelu', 'Conv', 'Identity']]
    assert max(s.scale.values()) == 2  # one MaxPool level
    assert all(st['halo'] >= 1 for st in s.stages)


@pytest.mark.parametrize('tile', [8, 16, [16, 32, 8], 64])
def test_matches_whole_volume(gn_unet_model, volume, reference, tile):
    s = TiledGroupNormSession(gn_unet_model, providers=['CPUExecutionProvider'], tile=tile)
    out = s.run(['seg'], {'scan': volume})[0]
    assert out.shape == reference.shape and out.dtype == np.float32
    # float32 rounding only; a wrong halo, tile offset or GroupNorm statistic gives errors of ~1e-1
    assert _rel_err(out, reference) < 1e-4


@pytest.mark.xfail(strict=True, reason='per-tile sums of x and x^2 are accumulated in float32 inside ORT '
                                       '(ReduceSum), so their rounding error grows with the tile size')
def test_precision_independent_of_tile_size(gn_unet_model, volume, reference):
    err = {t: _rel_err(TiledGroupNormSession(gn_unet_model, providers=['CPUExecutionProvider'], tile=t)
                       .run(['seg'], {'scan': volume})[0], reference) for t in (8, 64)}
    assert err[64] < 2 * err[8], err


def test_batch_is_per_sample(gn_unet_model, cpu_session):
    """GroupNorm statistics are per sample: a batch gives the same result as its elements one by one"""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, (1, 1, 16, 16, 16)), rng.normal(5, 3, (1, 1, 16, 16, 16))]).astype(np.float32)
    ref = cpu_session(gn_unet_model).run(['seg'], {'scan': x})[0]
    out = TiledGroupNormSession(gn_unet_model, providers=['CPUExecutionProvider'], tile=8).run(['seg'], {'scan': x})[0]
    assert _rel_err(out, ref) < 1e-4


def test_session_reused_for_other_shapes(gn_unet_model, cpu_session):
    s = TiledGroupNormSession(gn_unet_model, providers=['CPUExecutionProvider'], tile=16)
    for shape in [(16, 16, 16), (40, 24, 32)]:
        x = np.random.default_rng(2).normal(0, 1, (1, 1, *shape)).astype(np.float32)
        ref = cpu_session(gn_unet_model).run(['seg'], {'scan': x})[0]
        assert _rel_err(s.run(['seg'], {'scan': x})[0], ref) < 1e-4


def test_unsupported_op_rejected(models_dir):
    path = make_gn_unet(models_dir / 'gn_unet_softmax.onnx', extra_node='Softmax')
    with pytest.raises(AssertionError, match='unsupported op Softmax'):
        TiledGroupNormSession(path, providers=['CPUExecutionProvider'])


@pytest.mark.gpu
@pytest.mark.skipif(not has_cuda(), reason='CUDAExecutionProvider not available')
def test_cuda_matches_cpu_whole_volume(gn_unet_model, volume, reference):
    s = TiledGroupNormSession(gn_unet_model, providers=[('CUDAExecutionProvider', {'use_tf32': 0})], tile=16)
    assert _rel_err(s.run(['seg'], {'scan': volume})[0], reference) < 1e-4
