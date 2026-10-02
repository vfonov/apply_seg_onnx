"""End to end: segment_with_onnx[_batched] on files, and the command line interface.

With the pointwise model and raw intensities the expected labels are pointwise_labels(input as loaded).
"""
import csv

import numpy as np
import pytest

from apply_seg_onnx import inference as I
from apply_seg_onnx.io import load_volume_np, save_volume
from apply_seg_onnx.nifti_io import have_nibabel
from conftest import AFF_1MM, phantom, pointwise_labels, pointwise_logits

needs_nibabel = pytest.mark.skipif(not have_nibabel, reason='nibabel not installed')


def softmax(x, axis=0):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@pytest.fixture
def scan(tmp_path):
    """write a phantom; returns (file name, data as loaded back, affine)"""
    def make(name='in.mnc', shape=(20, 24, 28), seed=0, aff=AFF_1MM):
        fn = str(tmp_path / name)
        save_volume(fn, phantom(shape, seed), aff)
        d, a = load_volume_np(fn, dtype='float32')  # MINC stores floats quantised: use what the pipeline reads
        return fn, d, a
    return make


def settings_for(model, **kw):
    return {'models': [model], 'n_classes': 3, 'patch_sz': [16, 16, 16], 'stride': [8, 8, 8], **kw}


def load_labels(fn):
    return load_volume_np(fn, dtype='int32')


# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('ext', ['mnc', pytest.param('nii.gz', marks=needs_nibabel)])
def test_single_scan(tmp_path, scan, pointwise_model, ext):
    fn, d, aff = scan(f'in.{ext}')
    out = str(tmp_path / f'seg.{ext}')
    assert I.segment_with_onnx([fn], out, settings_for(pointwise_model)) == out
    seg, seg_aff = load_labels(out)
    assert np.array_equal(seg, pointwise_labels(d))
    assert np.allclose(seg_aff, aff, atol=1e-5)


@pytest.mark.parametrize('whole', [False, True])
def test_whole_and_patches_agree(tmp_path, scan, pointwise_model, whole):
    fn, d, _ = scan()
    out = str(tmp_path / 'seg.mnc')
    I.segment_with_onnx([fn], out, settings_for(pointwise_model, whole=whole, quant_size=16))
    assert np.array_equal(load_labels(out)[0], pointwise_labels(d))


def test_batched_minibatch_and_measure(tmp_path, scan, pointwise_model):
    # minibatches: [a, b] same shape (stacked in one model call), [c, e] different shapes (run one by one),
    # [missing] skipped (a missing input skips its whole minibatch)
    scans = [scan('a.mnc', seed=1), scan('b.mnc', seed=2), scan('c.mnc', shape=(18, 22, 26), seed=3),
             scan('e.mnc', seed=4)]
    outs = [str(tmp_path / f'seg_{i}.mnc') for i in range(4)]
    csv_fn = str(tmp_path / 'vol.csv')
    I.segment_with_onnx_batched([s[0] for s in scans] + [str(tmp_path / 'missing.mnc')],
                                outs + [str(tmp_path / 'seg_missing.mnc')],
                                settings_for(pointwise_model, labels_desc=['one', 'two']),
                                minibatch_size=2, measure=csv_fn)
    for (_, d, _), o in zip(scans, outs):
        assert np.array_equal(load_labels(o)[0], pointwise_labels(d))
    with open(csv_fn) as f:
        rows = list(csv.DictReader(f))
    assert [r['scan'] for r in rows] == [s[0] for s in scans] + [str(tmp_path / 'missing.mnc')]
    for (_, d, _), r in zip(scans, rows):  # 1 mm^3 voxels
        lab = pointwise_labels(d)
        assert float(r['one']) == pytest.approx((lab == 1).sum()) and float(r['two']) == pytest.approx((lab == 2).sum())
    assert rows[-1]['one'] == 'nan'


def test_recover_keeps_existing_output(tmp_path, scan, pointwise_model):
    fn, d, aff = scan()
    out = str(tmp_path / 'seg.mnc')
    save_volume(out, np.full(d.shape, 2, np.uint8), aff)  # "previous run"
    I.segment_with_onnx_batched([fn], [out], settings_for(pointwise_model), recover=True)
    assert np.all(load_labels(out)[0] == 2)
    I.segment_with_onnx_batched([fn], [out], settings_for(pointwise_model))
    assert np.array_equal(load_labels(out)[0], pointwise_labels(d))


def test_error_is_skipped_unless_crash(tmp_path, scan, pointwise_model):
    fn, _, _ = scan()
    bad = settings_for(pointwise_model, patch_sz=[16, 16])  # wrong number of dimensions: fails in the batch loop
    out = str(tmp_path / 'seg.mnc')
    I.segment_with_onnx_batched([fn], [out], bad)  # error printed, batch skipped
    with pytest.raises(Exception):
        I.segment_with_onnx_batched([fn], [out], bad, crash=True)


def test_fuzzy_outputs_are_softmax(tmp_path, scan, pointwise_model):
    fn, d, _ = scan()
    out = str(tmp_path / 'seg.mnc')
    I.segment_with_onnx([fn], out, settings_for(pointwise_model), fuzzy=str(tmp_path / 'prob'))
    prob = np.stack([load_volume_np(str(tmp_path / f'prob_{c}.mnc'))[0] for c in range(3)])
    assert np.allclose(prob.sum(0), 1, atol=1e-3)  # stored as scaled 16-bit integers
    assert np.allclose(prob, softmax(pointwise_logits(d)), atol=1e-3)


def test_label_values(tmp_path, scan, pointwise_model):
    fn, d, _ = scan()
    out = str(tmp_path / 'seg.mnc')
    I.segment_with_onnx([fn], out, settings_for(pointwise_model, label_values=[0, 10, 20]))
    assert np.array_equal(load_labels(out)[0], np.array([0, 10, 20])[pointwise_labels(d)])


@pytest.mark.parametrize('ext', ['mnc', pytest.param('nii.gz', marks=needs_nibabel)])
@pytest.mark.parametrize('values, dtype', [([0, 2, 300], 'uint16'), ([0, 2, 70000], 'uint32')])
def test_label_values_above_255(tmp_path, scan, pointwise_model, ext, values, dtype):
    """the output uses the smallest unsigned type that holds the label values"""
    fn, d, _ = scan(f'in.{ext}')
    out = str(tmp_path / f'seg.{ext}')
    I.segment_with_onnx([fn], out, settings_for(pointwise_model, label_values=values))
    seg, _ = load_volume_np(out, dtype='native')
    assert seg.dtype == dtype
    assert np.array_equal(seg, np.array(values)[pointwise_labels(d)])


@pytest.mark.parametrize('flip_map', [[0, 1, 2], [1, 2, 0]])  # a swap would make two classes tie
def test_flip_tta(tmp_path, scan, pointwise_model, flip_map):
    fn, d, _ = scan()
    out = str(tmp_path / 'seg.mnc')
    I.segment_with_onnx([fn], out, settings_for(pointwise_model, augment_tta={'flip_x': flip_map}))
    # the pointwise model commutes with flipping: TTA averages softmax(l) with softmax(l)[flip_map]
    p = softmax(pointwise_logits(d))
    assert np.array_equal(load_labels(out)[0], np.argmax(0.5 * p + 0.5 * p[flip_map], axis=0))


def test_majority_vote(tmp_path, scan, pointwise_model, gn_unet_model):
    fn, d, _ = scan()
    out = str(tmp_path / 'seg.mnc')
    s = settings_for(pointwise_model, majority=True)
    s['models'] = [pointwise_model, gn_unet_model, pointwise_model]  # two of three votes for the pointwise labels
    I.segment_with_onnx([fn], out, s)
    assert np.array_equal(load_labels(out)[0], pointwise_labels(d))


def test_tiled_groupnorm_matches_plain_whole(tmp_path, scan, gn_unet_model):
    fn, _, _ = scan(shape=(24, 32, 40))
    outs = {}
    for tiled in (None, 16):
        outs[tiled] = str(tmp_path / f'seg_{tiled}.mnc')
        s = {'models': [gn_unet_model], 'n_classes': 3, 'whole': True, 'quant_size': 8, 'tiled_groupnorm': tiled}
        I.segment_with_onnx([fn], outs[tiled], s)
    a, b = load_labels(outs[None])[0], load_labels(outs[16])[0]
    assert (a == b).mean() > 0.999  # float32 rounding may flip an argmax tie here and there
    assert len(np.unique(a)) > 1


@needs_nibabel
def test_mindglide_pipeline_restores_geometry(tmp_path, scan, pointwise_model):
    aff = np.array([[-1.0, 0, 0, 90], [0, 1.0, 0, -120], [0, 0, -2.0, 70], [0, 0, 0, 1]])  # RPI, 2 mm slices
    fn, d, _ = scan('in.nii.gz', shape=(12, 24, 28), aff=aff)
    out = str(tmp_path / 'seg.nii.gz')
    s = settings_for(pointwise_model, reorient='RAS', crop_foreground=True, resample='mindglide', largest=True,
                     use_gaussian_weights=True, sigma_scale=0.125)
    I.segment_with_onnx([fn], out, s)
    seg, seg_aff = load_labels(out)
    assert seg.shape == d.shape and np.allclose(seg_aff, aff)
    assert set(np.unique(seg)) <= {0, 1, 2} and (seg > 0).any()
    assert not seg[d == 0].any() or (seg[d == 0] > 0).mean() < 0.05


def test_constant_channel(tmp_path, scan, pointwise_model):
    """an input given as a number is a constant-filled channel; the pointwise model only sees channel 0"""
    fn, d, _ = scan()
    loaded, _ = I.load_scan([fn, 2.5], {})
    assert loaded.shape == (1, 2, *d.shape) and np.all(loaded[0, 1] == 2.5)


# ---------------------------------------------------------------------------------------------------------
# command line
def test_cli_help(run_cli):
    assert '--config' in run_cli('--help').stdout


def test_cli_single_with_flags(tmp_path, scan, pointwise_model, run_cli):
    fn, d, _ = scan()
    out = tmp_path / 'seg.mnc'
    run_cli(fn, out, '--model', pointwise_model, '--cpu', '-n', '3', '--patch_sz', 16, 16, 16, '--stride', 8, 8, 8)
    assert np.array_equal(load_labels(str(out))[0], pointwise_labels(d))
    from minc2_simple import minc2_file
    m = minc2_file(str(out))
    hist = m.metadata()['']['history']
    m.close()
    assert str(out) in hist  # the command line is recorded


def test_cli_config_lists_and_measure(tmp_path, scan, pointwise_model, write_config, run_cli):
    scans = [scan('a.mnc', seed=1), scan('b.mnc', seed=2)]
    outs = [str(tmp_path / 'seg_a.mnc'), str(tmp_path / 'seg_b.mnc')]
    (tmp_path / 'in.txt').write_text('\n'.join(s[0] for s in scans) + '\n')
    (tmp_path / 'out.txt').write_text('\n'.join(outs) + '\n')
    cfg = write_config({**settings_for('pointwise.onnx'), 'labels_desc': {'1': 'one', '2': 'two'}})
    prefix = str(pointwise_model)[:-len('pointwise.onnx')]
    run_cli('--config', cfg, '--model_prefix', prefix, '--cpu', '--minibatch_size', 2,
            '--li', tmp_path / 'in.txt', '--lo', tmp_path / 'out.txt', '--measure', tmp_path / 'vol.csv')
    for (_, d, _), o in zip(scans, outs):
        assert np.array_equal(load_labels(o)[0], pointwise_labels(d))
    with open(tmp_path / 'vol.csv') as f:
        assert len(list(csv.DictReader(f))) == 2
