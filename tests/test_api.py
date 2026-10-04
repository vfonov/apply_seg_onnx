"""Python interface (apply_seg_onnx.segment / segment_batch): file paths in and out, configuration as a dict."""
import copy
import os
import pathlib

import numpy as np
import pytest

import apply_seg_onnx
from apply_seg_onnx.api import apply_model_prefix
from apply_seg_onnx.io import load_volume_np, save_volume
from conftest import AFF_1MM, phantom, pointwise_labels

CONFIG = {'n_classes': 3, 'patch_sz': [16, 16, 16], 'stride': [8, 8, 8]}


def write_scan(path, seed=0):
    save_volume(str(path), phantom((20, 24, 28), seed), AFF_1MM)
    return load_volume_np(str(path), dtype='float32')[0]  # as the pipeline reads it


def labels(path):
    return load_volume_np(str(path), dtype='int32')[0]


def test_segment(tmp_path, pointwise_model):
    d = write_scan(tmp_path / 'in.mnc')
    config = {'models': [pointwise_model], **CONFIG}
    before = copy.deepcopy(config)
    out = apply_seg_onnx.segment(str(tmp_path / 'in.mnc'), str(tmp_path / 'seg.mnc'), config, cpu=True)
    assert out == str(tmp_path / 'seg.mnc')
    assert np.array_equal(labels(out), pointwise_labels(d))
    assert config == before  # the caller's dict is left alone


def test_segment_pathlib_prefix_and_single_model_name(tmp_path, pointwise_model):
    d = write_scan(tmp_path / 'in.mnc')
    config = {'models': os.path.basename(pointwise_model), **CONFIG}  # one name instead of a list
    prefix = os.path.dirname(pointwise_model) + os.sep
    out = apply_seg_onnx.segment(tmp_path / 'in.mnc', tmp_path / 'seg.mnc', config, model_prefix=prefix, cpu=True)
    assert out == str(tmp_path / 'seg.mnc')
    assert np.array_equal(labels(out), pointwise_labels(d))
    assert config['models'] == os.path.basename(pointwise_model)
    # a list of channels is one scan
    out2 = apply_seg_onnx.segment([tmp_path / 'in.mnc'], tmp_path / 'seg2.mnc', config, model_prefix=prefix, cpu=True)
    assert np.array_equal(labels(out2), pointwise_labels(d))


def test_segment_measure(tmp_path, pointwise_model):
    d = write_scan(tmp_path / 'in.mnc')
    config = {'models': [pointwise_model], 'labels_desc': ['one', 'two'], **CONFIG}
    apply_seg_onnx.segment(tmp_path / 'in.mnc', tmp_path / 'seg.mnc', config, cpu=True, measure=tmp_path / 'vol.csv')
    header, row = (tmp_path / 'vol.csv').read_text().strip().splitlines()[:2]
    assert 'one' in header and 'two' in header
    assert str(float(np.sum(pointwise_labels(d) == 1)))[:-2] in row  # 1 mm voxels: volume = count


def test_segment_batch(tmp_path, pointwise_model):
    ds = [write_scan(tmp_path / f'in{i}.mnc', seed=i) for i in range(3)]
    outs = apply_seg_onnx.segment_batch([tmp_path / f'in{i}.mnc' for i in range(3)],
                                        [tmp_path / f'seg{i}.mnc' for i in range(3)],
                                        {'models': [pointwise_model], **CONFIG}, cpu=True, minibatch_size=2)
    assert outs == [str(tmp_path / f'seg{i}.mnc') for i in range(3)]
    for d, o in zip(ds, outs):
        assert np.array_equal(labels(o), pointwise_labels(d))


def test_errors(tmp_path, pointwise_model):
    config = {'models': [pointwise_model], **CONFIG}
    with pytest.raises(FileNotFoundError):
        apply_seg_onnx.segment(tmp_path / 'missing.mnc', tmp_path / 'seg.mnc', config, cpu=True)
    with pytest.raises(FileNotFoundError):  # batch: errors are raised unless skip_errors
        apply_seg_onnx.segment_batch([tmp_path / 'missing.mnc'], [tmp_path / 'seg.mnc'], config, cpu=True)
    assert apply_seg_onnx.segment_batch([tmp_path / 'missing.mnc'], [tmp_path / 'seg.mnc'], config, cpu=True,
                                        skip_errors=True) == [str(tmp_path / 'seg.mnc')]
    assert not (tmp_path / 'seg.mnc').exists()
    with pytest.raises(ValueError, match='inputs'):
        apply_seg_onnx.segment_batch(['a.mnc', 'b.mnc'], ['a_seg.mnc'], config, cpu=True)
    with pytest.raises(ValueError, match='models'):
        apply_seg_onnx.segment('a.mnc', 'b.mnc', CONFIG, cpu=True)
    with pytest.raises(TypeError, match='dict'):
        apply_seg_onnx.segment('a.mnc', 'b.mnc', 'config.json', cpu=True)


def test_apply_model_prefix():
    c = apply_model_prefix({'models': pathlib.Path('m.onnx'), 'reference': 'ref.mnc'}, 'dir/')
    assert c == {'models': ['dir/m.onnx'], 'reference': 'dir/ref.mnc'}
    assert apply_model_prefix({'models': ['a', 'b']}) == {'models': ['a', 'b']}


def test_load_config(tmp_path):
    (tmp_path / 'c.json').write_text('{"models": ["m.onnx"], "whole": true}')
    assert apply_seg_onnx.load_config(tmp_path / 'c.json') == {'models': ['m.onnx'], 'whole': True}
