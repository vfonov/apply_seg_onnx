"""Example configs shipped in examples/: readable, and every key is documented in the README."""
import glob
import json
import os
import re

import pytest

from conftest import ROOT

CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'examples', '*.json')))


def documented_keys():
    readme = open(os.path.join(ROOT, 'README.md')).read()
    reference = readme[readme.index('## Config file reference'):readme.index('### Order of the steps')]
    keys = set()
    for row in re.findall(r'^\| (`[^|]*) \|', reference, re.M):  # first column: `key` or `key`, `key`
        keys.update(re.findall(r'`([a-z0-9_]+)`', row))
    return keys


def test_examples_present():
    assert len(CONFIGS) >= 4


@pytest.mark.parametrize('path', CONFIGS, ids=os.path.basename)
def test_example_config(path):
    with open(path) as f:
        config = json.load(f)
    assert isinstance(config['models'], list) and config['models']
    assert all(os.path.basename(m) == m for m in config['models'])  # bare names: used with --model_prefix
    unknown = set(config) - documented_keys()
    assert not unknown, f"keys not in the README config reference: {sorted(unknown)}"
    assert os.path.basename(path) in open(os.path.join(ROOT, 'README.md')).read()


# ---------------------------------------------------------------------------------------------------------
# The example pipelines on a real scan. Models and data are not in the repository: they come from the
# example data archive (examples/README.md) and the tests are skipped without them.
DATA_ROOT = os.environ.get('APPLY_SEG_ONNX_EXAMPLES', os.path.join(ROOT, 'examples'))
MODELS = os.path.join(DATA_ROOT, 'models')
SCAN_NAME = 'subject43_1_t2w'
SCAN = os.path.join(DATA_ROOT, 'data', SCAN_NAME + '.mnc')

# agreement with the reference outputs of the archive (made on a GPU in fp32); the margins cover
# CPU / GPU / TF32 rounding
MIN_LABEL_AGREEMENT = 0.9999  # fraction of voxels with the same label
MAX_MEAN_DIFF = 0.01         # regression outputs: mean and maximum absolute difference, intensity range 0..128
MAX_DIFF = 0.5


def example_files(path):
    """(config, missing files) of an example"""
    with open(path) as f:
        config = json.load(f)
    needed = [SCAN] + [os.path.join(MODELS, m) for m in config['models']]
    return config, [f for f in needed if not os.path.exists(f)]


@pytest.mark.examples
@pytest.mark.parametrize('path', CONFIGS, ids=os.path.basename)
def test_example_on_scan(path, tmp_path):
    import numpy as np
    import apply_seg_onnx
    from apply_seg_onnx.io import load_volume_np
    from conftest import has_cuda

    config, missing = example_files(path)
    if missing:
        pytest.skip(f"example data not installed: {', '.join(os.path.relpath(f, DATA_ROOT) for f in missing)}")
    name = os.path.splitext(os.path.basename(path))[0]
    continuous = config.get('continuous', False)
    dtype = 'float32' if continuous else 'int32'
    cpu = not has_cuda() or bool(os.environ.get('APPLY_SEG_ONNX_TEST_CPU'))

    out = apply_seg_onnx.segment(SCAN, tmp_path / f'{name}.mnc', config, model_prefix=MODELS + os.sep, cpu=cpu,
                                 measure=tmp_path / 'volumes.csv')
    result, aff = load_volume_np(out, dtype=dtype)
    scan, scan_aff = load_volume_np(SCAN, dtype='float32')
    assert np.all(np.isfinite(result))
    if continuous:
        # saved on the 1 mm grid: the same field of view
        assert config.get('save_uniformized') and result.shape != scan.shape
        assert result.min() >= config['output_clip'][0] - 64 and result.max() <= config['output_clip'][1] + 64
        assert result.max() > 64  # an image, not zeros (the unsharp mask goes beyond the clip range)
    else:
        assert result.shape == scan.shape and np.allclose(aff, scan_aff, atol=1e-4)
        allowed = config.get('label_values', list(range(config['n_classes'])))
        found = np.unique(result)
        assert set(found.tolist()) <= set(allowed)
        assert len(found) > len(allowed) // 2  # most structures are found
        rows = open(tmp_path / 'volumes.csv').read().strip().splitlines()
        assert len(rows) == 2 and rows[0].count(',') >= len(config['labels_desc'])

    reference = os.path.join(DATA_ROOT, 'data', 'reference', f'{SCAN_NAME}_{name}.mnc')
    if not os.path.exists(reference):
        return
    expected = load_volume_np(reference, dtype=dtype)[0]
    assert result.shape == expected.shape
    if continuous:
        d = np.abs(result - expected)
        assert d.mean() <= MAX_MEAN_DIFF and d.max() <= MAX_DIFF, (d.mean(), d.max())
    else:
        agreement = np.mean(result == expected)
        assert agreement >= MIN_LABEL_AGREEMENT, agreement
