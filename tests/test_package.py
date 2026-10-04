"""Package-level checks."""
import os
import subprocess
import sys

from conftest import ROOT

MODULES = ['inference', 'onnx_tiled', 'volume', 'postprocess', 'io', 'minc_io', 'nifti_io', 'geo']


def test_no_torch_import():
    """no module of the package may import torch (checked in a fresh interpreter)"""
    code = ('import sys, importlib; '
            + '; '.join(f'importlib.import_module("apply_seg_onnx.{m}")' for m in MODULES)
            + '; sys.exit(1 if "torch" in sys.modules else 0)')
    subprocess.run([sys.executable, '-c', code], check=True, env={**os.environ, 'PYTHONPATH': ROOT})


def test_no_torch_in_sources():
    """numpy/ONNX only: no source file imports torch or handles torch tensors"""
    import re
    pkg = os.path.join(ROOT, 'apply_seg_onnx')
    for name in sorted(os.listdir(pkg)):
        if name.endswith('.py'):
            src = open(os.path.join(pkg, name)).read()
            assert not re.search(r'^\s*(import|from)\s+torch\b', src, re.M), name
            assert 'volume_tensor' not in src, name


def test_save_minc_volume_needs_numpy_array(tmp_path):
    import numpy as np
    import pytest
    from apply_seg_onnx.minc_io import save_minc_volume
    with pytest.raises(TypeError):
        save_minc_volume(str(tmp_path / 'x.mnc'), [[[0.0]]], np.eye(4))
