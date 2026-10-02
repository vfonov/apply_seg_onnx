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
