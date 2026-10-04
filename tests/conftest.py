"""Shared fixtures: small synthetic ONNX models and volumes (no PyTorch needed)."""
import json
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import helper, numpy_helper, TensorProto

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pointwise model: logits_c = W[c] * x + B[c]. The label of a voxel depends only on its intensity:
# argmax is class 0 for x < 1, class 1 for 1 < x < 2, class 2 for x > 2 (no ties for non-integer x).
POINTWISE_W = np.array([0.0, 1.0, 2.0], np.float32)
POINTWISE_B = np.array([0.0, -1.0, -3.0], np.float32)


def pointwise_labels(x):
    """expected labels of the pointwise model"""
    return np.argmax(POINTWISE_W[:, None] * np.ravel(x)[None] + POINTWISE_B[:, None], axis=0).reshape(np.shape(x))


def pointwise_logits(x):
    """expected logits (C, ...) of the pointwise model"""
    return POINTWISE_W.reshape(-1, *([1] * np.ndim(x))) * x[None] + POINTWISE_B.reshape(-1, *([1] * np.ndim(x)))


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def make_pointwise_model(path, out_name='seg'):
    inits = [numpy_helper.from_array(POINTWISE_W.reshape(3, 1, 1, 1, 1), 'w'),
             numpy_helper.from_array(POINTWISE_B, 'b')]
    nodes = [helper.make_node('Conv', ['scan', 'w', 'b'], [out_name])]
    graph = helper.make_graph(
        nodes, 'pointwise',
        [helper.make_tensor_value_info('scan', TensorProto.FLOAT, ['N', 1, 'X', 'Y', 'Z'])],
        [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, ['N', 3, 'X', 'Y', 'Z'])], inits)
    return _save(graph, path)


def _groupnorm(nodes, inits, x, c, groups, name):
    """GroupNorm in the form written by common exporters: Reshape -> InstanceNormalization -> Reshape(Shape) -> Mul -> Add"""
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


def make_gn_unet(path, extra_node=None):
    """two-level U-Net with GroupNorm before every Conv (layer order 'gcl', as WMH-SynthSeg), 3 output classes.
    extra_node: op type inserted before the output (to test unsupported ops)"""
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
    if extra_node:
        nodes.append(helper.make_node(extra_node, ['c4_c'], ['seg']))
    else:
        nodes.append(helper.make_node('Identity', ['c4_c'], ['seg']))
    graph = helper.make_graph(
        nodes, 'gn_unet',
        [helper.make_tensor_value_info('scan', TensorProto.FLOAT, ['N', 1, 'X', 'Y', 'Z'])],
        [helper.make_tensor_value_info('seg', TensorProto.FLOAT, ['N', 3, 'X', 'Y', 'Z'])], inits)
    return _save(graph, path)


@pytest.fixture(scope='session')
def models_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('models')


@pytest.fixture(scope='session')
def pointwise_model(models_dir):
    return make_pointwise_model(models_dir / 'pointwise.onnx')


@pytest.fixture(scope='session')
def gn_unet_model(models_dir):
    return make_gn_unet(models_dir / 'gn_unet.onnx')


@pytest.fixture(scope='session')
def cpu_session():
    def make(path):
        return onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
    return make


def phantom(shape=(20, 24, 28), seed=0):
    """(k,j,i)-ordered volume: zero background, a blob of intensities in (0.1, 3) avoiding integers"""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing='ij')
    inside = (z / 0.8) ** 2 + (y / 0.7) ** 2 + (x / 0.75) ** 2 < 1
    v = np.zeros(shape, np.float32)
    v[inside] = rng.uniform(0.1, 3.0, inside.sum()).astype(np.float32)
    v[inside & (np.abs(v - np.round(v)) < 0.02)] += 0.05  # keep away from the class boundaries 1 and 2
    return v


AFF_1MM = np.array([[1.0, 0, 0, -10], [0, 1.0, 0, -12], [0, 0, 1.0, -14], [0, 0, 0, 1]])


@pytest.fixture
def run_cli():
    """run `python -m apply_seg_onnx args...` with the package on PYTHONPATH; returns CompletedProcess"""
    def run(*args, check=True):
        env = {**os.environ, 'PYTHONPATH': ROOT + os.pathsep + os.environ.get('PYTHONPATH', '')}
        r = subprocess.run([sys.executable, '-m', 'apply_seg_onnx', *map(str, args)], env=env,
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise AssertionError(f"apply_seg_onnx failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        return r
    return run


@pytest.fixture
def write_config(tmp_path):
    def write(settings, name='config.json'):
        p = tmp_path / name
        p.write_text(json.dumps(settings))
        return str(p)
    return write


def has_cuda():
    return 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
