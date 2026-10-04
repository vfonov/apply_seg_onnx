#!/usr/bin/env python3
"""
Batch-dynamic copy of the SynthSR ONNX model (tf2onnx export with a fixed batch of 1).

The export moves the single channel with Unsqueeze+Reshape at the input and Reshape+Squeeze at the
output, which only works for one scan. Both pairs are replaced by Transpose nodes and the batch
dimension is made symbolic, so the flipped copy of test-time augmentation can go in the same call.

The 2x nearest-neighbour upsampling is exported as Unsqueeze/Tile/Reshape chains; ONNX Runtime refuses
Tile outputs above 4 GiB, which two whole 1 mm scans exceed. Every Tile (one axis repeated twice) is
replaced by the equivalent Concat of the tensor with itself. Weights are untouched.

    python synthsr_make_batch_dynamic.py fixed_batch.onnx synthsr_v20_230130_batch.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as rt
from onnx import helper, numpy_helper


def make_batch_dynamic(model):
    g = model.graph
    nodes = list(g.node)
    produced = {o: n for n in nodes for o in n.output}
    in_name, out_name = g.input[0].name, g.output[0].name

    # head: scan (N,X,Y,Z,1) -> Unsqueeze -> Reshape -> (1,1,X,Y,Z)
    uns = next(n for n in nodes if n.op_type == 'Unsqueeze' and n.input[0] == in_name)
    rs_in = next(n for n in nodes if n.op_type == 'Reshape' and n.input[0] == uns.output[0])
    # tail: (1,1,X,Y,Z) -> Reshape -> Squeeze -> scan_out (1,X,Y,Z,1)
    sq = produced[out_name]
    assert sq.op_type == 'Squeeze', sq.op_type
    rs_out = produced[sq.input[0]]
    assert rs_out.op_type == 'Reshape', rs_out.op_type

    t_in = helper.make_node('Transpose', [in_name], [rs_in.output[0]], perm=[0, 4, 1, 2, 3], name='scan_to_ncxyz')
    t_out = helper.make_node('Transpose', [rs_out.input[0]], [out_name], perm=[0, 2, 3, 4, 1], name='scan_out_to_nxyzc')

    new_nodes = []
    for n in nodes:
        if n is uns:
            new_nodes.append(t_in)
        elif n is sq:
            new_nodes.append(t_out)
        elif n is rs_in or n is rs_out:
            continue
        else:
            new_nodes.append(n)

    # upsampling: Tile(x, repeats with a single 2 at axis k) == Concat([x, x], axis=k)
    consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    n_tile = 0
    for i, n in enumerate(new_nodes):
        if n.op_type != 'Tile':
            continue
        repeats = consts[n.input[1]]
        axis = int(np.argmax(repeats))
        assert repeats[axis] == 2 and repeats.sum() == len(repeats) + 1, repeats
        new_nodes[i] = helper.make_node('Concat', [n.input[0], n.input[0]], list(n.output), axis=axis, name=n.name)
        n_tile += 1
    assert n_tile == 12, n_tile
    del g.node[:]
    g.node.extend(new_nodes)

    # constants of the removed nodes
    used = {i for n in new_nodes for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)

    for v in list(g.input) + list(g.output):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField('dim_value')
        d.dim_param = 'N'
    del g.value_info[:]
    onnx.checker.check_model(model)
    return model


def main():
    src, dst = sys.argv[1:3]
    model = make_batch_dynamic(onnx.load(src))
    onnx.save(model, dst)

    # batch of 1 must reproduce the original, batch of 2 must run
    x = np.random.default_rng(0).random((2, 64, 96, 32, 1), dtype=np.float32)
    s0 = rt.InferenceSession(src, providers=['CPUExecutionProvider'])
    s1 = rt.InferenceSession(dst, providers=['CPUExecutionProvider'])
    i0, o0 = s0.get_inputs()[0].name, s0.get_outputs()[0].name
    ref = np.concatenate([s0.run([o0], {i0: x[i:i + 1]})[0] for i in range(2)])
    one = s1.run([o0], {i0: x[:1]})[0]
    two = s1.run([o0], {i0: x})[0]
    print(f"{dst}: batch 1 max|d| = {np.abs(one - ref[:1]).max():g}, batch 2 max|d| = {np.abs(two - ref).max():g}")


if __name__ == '__main__':
    main()
