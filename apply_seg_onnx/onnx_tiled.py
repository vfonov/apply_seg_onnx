#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whole-volume inference of GroupNorm networks with bounded GPU memory (no PyTorch dependency).

GroupNorm statistics are computed over the whole input, so patch-based inference of a GroupNorm network
is not the same as whole-volume inference. GroupNorm is the only non-local operation of networks such as
the WMH-SynthSeg 3D U-Net (Conv, LeakyRelu, MaxPool, nearest Resize, Concat are all local), so the network
can be evaluated exactly by stages: the graph is cut at every GroupNorm; each stage (the local nodes between
GroupNorms) is run tile by tile on the GPU with a halo covering its receptive field, and its outputs are stored
as whole-volume arrays in host memory. Each stage also returns per-channel sums of x and x^2 over the tile cores,
which give the exact GroupNorm statistics used (as a per-channel scale/shift) by the following stages.

GroupNorm is recognised in the form written by common exporters:
    Reshape(x, [N, G, -1]) -> InstanceNormalization -> Reshape(., Shape(x)) [-> Mul gamma -> Add beta]
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
try:
    import onnxruntime
except ImportError as e:
    raise ImportError("apply_seg_onnx needs ONNX Runtime >= 1.18: pip install 'apply_seg_onnx[gpu]' (onnxruntime-gpu) "
                      "or 'apply_seg_onnx[cpu]' (onnxruntime), or conda install -c conda-forge onnxruntime") from e

LOCAL_OPS = {'Conv', 'LeakyRelu', 'Relu', 'Sigmoid', 'Tanh', 'Elu', 'MaxPool', 'AveragePool', 'Resize',
             'Concat', 'Add', 'Mul', 'Sub', 'Div', 'Identity'}


_shared_cpu_arena = False


def _register_shared_cpu_arena():
    """register a process-wide CPU arena allocator, used by sessions with session.use_env_allocators=1"""
    global _shared_cpu_arena
    if not _shared_cpu_arena:
        mem_info = onnxruntime.OrtMemoryInfo('Cpu', onnxruntime.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0,
                                             onnxruntime.OrtMemType.DEFAULT)
        onnxruntime.create_and_register_allocator(mem_info, onnxruntime.OrtArenaCfg(0, -1, -1, -1))
        _shared_cpu_arena = True


def _attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


class TiledGroupNormSession:
    """
    Drop-in replacement for onnxruntime.InferenceSession.run() for single-input 3D GroupNorm networks,
    computing the whole-volume result with GPU memory bounded by the tile size.

    Why: GroupNorm normalizes every group of channels with the mean/variance taken over the *whole* input.
    Running such a network on patches normalizes every patch with its own statistics, which changes the
    result (for WMH-SynthSeg, 128^3 patches agree with the whole-volume labels on only ~95 % of brain voxels).
    Running the whole volume at once needs too much GPU memory (> 48 GB for a 160x256x256 T1 with ORT).

    How: GroupNorm is the only operation that needs the whole volume; all the others are local (each output
    voxel depends on a small neighbourhood of input voxels). So the graph is evaluated stage by stage:

      1. Graph analysis (__init__ -> _find_groupnorms, _plan):
         - every GroupNorm pattern (see the module docstring) is collapsed to one "GroupNorm" with
           input x (pre-norm tensor), output y, G groups, InstanceNormalization scale/bias, gamma/beta, eps;
         - the remaining (local) nodes are split into stages: a stage is every node that can be computed
           from the tensors already stored in host memory and from the outputs of GroupNorms whose
           statistics are known. Stages therefore end where a GroupNorm input (or a graph output) is produced.
           For the WMH-SynthSeg U-Net this gives 18 stages, e.g. GN -> Conv -> LeakyRelu [-> MaxPool],
           or GN -> Conv -> LeakyRelu -> Resize -> Concat(skip);
         - every tensor gets a 'scale' (downsampling factor relative to the input: 1, 2, 4, 8, 16) and
           every stage a 'halo' (receptive-field radius, in input voxels: radius of each Conv times its scale,
           plus one coarse voxel per Resize).
      2. Stage models (_stage_session): each stage becomes a small ONNX model (one ORT session each) with
           inputs:  stored tensors it reads; for each GroupNorm it applies, the pre-norm tensor x plus
                    '<y>__scale' and '<y>__shift' (1,C,1,1,1), so that y = x * scale + shift;
                    for each GroupNorm input it produces, '<x>__start' / '<x>__end' (tile core, int64[3])
           outputs: the tensors needed later (stored in host memory), and for each GroupNorm input it
                    produces '<x>__sum' / '<x>__sq': per-channel sums of x and x^2 over the tile core.
      3. Execution (_run_one), for each stage in order:
         - the GroupNorm statistics of its inputs, accumulated over all tiles of the previous stages
           (float64), are turned into the per-channel scale/shift (_affine);
         - the volume is covered by tile cores of `tile` voxels (_tiles); each tile is cut out of the stored
           tensors with the halo added (clipped at the volume border, where the Conv zero padding then
           matches the whole-volume computation), run on the device, and only its core is written back;
         - stored tensors are freed after the last stage that reads them.
      The result equals whole-volume inference up to floating point summation order (WMH-SynthSeg: logits
      within ~1e-4..1e-2 of plain whole-volume inference, ~1e-6 of the voxels change label).

    Memory: GPU memory is bounded by the largest stage on one tile (WMH-SynthSeg, tile 128: ~9.5 GB;
    96: ~8 GB; 64: ~4.5 GB). Host memory holds the whole-volume tensors that are still needed (~25 GB for a
    160x256x256 volume with this U-Net). Each stage session's CUDA arena grows only as requested and is shrunk
    after every run; all stage sessions share one CPU arena.

    Supported graphs: one float input (N,C,X,Y,Z); local ops in LOCAL_OPS; Conv with any kernel/dilation and
    equal strides; non-overlapping, unpadded pooling; nearest Resize with constant scales. Input sizes must be
    multiples of the largest downsampling factor (16 for a 5-level U-Net). Batch elements are run one by one
    (GroupNorm statistics are per sample).

    Args:
        model:        ONNX file name (external data next to it is loaded)
        sess_options: onnxruntime.SessionOptions used for every stage session (gets
                      'session.use_env_allocators'=1 for the shared CPU arena); a new one if None
        providers:    as for onnxruntime.InferenceSession; CUDAExecutionProvider options are kept, with
                      arena_extend_strategy='kSameAsRequested' added
        tile:         size of the tile cores in input voxels (int or 3 ints), rounded to a multiple of the
                      largest downsampling factor; smaller tiles use less GPU memory but are slower
                      (more halo recomputation and more, smaller kernel launches)

    Attributes (after construction):
        gns:      {GroupNorm output name: dict(x, y, groups, s, b, gamma, beta, eps)}
        stages:   list of dict(nodes, raw_in, gn_in, outputs, halo):
                  raw_in  - stored tensors read directly, gn_in - GroupNorm outputs applied in the stage,
                  outputs - tensors stored after the stage, halo - in input voxels
        scale:    {tensor name: downsampling factor relative to the input}
        last_use: {stored tensor name: index of the last stage reading it}

    Example:
        sess = TiledGroupNormSession('model.onnx', providers=['CUDAExecutionProvider'], tile=96)
        seg = sess.run(['seg'], {'scan': volume[None, None]})[0]
    """

    def __init__(self, model, sess_options=None, providers=None, tile=128):
        self.model = onnx.load(model)
        # all stage sessions share one CPU memory arena (otherwise each would keep its own peak allocation)
        self.sess_options = sess_options if sess_options is not None else onnxruntime.SessionOptions()
        _register_shared_cpu_arena()
        self.sess_options.add_session_config_entry('session.use_env_allocators', '1')
        # one session per stage: every session has its own CUDA arena, which is grown only as requested and
        # released after each run, so that the GPU memory is bounded by the largest tile of a single stage
        self.providers, self.run_options = [], None
        for p in (providers or []):
            name, opts = (p if isinstance(p, tuple) else (p, {}))
            if name == 'CUDAExecutionProvider':
                opts = dict(opts, arena_extend_strategy='kSameAsRequested')
                self.run_options = onnxruntime.RunOptions()
                self.run_options.add_run_config_entry('memory.enable_memory_arena_shrinkage',
                                                      f"gpu:{opts.get('device_id', 0)}")
            self.providers.append((name, opts))
        self.providers = self.providers or None
        self.tile = list(tile) if isinstance(tile, (list, tuple)) else [tile] * 3
        g = self.model.graph
        self.init = {i.name: i for i in g.initializer}
        inputs = [i.name for i in g.input if i.name not in self.init]
        assert len(inputs) == 1, "only single-input models are supported"
        self.input_name = inputs[0]
        self.output_names = [o.name for o in g.output]
        self.producer = {o: n for n in g.node for o in n.output}
        self.consumers = {}
        for n in g.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)
        self._find_groupnorms()
        self._plan()
        self._sessions = {}

    # ----------------------------------------------------------------------------------------------
    # graph analysis
    def _const(self, name):
        """value of an initializer as a numpy array, None if `name` is not an initializer"""
        return numpy_helper.to_array(self.init[name]) if name in self.init else None

    def _find_groupnorms(self):
        """
        Find the GroupNorm patterns Reshape(x) -> InstanceNormalization -> Reshape(Shape(x)) [-> Mul -> Add],
        store them in self.gns (keyed by the pattern's output name y) and put all other nodes in
        self.local_nodes, checking that they are all supported local ops.
        """
        self.gns, gn_nodes = {}, set()
        for n in self.model.graph.node:
            if n.op_type != 'InstanceNormalization':
                continue
            r1 = self.producer[n.input[0]]
            r2 = self.consumers[n.output[0]][0]
            assert r1.op_type == 'Reshape' and r2.op_type == 'Reshape', f"unsupported normalization at {n.name}"
            x = r1.input[0]
            nodes = [r1, n, r2]
            shape_node = self.producer.get(r2.input[1])
            if shape_node is not None and shape_node.op_type == 'Shape':
                nodes.append(shape_node)
            s, b = self._const(n.input[1]).astype(np.float64), self._const(n.input[2]).astype(np.float64)
            y, gamma, beta = r2.output[0], None, None
            cons = self.consumers.get(y, [])
            if len(cons) == 1 and cons[0].op_type == 'Mul' and self._const(cons[0].input[1]) is not None:
                gamma = self._const(cons[0].input[1]).reshape(-1).astype(np.float64)
                nodes.append(cons[0])
                y = cons[0].output[0]
                cons = self.consumers.get(y, [])
                if len(cons) == 1 and cons[0].op_type == 'Add' and self._const(cons[0].input[1]) is not None:
                    beta = self._const(cons[0].input[1]).reshape(-1).astype(np.float64)
                    nodes.append(cons[0])
                    y = cons[0].output[0]
            self.gns[y] = dict(x=x, y=y, groups=len(s), s=s, b=b, gamma=gamma, beta=beta,
                               eps=_attrs(n).get('epsilon', 1e-5))
            gn_nodes.update(id(m) for m in nodes)
        self.local_nodes = [n for n in self.model.graph.node if id(n) not in gn_nodes]
        for n in self.local_nodes:
            assert n.op_type in LOCAL_OPS, f"unsupported op {n.op_type} ({n.name}) for tiled inference"

    def _plan(self):
        """
        Split self.local_nodes into self.stages (see the class docstring). Greedy: starting from the stored
        ('materialized') tensors - first only the graph input - and the GroupNorm outputs whose input is stored,
        a stage takes every remaining node whose inputs are available (repeated until nothing changes).
        Its outputs are the produced tensors needed outside the stage: graph outputs, GroupNorm inputs and
        tensors read by later stages (e.g. U-Net skip connections). Also computes self.scale (downsampling
        factor of every tensor), each stage's halo (conservative: sum of the receptive-field radii of its
        nodes, in input voxels) and self.last_use (to free stored tensors).
        """
        scale = {self.input_name: 1}
        materialized = {self.input_name}
        gn_ready = {y for y, gn in self.gns.items() if gn['x'] in materialized}
        remaining = list(self.local_nodes)
        self.stages = []
        gn_x = {gn['x'] for gn in self.gns.values()}
        while remaining:
            ready = materialized | gn_ready
            nodes, produced = [], set()
            changed = True
            while changed:
                changed = False
                for n in remaining:
                    if n in nodes:
                        continue
                    if all((i == '' or i in self.init or i in ready or i in produced) for i in n.input):
                        nodes.append(n)
                        produced.update(n.output)
                        changed = True
            assert nodes, "could not schedule the graph for tiled inference"
            remaining = [n for n in remaining if n not in nodes]
            node_ids = {id(n) for n in nodes}
            outputs = [t for t in produced
                       if t in self.output_names or t in gn_x
                       or any(id(c) not in node_ids for c in self.consumers.get(t, []))]
            used = [i for n in nodes for i in n.input if i and i not in self.init and i not in produced]
            raw_in = sorted({i for i in used if i in materialized})
            gn_in = sorted({i for i in used if i in gn_ready and i not in materialized})
            # tensor scales and halo (in input voxels), conservative: sum of the radii of the stage's nodes
            for y in gn_in:
                scale[y] = scale[self.gns[y]['x']]
            halo = 0
            for n in nodes:
                si = scale[n.input[0]]
                a = _attrs(n)
                if n.op_type == 'Conv':
                    k = a.get('kernel_shape') or list(self._const(n.input[1]).shape[2:])
                    d = a.get('dilations', [1] * len(k))
                    st = a.get('strides', [1] * len(k))
                    halo += max((kk // 2) * dd for kk, dd in zip(k, d)) * si
                    so = si * st[0]
                elif n.op_type in ('MaxPool', 'AveragePool'):
                    st = a.get('strides', a['kernel_shape'])
                    assert list(a['kernel_shape']) == list(st) and not any(a.get('pads', [0])), \
                        f"{n.name}: only non-overlapping pooling is supported"
                    so = si * st[0]
                elif n.op_type == 'Resize':
                    sc = self._const(n.input[2]) if len(n.input) > 2 and n.input[2] else None
                    assert sc is not None and a.get('mode', b'nearest') in (b'nearest', 'nearest'), \
                        f"{n.name}: only nearest Resize with constant scales is supported"
                    so = si / sc[2]
                    halo += si
                else:
                    so = si
                for o in n.output:
                    scale[o] = so
            self.stages.append(dict(nodes=nodes, raw_in=raw_in, gn_in=gn_in, outputs=outputs, halo=halo))
            materialized |= set(outputs)
            gn_ready |= {y for y, gn in self.gns.items() if gn['x'] in materialized}
        self.scale = scale
        self.max_scale = int(max(scale.values()))
        # last stage using every materialized tensor, to free memory
        self.last_use = {}
        for k, st in enumerate(self.stages):
            for t in st['raw_in'] + [self.gns[y]['x'] for y in st['gn_in']]:
                self.last_use[t] = k

    # ----------------------------------------------------------------------------------------------
    # stage models
    def _stage_session(self, k):
        """
        ORT session for stage k, built on first use. The stage model applies the GroupNorms as
        y = x * '<y>__scale' + '<y>__shift', then the stage's nodes (with their initializers), and returns
        the stage outputs plus, for every GroupNorm input x it produces, per-channel sums '<x>__sum' and
        '<x>__sq' of x and x^2 over the tile core given by '<x>__start'/'<x>__end' (spatial, in x's grid).
        """
        if k in self._sessions:
            return self._sessions[k]
        st = self.stages[k]
        g_inputs, nodes, inits = [], [], []
        vi = lambda name, et=TensorProto.FLOAT, shape=None: helper.make_tensor_value_info(name, et, shape)
        for t in st['raw_in']:
            g_inputs.append(vi(t, shape=[1, None, None, None, None]))
        for y in st['gn_in']:
            x = self.gns[y]['x']
            if x not in st['raw_in']:
                g_inputs.append(vi(x, shape=[1, None, None, None, None]))
            g_inputs += [vi(y + '__scale', shape=[1, None, 1, 1, 1]), vi(y + '__shift', shape=[1, None, 1, 1, 1])]
            nodes += [helper.make_node('Mul', [x, y + '__scale'], [y + '__mul']),
                      helper.make_node('Add', [y + '__mul', y + '__shift'], [y])]
        nodes += st['nodes']
        needed = {i for n in st['nodes'] for i in n.input}
        inits += [self.init[i] for i in needed if i in self.init]
        g_outputs = [vi(o) for o in st['outputs']]
        gn_x = {gn['x'] for gn in self.gns.values()}
        if any(o in gn_x for o in st['outputs']):
            inits.append(numpy_helper.from_array(np.array([2, 3, 4], dtype=np.int64), '__axes'))
            inits.append(numpy_helper.from_array(np.array([0, 2, 3, 4], dtype=np.int64), '__raxes'))
        for o in st['outputs']:
            if o in gn_x:  # per-channel sums over the tile core, for GroupNorm statistics
                g_inputs += [vi(o + '__start', TensorProto.INT64, [3]), vi(o + '__end', TensorProto.INT64, [3])]
                nodes += [helper.make_node('Slice', [o, o + '__start', o + '__end', '__axes'], [o + '__core']),
                          helper.make_node('ReduceSum', [o + '__core', '__raxes'], [o + '__sum'], keepdims=0),
                          helper.make_node('Mul', [o + '__core', o + '__core'], [o + '__core2']),
                          helper.make_node('ReduceSum', [o + '__core2', '__raxes'], [o + '__sq'], keepdims=0)]
                g_outputs += [vi(o + '__sum'), vi(o + '__sq')]
        graph = helper.make_graph(nodes, f'stage{k}', g_inputs, g_outputs, initializer=inits)
        m = helper.make_model(graph, opset_imports=self.model.opset_import)
        m.ir_version = self.model.ir_version
        sess = onnxruntime.InferenceSession(m.SerializeToString(), self.sess_options, providers=self.providers)
        self._sessions[k] = sess
        return sess

    # ----------------------------------------------------------------------------------------------
    def _affine(self, y, sums, sqs, n_vox):
        """
        Per-channel scale a and shift b such that a*x + b equals the GroupNorm output y with whole-volume
        statistics. sums, sqs: per-channel sums of x and x^2 over the whole volume (float64);
        n_vox: number of voxels of x. With channels grouped into G groups of C/G channels:
            mean_g = sum over the group / (n_vox*C/G),  var_g = E[x^2] - mean_g^2  (biased, as GroupNorm)
            InstanceNormalization:  s_g * (x - mean_g) / sqrt(var_g + eps) + b_g
            then gamma_c * (.) + beta_c
        Returns float32 arrays of shape (1, C, 1, 1, 1).
        """
        gn = self.gns[y]
        C, G = len(sums), gn['groups']
        cpg = C // G
        mean = sums.reshape(G, cpg).sum(1) / (n_vox * cpg)
        var = np.maximum(sqs.reshape(G, cpg).sum(1) / (n_vox * cpg) - mean * mean, 0.0)
        inv = gn['s'] / np.sqrt(var + gn['eps'])          # per group
        a = np.repeat(inv, cpg)
        b = np.repeat(gn['b'] - inv * mean, cpg)
        if gn['gamma'] is not None:
            a, b = a * gn['gamma'], b * gn['gamma']
        if gn['beta'] is not None:
            b = b + gn['beta']
        return a.astype(np.float32).reshape(1, C, 1, 1, 1), b.astype(np.float32).reshape(1, C, 1, 1, 1)

    def _tiles(self, size, halo):
        """
        Yield (core, region) for every tile of a volume of `size` input voxels: core = [(start, end)] * 3,
        the part written back; region = core grown by the halo (rounded up to the largest downsampling factor)
        and clipped to the volume, the part fed to the stage. Cores tile the volume without overlap and start
        at multiples of the largest downsampling factor, so they map to whole voxels on every grid.
        """
        q = self.max_scale
        h = int(np.ceil(halo / q)) * q
        axes = []
        for n, t in zip(size, self.tile):
            t = max(q, int(round(t / q)) * q)
            axes.append([(a, min(a + t, n)) for a in range(0, n, t)])
        for cx in axes[0]:
            for cy in axes[1]:
                for cz in axes[2]:
                    core = [cx, cy, cz]
                    reg = [(max(0, a - h), min(n, b + h)) for (a, b), n in zip(core, size)]
                    yield core, reg

    def _run_one(self, x):
        """
        Whole-volume forward pass of one sample x (1, C, X, Y, Z) float32, stage by stage (see the class
        docstring). Stored tensors are kept without the batch axis (C, X/s, Y/s, Z/s); the statistics of the
        input itself (if it feeds a GroupNorm) are computed directly. Returns {output name: (1, ...) array}.
        """
        size = list(x.shape[2:])
        q = self.max_scale
        assert all(s % q == 0 for s in size), f"input size {size} must be a multiple of {q}"
        tensors = {self.input_name: x[0]}
        stats = {}
        if self.input_name in {gn['x'] for gn in self.gns.values()}:
            xd = x[0].astype(np.float64)
            stats[self.input_name] = (xd.sum(axis=(1, 2, 3)), (xd * xd).sum(axis=(1, 2, 3)))
            del xd
        gn_x = {gn['x'] for gn in self.gns.values()}
        for k, st in enumerate(self.stages):
            sess = self._stage_session(k)
            aff = {}
            for y in st['gn_in']:
                xg = self.gns[y]['x']
                n_vox = int(np.prod([s // self.scale[xg] for s in size]))
                aff[y] = self._affine(y, *stats[xg], n_vox)
            outs, acc = {}, {}
            for core, reg in self._tiles(size, st['halo']):
                feed = {}
                for t in set(st['raw_in']) | {self.gns[y]['x'] for y in st['gn_in']}:
                    s = int(self.scale[t])
                    feed[t] = np.ascontiguousarray(
                        tensors[t][None, :, reg[0][0] // s:reg[0][1] // s, reg[1][0] // s:reg[1][1] // s,
                                   reg[2][0] // s:reg[2][1] // s])
                for y in st['gn_in']:
                    feed[y + '__scale'], feed[y + '__shift'] = aff[y]
                for o in st['outputs']:
                    if o in gn_x:
                        s = self.scale[o]
                        feed[o + '__start'] = np.array([int((c[0] - r[0]) / s) for c, r in zip(core, reg)], np.int64)
                        feed[o + '__end'] = np.array([int((c[1] - r[0]) / s) for c, r in zip(core, reg)], np.int64)
                names = list(st['outputs']) + [o + suf for o in st['outputs'] if o in gn_x for suf in ('__sum', '__sq')]
                res = dict(zip(names, sess.run(names, feed, self.run_options)))
                for o in st['outputs']:
                    s = self.scale[o]
                    r = res[o][0]
                    if o not in outs:
                        outs[o] = np.empty((r.shape[0], *[int(n / s) for n in size]), dtype=r.dtype)
                    c0 = [int(c[0] / s) for c in core]
                    c1 = [int(c[1] / s) for c in core]
                    r0 = [int((c[0] - rr[0]) / s) for c, rr in zip(core, reg)]
                    outs[o][:, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] = \
                        r[:, r0[0]:r0[0] + c1[0] - c0[0], r0[1]:r0[1] + c1[1] - c0[1], r0[2]:r0[2] + c1[2] - c0[2]]
                    if o in gn_x:
                        a = acc.setdefault(o, [0.0, 0.0])
                        a[0] = a[0] + res[o + '__sum'].astype(np.float64)
                        a[1] = a[1] + res[o + '__sq'].astype(np.float64)
            tensors.update(outs)
            stats.update({o: tuple(v) for o, v in acc.items()})
            for t, last in self.last_use.items():
                if last == k and t in tensors and t not in self.output_names:
                    del tensors[t]
        return {o: tensors[o][None] for o in self.output_names}

    def run(self, output_names, input_feed, run_options=None):
        """
        Same call as onnxruntime.InferenceSession.run(): output_names (None = all graph outputs),
        input_feed {input name: (N, C, X, Y, Z) array}. The N samples are run one by one; run_options is
        ignored (the stage sessions use their own, see __init__). Returns a list of (N, ...) arrays.
        """
        x = np.asarray(input_feed[self.input_name], dtype=np.float32)
        res = [self._run_one(x[b:b + 1]) for b in range(x.shape[0])]
        names = output_names or self.output_names
        return [np.concatenate([r[o] for r in res], axis=0) for o in names]
