"""Python interface: files in, files out, configuration as a dict.

    import apply_seg_onnx

    config = {"models": ["model.onnx"], "n_classes": 3, "patch_sz": 64, "stride": 32}
    apply_seg_onnx.segment("in.mnc", "seg.mnc", config)
    apply_seg_onnx.segment_batch(["a.mnc", "b.mnc"], ["a_seg.mnc", "b_seg.mnc"], config, minibatch_size=2)

The config keys are those of the JSON config file of the command line (README, "Config file reference").
"""
import copy
import json
import os

from .inference import segment_with_onnx, segment_with_onnx_batched


def load_config(path):
    """Read a JSON config file into a dict."""
    with open(path, 'r') as f:
        return json.load(f)


def apply_model_prefix(config, model_prefix=None):
    """
    Copy of config with `models` as a list and model_prefix prepended (as a string, like --model_prefix)
    to every model and to `reference`. The dict passed in is not modified.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config: expected a dict, got {type(config).__name__}")
    config = copy.deepcopy(config)
    models = config.get('models', None)
    if models is None or (isinstance(models, (list, tuple)) and len(models) == 0):
        raise ValueError("config: 'models' is required")
    models = [os.fspath(m) for m in models] if isinstance(models, (list, tuple)) else [os.fspath(models)]
    if model_prefix is not None:
        model_prefix = os.fspath(model_prefix)
        models = [model_prefix + m for m in models]
        if config.get('reference', None) is not None:
            config['reference'] = model_prefix + os.fspath(config['reference'])
    config['models'] = models
    return config


def _path(p):
    return os.fspath(p) if isinstance(p, (str, os.PathLike)) else p


def _scan(scan):
    """a scan: one file, or a list of channels (files, or numbers for constant-filled channels)"""
    if isinstance(scan, (list, tuple)):
        return [_path(c) for c in scan]
    return _path(scan)


def segment(input, output, config, *, model_prefix=None, cpu=False, threads=0, device_id=None, use_tf32=False,
            measure=None, fuzzy=None, history=None):
    """
    Run the pipeline described by config on one scan and write the result to a file.

    Args:
        input:   path of the scan (.mnc, .nii, .nii.gz), or a list of input channels for multi-channel
                 models: paths, or numbers for constant-filled channels
        output:  path of the output volume, in the format given by its extension
        config:  dict with the config keys (`models` is required); not modified
        model_prefix: string prepended to every model name and to `reference`
        cpu:     run on the CPU; by default the CUDA execution provider is used, as on the command line
        threads: ONNX Runtime intra-op threads (0: its default)
        device_id: GPU index
        use_tf32: allow TF32 convolutions on the GPU
        measure: csv file for the label volumes (needs `labels_desc` in config)
        fuzzy:   prefix for the per-class probability maps (default: config key `fuzzy`)
        history: history string stored in the output (default: config key `history`)
    Returns:
        the output path
    Errors are raised (missing input: FileNotFoundError).
    """
    channels = _scan(input)
    if not isinstance(channels, list):
        channels = [channels]
    return segment_with_onnx(channels, _path(output), apply_model_prefix(config, model_prefix),
                             cpu=cpu, threads=threads, history=history, device_id=device_id, use_tf32=use_tf32,
                             measure=_path(measure), fuzzy=_path(fuzzy))


def segment_batch(inputs, outputs, config, *, model_prefix=None, cpu=False, threads=0, device_id=None,
                  use_tf32=False, minibatch_size=1, measure=None, fuzzy=None, history=None, progress=False,
                  recover=False, skip_errors=False):
    """
    Run the pipeline on several scans with the models loaded once.

    Args:
        inputs:  list of scans; each is a path or a list of channels (see segment)
        outputs: list of output paths, one per scan
        minibatch_size: scans per model call (scans of different shape are run one by one)
        measure: csv file for the label volumes, one row per scan
        fuzzy:   prefix for the probability maps: <prefix>_<scan>_<class>.<ext>
        progress: progress bar (needs tqdm)
        recover: skip scans whose output already exists
        skip_errors: report a failed minibatch (or one with missing inputs) and go on with the next one
                     instead of raising
        other arguments: as in segment
    Returns:
        the list of output paths
    """
    inputs = [_scan(s) for s in inputs]
    outputs = [_path(o) for o in outputs]
    if len(inputs) != len(outputs):
        raise ValueError(f"{len(inputs)} inputs for {len(outputs)} outputs")
    if not skip_errors:
        missing = [c for scan, out in zip(inputs, outputs) if not (recover and os.path.exists(out))
                   for c in (scan if isinstance(scan, list) else [scan]) if isinstance(c, str) and not os.path.exists(c)]
        if missing:
            raise FileNotFoundError(f"Input file(s) do not exist: {missing}")
    segment_with_onnx_batched(inputs, outputs, apply_model_prefix(config, model_prefix),
                              cpu=cpu, threads=threads, history=history, device_id=device_id, use_tf32=use_tf32,
                              minibatch_size=minibatch_size, fuzzy=_path(fuzzy), progress=progress,
                              measure=_path(measure), recover=recover, crash=not skip_errors)
    return outputs
