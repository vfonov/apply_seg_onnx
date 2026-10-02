"""apply_seg_onnx.postprocess: largest connected component, label volumes, CSV output."""
import csv
import json
import math

import numpy as np
import pytest

from apply_seg_onnx.postprocess import find_largest_component, measure_volumes, save_measurements


def two_blobs_touching_diagonally():
    a = np.zeros((10, 10, 10), bool)
    a[1:4, 1:4, 1:4] = True    # 27 voxels
    a[4:6, 4:6, 4:6] = True    # 8 voxels, touches the first one only at a corner
    a[7:9, 7:9, 0:3] = True    # 12 voxels, separate
    return a


def test_largest_component_connectivity():
    a = two_blobs_touching_diagonally()
    six = find_largest_component(a, connectivity=1)
    assert six.sum() == 27 and six[2, 2, 2] and not six[4, 4, 4]
    full = find_largest_component(a, connectivity=3)   # corner contact joins the first two blobs
    assert full.sum() == 35 and full[4, 4, 4] and not full[8, 8, 1]


def test_largest_component_empty_and_labels():
    assert not find_largest_component(np.zeros((4, 4, 4), bool)).any()
    lab = np.zeros((6, 6, 6), np.uint8)
    lab[0:2, 0:2, 0:2] = 1
    lab[3:6, 3:6, 3:6] = 2  # nonzero labels are foreground, whatever their value
    assert find_largest_component(lab, connectivity=1).sum() == 27


AFF = np.diag([1.0, 1.5, 2.0, 1.0])  # 3 mm^3 voxels


@pytest.mark.parametrize('desc', [{'1': 'one', 2: 'two'}, ['one', 'two'], 'json'])
def test_measure_volumes(tmp_path, desc):
    if desc == 'json':
        desc = str(tmp_path / 'labels.json')
        with open(desc, 'w') as f:
            json.dump({'1': 'one', '2': 'two'}, f)
    seg = np.zeros((4, 4, 4), np.uint8)
    seg[0, 0, :3] = 1
    seg[1:3, 1:3, 1:3] = 2
    m = measure_volumes(seg, AFF, desc, out_seg_f='out.mnc', in_scan='in.mnc')
    assert m == {'scan': 'in.mnc', 'segmentation': 'out.mnc', 'one': pytest.approx(9.0), 'two': pytest.approx(24.0)}


def test_measure_volumes_missing_and_from_file(tmp_path):
    m = measure_volumes(None, None, ['one'], out_seg_f='missing.mnc', in_scan='in.mnc')
    assert math.isnan(m['one'])
    from apply_seg_onnx.io import save_volume
    seg = np.zeros((4, 4, 4), np.uint8)
    seg[:2] = 1
    fn = str(tmp_path / 'seg.mnc')
    save_volume(fn, seg, AFF)
    m = measure_volumes(None, None, ['one'], out_seg_f=fn, in_scan='in.mnc', load_output=True)
    assert m['one'] == pytest.approx(32 * 3.0)
    with pytest.raises(ValueError):
        measure_volumes(seg, AFF, 42)


def test_save_measurements(tmp_path):
    rows = [measure_volumes(np.ones((2, 2, 2), np.uint8), AFF, ['one'], 'a_seg.mnc', 'a.mnc'),
            measure_volumes(None, None, ['one'], 'b_seg.mnc', 'b.mnc')]
    fn = str(tmp_path / 'vol.csv')
    save_measurements(fn, rows)
    with open(fn) as f:
        r = list(csv.DictReader(f))
    assert list(r[0].keys()) == ['one', 'scan', 'segmentation']  # sorted columns
    assert float(r[0]['one']) == pytest.approx(24.0) and r[1]['one'] == 'nan' and r[1]['scan'] == 'b.mnc'


@pytest.mark.parametrize('ext', ['mnc', pytest.param('nii.gz', marks=pytest.mark.skipif(
    not __import__('apply_seg_onnx.nifti_io', fromlist=['x']).have_nibabel, reason='nibabel not installed'))])
@pytest.mark.parametrize('label, dtype', [(40000, np.uint16), (70000, np.uint32)])
def test_measure_volumes_large_labels_from_file(tmp_path, ext, label, dtype):
    from apply_seg_onnx.io import save_volume
    seg = np.zeros((4, 4, 4), dtype)
    seg[:2] = label
    seg[3, 3, 3] = 1
    fn = str(tmp_path / f'seg.{ext}')
    save_volume(fn, seg, AFF)
    m = measure_volumes(None, None, {str(label): 'big', '1': 'one'}, out_seg_f=fn, in_scan='in', load_output=True)
    assert m['big'] == pytest.approx(32 * 3.0) and m['one'] == pytest.approx(3.0)
