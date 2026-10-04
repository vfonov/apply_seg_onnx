#! /bin/bash
# Pack the example models and the test scan (not kept in git) into one archive:
#   examples/make_data_archive.sh [output.tar.gz]
# Unpacked at the root of the repository (tar xzf ...) it restores examples/models and examples/data.
set -e
cd "$(dirname "$0")/.."
out=${1:-apply_seg_onnx_example_data.tar.gz}

files="examples/DATA.md
examples/models/_20240404_conjurer_trained_dice_7733.onnx
examples/models/WMH-SynthSeg_v10_231110_new.onnx
examples/models/WMH-SynthSeg_v10_231110_new.onnx.data
examples/models/synthsr_v20_230130_batch.onnx
examples/data/subject43_1_t2w.mnc
examples/data/reference/subject43_1_t2w_mindglide.mnc
examples/data/reference/subject43_1_t2w_wmh_synthseg.mnc
examples/data/reference/subject43_1_t2w_synthsr.mnc
examples/data/reference/subject43_1_t2w_synthsr_no_tta.mnc"

for f in $files; do
    [[ -e $f ]] || { echo "missing: $f" >&2; exit 1; }
done
sha256sum $files > examples/data/SHA256SUMS
tar --owner=0 --group=0 --numeric-owner -czf "$out" $files examples/data/SHA256SUMS
sha256sum "$out" > "$out.sha256"
ls -l "$out" "$out.sha256"
