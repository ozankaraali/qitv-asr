#!/usr/bin/env bash
# A streaming FastConformer (.nemo) as the released sherpa-onnx streaming transducer files: fc_export.py pins the
# look-ahead (default 13 right frames = 1,040 ms), exports encoder/decoder/joiner ONNX with sherpa-onnx's streaming
# metadata and tokens.txt, puts NeMo's features and NeMo's first streaming chunk in front of the encoder, and makes
# the int8 files (see its docstring); then, with a check manifest, a quick decode through sherpa-onnx's
# OnlineRecognizer as the released numbers were read: normalize_samples=False, 0.16 s of silence before each clip and
# 3 s after, fed in 0.1 s pieces, modified beam search with 4 paths, blank penalty 0.5. CPU only.
#
# usage: bash fc_export.sh <model.nemo> <out dir> [right context frames, default 13]
# env: PYTHON (one with NeMo, onnx, onnxruntime and sherpa-onnx; default: python), CHECK=<manifest.jsonl> (a few
#      clips are read and scored), CHECK_CLIPS (default 5), THREADS (default 4)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 2 ] || { sed -n "9,11p" "$0"; exit 2; }
src=$1
out=$2
right=${3:-13}
threads=${THREADS:-4}
PY=${PYTHON:-python}
[ -f "$src" ] || { echo "no such model: $src" >&2; exit 1; }
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=$threads nice -n 15 "$PY" "$here/fc_export.py" "$src" "$out" "$right" 2>&1 \
  | grep -v "pre-processing before quantization"
for f in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do
  [ -s "$out/$f" ] || { echo "missing $out/$f" >&2; exit 1; }
done
if [ -n "${CHECK:-}" ]; then
  head -n "${CHECK_CLIPS:-5}" "$CHECK" > "$out/check_manifest.jsonl"
  "$PY" "$here/../eval/read.py" sherpa-stream "$out" "$out/check_manifest.jsonl" "$out/check_hyps.jsonl" \
    --int8 --raw --lead-s 0.16 --tail-s 3 --paths 4 --blank-penalty 0.5 --workers 1 --threads "$threads"
  "$PY" "$here/../eval/score.py" "$out/check_manifest.jsonl" "export=$out/check_hyps.jsonl"
fi
ls -l "$out"
