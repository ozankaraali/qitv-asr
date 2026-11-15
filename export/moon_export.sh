#!/bin/bash
# A Turkish Moonshine checkpoint (a model folder with tr.model) as sherpa-onnx Moonshine v2 files: encoder_model and
# decoder_model_merged (int8, the conv frontend kept fp32) and a base64 tokens.txt from our SentencePiece.
#
# The ONNX export and quantisation are WorkmindAI/stt-models' reproducible pipeline (MIT; its scripts/export_onnx.py
# and scripts/quantize_onnx.py, with its patch for optimum-onnx). The released files were made at stt-models commit
# 91d5726 with transformers 5.14.1 and optimum-onnx fc357f9 (patched) on the CPU.
#
# The weights are quantised per channel (one scale for each output row): stt-models' quantize_onnx.py with
# per_channel=True instead of its per-tensor default. Per tensor cost the medium 0.6-0.8 points of words wrong on
# read speech, at the same size and speed.
#
# usage: STT_MODELS=<a clone of github.com/WorkmindAI/stt-models> bash moon_export.sh <checkpoint dir> <out dir>
#   (the first run makes a venv inside the clone, as its README does)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
CK=$(cd "${1:?checkpoint dir}" && pwd)
OUT=$(mkdir -p "${2:?out dir}" && cd "$2" && pwd)
STT=${STT_MODELS:?set STT_MODELS to a clone of WorkmindAI/stt-models}
UV=${UV:-uv}
log() { echo "$(date +%T) $*"; }
cd "$STT"
if [ ! -x .venv/bin/python3 ]; then
    log "venv"
    $UV venv -q .venv --python 3.12
    $UV pip install -q --python .venv/bin/python3 --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
    $UV pip install -q --python .venv/bin/python3 transformers==5.14.1 soundfile librosa numpy accelerate sentencepiece protobuf
    $UV pip install -q --python .venv/bin/python3 optimum==2.2.0 onnx==1.22.0 onnxruntime==1.28.0 onnxscript==0.7.1 onnxslim==0.1.94
    [ -d optimum-onnx ] || git clone -q --depth 50 https://github.com/huggingface/optimum-onnx.git
    (cd optimum-onnx && git checkout -q fc357f980340e21a930698d03794ec63392ceb6f && git apply ../patches/optimum-onnx-moonshine-streaming.patch)
    $UV pip install -q --python .venv/bin/python3 -e ./optimum-onnx --no-deps
fi
[ -f "$CK/tr.model" ] || { echo "no tr.model in $CK (moon_train.py copies the --sp model there)" >&2; exit 1; }
log "export $CK -> $OUT"
OMP_NUM_THREADS=${THREADS:-8} nice -n 10 .venv/bin/python3 scripts/export_onnx.py "$CK" "$OUT"
log "quantize (per-channel weights)"
Q=$(mktemp -d)/quantize_onnx_per_channel.py
sed "s/per_channel=False,/per_channel=True,/" scripts/quantize_onnx.py > "$Q"
grep -q "per_channel=True," "$Q" || { echo "scripts/quantize_onnx.py has no per_channel=False to switch" >&2; exit 1; }
nice -n 10 .venv/bin/python3 "$Q" "$OUT/encoder_model.onnx" "$OUT/onnx" q8
nice -n 10 .venv/bin/python3 "$Q" "$OUT/decoder_model_merged.onnx" "$OUT/onnx" q8
log "tokens"
.venv/bin/python3 "$here/moon_tokens.py" "$CK/tr.model" "$OUT/onnx/tokens.txt"
ls -la "$OUT/onnx"
log "done: $OUT/onnx holds encoder_model_quantized.onnx, decoder_model_merged_quantized.onnx and tokens.txt"
