"""The Turkish streaming FastConformer as the released sherpa-onnx streaming transducer files (int8), after
sherpa-onnx's own scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer.py.

1. fp32 ONNX: the model pinned to one look-ahead first ([70, 13] = 1.04 s; it was trained with four: 0, 80, 480 and
   1,040 ms), then encoder/decoder/joiner with sherpa-onnx's streaming metadata on the encoder, and tokens.txt. Only
   the transducer branch is exported (the hybrid's CTC head is not).
2. The encoder gets two steps put in front of its graph, then dynamic int8 with per-channel weights; its subsampling
   front end (the nodes named /pre_encode/...) stays float32:
   - NeMo's own features (add_scaled): read with OnlineRecognizer.from_transducer(..., normalize_samples=False),
     sherpa-onnx scales the samples by 32768 before its fbank, so its log-mel is log(max(E * 2^30, eps)), and
     log(exp(f - 30 ln 2) + 2^-24) is then NeMo's log(E + 2^-24) to float precision. Left at the default
     (normalize_samples=True), the features are not the ones the encoder now expects, and it writes nothing.
   - NeMo's first streaming chunk (add_first16): see its docstring. With 0.16 s of silence fed before the speech,
     every window the encoder sees is the one NeMo's own streaming reads.
3. Decoder: dynamic int8 (QUInt8). Joiner: dynamic int8 with per-channel weights.
4. tokens.txt beside them. The fp32 files go to <out>/fp32/ and are removed at the end (kept with --keep-fp32).

The released files were read with normalize_samples=False, 0.16 s of silence before each clip and 3 s after
(eval/read.py sherpa-stream --int8 --raw --lead-s 0.16 --tail-s 3). CPU is enough.
usage: python fc_export.py <model.nemo> <out dir> [right context frames, default 13] [--keep-fp32]"""
import argparse
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper
from onnxruntime.quantization import QuantType, quantize_dynamic

PRE_ENCODE = r"^/pre_encode/"  # the encoder nodes left in float32


def add_meta_data(filename, meta_data):
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    onnx.save(model, filename)


@torch.no_grad()
def export_fp32(src, out, right):
    """encoder/decoder/joiner.onnx (fp32) and tokens.txt; sherpa-onnx's streaming metadata on the encoder."""
    import nemo.collections.asr as nemo_asr

    out.mkdir(parents=True, exist_ok=True)
    m = nemo_asr.models.ASRModel.restore_from(src, map_location="cpu")
    m.encoder.set_default_att_context_size([70, right])
    m.encoder.setup_streaming_params()
    with open(out / "tokens.txt", "w", encoding="utf-8") as f:
        for i, s in enumerate(m.joint.vocabulary):
            f.write(f"{s} {i}\n")
        f.write(f"<blk> {i + 1}\n")
    if hasattr(m, "ctc_decoder"):  # a hybrid: its transducer branch alone
        m.change_decoding_strategy(decoder_type="rnnt")
    m.eval()
    cfg = m.encoder.streaming_cfg
    chunk = cfg.chunk_size[1] if isinstance(cfg.chunk_size, list) else cfg.chunk_size
    pre = cfg.pre_encode_cache_size[1] if isinstance(cfg.pre_encode_cache_size, list) else cfg.pre_encode_cache_size
    m.set_export_config({"decoder_type": "rnnt", "cache_support": True})
    m.encoder.export(str(out / "encoder.onnx"))
    m.decoder.export(str(out / "decoder.onnx"))
    m.joint.export(str(out / "joiner.onnx"))
    normalize = m.cfg.preprocessor.normalize
    meta = {
        "vocab_size": m.decoder.vocab_size,
        "window_size": chunk + pre,
        "chunk_shift": chunk,
        "normalize_type": "" if normalize == "NA" else normalize,
        "cache_last_channel_dim1": len(m.encoder.layers),
        "cache_last_channel_dim2": cfg.last_channel_cache_size,
        "cache_last_channel_dim3": m.encoder.d_model,
        "cache_last_time_dim1": len(m.encoder.layers),
        "cache_last_time_dim2": m.encoder.d_model,
        "cache_last_time_dim3": m.encoder.conv_context_size[0],
        "pred_rnn_layers": m.decoder.pred_rnn_layers,
        "pred_hidden": m.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": type(m).__name__,  # EncDecHybridRNNTCTCBPEModel
        "version": "1",
        "model_author": "QiTV (fine-tuned from NVIDIA stt_en_fastconformer_hybrid_large_streaming_multi, CC BY 4.0)",
        "comment": f"Turkish streaming transducer, look-ahead {right * 80} ms; only the transducer branch is exported",
    }
    add_meta_data(str(out / "encoder.onnx"), meta)
    print(meta)
    print("chunk", chunk, "pre-encode cache", pre, flush=True)


def add_scaled(model):
    """NeMo's features from sherpa-onnx's fbank read with normalize_samples=False: the samples are scaled by 32768
    first, so sherpa-onnx's log-mel is log(max(E * 2^30, eps)); log(exp(f - 30 ln 2) + 2^-24) is NeMo's
    log(E + 2^-24) to float precision (the floor now sits at E = 1.1e-16, far under NeMo's 2^-24 guard)."""
    g = model.graph
    name = "audio_signal"
    fixed = "audio_signal_nemo_scale"
    for n in g.node:
        for i, x in enumerate(n.input):
            if x == name:
                n.input[i] = fixed

    def k(nm, v):
        return helper.make_node("Constant", [], [nm], value=helper.make_tensor(nm + "_t", TensorProto.FLOAT, [], [v]))

    nodes = [
        k("fix_shift", float(30 * np.log(2.0))), k("fix_guard", 2.0 ** -24),
        helper.make_node("Sub", [name, "fix_shift"], ["fix_s"]),
        helper.make_node("Exp", ["fix_s"], ["fix_e"]),
        helper.make_node("Add", ["fix_e", "fix_guard"], ["fix_a"]),
        helper.make_node("Log", ["fix_a"], [fixed]),
    ]
    for j, n in enumerate(nodes):
        g.node.insert(j, n)
    return model


def add_first16(model):
    """NeMo's first streaming chunk inside sherpa-onnx's fixed windows. sherpa-onnx steps the encoder over 121-frame
    windows shifted by 112 from frame 0, and the graph drops the first 2 subsampled frames of every window (NeMo's
    drop_extra_pre_encoded, meant for the windows after the first); NeMo's own streaming reads its first chunk as 105
    frames with nothing dropped, so its chunk grid sits 16 feature frames earlier. Here, on a call with
    cache_last_channel_len == 0 (a fresh or reset stream), the window's first 16 frames are skipped and nothing is
    dropped: given 0.16 s of sound before the speech (2560 samples, e.g. silence), every window then is NeMo's.
    One stream per call (the condition is read from batch item 0), as sherpa-onnx's decode_stream does."""
    g = model.graph
    i64 = TensorProto.INT64

    def k(nm, v):
        return helper.make_node("Constant", [], [nm], value=helper.make_tensor(nm + "_t", i64, [len(v)], v))

    def ks(nm, v):
        return helper.make_node("Constant", [], [nm], value=helper.make_tensor(nm + "_t", i64, [], [v]))

    nodes = [
        k("f16_zero_idx", [0]), ks("f16_zero", 0), k("f16_skip", [16]), k("f16_none", [0]), k("f16_two", [2]),
        k("f16_big", [2 ** 62]), k("f16_axis2", [2]),
        helper.make_node("Gather", ["cache_last_channel_len", "f16_zero_idx"], ["f16_len0"], axis=0),
        helper.make_node("Equal", ["f16_len0", "f16_none"], ["f16_first"]),
        helper.make_node("Where", ["f16_first", "f16_skip", "f16_none"], ["f16_start"]),
        helper.make_node("Where", ["f16_first", "f16_none", "f16_two"], ["f16_drop"]),
        helper.make_node("Slice", ["audio_signal", "f16_start", "f16_big", "f16_axis2"], ["f16_audio"]),
        helper.make_node("Sub", ["length", "f16_start"], ["f16_length"]),
        helper.make_node("Squeeze", ["f16_drop", "f16_zero_idx"], ["f16_drop_scalar"]),
    ]
    rewired = {"audio": 0, "length": 0, "slice": 0, "sub": 0}
    for n in g.node:  # node names as NeMo 3.0 exports a cache-aware FastConformer encoder
        if n.name == "/Transpose_2" and n.input[0] == "audio_signal":
            n.input[0] = "f16_audio"
            rewired["audio"] += 1
        elif n.name == "/pre_encode/conv/Cast" and n.input[0] == "length":
            n.input[0] = "f16_length"
            rewired["length"] += 1
        elif n.name == "/Slice" and n.op_type == "Slice":
            n.input[1] = "f16_drop"
            rewired["slice"] += 1
        elif n.name == "/Sub" and n.op_type == "Sub":
            n.input[1] = "f16_drop_scalar"
            rewired["sub"] += 1
    assert all(v == 1 for v in rewired.values()), f"not the encoder graph this step expects: {rewired}"
    for j, n in enumerate(nodes):
        g.node.insert(j, n)
    return model


def encoder_int8(src, out):
    """The fp32 encoder with NeMo's first chunk and NeMo's features in front, then dynamic int8 (QUInt8, per-channel
    weights) with the subsampling front end left in float32; sherpa-onnx's metadata kept."""
    model = onnx.load(src)
    meta = {p.key: p.value for p in model.metadata_props}
    model = add_scaled(add_first16(model))
    excl = [n.name for n in model.graph.node if re.search(PRE_ENCODE, n.name)]
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staged = Path(tmp) / "src.onnx"
        onnx.save(model, staged)
        quantize_dynamic(model_input=str(staged), model_output=str(out), weight_type=QuantType.QUInt8,
                         per_channel=True, reduce_range=False, nodes_to_exclude=excl)
    model = onnx.load(out)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = key, value
    onnx.save(model, out)
    print(f"{out}: {len(excl)} front-end nodes left in float32", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a cache-aware streaming FastConformer .nemo")
    ap.add_argument("out", help="output folder")
    ap.add_argument("right", nargs="?", type=int, default=13, help="right context frames (80 ms each), default 13")
    ap.add_argument("--keep-fp32", action="store_true", help="keep the fp32 export in <out>/fp32/")
    a = ap.parse_args()
    out = Path(a.out)
    fp32 = out / "fp32"
    export_fp32(a.model, fp32, a.right)
    encoder_int8(fp32 / "encoder.onnx", out / "encoder.int8.onnx")
    quantize_dynamic(model_input=str(fp32 / "decoder.onnx"), model_output=str(out / "decoder.int8.onnx"),
                     weight_type=QuantType.QUInt8)
    quantize_dynamic(model_input=str(fp32 / "joiner.onnx"), model_output=str(out / "joiner.int8.onnx"),
                     weight_type=QuantType.QUInt8, per_channel=True)
    shutil.copy(fp32 / "tokens.txt", out / "tokens.txt")
    if not a.keep_fp32:
        shutil.rmtree(fp32)
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        print(f"{out / f}: {(out / f).stat().st_size / 1e6:.1f} MB")
    print("Read with sherpa_onnx.OnlineRecognizer.from_transducer(..., normalize_samples=False), 0.16 s of silence "
          "before the speech and about 3 s after it.")


if __name__ == "__main__":
    main()
