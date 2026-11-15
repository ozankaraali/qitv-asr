"""Read every clip of a test manifest with one model and keep each reading (hyps jsonl: audio_filepath, ref, hyp).
Score them with eval/score.py. Clips already in the output are not read again, so a stopped run goes on.

Each model runs as it is deployed:
- moonshine: a Moonshine export (encoder_model_quantized.onnx, decoder_model_merged_quantized.onnx, tokens.txt)
  through sherpa-onnx's Moonshine v2 loader, plain (greedy) decoding, the clip padded to whole 80-sample frames,
  no gain. (The released Moonshine numbers.)
- sherpa-stream: a streaming transducer (encoder/decoder/joiner[.int8].onnx, tokens.txt) through sherpa-onnx's
  OnlineRecognizer: --lead-s of silence, the clip and --tail-s of silence fed in 0.1 s pieces (decoding whatever is
  ready after each), then input_finished; modified beam search with --paths paths, the blank's score lowered by
  --blank-penalty; --raw passes normalize_samples=False (the samples' scale is left to the model). One segment a
  clip (no endpoints). The released FastConformer numbers: its int8 files (export/fc_export.py) with --int8 --raw
  --lead-s 0.16 --tail-s 3 --paths 4 --blank-penalty 0.5. The comparison model mihu-v2 was read this way as its
  config.json sets it (4 paths, 1 s of silence after, no penalty).
- sherpa-offline-nemo: an offline NeMo transducer through sherpa-onnx's OfflineRecognizer (model_type
  nemo_transducer), at input gain --gain with --pad-s of silence on each side. The comparison model
  ufakzeka-kulak was read this way, as its model card says (gain 8, 0.25 s).
- nemo: a FastConformer .nemo through NeMo at a 1,040 ms look-ahead ([70, 13]), batched beam search (malsd) with
  --paths paths, the blank's score lowered by --blank-penalty before any decoding decision; on the GPU under a
  hard memory cap (--gpu-gb) or on the CPU. (The released FastConformer's .nemo numbers: 4 paths, penalty 0.5.)

The reference kept beside each reading is the row's text_spoken, else text, else text_raw.
A reading can depend on --threads (onnxruntime's int8 kernels split their sums by thread): the released models were
read with one thread a worker (--threads 1), mihu-v2 with 4.

usage: python read.py <kind> <model dir or .nemo> <test.jsonl> <hyps.jsonl> [--workers 4] [--threads 2]
       [--paths 4] [--blank-penalty 0] [--lead-s 0] [--tail-s 1.0] [--raw] [--int8] [--gain 8] [--pad-s 0.25]
       [--gpu-gb 0]"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

KINDS = ("moonshine", "sherpa-stream", "sherpa-offline-nemo", "nemo")
RIGHT = 13  # FastConformer look-ahead frames (80 ms each): 1,040 ms
_read = None


def reference(r):
    return r.get("text_spoken") or r.get("text") or r.get("text_raw") or ""


def _init(kind, d, a):
    """A worker's recognizer. An error here (a missing file) is kept and given back for every clip instead of
    raised: a pool whose initializer raises starts new workers forever."""
    global _read
    try:
        _make_reader(kind, d, a)
    except Exception as error:
        message = f"the model did not load: {type(error).__name__}: {error}"

        def read(x, sr):
            raise RuntimeError(message)
        _read = read


def _make_reader(kind, d, a):
    global _read
    import sherpa_onnx
    from array import array

    d = Path(d)
    if kind == "moonshine":
        rec = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
            encoder=str(d / "encoder_model_quantized.onnx"), decoder=str(d / "decoder_model_merged_quantized.onnx"),
            tokens=str(d / "tokens.txt"), num_threads=a["threads"])

        def read(x, sr):
            sound = array("f", x.tolist())
            sound.extend(array("f", bytes(4 * (-len(sound) % 80))))
            s = rec.create_stream()
            s.accept_waveform(sr, sound)
            rec.decode_stream(s)
            return s.result.text.strip()
    elif kind == "sherpa-stream":
        q = ".int8" if a["int8"] else ""
        rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=str(d / f"encoder{q}.onnx"), decoder=str(d / f"decoder{q}.onnx"),
            joiner=str(d / f"joiner{q}.onnx"), num_threads=a["threads"], sample_rate=16000, feature_dim=80,
            decoding_method="modified_beam_search" if a["paths"] > 1 else "greedy_search",
            max_active_paths=a["paths"], blank_penalty=a["blank_penalty"], normalize_samples=not a["raw"])

        def read(x, sr):
            x = np.concatenate([np.zeros(int(sr * a["lead_s"]), dtype=np.float32), x,
                                np.zeros(int(sr * a["tail_s"]), dtype=np.float32)])
            s = rec.create_stream()
            for i in range(0, len(x), sr // 10):
                s.accept_waveform(sr, x[i: i + sr // 10])
                while rec.is_ready(s):
                    rec.decode_stream(s)
            s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            return rec.get_result(s).strip()
    else:
        q = ".int8" if a["int8"] else ""
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / f"encoder{q}.onnx"), decoder=str(d / f"decoder{q}.onnx"), joiner=str(d / f"joiner{q}.onnx"),
            tokens=str(d / "tokens.txt"), num_threads=a["threads"], model_type="nemo_transducer")

        def read(x, sr):
            pad = np.zeros(int(sr * a["pad_s"]), dtype=np.float32)
            s = rec.create_stream()
            s.accept_waveform(sr, np.concatenate([pad, x * a["gain"], pad]))
            rec.decode_stream(s)
            return s.result.text.strip()
    _read = read


def _one(row):
    import soundfile as sf

    try:
        x, sr = sf.read(row["audio_filepath"], dtype="float32", always_2d=True)
        x = np.ascontiguousarray(x.mean(axis=1), dtype=np.float32)
        hyp = _read(x, sr)
        return {"audio_filepath": row["audio_filepath"], "ref": reference(row), "hyp": hyp, "_audio": len(x) / sr}
    except Exception as error:  # not kept: the clip is read again next time
        return {"audio_filepath": row["audio_filepath"], "error": f"{type(error).__name__}: {error}"[:200]}


def run_nemo(path, todo, out, a):
    """A FastConformer through NeMo, on the GPU under a hard cap of --gpu-gb (the CUDA context's ~0.5 GB included)
    or on the CPU (--gpu-gb 0)."""
    import torch
    from omegaconf import open_dict

    import nemo.collections.asr as nemo_asr

    torch.set_num_threads(a.threads)
    device = "cpu"
    if a.gpu_gb > 0:
        free, total = torch.cuda.mem_get_info()
        if free < a.gpu_gb * 2**30:
            sys.exit(f"only {free / 2**30:.1f} GB free on the GPU (needs {a.gpu_gb})")
        torch.cuda.set_per_process_memory_fraction((a.gpu_gb - 0.5) * 2**30 / total)
        device = "cuda"
    asr = nemo_asr.models.ASRModel.restore_from(str(path), map_location=device)
    asr.eval()
    if hasattr(asr.encoder, "set_default_att_context_size"):
        asr.encoder.set_default_att_context_size([70, RIGHT])
    blank = asr.joint.num_classes_with_blank - 1
    after = asr.joint.joint_after_projection
    penalty = a.blank_penalty

    def with_penalty(*args, **kwargs):  # the blank's score lowered before any decoding decision
        o = after(*args, **kwargs)
        if penalty:
            o = o.clone()
            o[..., blank] -= penalty
        return o

    asr.joint.joint_after_projection = with_penalty
    cfg = asr.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "malsd_batch"
        cfg.beam.beam_size = a.paths
        cfg.beam.return_best_hypothesis = True
    asr.change_decoding_strategy(cfg, decoder_type="rnnt", verbose=False)

    def transcribe(paths, bs):
        with torch.inference_mode():
            res = asr.transcribe(paths, batch_size=bs, verbose=False, num_workers=0)
        return [h.text if hasattr(h, "text") else str(h) for h in (res[0] if isinstance(res, tuple) else res)]

    todo.sort(key=lambda r: -(r.get("duration") or 0))
    failed = 0
    for i in range(0, len(todo), 64):
        part = todo[i: i + 64]
        try:
            hyps = transcribe([r["audio_filepath"] for r in part], 4)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hyps = []
            for r in part:  # one at a time; a clip that still does not fit is left for next time
                try:
                    hyps += transcribe([r["audio_filepath"]], 1)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    hyps.append(None)
        for r, h in zip(part, hyps):
            if h is None:
                failed += 1
                continue
            out.write(json.dumps({"audio_filepath": r["audio_filepath"], "ref": reference(r), "hyp": h},
                                 ensure_ascii=False) + "\n")
        out.flush()
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("model")
    ap.add_argument("manifest")
    ap.add_argument("hyps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--paths", type=int, default=4)
    ap.add_argument("--blank-penalty", type=float, default=0.0)
    ap.add_argument("--lead-s", type=float, default=0.0, help="sherpa-stream: silence fed before the clip")
    ap.add_argument("--tail-s", type=float, default=1.0, help="sherpa-stream: silence fed after the clip")
    ap.add_argument("--raw", action="store_true",
                    help="sherpa-stream: normalize_samples=False (export/fc_export.py's encoder expects it)")
    ap.add_argument("--int8", action="store_true", help="sherpa-*: the .int8.onnx files")
    ap.add_argument("--gain", type=float, default=8.0, help="sherpa-offline-nemo: input gain")
    ap.add_argument("--pad-s", type=float, default=0.25, help="sherpa-offline-nemo: silence on each side")
    ap.add_argument("--gpu-gb", type=float, default=0.0, help="nemo: on the GPU, at most this many GB (0: CPU)")
    a = ap.parse_args()
    q = ".int8" if a.int8 else ""
    need = {"moonshine": ["encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx", "tokens.txt"],
            "nemo": []}.get(a.kind, [f"encoder{q}.onnx", f"decoder{q}.onnx", f"joiner{q}.onnx", "tokens.txt"])
    missing = [f for f in need if not (Path(a.model) / f).is_file()] or ([a.model] if not Path(a.model).exists() else [])
    if missing:
        sys.exit(f"{a.model}: missing {', '.join(missing)}")
    rows = [json.loads(line) for line in open(a.manifest, encoding="utf-8") if line.strip()]
    rows = [r for r in rows if Path(r["audio_filepath"]).exists()]
    have = set()
    if os.path.exists(a.hyps):
        for line in open(a.hyps, encoding="utf-8"):
            try:
                have.add(json.loads(line)["audio_filepath"])
            except (ValueError, KeyError):
                pass  # a line cut by a stopped run: that clip is read again
    todo = [r for r in rows if r["audio_filepath"] not in have]
    print(f"{len(rows)} clips, {len(rows) - len(todo)} read before, {len(todo)} to read", flush=True)
    if not todo:
        return
    t0 = time.time()
    with open(a.hyps, "a", encoding="utf-8") as out:
        if a.kind == "nemo":
            failed = run_nemo(a.model, todo, out, a)
            print(f"read {len(todo) - failed} clips in {(time.time() - t0) / 60:.1f} min"
                  + (f"; {failed} out of GPU memory (not kept)" if failed else ""), flush=True)
            return
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = str(a.threads)
        opts = {"threads": a.threads, "paths": a.paths, "blank_penalty": a.blank_penalty, "lead_s": a.lead_s,
                "tail_s": a.tail_s, "raw": a.raw, "int8": a.int8, "gain": a.gain, "pad_s": a.pad_s}
        todo.sort(key=lambda r: -(r.get("duration") or 0))  # long clips first: the pool ends evenly
        audio, errors = 0.0, []
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.kind, a.model, opts)) as pool:
            for res in pool.imap_unordered(_one, todo, chunksize=2):
                if "error" in res:
                    errors.append(res)
                    if res["error"].startswith("RuntimeError: the model did not load"):
                        pool.terminate()
                        sys.exit(res["error"])
                    continue
                audio += res.pop("_audio")
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()
    took = time.time() - t0
    print(f"read {len(todo) - len(errors)} clips ({audio / 3600:.2f} h) in {took / 60:.1f} min, "
          f"{audio / max(took, 1e-9):.0f}x real time on {a.workers}x{a.threads} threads"
          + (f"; {len(errors)} failed (not kept), e.g. {errors[0]['error']}" if errors else ""), flush=True)


if __name__ == "__main__":
    main()
