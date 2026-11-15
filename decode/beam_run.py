"""Decoder settings compared on one test manifest: each clip goes through a Moonshine export's encoder once (its
output can be kept, --keep, and reused by later runs), then through every decoder config given.

A config is tag=k[,lm,lam,beta]: k 1 is greedy; k > 1 a beam of k paths; lm a .qng file (decode/qng.py) or "-" for
none; lam the LM weight; beta a bonus per token. QiTV's optional decoder is "b4lm=4,<lm.qng>,0.1,0.5" (4 paths,
weight 0.1, bonus 0.5); the beam may end a hypothesis only where the model's end score is within 1.0 of its best
token, and a hypothesis that starts looping is dropped.

Readings go to <out dir>/<tag>.jsonl (audio_filepath, hyp, tokens, the clip's CPU seconds; one thread per worker),
ready for eval/score.py. Configs whose readings are complete are skipped.

usage: python beam_run.py <export onnx dir> <test.jsonl> <out dir> --config greedy=1 --config b4=4
       --config b4lm=4,lm.qng,0.1,0.5 [--workers 4] [--keep] [--limit N]"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

W = {}
A = None
CONFIGS = []


def parse(spec):
    tag, _, rest = spec.partition("=")
    parts = rest.split(",")
    k = int(parts[0])
    lm = parts[1] if len(parts) > 1 and parts[1] != "-" else ""
    lam = float(parts[2]) if len(parts) > 2 else 0.0
    beta = float(parts[3]) if len(parts) > 3 else 0.0
    return {"tag": tag, "k": k, "lm": lm, "lam": lam, "beta": beta}


def init(args, configs):
    global A, CONFIGS
    from moonbeam import Moon
    from qng import NgramLM

    A, CONFIGS = args, configs
    W["m"] = Moon(args.model, threads=1, encoder=True)
    W["lm"] = {c["lm"]: NgramLM(c["lm"]) for c in configs if c["lm"]}


def enc_path(i):
    return Path(A.out) / "enc" / f"{i}.npy"


def work(item):
    i, r = item
    m = W["m"]
    p = enc_path(i)
    out = {"audio_filepath": r["audio_filepath"], "audio_s": r.get("duration") or 0.0}
    if A.keep and p.exists():
        h = np.load(p)
        frames = int(open(str(p) + ".frames").read())
        if h.shape[1] == 0:
            h = None
        out["enc_s"] = 0.0
    else:
        import soundfile as sf

        x, sr = sf.read(r["audio_filepath"], dtype="float32", always_2d=True)
        x = np.ascontiguousarray(x.mean(axis=1), dtype=np.float32)
        assert sr == 16000, sr
        out["audio_s"] = len(x) / sr
        t0 = time.process_time()
        h, frames = m.encode(x)
        out["enc_s"] = time.process_time() - t0
        if A.keep:
            np.save(p, h if h is not None else np.zeros((1, 0, 1), np.float32))
            with open(str(p) + ".frames", "w") as f:
                f.write(str(frames))
    res = {}
    for c in CONFIGS:
        t0 = time.process_time()
        if c["k"] == 1:
            toks = m.greedy(h, frames)
        else:
            toks = m.beam(h, frames, k=c["k"], lm=W["lm"].get(c["lm"]), lam=c["lam"], beta=c["beta"],
                          eos_gap=1.0, drop_loops=True)
        res[c["tag"]] = {"hyp": m.text(toks), "tokens": toks, "dec_s": time.process_time() - t0}
    out["res"] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="the export's onnx folder (encoder_model_quantized.onnx, ...)")
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--keep", action="store_true", help="keep (and reuse) the encoder output")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.manifest, encoding="utf-8") if line.strip()]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    configs = []
    for spec in a.config:
        c = parse(spec)
        p = out / f"{c['tag']}.jsonl"
        if p.exists() and sum(1 for _ in open(p)) >= len(rows):
            print(f"{c['tag']}: already read", flush=True)
            continue
        configs.append(c)
    if not configs:
        return
    (out / "enc").mkdir(parents=True, exist_ok=True)
    items = sorted(enumerate(rows), key=lambda x: -(x[1].get("duration") or 0))
    t0 = time.time()
    res = []
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(a, configs)) as pool:
        for r in pool.imap_unordered(work, items, chunksize=1):
            res.append(r)
    audio = sum(r["audio_s"] for r in res)
    enc = sum(r["enc_s"] for r in res)
    for c in configs:
        tag = c["tag"]
        with open(out / f"{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in res:
                row = {"audio_filepath": r["audio_filepath"], "audio_s": r["audio_s"], **r["res"][tag]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        dec = sum(r["res"][tag]["dec_s"] for r in res)
        print(f"{tag}: {len(res)} clips {audio / 3600:.2f} h; per-thread RTF dec {dec / max(audio, 1e-9):.4f}"
              + (f" enc {enc / max(audio, 1e-9):.4f}" if enc else ""), flush=True)
    print(f"{time.time() - t0:.0f} s wall", flush=True)


if __name__ == "__main__":
    main()
