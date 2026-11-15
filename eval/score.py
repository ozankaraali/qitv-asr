"""Score readings (eval/read.py's hyps jsonl, or any jsonl with audio_filepath and hyp) against a test manifest, the
same way for every model: words wrong (WER) with a 95 % bootstrap interval, letters wrong (CER), the strict WER,
the paired difference to a base model on the same resamples, and (--punct) sentence-end, comma and question-mark
F1. The scorers are in common/scoring.py.

- References: each row's text_spoken, else text, else text_raw. A clip whose reference has no words under the
  scorer (a lone "(Müzik)") is out for everyone, as is a clip whose audio file is gone.
- --fix-thousands: FLEURS writes 40.000 as "40. 000"; read it as 40.000 (FLEURS only).
- --scorer en: the English scorer (num2words needed).
- A model that has not read every clip is listed as partial and left out of the table.

usage: python score.py <test.jsonl> <name=hyps.jsonl> [...] [--base NAME] [--scorer tr|en] [--fix-thousands]
       [--punct] [--json out.json]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.scoring import (PCLASSES, align, edits, en_words, fixed_ref, interval, letter_edits, paired,  # noqa: E402
                            punct_counts, raw_tokens, resamples, shown_text, strict_words, wer, words)
from common.tr_text import pc_text  # noqa: E402


def load_rows(path, scorer, fix):
    rows, gone, empty = [], 0, 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ref = r.get("text_spoken") or r.get("text") or r.get("text_raw") or ""
        fixed = fixed_ref(ref) if fix else ref
        w = en_words(fixed) if scorer == "en" else words(fixed)
        if not w:
            empty += 1
            continue
        if not Path(r["audio_filepath"]).exists():
            gone += 1
            continue
        rows.append({"audio_filepath": r["audio_filepath"], "ref": fixed, "words": w,
                     "strict": strict_words(pc_text(fixed)), "duration": r.get("duration")})
    return rows, gone, empty


def load_hyps(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "hyp" in r:
            out[r["audio_filepath"]] = r["hyp"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("models", nargs="+", help="name=hyps.jsonl")
    ap.add_argument("--base", default="", help="paired differences against this model (default: the first)")
    ap.add_argument("--scorer", default="tr", choices=["tr", "en"])
    ap.add_argument("--fix-thousands", action="store_true")
    ap.add_argument("--punct", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows, gone, empty = load_rows(a.manifest, a.scorer, a.fix_thousands)
    name = Path(a.manifest).stem
    idx = resamples(name, len(rows))
    n = np.array([len(r["words"]) for r in rows], dtype=np.int64)
    nc = np.array([len(" ".join(r["words"])) for r in rows], dtype=np.int64)
    ns = np.array([len(r["strict"]) for r in rows], dtype=np.int64)
    hours = sum(r["duration"] or 0 for r in rows) / 3600
    print(f"{name}: {len(rows)} clips, {int(n.sum())} words, {hours:.2f} h"
          + (f"; {gone} clips without audio" if gone else "") + (f"; {empty} with an empty reference" if empty else ""))
    scored, summary = {}, {}
    for spec in a.models:
        label, _, path = spec.partition("=")
        hyps = load_hyps(path)
        got = [hyps.get(r["audio_filepath"]) for r in rows]
        k = sum(g is not None for g in got)
        if k < len(rows):
            print(f"  {label}: partial, {k} of {len(rows)} clips read; left out")
            continue
        conv = (lambda t: en_words(t)) if a.scorer == "en" else (lambda t: words(shown_text(t)))
        hw = [conv(g) for g in got]
        e = np.array([edits(r["words"], h) for r, h in zip(rows, hw)], dtype=np.int64)
        ec = np.array([letter_edits(" ".join(r["words"]), " ".join(h)) for r, h in zip(rows, hw)], dtype=np.int64)
        es = np.array([edits(r["strict"], strict_words(g)) for r, g in zip(rows, got)], dtype=np.int64)
        ops = [align(r["words"], h)[0] for r, h in zip(rows, hw)]
        sub = sum(1 for o, r, h in zip(ops, rows, hw) for i, j in o if i is not None and j is not None
                  and r["words"][i] != h[j])
        dele = sum(1 for o in ops for i, j in o if j is None)
        ins = sum(1 for o in ops for i, j in o if i is None)
        scored[label] = {"e": e, "ec": ec, "es": es, "got": got}
        lo, hi = interval(e, n, idx)
        clo, chi = interval(ec, nc, idx)
        rec = {"wer": wer(e, n), "lo": lo, "hi": hi, "cer": wer(ec, nc), "cer_lo": clo, "cer_hi": chi,
               "strict_wer": wer(es, ns), "wrong": 100 * sub / max(1, n.sum()), "missed": 100 * dele / max(1, n.sum()),
               "extra": 100 * ins / max(1, n.sum())}
        if a.punct and a.scorer == "tr":
            counts = [punct_counts(raw_tokens(r["ref"]), raw_tokens(shown_text(g))) for r, g in zip(rows, got)]
            for cls in PCLASSES:
                tp, fp, fn = (sum(c[cls][s] for c in counts) for s in range(3))
                rec[f"f1_{cls}"] = 100 * 2 * tp / max(1, 2 * tp + fp + fn)
        summary[label] = rec
    if not scored:
        return
    base = a.base or next(iter(scored))
    print()
    print("| Model | WER % (95 % CI) | wrong / missed / extra | CER % (95 % CI) | strict WER % |"
          + (f" vs {base} (points) |" if len(scored) > 1 else "")
          + (" F1 end / inside / comma / question |" if a.punct and a.scorer == "tr" else ""))
    print("|---|---|---|---|---|" + ("---|" if len(scored) > 1 else "") + ("---|" if a.punct and a.scorer == "tr" else ""))
    for label, s in scored.items():
        r = summary[label]
        line = (f"| {label} | {r['wer']:.1f} ({r['lo']:.1f}–{r['hi']:.1f}) | {r['wrong']:.1f} / {r['missed']:.1f} / "
                f"{r['extra']:.1f} | {r['cer']:.1f} ({r['cer_lo']:.1f}–{r['cer_hi']:.1f}) | {r['strict_wer']:.1f} |")
        if len(scored) > 1:
            if label == base or base not in scored:
                line += " |"
            else:
                d, dlo, dhi = paired(s["e"], scored[base]["e"], n, idx)
                r.update(d=d, dlo=dlo, dhi=dhi)
                line += f" {d:+.1f} ({dlo:+.1f} to {dhi:+.1f}) |"
        if a.punct and a.scorer == "tr":
            line += " " + " / ".join(f"{r[f'f1_{c}']:.0f}" for c in ("e", "i", "c", "q")) + " |"
        print(line)
    if a.json:
        json.dump({"set": name, "clips": len(rows), "words": int(n.sum()), "models": summary}, open(a.json, "w"),
                  indent=1, default=float)


if __name__ == "__main__":
    main()
