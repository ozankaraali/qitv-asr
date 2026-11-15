"""A test set from recordings you have the rights to use and their hand-made subtitles (WebVTT or SRT), cut on the
subtitles' own cues. No speech model chooses anything: the only listener is a voice detector (Silero VAD), and it
only looks at timing. This is how the Creative Commons conversation test was built; it suits TV drama, talk shows,
vlogs or live streams just as well.

- Text: the subtitler's, as written. Out: formatting tags (<i>, <font>, {\\an8}), HTML entities read, notes in
  brackets or between stars ([Müzik], (gülüyor), *Müzik*; round brackets also hold a speaker's name before a line,
  or the uploader's own written remarks), a dialogue dash at a line's start, a speaker label at a line's start (all
  caps, "SUNUCU: ", or a capitalised name and a colon). A cue with nothing left, or one carrying music notes
  (singing or music), has no speech: out.
- Cues: one clip a cue, from 0.1 s before its start to 0.2 s after its end, never into a neighbouring cue's time.
  A cue under 1.0 s joins the cue next to it when the gap is 0.3 s or less (and the two run 12 s or less); cues that
  overlap by 0.3 s or more are one clip. Clips under 0.5 s and over 30 s are out.
- Timing check (loose, model-neutral): the share of 32 ms steps in the clip where the detector hears a voice (above
  0.35). Under 0.15: the cue's sound is elsewhere (or there is none): out. A clip whose text needs over 30 letters a
  second (fast Turkish runs 15-20) is out too: its cue's time cannot hold its words. A track whose cues sit 0.5 s or
  more off the voice as a whole (the best whole-track shift within +-3 s covers 10 points more of the cue time with
  voice) is reported, and left out with --drop-shifted.
- Uploaded subtitles often shorten or rephrase what is said. The checked conversation test kept only the clips whose
  text the forced aligner finds in the sound, edges included: data/mms_check.py in fixed mode on this list (see
  README.md).
- Balance: per group (a channel, a show, a speaker; the item's "group") at most --group-items recordings and
  --group-min minutes of clips; each recording gets an even share, its clips taken in time order; a recording's
  unused share goes to the others.

The items list is a jsonl, one recording a line: {"id", "audio" (a 16 kHz mono file), "subtitles" (.vtt or .srt),
optional "group", "title", "url", "license"}. Every dropped cue and why goes to <out>/dropped.tsv, the per-recording
timing measures to <out>/recordings.tsv, and, when the items carry a url or licence, an attribution table to
<out>/ATTRIBUTION.md.

usage: python build_subtitle_test.py <items.jsonl> <out dir> --vad silero_vad.onnx [--name subs_test]
       [--group-min 20] [--group-items 3] [--drop-shifted] [--dry]
writes <out dir>/<name>.jsonl {audio_filepath, duration, text, item, group, start, end, ...} and the clips
(16 kHz mono Opus) under <out dir>/<name>/."""
import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
STEP = 512  # the detector's 32 ms
TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")
TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")
# [Müzik], (gülüyor), *Müzik*; "5*4=20 cm" is no note (digits)
NOTE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[A-Za-zÇĞİÖŞÜçğıöşüâîû ]+\*")
MUSIC = re.compile(r"[♪♫🎵🎶]")
DASH = re.compile(r"^\s*[-–—]+\s*")
LABEL = re.compile(r"^\s*([A-ZÇĞİÖŞÜÂÎÛ][A-ZÇĞİÖŞÜÂÎÛ0-9 .]{1,24}):\s+")
NLABEL = re.compile(r"^\s*[A-ZÇĞİÖŞÜ][a-zçğıöşü]{1,15}:\s+")  # a speaker's name before a colon
MAX_RATE = 30  # letters a second a cue's time can hold


def secs(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def raw_cues(path):
    out, cur = [], None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        m = TIME.search(line)
        if m:
            cur = [secs(*m.groups()[:4]), secs(*m.groups()[4:]), []]
        elif cur is not None:
            if line.strip():
                cur[2].append(line)
            else:
                if cur[2]:
                    out.append((cur[0], cur[1], cur[2]))
                cur = None
    return sorted(out, key=lambda c: (c[0], c[1]))


def clean(lines, notes):
    """The subtitler's text with only formatting, bracketed sound notes, line-start dashes and speaker labels out."""
    kept = []
    for ln in lines:
        t = html.unescape(TAG.sub("", ln)).replace(" ", " ")
        for _ in NOTE.findall(t):
            notes["bracketed note"] += 1
        t = NOTE.sub(" ", t)
        m = LABEL.match(DASH.sub("", t)) or NLABEL.match(DASH.sub("", t))
        if m:
            notes["speaker label"] += 1
            t = DASH.sub("", t)[m.end():]
        if DASH.match(t):
            notes["dialogue dash"] += 1
        t = DASH.sub("", t)
        kept.append(t)
    return " ".join(" ".join(kept).split())


class Vad:
    """Silero VAD as an ONNX file with inputs x (512 samples), h and c (as sherpa-onnx ships it)."""

    def __init__(self, path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), sess_options=o)

    def probs(self, x):
        n = len(x) // STEP
        frames = x[:n * STEP].reshape(n, STEP).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        for i in range(n):
            p, h, c = self.s.run(None, {"x": frames[i][None], "h": h, "c": c})
            out[i] = p[0][0]
        return out


def item_rows(item, vad, dropped, notes):
    vid = item["id"]
    x, sr = sf.read(str(item["audio"]), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    assert sr == SR, f"{item['audio']}: {sr} Hz (16 kHz mono expected)"
    total = len(x) / SR
    cues = []
    for s, e, lines in raw_cues(item["subtitles"]):
        raw = " | ".join(" ".join(TAG.sub("", ln).split()) for ln in lines)
        if MUSIC.search(" ".join(lines)):
            dropped.append((vid, s, e, "music notes (singing or music, no talk)", raw))
            continue
        t = clean(lines, notes)
        if not re.search(r"[^\W\d_]", t):
            dropped.append((vid, s, e, "no speech (only a sound note or nothing)", raw))
            continue
        if e <= s:
            dropped.append((vid, s, e, "a cue that ends before it starts", raw))
            continue
        cues.append([s, e, t, raw])
    # light merging: overlapping cues, and a cue under 1 s with the next one close by
    merged = []
    for c in cues:
        if merged:
            g = merged[-1]
            overlap = g[1] - c[0]
            short = (g[1] - g[0] < 1.0 or c[1] - c[0] < 1.0) and c[0] - g[1] <= 0.3
            if (overlap >= 0.3 or short) and max(g[1], c[1]) - g[0] <= (15.0 if overlap >= 0.3 else 12.0):
                notes["merged: overlap" if overlap >= 0.3 else "merged: short cue"] += 1
                g[1], g[2], g[3] = max(g[1], c[1]), g[2] + " " + c[2], g[3] + " || " + c[3]
                continue
        merged.append(list(c))
    p = vad.probs(x)
    voice = p > 0.35
    fr = STEP / SR
    # the whole track against the voice: the share of cue time with voice at each shift
    mask = np.zeros(len(voice), dtype=bool)
    for s, e, *_ in merged:
        mask[int(s / fr):int(e / fr) + 1] = True
    best, at0 = (0.0, 0.0), 0.0
    if mask.sum():
        for k in range(-int(3 / fr), int(3 / fr) + 1):
            sh = np.roll(mask, k)
            cov = float((sh & voice).sum() / sh.sum())
            if k == 0:
                at0 = cov
            if cov > best[1]:
                best = (k * fr, cov)
    rows = []
    for i, (s, e, t, raw) in enumerate(merged):
        prev_e = merged[i - 1][1] if i else 0.0
        next_s = merged[i + 1][0] if i + 1 < len(merged) else total
        a = max(0.0, s - 0.1, prev_e if prev_e <= s else s)
        b = min(total, e + 0.2, next_s if next_s >= e else e)
        if b - a < 0.5:
            dropped.append((vid, s, e, "under 0.5 s", raw))
            continue
        if b - a > 30.0:
            dropped.append((vid, s, e, "over 30 s", raw))
            continue
        lo, hi = int(a / fr), max(int(a / fr) + 1, int(b / fr))
        share = float(voice[lo:hi].mean()) if hi <= len(voice) else 0.0
        if share < 0.15:
            dropped.append((vid, s, e, f"no voice in the cue's time (voice in {share:.0%} of it): sound elsewhere", raw))
            continue
        rate = len(re.findall(r"[^\W\d_]", t)) / (b - a)
        if rate > MAX_RATE:
            dropped.append((vid, s, e, f"its words cannot fit the cue's time ({rate:.0f} letters a second): timing off",
                            raw))
            continue
        rows.append({"a": a, "b": b, "text": t, "voice": share})
    return x, rows, {"shift_s": round(best[0], 2), "cover_best": round(best[1], 3), "cover_at_0": round(at0, 3),
                     "shifted": abs(best[0]) >= 0.5 and best[1] - at0 >= 0.10, "cues": len(cues), "clips": len(rows),
                     "clip_min": round(sum(r["b"] - r["a"] for r in rows) / 60, 2), "sound_min": round(total / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--vad", required=True, help="silero_vad.onnx (inputs x, h, c)")
    ap.add_argument("--name", default="subs_test")
    ap.add_argument("--group-min", type=float, default=20.0, help="minutes of clips per group at most")
    ap.add_argument("--group-items", type=int, default=3, help="recordings per group at most")
    ap.add_argument("--drop-shifted", action="store_true")
    ap.add_argument("--dry", action="store_true", help="measure and report, write no clips or lists")
    a = ap.parse_args()
    items = [json.loads(line) for line in open(a.items, encoding="utf-8") if line.strip()]
    out = Path(a.out)
    clips_dir = out / a.name
    out.mkdir(parents=True, exist_ok=True)
    by_group = defaultdict(list)
    for it in items:
        by_group[it.get("group") or it["id"]].append(it)
    vad = Vad(a.vad)
    dropped, notes, measures, chosen = [], Counter(), [], []
    for group, its in by_group.items():
        measured = []
        for it in its:
            if len(measured) == a.group_items:
                measures.append((it["id"], {"note": f"the group has {a.group_items} recordings already"}))
                continue
            x, rows, m = item_rows(it, vad, dropped, notes)
            measures.append((it["id"], m))
            if m["shifted"] and a.drop_shifted:
                for r in rows:
                    dropped.append((it["id"], r["a"], r["b"], f"the whole track is {m['shift_s']} s off the voice",
                                    r["text"]))
                continue
            if rows:
                measured.append((it, x, rows))
        budget = a.group_min * 60
        order = sorted(measured, key=lambda t: sum(r["b"] - r["a"] for r in t[2]))
        left = len(order)
        for it, x, rows in order:
            share = budget / left
            took, used = [], 0.0
            for r in rows:
                d = r["b"] - r["a"]
                if used + d > share:
                    dropped.append((it["id"], r["a"], r["b"], "past the group's share of minutes (not a fault)", r["text"]))
                    continue
                took.append(r)
                used += d
            budget -= used
            left -= 1
            chosen.append((it, x, took))
    lines, kept = [], []
    if not a.dry:
        clips_dir.mkdir(parents=True, exist_ok=True)
    for it, x, took in chosen:
        if took:
            kept.append(it)
        for r in took:
            p = clips_dir / f"{it['id']}_{int(round(r['a'] * 1000)):08d}.ogg"
            if not a.dry:
                sf.write(p, x[int(r["a"] * SR):int(r["b"] * SR)], SR, format="OGG", subtype="OPUS")
            row = {"audio_filepath": str(p.resolve()), "duration": round(r["b"] - r["a"], 3), "text": r["text"],
                   "item": it["id"], "group": it.get("group") or it["id"], "start": round(r["a"], 3),
                   "end": round(r["b"], 3)}
            row.update({k: it[k] for k in ("title", "url", "license") if it.get(k)})
            lines.append(row)
    (out / "dropped.tsv").write_text("".join(f"{v}\t{s:.3f}\t{e:.3f}\t{why}\t{raw}\n" for v, s, e, why, raw in dropped),
                                     encoding="utf-8")
    (out / "recordings.tsv").write_text("".join(f"{v}\t{json.dumps(m, ensure_ascii=False)}\n" for v, m in measures),
                                        encoding="utf-8")
    hours = sum(r["duration"] for r in lines) / 3600
    groups = {it.get("group") or it["id"] for it in kept}
    print(f"{len(lines)} clips, {hours:.2f} h, {len(kept)} recordings, {len(groups)} groups")
    print("text notes:", dict(notes))
    print("dropped:", dict(Counter(re.sub(r" \(voice in .*?\)|-?\d+(\.\d+)? s off", "", d[3]) for d in dropped)))
    for v, m in measures:
        if m.get("shifted"):
            print(f"  shifted track: {v} {m}")
    if a.dry:
        return
    (out / f"{a.name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines),
                                         encoding="utf-8")
    if any(it.get("url") or it.get("license") for it in kept):
        per = Counter()
        for r in lines:
            per[r["item"]] += r["duration"]
        md = ["# Attribution", "", f"Every clip of `{a.name}.jsonl` is cut from one of the recordings below; its text is "
              "the recording's own subtitle track.", "",
              "| # | Group | Title | URL | Licence | Clips (min) |", "|---|---|---|---|---|---|"]
        for i, it in enumerate(kept, 1):
            cell = lambda k: str(it.get(k) or "").replace("|", "\\|")  # noqa: E731
            md.append(f"| {i} | {cell('group')} | {cell('title')} | {cell('url')} | {cell('license')} | "
                      f"{per[it['id']] / 60:.1f} |")
        md += ["", f"{len(lines)} clips, {hours:.2f} h, {len(kept)} recordings, {len(groups)} groups."]
        (out / "ATTRIBUTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
