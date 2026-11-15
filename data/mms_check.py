"""The forced-aligner check: does a clip's text fit its sound, and do clip and text start and end together? The only
judge is a forced aligner (torchaudio's MMS_FA: letters, 20 ms frames) that is told the text and never writes a reading
of its own. No recognizer's reading keeps or drops a clip: a listener, or the subtitle times, only place the text
roughly. This step chose the final round's re-cut clips (the talk list, the re-cut parliament sittings, the TV
subtitle lines no earlier clip covered) and the final round's YODAS v3 spans; the checked conversation test was
checked the same way. The human text stays the authority: the check is about place and timing, never about content.

Two modes:
- window (subtitle lines, minutes, transcripts): the clip's own words +-4 words of its recording's text; the aligner
  takes those words in order, free to start at any word and to end at any word, and must account for every frame of
  the clip (CTC Viterbi). The words it covers become the clip's text (text_raw; the placed text is kept as
  text_placed), so the aligner sets the edges: a word the sound lacks at an edge falls out, a neighbouring line's
  word said inside the clip comes in.
- fixed (a span list with its own transcripts, a test set): the clip's text forced over the whole clip, first word
  to last; never trimmed or extended.

Every frame of the forced path has a cost (how far the forced letter or blank is below that frame's best
log-probability). Measures:
- fit: minus the path's cost per letter;
- blank_in / blank_out: the blank-path cost up to the end of the first word / from the start of the last word
  (speech the text lacks at an edge);
- tpl_first / tpl_last: the first / last word's own letter cost per letter (text the speech lacks at an edge).

The bars come from controls only (no reader, no model's score), on the same clips with the same aligner, and are
fixed in a file before any clip is checked:
- foreign: as many words from elsewhere in the same recording's text (a window starting 300+ words away; 20 tries),
  or with --foreign other the text of a clip of another recording with about as many letters (the recordings must
  not repeat each other's sentences: a repeated sentence fits like the clip's own): its fit. The fit bar:
  above 99 % of foreign texts with as many letters (bins 0-20, 20-40, 40-70, 70+ letters; a bin with under 100
  foreign texts takes the bar of the bin below; the lowest takes the 99 % level of the lowest bins pooled to 100).
- edge cut: the text minus its first word (or its last): the blank cost at that edge. The edge-blank bar catches 95 %
  of these controls whose word has 4+ letters.
- edge extra: the recording's word before the first (after the last) forced in: its letter cost per letter. The
  edge-letter bar catches 95 % of them.
One rule for the three: an edge misplaced by one word of four letters or more is caught 95 % of the time. Calibrate
per source (sound and text differ), on clips whose text is known to be in the sound (--calib-max-cer keeps rows a
listener located; the re-cut's TV controls ran on the lines its re-sync step had placed). With --placed-controls
(the checked test, where some cue texts are not in their clip at all): edge controls only from clips whose own text
passes the fit bar, and an edge-cut control that leaves no trace (cost 0: the word is not said there) left out.

The decision, the first rule a clip fails: no word the aligner can read; no path; misplaced (fits no better than
foreign text); edge: speech the text lacks; edge: text the speech lacks; under --min-words words (the covered words
in window mode; 3 for training lists). Window mode: two kept clips that claim the same word of the text both go.
Inside a clip nothing is judged: a word the minutes or the subtitles leave out (or add) between the first and the
last word is the human text's own content.

Input: a manifest of clips (audio_filepath, duration; offset when the clip sits inside a longer file), the text in
--field (default text_raw, else text). A clip's recording is its recording/item/episode/session/video_id field, else
its name (<recording>_<start ms>[_<end ms>]); its place in time is its start field, else its name. A recording's text
is every row's text in time order: rows with "context": true add their words and are never checked (lines already
used, spans a text rule left out). Rows written by long_align.py --placed instead carry "words" (a JSON list of the
recording's whole text) and their placed range in it ("w0", "w1").
--need-marks: a row whose text has no punctuation mark at all is left out before the check (it stays context): the
models write cased, punctuated captions, and the final round's YODAS v3 spans were taken only with marks.

usage: python mms_check.py controls <clips.jsonl> <controls.jsonl> [--mode window|fixed] [--foreign same|other]
           [--recordings 40 --min-clips 30 --seed 1] [--calib-max-cer 0.15] [--need-marks] [--device cpu]
       python mms_check.py bars <controls.jsonl> <bars.json> [--placed-controls]   (never over an existing file)
       python mms_check.py check <clips.jsonl> <bars.json> <kept.jsonl> [--min-words 3] [--need-marks]
           [--part 0/4]
check also writes <kept>.decisions.tsv (every clip: its decision and measures). MMS_FA's weights come through
torchaudio on first use. The aligner runs on the GPU when there is one (--device), the path search on the CPU."""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from long_align import SR, keys_of  # noqa: E402

from common.tr_numbers import spoken  # noqa: E402

HOP_S = 320 / SR  # MMS: 20 ms frames
E = 4  # window mode: words of context on each side of a clip's own words
FAR = 300  # a foreign window starts at least this many words away
TRIES = 20
CATCH, MIN_LETTERS, FIT_Q, MIN_FOREIGN = 0.95, 4, 0.99, 100
BINS = (0, 20, 40, 70, 10 ** 6)
MARKS = re.compile(r"[.,?!;:…]")
NAME = re.compile(r"^(.+?)_(\d{6,})(?:_(\d{6,}))?$")
REC_FIELDS = ("recording", "item", "episode", "session", "video_id")
NINF = np.float32(-1e30)


# ------------------------------------------------------------------ the aligner

class Aligner:
    """torchaudio's MMS_FA: each clip heard alone (no padding from a batch), frame log-probabilities minus each
    frame's best; a word's letters as MMS reads them (long_align.keys_of: numbers spelled out, marks and accents off;
    dotless ı as i)."""

    def __init__(self, device, threads):
        import torch
        import torchaudio

        torch.set_num_threads(threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        bundle = torchaudio.pipelines.MMS_FA
        self.torch, self.device = torch, device
        self.model = bundle.get_model(with_star=False).eval().to(device)
        self.dict = bundle.get_dict(star=None)
        self.blank = self.dict["-"]
        self.cache = {}

    def tokens(self, word):
        t = self.cache.get(word)
        if t is None:
            s = "".join(keys_of(word)).replace("ı", "i")
            t = self.cache[word] = [self.dict[c] for c in s if c in self.dict and c != "-"]
        return t

    def scores(self, x):
        torch = self.torch
        with torch.inference_mode():
            em, lens = self.model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device),
                                  torch.tensor([len(x)]).to(self.device))
            lp = torch.log_softmax(em.float(), -1)[0, :int(lens[0])].cpu().numpy()
        return lp - lp.max(axis=1, keepdims=True)


def _forward_np(Rl, skip, dp):
    T, S = Rl.shape
    bp = np.zeros((T, S), np.int8)
    c1 = np.empty(S, np.float32)
    c2 = np.empty(S, np.float32)
    for t in range(1, T):
        c1[0] = NINF
        c1[1:] = dp[:-1]
        c2[:2] = NINF
        c2[2:] = dp[:-2]
        c2[~skip] = NINF
        best = dp.copy()
        ch = np.zeros(S, np.int8)
        m1 = c1 > best
        best[m1] = c1[m1]
        ch[m1] = 1
        m2 = c2 > best
        best[m2] = c2[m2]
        ch[m2] = 2
        dp = best + Rl[t]
        bp[t] = ch
    return dp, bp


try:  # the same steps compiled (the per-frame loop is the check's CPU cost); the same paths
    from numba import njit

    @njit(cache=True)
    def _forward(Rl, skip, dp):
        T, S = Rl.shape
        bp = np.zeros((T, S), np.int8)
        new = np.empty(S, np.float32)
        for t in range(1, T):
            for s in range(S):
                best = dp[s]
                ch = 0
                if s >= 1:
                    c1 = dp[s - 1]
                    if c1 > best:
                        best = c1
                        ch = 1
                if s >= 2 and skip[s]:
                    c2 = dp[s - 2]
                    if c2 > best:
                        best = c2
                        ch = 2
                new[s] = best + Rl[t, s]
                bp[t, s] = ch
            for s in range(S):
                dp[s] = new[s]
        return dp, bp
except ImportError:
    _forward = _forward_np


def viterbi(R, toks, wtok, starts, ends, blank=0):
    """CTC Viterbi of a run of words on a clip, free to start at the first letter of any word in `starts` and to end
    on the last letter of any word in `ends`, covering every frame. R [T, V]; toks: flat letter ids; wtok: per word
    (first token, end token). The state of every frame (even: the blank before token s // 2, odd: token s // 2), or
    None when the text cannot fit."""
    T, L = len(R), len(toks)
    if L == 0 or T == 0:
        return None
    S = 2 * L + 1
    toks = np.asarray(toks)
    lab = np.full(S, blank)
    lab[1::2] = toks
    skip = np.zeros(S, np.bool_)
    skip[3::2] = toks[1:] != toks[:-1]
    dp = np.full(S, NINF, np.float32)
    for k in starts:
        a, b = wtok[k]
        if b > a:
            dp[2 * a] = R[0, blank]
            dp[2 * a + 1] = R[0, toks[a]]
    dp, bp = _forward(np.ascontiguousarray(R[:, lab], dtype=np.float32), skip, dp)
    fin = [x for k in ends for x in ((2 * wtok[k][1] - 1, 2 * wtok[k][1]) if wtok[k][1] > wtok[k][0] else ())]
    if not fin:
        return None
    fin = np.array(fin)
    s = int(fin[np.argmax(dp[fin])])
    if dp[s] < -1e29:
        return None
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    return path


def measure(R, path, toks, wtok, wletters, blank=0):
    """What the forced path says: the words it covers (first, last: positions in the run) and the costs (-R, >= 0)."""
    T = len(R)
    lab = np.full(2 * len(toks) + 1, blank)
    lab[1::2] = toks
    cost = -R[np.arange(T), lab[path]]
    tokframe = path % 2 == 1
    tidx = np.where(tokframe, path // 2, -1)
    tok2word = np.empty(len(toks), np.int32)
    for k, (a, b) in enumerate(wtok):
        tok2word[a:b] = k
    fr = np.flatnonzero(tokframe)
    if len(fr) == 0:
        return None
    wf = tok2word[tidx[fr]]
    first, last = int(wf[0]), int(wf[-1])
    spans = {}
    for f, k in zip(fr, wf):
        k = int(k)
        spans[k] = (spans[k][0], int(f)) if k in spans else (int(f), int(f))
    covered = [k for k in range(first, last + 1) if wtok[k][1] > wtok[k][0]]
    if any(k not in spans for k in covered):
        return None
    wcost = [float(cost[spans[k][0]:spans[k][1] + 1].sum()) for k in covered]
    n_tok = sum(wtok[k][1] - wtok[k][0] for k in covered)
    e0, s1 = spans[first][1], spans[last][0]
    tpl = [float(np.where(tokframe[spans[k][0]:spans[k][1] + 1], cost[spans[k][0]:spans[k][1] + 1], 0.0).sum())
           / max(1, wletters[k]) for k in covered]
    return {"first": first, "last": last, "n_tok": n_tok, "fit": float(-cost.sum() / max(1, n_tok)),
            "lead": float(cost[:spans[first][0]].sum()), "trail": float(cost[spans[last][1] + 1:].sum()),
            "w_first": wcost[0], "w_last": wcost[-1],
            "blank_in": float(np.where(tokframe[:e0 + 1], 0.0, cost[:e0 + 1]).sum()),
            "blank_out": float(np.where(tokframe[s1:], 0.0, cost[s1:]).sum()),
            "tpl_first": tpl[0], "tpl_last": tpl[-1],
            "t0": spans[first][0] * HOP_S, "t1": (spans[last][1] + 1) * HOP_S}


def align(al, R, words, ids, starts=None, ends=None):
    """The aligner on the words words[i] for i in ids, in order (an item that is a string is put in as it is), free to
    start at a word in `starts` and to end at a word in `ends` (indices into words; default any). first and last of
    the result are indices into words."""
    toks, wtok, wl = [], [], []
    for i in ids:
        t = al.tokens(i if isinstance(i, str) else words[i])
        wtok.append((len(toks), len(toks) + len(t)))
        toks += t
        wl.append(len(t))
    pos = list(range(len(ids)))
    st = pos if starts is None else [n for n, i in enumerate(ids) if not isinstance(i, str) and i in set(starts)]
    en = pos if ends is None else [n for n, i in enumerate(ids) if not isinstance(i, str) and i in set(ends)]
    path = viterbi(R, toks, wtok, st, en, al.blank)
    if path is None:
        return None
    m = measure(R, path, np.asarray(toks), wtok, wl, al.blank)
    if m is None:
        return None
    m["first"], m["last"] = ids[m["first"]], ids[m["last"]]
    return m


KEEP = ("fit", "n_tok", "first", "last", "blank_in", "blank_out", "tpl_first", "tpl_last", "lead", "trail", "t0", "t1")


def kept_fields(m):
    return None if m is None else {k: (round(m[k], 4) if isinstance(m[k], float) else m[k]) for k in KEEP}


# ------------------------------------------------------------------ the rows and their recordings

def text_of(r, fields):
    for f in fields:
        if r.get(f):
            return r[f]
    return ""


def recording_of(r):
    for f in REC_FIELDS:
        if r.get(f):
            return str(r[f])
    m = NAME.match(Path(r["audio_filepath"]).stem)
    return m.group(1) if m else Path(r["audio_filepath"]).stem


def start_of(r, k):
    if r.get("start") is not None:
        return float(r["start"])
    m = NAME.match(Path(r["audio_filepath"]).stem)
    if m:
        return int(m.group(2)) / 1000
    return float(r["offset"]) if r.get("offset") is not None else float(k)


def load_rows(path, a):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    fields = a.field.split(",")
    for k, r in enumerate(rows):
        r["_k"], r["_text"] = k, text_of(r, fields)
        r["_why"] = None
        if not r.get("context"):
            if a.need_marks and not MARKS.search(r["_text"]):
                r["_why"] = "no punctuation mark in the text"
    return rows


def recordings(rows, al, mode):
    """{recording: (words, [(row, first word, last word), ...])}: the recording's text as words (fixed mode: only
    words with a letter the aligner reads) and each checked row's own range in it (None: not placed)."""
    by = defaultdict(list)
    for k, r in enumerate(rows):
        by[recording_of(r)].append(r)
    out, words_files = {}, {}
    for rec, rs in by.items():
        rs.sort(key=lambda r: (start_of(r, r["_k"]), r["_k"]))
        placed = [r for r in rs if r.get("words")]
        if placed:  # long_align.py --placed: the whole text and each clip's placed range
            f = placed[0]["words"]
            if f not in words_files:
                words_files[f] = json.loads(Path(f).read_text(encoding="utf-8"))
            words = words_files[f]
            items = [(r, r.get("w0"), r.get("w1")) for r in rs if not r.get("context") and not r["_why"]]
            out[rec] = (words, [(r, w0, w1) if w0 is not None and w1 is not None else (r, None, None)
                                for r, w0, w1 in items])
            continue
        words, items = [], []
        for r in rs:
            ws = r["_text"].split()
            if mode == "fixed":
                ws = [w for w in ws if al.tokens(w)]
            if not r.get("context") and not r["_why"]:
                items.append((r, len(words), len(words) + len(ws) - 1) if ws else (r, None, None))
            words += ws
        out[rec] = (words, items)
    return out


def sound(r):
    import soundfile as sf

    if r.get("offset") is not None:
        with sf.SoundFile(r["audio_filepath"]) as f:
            sr = f.samplerate
            f.seek(int(round(float(r["offset"]) * sr)))
            x = f.read(int(round(float(r["duration"]) * sr)), dtype="float32", always_2d=True)
    else:
        x, sr = sf.read(r["audio_filepath"], dtype="float32", always_2d=True)
    assert sr == SR, f"{r['audio_filepath']}: {sr} Hz (16 kHz mono expected)"
    return np.ascontiguousarray(x.mean(axis=1))


# ------------------------------------------------------------------ the best fit and the controls

def best_fit(al, R, words, w0, w1, mode):
    if mode == "fixed":
        ids = list(range(w0, w1 + 1))
        return align(al, R, words, ids, starts=[ids[0]], ends=[ids[-1]]), None
    a, b = max(0, w0 - E), min(len(words) - 1, w1 + E)
    return align(al, R, words, list(range(a, b + 1))), (a, b)


def controls_of(al, R, words, w0, w1, mode, rng, other):
    """The best fit and the controls of one clip (see the docstring)."""
    N = len(words)
    best, win = best_fit(al, R, words, w0, w1, mode)
    out = {"best": kept_fields(best)}
    letters = lambda k: len(al.tokens(words[k]))  # noqa: E731
    if mode == "fixed":
        ids = list(range(w0, w1 + 1))
        fixed = lambda xs: align(al, R, words, xs, starts=[xs[0]], ends=[xs[-1]])  # noqa: E731
        n = len(ids)
        if other is not None:
            f = fixed_text(al, R, other)
            if f is not None:
                out["foreign"] = {"fit": f["fit"], "letters": f["n_tok"]}
        else:
            for _ in range(TRIES):
                fa = rng.randrange(0, max(1, N - n))
                if abs(fa - w0) >= FAR:
                    f = fixed(list(range(fa, min(N - 1, fa + n - 1) + 1)))
                    if f is not None:
                        out["foreign"] = {"fit": f["fit"], "letters": f["n_tok"]}
                    break
        if n >= 2:
            m = fixed(ids[1:])
            if m is not None:
                out["cut_first"] = {"blank": m["blank_in"], "letters": letters(ids[0])}
            m = fixed(ids[:-1])
            if m is not None:
                out["cut_last"] = {"blank": m["blank_out"], "letters": letters(ids[-1])}
        if w0 > 0:
            m = fixed([w0 - 1] + ids)
            if m is not None:
                out["extra_first"] = {"tpl": m["tpl_first"], "letters": letters(w0 - 1)}
        if w1 + 1 < N:
            m = fixed(ids + [w1 + 1])
            if m is not None:
                out["extra_last"] = {"tpl": m["tpl_last"], "letters": letters(w1 + 1)}
        return out
    a, b = win
    size = b - a + 1
    for _ in range(TRIES):
        fa = rng.randrange(0, max(1, N - size))
        if abs(fa - a) >= FAR:
            f = align(al, R, words, list(range(fa, min(N - 1, fa + size - 1) + 1)))
            if f is not None:
                out["foreign"] = {"fit": f["fit"], "letters": f["n_tok"]}
            break
    if best is None:
        return out
    first, last = best["first"], best["last"]
    if last > first:
        m = align(al, R, words, list(range(first + 1, b + 1)))
        if m is not None:
            out["cut_first"] = {"blank": m["blank_in"], "letters": letters(first)}
        m = align(al, R, words, list(range(a, last)))
        if m is not None:
            out["cut_last"] = {"blank": m["blank_out"], "letters": letters(last)}
    prev = first - 1
    while prev >= 0 and not al.tokens(words[prev]):
        prev -= 1
    if prev >= 0:
        m = align(al, R, words, list(range(prev, last + 1)), starts=[prev], ends=[last])
        if m is not None:
            out["extra_first"] = {"tpl": m["tpl_first"], "letters": letters(prev)}
    nxt = last + 1
    while nxt < N and not al.tokens(words[nxt]):
        nxt += 1
    if nxt < N:
        m = align(al, R, words, list(range(first, nxt + 1)), starts=[first], ends=[nxt])
        if m is not None:
            out["extra_last"] = {"tpl": m["tpl_last"], "letters": letters(nxt)}
    return out


def fixed_text(al, R, text_words):
    ws = [w for w in text_words if al.tokens(w)]
    if not ws:
        return None
    return align(al, R, ws, list(range(len(ws))), starts=[0], ends=[len(ws) - 1])


def controls(a):
    if a.mode == "window" and a.foreign == "other":
        sys.exit("--foreign other is for fixed mode (window mode takes its foreign window from the same recording)")
    rows = load_rows(a.clips, a)
    al = Aligner(a.device, a.threads)
    recs = recordings(rows, al, a.mode)

    def usable(r):
        return a.calib_max_cer is None or (r.get("cer") is not None and r["cer"] <= a.calib_max_cer)

    names = sorted(rec for rec, (_, items) in recs.items()
                   if sum(w0 is not None and usable(r) for r, w0, _ in items) >= a.min_clips)
    if a.recordings:
        names = sorted(random.Random(a.seed).sample(names, min(a.recordings, len(names))))
    by_letters = defaultdict(list)  # --foreign other: the clips' texts by letters (tens), with their recording
    if a.foreign == "other":
        for rec, (words, items) in recs.items():
            for r, w0, w1 in items:
                if w0 is not None:
                    ws = words[w0:w1 + 1]
                    by_letters[min(sum(len(al.tokens(w)) for w in ws) // 10, 30)].append((rec, ws))
    t0, n = time.time(), 0
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"controls": 1, "mode": a.mode, "foreign": a.foreign, "clips": a.clips,
                            "recordings": len(names)}) + "\n")
        for k, rec in enumerate(names):
            words, items = recs[rec]
            for r, w0, w1 in items:
                if w0 is None or not usable(r):
                    continue
                rng = random.Random(r["audio_filepath"])
                other = None
                if a.foreign == "other":
                    pool = [ws for rr, ws in by_letters[min(sum(len(al.tokens(w)) for w in words[w0:w1 + 1]) // 10, 30)]
                            if rr != rec]
                    other = rng.choice(pool) if pool else None
                R = al.scores(sound(r))
                res = controls_of(al, R, words, w0, w1, a.mode, rng, other)
                res["audio_filepath"] = r["audio_filepath"]
                f.write(json.dumps(res) + "\n")
                n += 1
            print(f"  {k + 1}/{len(names)} {rec}: {n} clips so far ({time.time() - t0:.0f} s)", flush=True)
    print(f"controls of {n} clips from {len(names)} recordings -> {a.out}", flush=True)


# ------------------------------------------------------------------ the bars

def fit_bar(bars, letters):
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= letters < hi:
            return bars[f"{lo}-{hi}"]["bar"]


def bars(a):
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists: the bars are fixed once, before any clip is checked")
    lines = [json.loads(x) for x in open(a.controls, encoding="utf-8") if x.strip()]
    head, rows = lines[0], lines[1:]
    assert head.get("controls") == 1, f"{a.controls}: not a controls file"
    fb, prev = {}, None
    for lo, hi in zip(BINS, BINS[1:]):
        f = np.array([r["foreign"]["fit"] for r in rows if r.get("foreign") and lo <= r["foreign"]["letters"] < hi])
        bar = float(np.quantile(f, FIT_Q)) if len(f) >= MIN_FOREIGN else prev
        fb[f"{lo}-{hi}"] = {"bar": bar, "foreign": int(len(f))}
        prev = bar
    if fb[f"{BINS[0]}-{BINS[1]}"]["bar"] is None:  # no bin below the lowest: its foreign texts pooled upwards
        for k in range(2, len(BINS)):
            f = np.array([r["foreign"]["fit"] for r in rows if r.get("foreign") and r["foreign"]["letters"] < BINS[k]])
            if len(f) >= MIN_FOREIGN:
                break
        if len(f) == 0:
            sys.exit("no foreign controls: the recordings' texts are too short for a window 300 words away "
                     "(try --foreign other)")
        pooled = float(np.quantile(f, FIT_Q))
        for lo, hi in zip(BINS, BINS[1:]):
            if fb[f"{lo}-{hi}"]["bar"] is None:
                fb[f"{lo}-{hi}"].update(bar=pooled, pooled=f"0-{BINS[k]}: {len(f)} foreign texts")
    src = rows
    if a.placed_controls:
        src = [r for r in rows if r.get("best") and r["best"]["fit"] >= fit_bar(fb, r["best"]["n_tok"])]
    cut = np.array([r[k]["blank"] for r in src for k in ("cut_first", "cut_last")
                    if r.get(k) and r[k]["letters"] >= MIN_LETTERS])
    traceless = 0
    if a.placed_controls:
        traceless = int((cut <= 1e-6).sum())
        cut = cut[cut > 1e-6]
    extra = np.array([r[k]["tpl"] for r in src for k in ("extra_first", "extra_last") if r.get(k)])
    if not len(cut) or not len(extra):
        sys.exit("no edge controls")
    th = {"bars": 1, "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": head["mode"], "foreign": head["foreign"],
          "controls_file": str(a.controls), "recordings": head["recordings"], "clips": len(rows),
          "catch": CATCH, "min_letters": MIN_LETTERS, "fit_quantile": FIT_Q, "placed_controls": a.placed_controls,
          "fit_bar": fb,
          "edge_blank": {"max": float(np.quantile(cut, 1 - CATCH)), "controls": int(len(cut)),
                         "traceless_left_out": traceless,
                         "control": "first or last word (4+ letters) taken out: blank-path cost at that edge"},
          "edge_letter": {"max": float(np.quantile(extra, 1 - CATCH)), "controls": int(len(extra)),
                          "control": "the word before the first / after the last forced in: its letter cost per "
                                     "letter"}}
    if a.placed_controls:
        th["edge_controls_from"] = f"{len(src)} clips whose own text fits above the foreign bar"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(th, indent=1, ensure_ascii=False))
    tmp.replace(out)
    print(json.dumps(th, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ the check

def decide(m, th, n_words, mode, min_words):
    """The first rule the clip fails (None: kept)."""
    if n_words == 0:
        return "no word the aligner can read"
    if m is None:
        return "no path (the text cannot fit the clip)"
    if m["fit"] < fit_bar(th["fit_bar"], m["n_tok"]):
        return "misplaced: fits no better than foreign text"
    if max(m["blank_in"], m["blank_out"]) > th["edge_blank"]["max"]:
        return "edge: speech the text lacks"
    if max(m["tpl_first"], m["tpl_last"]) > th["edge_letter"]["max"]:
        return "edge: text the speech lacks"
    if (m["last"] - m["first"] + 1 if mode == "window" else n_words) < min_words:
        return f"under {min_words} words"
    return None


def check(a):
    th = json.loads(Path(a.bars).read_text())
    assert th.get("bars") == 1, f"{a.bars}: not a bars file"
    mode = th["mode"]
    rows = load_rows(a.clips, a)
    al = Aligner(a.device, a.threads)
    recs = recordings(rows, al, mode)
    part, of = (int(x) for x in a.part.split("/"))
    names = [rec for k, rec in enumerate(sorted(recs)) if k % of == part]
    t0 = time.time()
    tally, hours = Counter(), Counter()
    out = Path(a.out)
    with open(out, "w", encoding="utf-8") as fo, open(f"{out}.decisions.tsv", "w", encoding="utf-8") as fd:
        for r in rows:
            if r["_why"] and recording_of(r) in names:
                tally[r["_why"]] += 1
                fd.write(f"{r['audio_filepath']}\t{r['_why']}\n")
        for k, rec in enumerate(names):
            words, items = recs[rec]
            res = []
            for r, w0, w1 in items:
                if w0 is None:
                    res.append((r, None, "not placed in the recording's text" if r.get("words") else
                                "no word the aligner can read"))
                    continue
                m, _ = best_fit(al, al.scores(sound(r)), words, w0, w1, mode)
                n_words = sum(1 for w in words[w0:w1 + 1] if al.tokens(w))
                res.append((r, m, decide(m, th, n_words, mode, a.min_words)))
            if mode == "window":  # two kept clips never share a word of the text: both go
                kept = [j for j, (_, _, d) in enumerate(res) if d is None]
                for x, y in zip(kept, kept[1:]):
                    if res[x][1]["last"] >= res[y][1]["first"]:
                        res[x] = (*res[x][:2], "two clips claim the same words")
                        res[y] = (*res[y][:2], "two clips claim the same words")
            for r, m, d in res:
                tally[d or "kept"] += 1
                hours[d or "kept"] += float(r.get("duration") or 0) / 3600
                ms = "\t".join(f"{m[x]:.3f}" if m else ""
                               for x in ("fit", "blank_in", "blank_out", "tpl_first", "tpl_last"))
                fd.write(f"{r['audio_filepath']}\t{d or 'kept'}\t{ms}\t{m['n_tok'] if m else ''}\n")
                if d is not None:
                    continue
                row = {x: v for x, v in r.items() if not x.startswith("_")}
                if mode == "window":
                    row["text_placed"] = r["_text"]
                    row["text_raw"] = " ".join(words[m["first"]:m["last"] + 1]).strip(" ,;:")
                    row["text_spoken"] = spoken(row["text_raw"])
                    for x in ("text", "text_std", "heard", "cer", "w0", "w1"):  # they belonged to the placed text
                        row.pop(x, None)
                row["mms_fit"] = round(m["fit"], 3)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {k + 1}/{len(names)} {rec}: {sum(d is None for *_, d in res)} kept of {len(res)} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    print(f"bars fixed at {th['fixed_at']} ({mode} mode)")
    for d, n in tally.most_common():
        print(f"  {d}: {n} clips, {hours[d]:.2f} h")
    print(f"kept: {tally['kept']} clips -> {out} (every decision: {out}.decisions.tsv)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--field", default="text_raw,text", help="the text field(s), first present wins")
        p.add_argument("--need-marks", action="store_true", help="a text with no punctuation mark at all is left out")
        p.add_argument("--device", default="auto", help="the aligner's device: auto, cpu or cuda")
        p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("controls", help="the best fit and the controls of every clip (or a sample of recordings)")
    p.add_argument("clips")
    p.add_argument("out")
    p.add_argument("--mode", choices=("window", "fixed"), default="window")
    p.add_argument("--foreign", choices=("same", "other"), default="same",
                   help="foreign text from the same recording (300+ words away) or from another recording's clip")
    p.add_argument("--recordings", type=int, default=0, help="a random sample of this many recordings (0: all)")
    p.add_argument("--min-clips", type=int, default=1, help="sample only recordings with this many clips")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--calib-max-cer", type=float, default=None,
                   help="controls only on rows whose listener letter error (cer) is at most this")
    common(p)
    p = sub.add_parser("bars", help="the bars from the controls, fixed in a new file")
    p.add_argument("controls")
    p.add_argument("out")
    p.add_argument("--placed-controls", action="store_true",
                   help="edge controls only from clips that pass the fit bar; traceless edge cuts left out")
    p = sub.add_parser("check", help="every clip against the fixed bars; the kept rows")
    p.add_argument("clips")
    p.add_argument("bars")
    p.add_argument("out")
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--part", default="0/1", help="K/N: only every N-th recording from the K-th (to run N at once)")
    common(p)
    a = ap.parse_args()
    {"controls": controls, "bars": bars, "check": check}[a.cmd](a)


if __name__ == "__main__":
    main()
