"""Moonshine Streaming (English, MIT; Small or Medium) trained further on Turkish.

- Vocabulary: with --sp, a Turkish (or Turkish + English) SentencePiece replaces the base model's English one when
  the run starts from the English base (--base, no --init): the decoder's embeddings and output layer are resized,
  each new piece starting from the mean of the English pieces that spell it, so the decoder starts from what it
  knew. Labels are the pieces plus the end id (2); the model puts the start id (1) in front itself.
- The encoder streams only when given an attention mask (its sliding windows are built from it), so every batch
  passes one, in training and in decoding.
- Data: a weighted mix (data/build_mix.py's mix.json: one jsonl manifest per source and its sampling weight).
  Batches are cut from pools of clips sorted by length (--bucket), each source read in a shuffled order without
  repeats (--no-repeat).
- Augmentation: speed 0.9/1/1.1, room echo, loudness changes, backgrounds under speech (noise, music, babble from a
  MUSAN-style folder), broadcast-like channel effects with a clean share (--aug2), SpecAugment-style masks on the
  encoder's post-conv features (--specaug) and decoder dropout (--dec-dropout).
- Schedules: cosine over --hours; wsd (warm-up, stable at the peak, a cosine decay over the last --decay-frac; stop
  any time and finish with a short decay run); decay (no warm-up: --lr down to --final-lr over --hours from now,
  which is how a plateau state is cooled down).
- Checks every --eval-every steps on up to four dev lists (--val: a read-speech dev first, a drama dev second, then
  optional extra devs), greedy, scored with the strict scorer's normalisation and as written. Each check saves
  <out>/last and <out>/state.pt (resumable) and, with --snapshots N, the last N checks' weights for train/moon_average.py.

usage: python moon_train.py <out dir> --base <moonshine dir> --sp <tokenizer.model> --mix <mix.json>
       --val <dev1.jsonl,dev2.jsonl[,...]> [--init <model dir>] [--resume] [--hours 6] [--batch-s 240] [--lr 3e-4]
       [--musan-dir DIR] [--rirs FILE] [--bg-exclude FILE,...] [--dry]
Relative paths are read under --root (default: the environment variable QASR_ROOT, else the current folder)."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]  # tempo.py, common/
from common.scoring import edits, strict_words  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--root", default=os.environ.get("QASR_ROOT", "."), help="where relative paths are read from")
ap.add_argument("--base", required=True, help="the English Moonshine Streaming model folder (its processor is used)")
ap.add_argument("--hours", type=float, default=6.0)
ap.add_argument("--batch-s", type=float, default=240.0)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--dry", action="store_true", help="CPU, two steps on a few clips, nothing saved")
ap.add_argument("--accumulate", type=int, default=1, help="pieces of --batch-s seconds a step (memory)")
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (a fit test)")
ap.add_argument("--checkpointing", type=int, default=1, help="gradient checkpointing (1) or not (0)")
ap.add_argument("--sp", default="", help="a SentencePiece model in place of the base model's tokenizer ('' keeps theirs)")
ap.add_argument("--init", default="", help="a fine-tuned model dir (new pieces already in): go on from it, no swap")
ap.add_argument("--mix", required=True, help="mix.json: [{manifest_filepath, weight}, ...]")
ap.add_argument("--val", required=True,
                help="dev manifests, comma-separated: a read-speech dev, a drama dev, then up to two more")
ap.add_argument("--val-max", type=int, default=300, help="clips read from each dev list but the first")
ap.add_argument("--musan-dir", default="", help="a folder with noise/, music/ and speech/ (16 kHz wav or ogg) for "
                "backgrounds under speech; '' = none")
ap.add_argument("--rirs", default="", help="room impulse responses: a text file of paths, or a .jsonl(.gz) whose "
                "rows name a file in sources[0].source; '' = no echo")
ap.add_argument("--bg-exclude", default="", help="files never used as backgrounds (lists of paths, comma-separated): "
                "the sources of a no-speech or song test")
ap.add_argument("--musan", type=int, default=1, help="backgrounds under speech (1) or none (0)")
ap.add_argument("--augment", type=int, default=1, help="speed 0.9/1/1.1, room echo, loudness changes (1) or none (0)")
ap.add_argument("--specaug", type=float, default=0.0, help="share of training clips whose post-conv encoder features get "
                "small time masks (<=160 ms, about one per 3 s, so no word is lost) and two channel masks (each <=1/8 "
                "of the width); 0 = off")
ap.add_argument("--dec-dropout", type=float, default=0.0, help="dropout in the decoder's self/cross attention and on "
                "its MLP outputs; 0 = off (Moonshine's default)")
ap.add_argument("--aug2", type=int, default=0, help="broadcast-like channels (compressor, EQ, band limits, pitch, "
                "short dropouts) and a clean share with no added sound at all (1), or neither (0)")
ap.add_argument("--tempo", type=float, default=0.0, help="share of speech clips played faster with their pitch kept "
                "(tempo.py, WSOLA) at a factor drawn from [1.1, --tempo-max], in place of the speed step; labels as "
                "they are, and a clip only gets shorter, so --max-s holds; 0 = off (the released models: off)")
ap.add_argument("--tempo-max", type=float, default=1.6, help="--tempo: the fastest factor")
ap.add_argument("--babble", type=int, default=1, help="other voices under speech too, at a low level (on by default)")
ap.add_argument("--max-s", type=float, default=25.0, help="longest clip heard, s")
ap.add_argument("--bucket", type=int, default=1, help="draw this many batches' worth of clips, sort by length and "
                "cut batches from them: less padding (a mixed batch without it was about half sound)")
ap.add_argument("--best-by", default="mean", choices=["mean", "drama"],
                help="best = the mean of the first two dev lists, or the second (drama) dev alone")
ap.add_argument("--snapshots", type=int, default=0, help="keep the last N checks' weights as snap-<step> (to average)")
ap.add_argument("--no-repeat", type=int, default=0, help="draw each source in a shuffled order without repeats, each "
                "data worker its own slice (1), or with replacement (0: at ~1 pass a third of a source goes unheard)")
ap.add_argument("--carry", type=int, default=0, help="--bucket: the clips left over after cutting a pool into batches "
                "start the next pool (1) instead of making a small batch of their own (0)")
ap.add_argument("--lr-file", default="", help="a file whose first word is the peak learning rate (a runner may lower "
                "it between resumes); read only when named")
ap.add_argument("--schedule", default="cosine", choices=["cosine", "wsd", "decay"],
                help="cosine over --hours; wsd: warmup, stable at the peak, a cosine decay over the last --decay-frac; "
                "decay: no warmup, the peak --lr down to --final-lr over --hours from now (finishing a run that "
                "stopped high)")
ap.add_argument("--decay-frac", type=float, default=0.2)
ap.add_argument("--final-lr", type=float, default=0.0, help="the learning rate a wsd/decay schedule ends at "
                "(0: 2%% of --lr)")
ap.add_argument("--resume", action="store_true", help="go on from <out>/last + <out>/state.pt (optimizer, step, "
                "hours, time)")
A = ap.parse_args()
ROOT = Path(A.root)


def under_root(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


if A.lr_file and under_root(A.lr_file).exists():
    A.lr = float(under_root(A.lr_file).read_text().split()[0])

MAX_S = A.max_s
PAD, START, END = 0, 1, 2


class Pieces:
    """Text <-> ids: the Turkish SentencePiece, or their tokenizer (its own start id dropped)."""

    def __init__(self, processor):
        self.sp = None
        if A.sp:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=str(under_root(A.sp)))
        self.tok = processor.tokenizer

    def size(self):
        return self.sp.get_piece_size() if self.sp else len(self.tok)

    def encode(self, text):
        ids = self.sp.encode(text) if self.sp else self.tok(text, add_special_tokens=False).input_ids
        return ids + [END]

    def decode(self, ids):
        ids = [int(i) for i in ids if int(i) > 3]
        return self.sp.decode(ids) if self.sp else self.tok.decode(ids, skip_special_tokens=True)


def labels_of(pieces, texts):
    seqs = [pieces.encode(t) for t in texts]
    longest = max(len(q) for q in seqs)
    return torch.tensor([q + [-100] * (longest - len(q)) for q in seqs])


def swap_vocabulary(model, processor, pieces):
    """The decoder's embeddings and output layer for the Turkish pieces: each starts from the
    mean of the old (English) pieces that spell it."""
    old_in = model.get_input_embeddings().weight.data
    old_out = model.get_output_embeddings().weight.data
    n = pieces.size()
    new_in = old_in.new_zeros((n, old_in.shape[1]))
    new_out = old_out.new_zeros((n, old_out.shape[1]))
    tok = processor.tokenizer
    for i in range(n):
        if i in (PAD, START, END, 3):
            j = {PAD: tok.pad_token_id or 0, START: 1, END: 2, 3: tok.unk_token_id or 0}[i]
            new_in[i], new_out[i] = old_in[j], old_out[j]
            continue
        text = pieces.sp.id_to_piece(i).replace("\u2581", " ")
        old = tok(text, add_special_tokens=False).input_ids or [tok.unk_token_id or 0]
        new_in[i] = old_in[old].mean(0)
        new_out[i] = old_out[old].mean(0)
    model.resize_token_embeddings(n)
    model.get_input_embeddings().weight.data.copy_(new_in)
    model.get_output_embeddings().weight.data.copy_(new_out)
    model.config.vocab_size = n
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "vocab_size"):
        model.config.decoder.vocab_size = n


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# TV-like backgrounds: noise, music (with vocals too: background songs) and babble (3-7 speakers at once), under a
# clip at random levels. Drama seldom (it has its own music), the clean sources often, no-speech clips never. The
# share of clips that get one is set by the source's name (its manifest's file name).
AUG_SHARE = {"parliament": 0.4, "speeches": 0.4, "cv": 0.5, "fleurs": 0.5, "english": 0.5, "nonspeech": 0.0}
SNR_DB = {"noise": (5.0, 20.0), "music": (5.0, 15.0), "babble": (15.0, 25.0)}
BG_WEIGHT = {"noise": 0.425, "music": 0.425, "babble": 0.15}  # babble: a low amount, and quiet
# The FastConformer recipe besides (speed perturbation, room echo on 30 %), plus loudness changes for half the
# clips (whispers and shouts): Moonshine reads the raw waveform, so level reaches the model directly
SPEED = (0.9, 1.0, 1.1)
RIR_SHARE, GAIN_SHARE, GAIN_DB = 0.3, 0.5, (-15.0, 6.0)
TEMPO_MIN = 1.1  # --tempo: the slowest of its factors (speed's 1.1 is the fastest the other clips hear)


def aug_share(name):
    if name.startswith(("drama", "films")):  # their own music and sound already
        return 0.1
    if name.startswith("books"):  # clean read narration, like the read sets
        return 0.5
    if name == "nonspeech" or name.startswith("songs"):  # no speech under them (songs_tr: music already under the voice)
        return 0.0
    return AUG_SHARE.get(name, 0.4)


def background_pools():
    """Background files by kind; the files a test drew from (--bg-exclude) are never heard."""
    if not A.musan_dir:
        return {}
    musan = under_root(A.musan_dir)
    test = set()
    for listing in filter(None, A.bg_exclude.split(",")):
        test |= {line.strip() for line in open(under_root(listing), encoding="utf-8") if line.strip()}
    pools = {}
    kinds = (("noise", "noise"), ("music", "music")) + ((("babble", "speech"),) if A.babble else ())
    for kind, folder in kinds:
        found = []
        for f in sorted(list((musan / folder).rglob("*.wav")) + list((musan / folder).rglob("*.ogg"))):
            if str(f) in test:
                continue
            try:
                info = sf.info(str(f))
            except Exception:
                continue
            if info.samplerate == 16000 and info.frames > 16000:
                found.append((str(f), info.frames))
        if found:
            pools[kind] = found
    return pools


def piece(pool, n, rng):
    path, frames = rng.choice(pool)
    start = rng.randrange(frames - n) if frames > n else 0
    try:
        y, _ = sf.read(path, dtype="float32", start=start, frames=n)
    except Exception:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        return None
    return np.tile(y, n // len(y) + 1)[:n] if len(y) < n else y


def rir_pool():
    """Room impulse responses: a .jsonl(.gz) of recordings (each naming its file in sources[0].source, as lhotse
    writes them) or a plain list of paths."""
    if not A.rirs:
        return []
    import gzip

    path = under_root(A.rirs)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if ".jsonl" in path.name:
        return [json.loads(line)["sources"][0]["source"] for line in lines]
    return lines


def speed(x, rng):
    """Kaldi-style speed perturbation: tempo and pitch together, by resampling."""
    s = rng.choice(SPEED)
    if s == 1.0:
        return x
    from scipy.signal import resample_poly

    return resample_poly(x, 10, 9 if s == 0.9 else 11).astype(np.float32)


def faster(x, rng):
    """--tempo: the clip 1.1 to --tempo-max times as fast, its pitch kept (tempo.py's WSOLA, ~700x real time)."""
    from tempo import wsola

    return wsola(x, rng.uniform(TEMPO_MIN, A.tempo_max))


def echo(x, rirs, rng):
    """A simulated room: the clip convolved with a room impulse response, loudness kept."""
    try:
        h, _ = sf.read(rng.choice(rirs), dtype="float32")
    except Exception:
        return x
    if h.ndim > 1:
        h = h[:, 0]
    from scipy.signal import fftconvolve

    y = fftconvolve(x, h / (np.abs(h).max() + 1e-9))[: len(x)]
    y *= math.sqrt((float(np.mean(x ** 2)) + 1e-10) / (float(np.mean(y ** 2)) + 1e-10))
    return y.astype(np.float32)


def level(x, rng):
    """Quieter or louder as a whole; past full scale it saturates softly, like shouting into a limiter."""
    y = x * 10 ** (rng.uniform(*GAIN_DB) / 20)
    return (np.tanh(y) if float(np.abs(y).max()) > 1.0 else y).astype(np.float32)


# --aug2: each pass should sound new, both ways: a clean share hears no added sound at all, the rest the TV's own
# channel effects too.
CLEAN_SHARE = 0.4  # of speech clips: no echo, background or channel (speed and level still vary)
CHANNEL = {"compress": 0.0, "eq": 0.3, "band": 0.15, "pitch": 0.1, "dropout": 0.3}
# Moonshine normalises every 5 ms frame (MoonshineStreamingFrameCMVN), so a compressor or a gain barely changes what
# it hears (about 1 % of its features); EQ (26 %), band limits (56 %) and zeroed spans (100 %) do. A dropout is zeros,
# 1-3 spans of 20-60 ms, at most 6 % of a clip, never on clips under 2 s (a short word must not vanish from under its
# label)


def _biquad_peak(f0, gain_db, q, sr=16000):
    a = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / sr
    alpha = math.sin(w) / (2 * q)
    b = [1 + alpha * a, -2 * math.cos(w), 1 - alpha * a]
    den = [1 + alpha / a, -2 * math.cos(w), 1 - alpha / a]
    return [v / den[0] for v in b], [v / den[0] for v in den]


def channel(x, rng):
    """What broadcast does to a voice, each effect on its own share: a compressor (TV's even loudness),
    an EQ (two random bands, +-6 dB), a band limit (phone 300-3400 Hz, or a low-pass at 4-7 kHz), a pitch
    shift (+-1.5 semitones, tempo kept), short dropouts (one to three 30-80 ms holes, a codec's)."""
    from scipy.signal import butter, lfilter, sosfilt

    y = x
    if rng.random() < CHANNEL["compress"]:
        env = np.sqrt(lfilter([0.002], [1, -0.998], y.astype(np.float64) ** 2) + 1e-10)
        level_db = 20 * np.log10(env)
        thr, ratio = rng.uniform(-32.0, -18.0), rng.uniform(3.0, 6.0)
        gain_db = np.where(level_db > thr, (thr - level_db) * (1 - 1 / ratio), 0.0)
        y = y * 10 ** (gain_db / 20)
        y = y * math.sqrt((float(np.mean(x ** 2)) + 1e-10) / (float(np.mean(y ** 2)) + 1e-10))
    if rng.random() < CHANNEL["eq"]:
        for _ in range(2):
            b, a = _biquad_peak(rng.uniform(150, 6000), rng.uniform(-6, 6), rng.uniform(0.7, 2.0))
            y = lfilter(b, a, y)
    if rng.random() < CHANNEL["band"]:
        if rng.random() < 0.3:
            y = sosfilt(butter(6, [300, 3400], btype="band", fs=16000, output="sos"), y)
        else:
            y = sosfilt(butter(6, rng.uniform(4000, 7000), btype="low", fs=16000, output="sos"), y)
    if rng.random() < CHANNEL["pitch"]:
        # the pitch moved by resampling, the length kept by WSOLA (a phase-vocoder pitch shift left speech nearly
        # unreadable: 48-57 % WER on shifted dev sound against 18-27 % clean; this way costs 1-3 points)
        from scipy.signal import resample_poly
        from tempo import wsola

        r = 2 ** (rng.uniform(-1.5, 1.5) / 12)
        z = resample_poly(np.asarray(y, dtype=np.float32), 100, int(round(100 * r)))
        y = wsola(z, len(z) / len(y)).astype(np.float32)
    if rng.random() < CHANNEL["dropout"] and len(y) >= 32000:
        y = np.array(y, copy=True)
        room = int(0.06 * len(y))
        for _ in range(rng.randint(1, 3)):
            n = min(int(16000 * rng.uniform(0.02, 0.06)), room)
            if n <= 0:
                break
            at = rng.randrange(0, len(y) - n)
            y[at:at + n] = 0.0
            room -= n
    peak = float(np.abs(y).max())
    return (y / peak * 0.99 if peak > 1 else y).astype(np.float32)


def with_background(x, pools, rng):
    kinds = sorted(pools)
    kind = rng.choices(kinds, weights=[BG_WEIGHT[k] for k in kinds])[0]
    if kind == "babble":
        parts = [piece(pools["babble"], len(x), rng) for _ in range(rng.randint(3, 7))]
        parts = [q for q in parts if q is not None]
        if not parts:
            return x
        bg = np.sum(parts, axis=0)
    else:
        bg = piece(pools[kind], len(x), rng)
        if bg is None:
            return x
    snr = rng.uniform(*SNR_DB[kind])
    px, pb = float(np.mean(x ** 2)) + 1e-10, float(np.mean(bg ** 2)) + 1e-10
    y = x + bg * math.sqrt(px / (pb * 10 ** (snr / 10)))
    peak = float(np.abs(y).max())
    return (y / peak * 0.99).astype(np.float32) if peak > 1 else y.astype(np.float32)


class Mix(IterableDataset):
    """Endless batches (lists of (audio, text)) sampled from the sources by their weights, each
    batch up to --batch-s seconds of sound, features made in the worker."""

    def __init__(self, mix, processor):
        kept = [(load(under_root(m["manifest_filepath"])), m["weight"], Path(m["manifest_filepath"]).stem) for m in mix]
        kept = [k for k in kept if k[0]]
        self.sources = [(rows, w) for rows, w, _ in kept]
        self.aug = [aug_share(name) for _, _, name in kept]
        self.pools = background_pools() if A.musan else {}
        self.rirs = rir_pool() if A.augment else []
        self.speech = [name not in ("nonspeech", "songs") for _, _, name in kept]
        self.tempo = [ok and not name.startswith("songs") for ok, (_, _, name) in zip(self.speech, kept)]  # sung: no
        print("augment:", {"speed": SPEED, "echo": RIR_SHARE if self.rirs else 0, "level": GAIN_SHARE if A.augment else 0,
                           "rirs": len(self.rirs)}
              | ({"tempo": A.tempo, "factors": (TEMPO_MIN, A.tempo_max)} if A.tempo > 0 else {}), flush=True)
        print("backgrounds:", {k: len(v) for k, v in self.pools.items()},
              {name: share for (_, _, name), share in zip(kept, self.aug)}, flush=True)
        self.processor = processor

    def draw(self, rng, weights):
        """One clip, freshly augmented, or None."""
        k = rng.choices(range(len(self.sources)), weights=weights)[0]
        if A.no_repeat:
            order, at = self._order[k], self._at[k]
            if at >= len(order):  # this worker's slice heard once: a new shuffled pass
                rng.shuffle(order)
                at = 0
            r = self.sources[k][0][order[at]]
            self._at[k] = at + 1
        else:
            r = rng.choice(self.sources[k][0])
        if r["duration"] > MAX_S or r["duration"] < 0.5:
            return None
        try:
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
        except Exception:
            return None
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            return None
        clean = A.aug2 and self.speech[k] and rng.random() < CLEAN_SHARE  # --aug2: no added sound at all
        fast = A.tempo > 0 and self.tempo[k] and rng.random() < A.tempo  # off: no draw, the same sequence as before
        if fast:
            x = faster(x, rng)
        if A.augment and self.speech[k]:
            x = x if fast else speed(x, rng)
            if self.rirs and not clean and rng.random() < RIR_SHARE:
                x = echo(x, self.rirs, rng)
        if self.pools and not clean and rng.random() < self.aug[k]:
            x = with_background(x, self.pools, rng)
        if A.aug2 and not clean:
            x = channel(x, rng)
        if A.augment and rng.random() < GAIN_SHARE:
            x = level(x, rng)
        return x, r["text"]

    def batch(self, items):
        items.sort(key=lambda t: len(t[0]))
        feats = self.processor([x for x, _ in items], sampling_rate=16000, return_tensors="pt",
                               padding=True, return_attention_mask=True)
        return feats["input_values"], feats["attention_mask"], [t for _, t in items], sum(len(x) for x, _ in items) / 16000

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1234 + (info.id if info else 0) + int(time.time()))
        if A.no_repeat:  # this worker's slice of every source, in a shuffled order
            wid, nw = (info.id, info.num_workers) if info else (0, 1)
            self._order = []
            for rows, _ in self.sources:
                order = list(range(wid, len(rows), nw)) or list(range(len(rows)))
                rng.shuffle(order)
                self._order.append(order)
            self._at = [0] * len(self.sources)
        weights = [w for _, w in self.sources]
        carry = []  # --carry: the last pool's leftover clips
        while True:
            pool, total = carry, sum(len(t[0]) / 16000 for t in carry)
            carry = []
            while total < A.batch_s * A.bucket and len(pool) < 128 * A.bucket:
                got = self.draw(rng, weights)
                if got is not None:
                    pool.append(got)
                    total += len(got[0]) / 16000
            if A.bucket <= 1:
                yield self.batch(pool)
                continue
            pool.sort(key=lambda t: len(t[0]))  # neighbours in length share a batch: little padding
            batches, cur, cur_s = [], [], 0.0
            for item in pool:
                secs = len(item[0]) / 16000
                if cur and (cur_s + secs > A.batch_s or len(cur) >= 128):
                    batches.append(cur)
                    cur, cur_s = [], 0.0
                cur.append(item)
                cur_s += secs
            if cur and A.carry and batches:
                carry = cur  # they open the next pool, among clips of every length
            elif cur:
                batches.append(cur)
            rng.shuffle(batches)
            for b in batches:
                yield self.batch(b)


def evaluate(model, processor, pieces, rows, device, bs=16):
    model.eval()
    hyps = []
    with torch.inference_mode():
        for i in range(0, len(rows), bs):
            part = rows[i: i + bs]
            audio = [sf.read(r["audio_filepath"], dtype="float32")[0] for r in part]
            audio = [x.mean(axis=1) if x.ndim > 1 else x for x in audio]
            f = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
            longest = max(len(x) for x in audio) / 16000
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model.generate(input_values=f["input_values"].to(device), attention_mask=f["attention_mask"].to(device),
                                     max_new_tokens=int(8 * longest) + 16, decoder_start_token_id=START,
                                     eos_token_id=END, pad_token_id=PAD)
            hyps += [pieces.decode(o) for o in out]
    model.train()
    e = n = 0
    for r, h in zip(rows, hyps):  # the strict scorer's normalisation (digits spelled, case and marks aside)
        a, b = strict_words(r["text"]), strict_words(h)
        e += edits(a, b)
        n += len(a)
    # as written, as NeMo's own checks count it (capitals and punctuation count): the curve beside a
    # FastConformer's val_wer
    se = sn = 0
    for r, h in zip(rows, hyps):
        a, b = r["text"].split(), h.split()
        se += edits(a, b)
        sn += len(a)
    return 100 * e / max(1, n), hyps, 100 * se / max(1, sn)


def regularise(model):
    """--specaug: masks on the encoder's 20 ms post-conv features, training only (the embedder sits outside the
    checkpointed layers, so its masks are drawn once). --dec-dropout: attention dropout plus dropout on each decoder
    MLP's output; torch's checkpointing replays the same RNG, so recomputed layers get the same masks."""
    rng = random.Random(1234 + int(time.time()))
    if A.specaug > 0:
        def mask_features(module, inputs, out):
            if not module.training:
                return out
            h, pm = out
            n, t_all, width = h.shape
            keep = torch.ones(n, t_all, 1, device=h.device, dtype=h.dtype)
            chan = torch.ones(n, 1, width, device=h.device, dtype=h.dtype)
            valid = pm.sum(-1).tolist() if pm is not None else [t_all] * n
            for b in range(n):
                if rng.random() >= A.specaug:
                    continue
                t = int(valid[b])
                for _ in range(min(10, max(1, t // 150))):
                    w = rng.randint(0, 8)
                    if w and t > w:
                        s0 = rng.randint(0, t - w)
                        keep[b, s0:s0 + w] = 0
                for _ in range(2):
                    w = rng.randint(0, width // 8)
                    if w:
                        s0 = rng.randint(0, width - w)
                        chan[b, :, s0:s0 + w] = 0
            return h * keep * chan, pm
        model.model.encoder.embedder.register_forward_hook(mask_features)
    if A.dec_dropout > 0:
        p = A.dec_dropout
        for layer in model.model.decoder.layers:
            layer.self_attn.attention_dropout = p
            layer.encoder_attn.attention_dropout = p
            layer.mlp.register_forward_hook(lambda m, i, o: torch.nn.functional.dropout(o, p, m.training))
    print(f"regularisers: specaug {A.specaug}, decoder dropout {A.dec_dropout}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() and not A.dry else "cpu"
    out = Path(A.out)
    out.mkdir(parents=True, exist_ok=True)
    resumed = A.resume and (out / "state.pt").exists()
    base = under_root(A.base)
    src = out / "last" if resumed else (under_root(A.init) if A.init else base)
    processor = AutoProcessor.from_pretrained(base)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(src).to(device)
    print("model from", src, flush=True)
    if A.checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()  # 16 GB: activations recomputed in the backward pass
        model.config.use_cache = False
    regularise(model)
    model.train()
    pieces = Pieces(processor)
    if pieces.sp and src == base:  # only the English base needs the new pieces swapped in
        swap_vocabulary(model, processor, pieces)
        model.to(device)
    print("pieces:", pieces.size(), "| a sample's labels:", pieces.encode("Merhaba dünya, İstanbul'da."),
          "| decoder start", model.config.decoder_start_token_id, "eos", model.config.eos_token_id, flush=True)
    mix = json.load(open(under_root(A.mix)))
    data = Mix(mix, processor)
    loader = DataLoader(data, batch_size=None, num_workers=0 if A.dry else A.workers, prefetch_factor=None if A.dry else 4)
    opt = torch.optim.AdamW(model.parameters(), lr=A.lr, betas=(0.9, 0.98), weight_decay=0.01)
    vals = A.val.split(",")
    names = [Path(v).stem for v in vals]
    fleurs = load(under_root(vals[0]))  # the read-speech dev (FLEURS dev here), whole
    drama = load(under_root(vals[1]))[:A.val_max]  # the drama dev
    third = load(under_root(vals[2]))[:A.val_max] if len(vals) > 2 else []  # optional extra devs
    fourth = load(under_root(vals[3]))[:A.val_max] if len(vals) > 3 else []
    if A.dry:
        fleurs, drama = fleurs[:4], drama[:4]
    budget = A.hours * 3600 if not A.dry else 60
    step, seen, best, before = 0, 0.0, 1e9, 0.0
    if resumed:
        st = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        step, seen, best, before = st["step"], st["seen"], st["best"], st["elapsed"]
        print(f"resumed at step {step}, {seen / 3600:.0f} h heard, {before / 3600:.2f} h in", flush=True)
    started = time.time() - before
    decay_from = time.time()  # --schedule decay: its --hours count from this start, not the run's
    log = open(out / "train.log", "a")
    micro = 0
    for x, mask, texts, secs in loader:
        elapsed = time.time() - started
        clock = time.time() - decay_from if A.schedule == "decay" else elapsed
        if clock > budget or (A.dry and step >= 2) or (A.max_steps and step >= A.max_steps):
            break
        frac = min(1.0, clock / budget)
        final = A.final_lr or 0.02 * A.lr
        if A.schedule == "cosine":
            lr = A.lr * min(1.0, (step + 1) / A.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        elif A.schedule == "wsd":
            d = max(0.0, (frac - (1 - A.decay_frac)) / A.decay_frac)
            lr = min(1.0, (step + 1) / A.warmup) * (final + (A.lr - final) * 0.5 * (1 + math.cos(math.pi * d)))
        else:
            lr = final + (A.lr - final) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        labels = labels_of(pieces, texts)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_values=x.to(device), attention_mask=mask.to(device), labels=labels.to(device)).loss
            (loss / A.accumulate).backward()
        except torch.OutOfMemoryError:  # an unusually long batch: skipped, training goes on
            opt.zero_grad(set_to_none=True)
            loss = None
            torch.cuda.empty_cache()
            micro = 0
            log.write(f"step {step}: a batch of {float(secs):.0f} s did not fit; skipped\n")
            log.flush()
            continue
        seen += float(secs)
        micro += 1
        if micro % A.accumulate:
            continue
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0 or A.dry:
            msg = f"step {step} loss {loss.item():.3f} lr {lr:.2e} gn {gnorm:.2f} {seen / 3600:.1f} h heard {elapsed / 60:.0f} min"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        if step % A.eval_every == 0 or (A.dry and step == 2):
            wf, _, sf_ = evaluate(model, processor, pieces, fleurs, device)
            print(f"  (check: {names[0]} {wf:.1f}%)", flush=True)  # a stall watch may read the log's age
            wd, hd, sd = evaluate(model, processor, pieces, drama, device)
            print(f"  (check: {names[1]} {wd:.1f}%)", flush=True)
            wt, _, st_ = evaluate(model, processor, pieces, third, device) if third else (0.0, [], 0.0)
            ws, _, ss = evaluate(model, processor, pieces, fourth, device) if fourth else (0.0, [], 0.0)
            msg = (f"CHECK step {step} ({seen / 3600:.0f} h heard, {elapsed / 3600:.2f} h): {names[0]} {wf:.1f}% "
                   f"(as written {sf_:.1f}%), {names[1]} {wd:.1f}% (as written {sd:.1f}%)"
                   + (f", {names[2]} {wt:.1f}% (as written {st_:.1f}%)" if third else "")
                   + (f", {names[3]} {ws:.1f}% (as written {ss:.1f}%)" if fourth else "") + f"  e.g. {hd[0][:80]!r}")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            if not A.dry:
                model.save_pretrained(out / "last")
                processor.save_pretrained(out / "last")
                import shutil

                if A.sp:
                    shutil.copy(under_root(A.sp), out / "last" / "tr.model")
                if A.snapshots:  # the last N checks' weights, averaged after the run (moon_average.py)
                    model.save_pretrained(out / f"snap-{step:07d}")
                    for old in sorted(out.glob("snap-*"))[:-A.snapshots]:
                        shutil.rmtree(old)
                tmp = out / "state.pt.tmp"
                score = wd if A.best_by == "drama" else (wf + wd) / 2
                torch.save({"opt": opt.state_dict(), "step": step, "seen": seen, "best": min(best, score),
                            "elapsed": elapsed}, tmp)
                tmp.replace(out / "state.pt")
                if score < best:
                    best = score
                    model.save_pretrained(out / "best")
                    processor.save_pretrained(out / "best")
                    if A.sp:
                        shutil.copy(under_root(A.sp), out / "best" / "tr.model")
    if not A.dry:
        model.save_pretrained(out / "last")
        processor.save_pretrained(out / "last")
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    print("finished", step, "steps; peak GPU memory", f"{peak:.1f} GB;", f"{(time.time() - started) / max(1, step):.2f} s a step", flush=True)


if __name__ == "__main__":
    main()
    import os

    os._exit(0)
