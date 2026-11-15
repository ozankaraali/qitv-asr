"""A long recording matched to its written text (a session and its official minutes, a talk and its transcript, a
programme and its subtitles from another edit), as training clips whose text is the written one (capitals,
punctuation, proper spelling), the way parliament corpora such as EuroSpeech were built:

1. the sound is cut at pauses (by loudness against its surroundings; a voice detector heard a hall's constant
   murmur as one long speech) into pieces of 5-20 s;
2. a listener (any offline sherpa-onnx NeMo transducer, --listener) reads each piece; its reading only finds the
   place in the text, it never becomes the text;
3. all readings, in order, are aligned word by word with the whole text (digits spelled out, Turkish I/ı, marks and
   accents ignored);
4. each piece takes the text's words its reading lined up with; it is kept when the two differ by at most --max-cer
   of their letters, and when a number at either edge is heard whole.

With --placed (the final round's re-cut), step 4 keeps nothing and drops nothing: every piece whose reading lines up
with the text anywhere is written, with the word range its reading found (w0, w1, into <out dir>/words.json, the whole
text as words); a range longer than 4 words a second (+5) cannot have been said in the piece and is left out. Then
data/mms_check.py (window mode) lets the forced aligner set each clip's words and decide which stay: the listener
only placed them.

The text is a plain text file (paragraphs, one a line; speaker names and headings taken out beforehand) or a WebVTT
file (hand-made subtitles made for another upload of the same programme: its cues' text is used in order and its
timings are ignored).

usage: python long_align.py <sound> <text.txt | subs.vtt> <out dir> --listener <dir> [--gain 8] [--max-cer 0.1]
       [--threads 6] [--opus] [--placed]
writes <out dir>/clips/*.flac (or .ogg) and <out dir>/manifest.jsonl (audio_filepath, duration, text_raw, heard,
cer, start; with --placed also words, w0, w1). The listener's readings are cached beside the sound
(<sound>.heard.json)."""

import argparse
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tr_numbers import turkish_number  # noqa: E402

SR = 16000


def key(tok):
    t = unicodedata.normalize("NFKD", tok.replace("İ", "i").replace("I", "ı"))
    t = "".join(c for c in t if not unicodedata.combining(c)).casefold()
    t = unicodedata.normalize("NFC", t.replace("i̇", "i"))
    return "".join(c for c in t if c.isalnum())


NUM = re.compile(r"(%?)(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?:['’]?(\w*))?")


def keys_of(tok):
    """A word as one or more keys; numbers as Turkish says them: 2013'te -> iki bin on üçte,
    1.132 -> bin yüz otuz iki, 28,5 -> yirmi sekiz virgül beş, %25 -> yüzde yirmi beş."""
    m = NUM.fullmatch(tok.strip(".,;:?!\"“”()"))
    if m and len(m.group(2).replace(".", "")) < 13:
        words = (["yüzde"] if m.group(1) else []) + turkish_number(int(m.group(2).replace(".", ""))).split()
        if m.group(3):
            words += ["virgül"] + turkish_number(int(m.group(3))).split()
        if m.group(4):
            words[-1] += m.group(4)
        return [key(w) for w in words]
    k = key(tok)
    return [k] if k else []


def edges_heard(label, hyp_keys):
    """A clip whose text begins or ends with a number keeps it only when the whole number is heard
    at that edge (a cut inside "bin yüz | otuz iki" would teach the model the rest)."""
    ws = label.split()
    if not ws:
        return False
    for tok, at_end in ((ws[-1], True), (ws[0], False)):
        if any(c.isdigit() for c in tok):
            ks = keys_of(tok)
            seq = hyp_keys[-len(ks):] if at_end else hyp_keys[:len(ks)]
            if seq != ks:
                return False
    return True


def pieces(audio, lo=5.0, hi=20.0, pause=0.3, quiet_db=8.0):
    """Cuts at pauses by loudness: Silero heard the hall's constant murmur as one long speech (25
    pieces in 7 h). A frame (10 ms) is quiet when it is `quiet_db` under the loudness of the 30 s
    around it (its 70th percentile). A piece ends in the middle of a pause (a run of quiet frames of
    at least `pause` s) once it is `lo` s long; with no pause by `hi` s, at its quietest frame
    after 10 s."""
    hop = SR // 100
    n = len(audio) // hop
    db = np.empty(n, np.float32)
    step = 360000  # an hour of frames at a time: a 10 h session stays near its own 1.2 GB
    for i in range(0, n, step):
        m = min(step, n - i)
        x = audio[i * hop:(i + m) * hop].astype(np.float32).reshape(m, hop) / 32768.0
        db[i:i + m] = 10 * np.log10(np.mean(x * x, axis=1) + 1e-10)
    db = np.convolve(db, np.ones(5) / 5, mode="same")
    blk = 3000  # 30 s
    ref = np.array([np.percentile(db[max(0, i - blk // 2): i + blk // 2], 70) for i in range(0, n, 100)])
    ref = np.repeat(ref, 100)[:n]
    quiet = db < ref - quiet_db
    # pauses: runs of quiet frames at least `pause` long
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.astype(np.int8), [0]))))
    runs = [(r0, r1) for r0, r1 in zip(edges[::2], edges[1::2]) if r1 - r0 >= pause * 100]
    cuts, s = [], 0
    L, H = int(lo * 100), int(hi * 100)
    for r0, r1 in runs + [(n, n)]:
        mid = (r0 + r1) // 2
        while mid - s > H:  # no pause in time: the quietest frame between 10 s and hi
            w0 = s + 1000
            c = w0 + int(np.argmin(db[w0:s + H]))
            cuts.append((s, c))
            s = c
        if mid - s >= L or r0 == n:
            cuts.append((s, mid))
            s = mid
    spans = []
    for a, b in cuts:
        while a < b and quiet[a]:  # trim the pause off both ends, keep 0.1 s of it
            a += 1
        while b > a and quiet[b - 1]:
            b -= 1
        a, b = max(0, a - 10), min(n, b + 10)
        if (b - a) >= 150 and np.any(db[a:b] > ref[a:b] - 10):
            spans.append((a * hop, b * hop))
    return spans


def read_all(audio, spans, threads, listener, gain):
    """The listener's reading of each span (int16 audio; each piece scaled by gain / 32768). Some exports want the
    sound louder: the listener used for the released data took gain 8, as its model card says."""
    import sherpa_onnx
    listener = Path(listener)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(listener / "encoder.int8.onnx"), decoder=str(listener / "decoder.int8.onnx"),
        joiner=str(listener / "joiner.int8.onnx"), tokens=str(listener / "tokens.txt"),
        num_threads=threads, model_type="nemo_transducer")
    texts = [""] * len(spans)  # a span under 0.1 s (a cue past the sound's end, overlapping cues) is not read:
    real = [k for k, (a, b) in enumerate(spans) if b - a >= SR // 10]  # the encoder refuses an empty one
    for i in range(0, len(real), 16):
        streams = []
        for k in real[i:i + 16]:
            a, b = spans[k]
            s = rec.create_stream()
            s.accept_waveform(SR, audio[a:b].astype(np.float32) * (gain / 32768.0))
            streams.append(s)
        rec.decode_streams(streams)
        for k, s in zip(real[i:i + 16], streams):
            texts[k] = s.result.text.strip()
    return texts


def write_clip(path, samples, opus):
    if opus:
        sf.write(path, samples, SR, format="OGG", subtype="OPUS", compression_level=0.9)
    else:
        sf.write(path, samples, SR, format="FLAC")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("flac")
    ap.add_argument("minutes")
    ap.add_argument("out")
    ap.add_argument("--listener", required=True, help="an offline sherpa-onnx NeMo transducer folder (*.int8.onnx)")
    ap.add_argument("--gain", type=float, default=8.0, help="input gain for the listener")
    ap.add_argument("--max-cer", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--opus", action="store_true", help="clips as Ogg Opus, about 33 kbit/s (else FLAC)")
    ap.add_argument("--placed", action="store_true",
                    help="every placed piece with its word range, no letter-error test (for data/mms_check.py)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        audio, sr = sf.read(a.flac, dtype="int16")  # 16-bit in memory: half of float32
    except sf.LibsndfileError:  # libsndfile calls a few good Ogg Opus files "malformed": ffmpeg reads them
        import subprocess
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", a.flac, "-f", "s16le",
                            "-ac", "1", "-ar", str(SR), "-"], capture_output=True, check=True)
        audio, sr = np.frombuffer(r.stdout, dtype=np.int16).copy(), SR
    assert sr == SR and audio.ndim == 1
    from rapidfuzz.distance import Levenshtein
    cache = Path(a.flac).with_suffix(".heard.json")  # the same pieces' readings, kept from an earlier run
    try:
        c = json.loads(cache.read_text())
        assert c["samples"] == len(audio)
        spans, texts = [tuple(x) for x in c["spans"]], c["texts"]
        print(f"{len(audio) / SR / 3600:.2f} h of sound, {len(spans)} pieces read before", flush=True)
    except (OSError, ValueError, KeyError, AssertionError):
        spans = pieces(audio)
        print(f"{len(audio) / SR / 3600:.2f} h of sound, {len(spans)} pieces, "
              f"{sum(b - s for s, b in spans) / SR / 3600:.2f} h of speech ({time.time() - t0:.0f} s)", flush=True)
        texts = read_all(audio, spans, a.threads, a.listener, a.gain)
        print(f"read ({time.time() - t0:.0f} s)", flush=True)
        try:  # kept beside the sound: aligning it to another text (another edition) skips the reading
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps({"samples": len(audio), "spans": [[int(x), int(y)] for x, y in spans], "texts": texts},
                                      ensure_ascii=False))
            tmp.replace(cache)
        except OSError:
            pass

    # the text as words (surface) and keys, each key knowing its word
    if a.minutes.endswith(".vtt"):  # hand-made subtitles of another upload: their text, in order
        from subs_align import cues, join_text
        words = join_text([t for _, _, t in cues(a.minutes)]).split()
    else:
        words = []
        for par in Path(a.minutes).read_text(encoding="utf-8").splitlines():
            par = re.sub(r"\s*\([^)]*\)", "", par)  # (Alkışlar), (Gürültüler): notes, not speech
            par = re.sub(r"\s*\[[^\]]*\]", "", par)
            words += par.replace("…", "...").split()
    rk, rown = [], []
    for i, w in enumerate(words):
        for k in keys_of(w):
            rk.append(k)
            rown.append(i)
    hk, hown = [], []
    for j, t in enumerate(texts):
        for w in t.split():
            for k in keys_of(w):
                hk.append(k)
                hown.append(j)
    print(f"text {len(words)} words, readings {len(hk)} words; aligning", flush=True)
    to_ref = [None] * len(hk)
    same = [False] * len(hk)
    for op in Levenshtein.opcodes(hk, rk):
        if op.tag in ("equal", "replace"):
            n = min(op.src_end - op.src_start, op.dest_end - op.dest_start)
            for d in range(n):
                to_ref[op.src_start + d] = op.dest_start + d
                same[op.src_start + d] = op.tag == "equal"
    print(f"aligned ({time.time() - t0:.0f} s)", flush=True)

    first = {}
    for x, j in enumerate(hown):
        first.setdefault(j, []).append(x)
    kept, rows, cers = 0.0, [], []
    words_file = (out / "words.json").resolve()
    if a.placed:
        words_file.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    pool, jobs = ThreadPoolExecutor(4), []  # libsndfile lets go of the GIL: clips encode in parallel
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for j, (s, e) in enumerate(spans):
            xs = first.get(j, [])
            mapped = [to_ref[x] for x in xs if to_ref[x] is not None]
            if not mapped or (not a.placed and (len(xs) < 3 or sum(same[x] for x in xs) < 0.5 * len(xs))):
                continue
            lo, hi = rown[min(mapped)], rown[max(mapped)]
            if hi - lo + 1 > (4.0 * (e - s) / SR + 5 if a.placed else 2.5 * len(texts[j].split()) + 5):
                continue
            label = " ".join(words[lo:hi + 1]).strip(" ,;:")
            ref_s = " ".join(k for w in words[lo:hi + 1] for k in keys_of(w))
            hyp_s = " ".join(hk[x] for x in xs)
            cer = Levenshtein.normalized_distance(hyp_s, ref_s)
            cers.append(cer)
            if not a.placed and (cer > a.max_cer or not edges_heard(label, [hk[x] for x in xs])):
                continue
            name = f"{Path(a.flac).stem}_{s // 16:09d}_{e // 16:09d}." + ("ogg" if a.opus else "flac")
            jobs.append(pool.submit(write_clip, out / "clips" / name, audio[s:e], a.opus))
            dur = (e - s) / SR
            kept += dur
            row = {"audio_filepath": str((out / "clips" / name).resolve()), "duration": round(dur, 3),
                   "text_raw": label, "heard": texts[j], "cer": round(cer, 4), "start": round(s / SR, 2)}
            if a.placed:
                row.update(recording=Path(a.flac).stem, words=str(words_file), w0=lo, w1=hi)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for j in jobs:
        j.result()
    pool.shutdown()
    speech = sum(b - s for s, b in spans) / SR
    (out / "summary.json").write_text(json.dumps({"sound_h": round(len(audio) / SR / 3600, 3),
        "speech_h": round(speech / 3600, 3), "kept_h": round(kept / 3600, 3)}))
    print(f"kept {kept / 3600:.2f} h of {speech / 3600:.2f} h of speech ({kept / max(1, speech):.0%}); "
          f"cer of the matched: median {np.median(cers):.3f}, under 0.1: {np.mean(np.array(cers) < 0.1):.0%}, "
          f"under 0.2: {np.mean(np.array(cers) < 0.2):.0%} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
