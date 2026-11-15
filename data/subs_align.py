"""A recording and its hand-made Turkish subtitles (WebVTT; for example the subtitles made for deaf and
hard-of-hearing viewers) -> clips whose text is the subtitles' own.

Cues are joined into pieces of up to --max-piece seconds when the gap between them is short. Sound notes such as
(Hüzünlü müzik), position tags, all-caps speaker labels and the dialogue dashes are dropped, the continuation dots of
a split line ("Ana..." / "...sen de") are joined, and accessibility credits (audio description, sign language,
technical credits, web addresses) are left out.

- Training clips (default): a listener (any offline sherpa-onnx NeMo transducer, --listener) reads each piece, and
  a piece is kept when its reading and the text differ by at most --max-cer of their letters (interjections and
  written laughter aside with --interjections), it lasts 0.8 s or more, and a number at either edge is heard whole.
  The listener only checks the timing and the match: the text is always the subtitler's.
- A test set (--keep-all): every line whatever the listener heard (only song lines and pieces under 0.5 s out);
  with --max-piece 10, lines run together into caption-sized pieces of up to 10 s. The listener's letter error is
  kept as a field (cer) and never decides.

usage: python subs_align.py <sound> <subs.vtt> <out dir> --listener <dir> [--gain 8] [--max-cer 0.15]
       [--threads 4] [--opus] [--keep-all] [--max-piece 15] [--interjections] [--shift S] [--dialect]
writes <out dir>/clips/ and <out dir>/manifest.jsonl (audio_filepath, duration, text_raw, heard, cer, start)."""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).parent))
from long_align import SR, edges_heard, keys_of, read_all  # noqa: E402

# accessibility credits in subtitles made for deaf and hard-of-hearing viewers: not dialogue, never said in the
# broadcast (only music under them); also the note on the "*" dialect marker
CREDIT = re.compile(
    r"www\.|\.org\b|\.com\b|[Ss]esli [Bb]etimleme|betimlemesi|Betimleme Metin|Metin Yazar|Seslendiren\s*:|[İi]şaret [Dd]ili|"
    r"Son Kontroller|Teknik Yap[ıi]m|[Aa]yrıntılı [Aa]ltyazı|Derneğine yaptırılmıştır|tarafından\.\.\.|"
    r"[ÇçCc]evirmen\w*\s*:|Editör\w*\s*:|Montaj\s*:|Bu dizide\b.*\bkonuşmaktadır")
TIME = re.compile(r"(\d+):(\d\d):(\d\d)\.(\d{3})\s*-->\s*(\d+):(\d\d):(\d\d)\.(\d{3})")


def secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def cues(path):
    out, cur, credit = [], None, False
    for line in open(path, encoding="utf-8", errors="replace").read().splitlines() + [""]:
        m = TIME.search(line)
        if m:
            cur = [secs(*m.groups()[:4]), secs(*m.groups()[4:]), []]
        elif cur is not None:
            if line.strip():
                cur[2].append(line.strip())
            else:
                text = clean(" ".join(cur[2]))
                credit = bool(CREDIT.search(text)) or (credit and text.startswith("..."))  # a credit's names go on
                if text and not credit:
                    out.append((cur[0], cur[1], text))
                cur = None
    return out


TAG = re.compile(r"\{\\[^}]*\}")  # "{\an8}": a position tag inside the text
LABEL = re.compile(r"(^|[.!?…\-–]\s*)[A-ZÇĞİÖŞÜÂÎÛ]{2,}(?:\s+[A-ZÇĞİÖŞÜÂÎÛ]{2,})*\s*:\s*")  # "DIŞ SES:" a speaker label


def marks_out(t):
    """Position tags and all-caps speaker labels out (neither is said)."""
    return " ".join(LABEL.sub(r"\1", TAG.sub(" ", t)).split())


def clean(t):
    t = marks_out(t)
    t = re.sub(r"<[^>]+>", "", t)  # <i>, <c.colour>
    t = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", t)  # (Jenerik), (At kişneme sesleri), [Müzik]
    t = re.sub(r"(^|\s)[-–]+\s*(?=\S)", r"\1", t)  # dialogue dashes, with or without a space: "-Evet."
    t = t.replace("…", "...").replace("*", "")  # "*Lan göbel!": the star marks dialect, not a sound
    return " ".join(t.split())


def join_text(parts):
    text = ""
    for p in parts:
        if text.endswith("...") and p.startswith("..."):
            text = text[:-3] + " " + p[3:]  # "Ana..." + "...sen de" -> "Ana sen de"
        else:
            text = (text + " " + p).strip()
    return " ".join(text.lstrip(".").split())


def dialect_fold(k):
    """Heavy-dialect films (the subtitler writes the dialect; the listener hears standard Turkish): k/g/h one
    sound (gız/kız, yasah/yasak), e ı i u ü one vowel (ölürem/ölürüm, oldı/oldu). Only for comparing;
    the text stays the subtitler's."""
    return re.sub("[eıiuü]", "i", re.sub("[gkh]", "k", k))


# interjections ("eeeh yeter!" must keep its "eeeh"), in key form (long_align.key: lower case, ö ü ç ş ğ without
# marks): no listener writes them, so a line is not judged on them; the text keeps them
INJ = re.compile(r"^(e+h*|a+h+|aa+|ı+h+|ıı+|o+h+|oo+|u+h+|uu+|h[ıi]+h*|hı+m+|hm+|o+f+|u+f+|of+|vay+|hey+|ay+|ah+|oh+|eh+|"
                 r"ee+h*|hah+|heh+|ihh+|ii+h*|ayy+|aah+|oho+|cık+)$")
# written laughter, key form, after stretched letters are collapsed ("Hahahahaaa" -> hahaha); a spaced "ha ha ha"
# is laughter word by word
LAUGH = re.compile(r"^(a?(ha)+h?|(he)+h?|(hı)+h?|(hi)+h?|(ho)+h?|kı(kı)+|ki(ki)+|ah(ah)+|eh(eh)+)$")


def nonword(k):
    """An interjection or written laughter (a key): no listener writes them, so no line is judged on them."""
    c = re.sub(r"(.)\1{2,}", r"\1", k)
    return bool(INJ.match(k) or INJ.match(c) or LAUGH.match(c))


def pieces(cs, gap=0.6, hi=15.0):
    out = []
    for s, e, t in cs:
        if out and s - out[-1][1] <= gap and e - out[-1][0] <= hi:
            out[-1] = (out[-1][0], e, out[-1][2] + [t])
        else:
            out.append((s, e, [t]))
    return [(s, e, join_text(ts)) for s, e, ts in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sound")
    ap.add_argument("vtt")
    ap.add_argument("out")
    ap.add_argument("--listener", required=True, help="an offline sherpa-onnx NeMo transducer folder (*.int8.onnx)")
    ap.add_argument("--gain", type=float, default=8.0, help="input gain for the listener")
    ap.add_argument("--max-cer", type=float, default=0.15)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--opus", action="store_true")
    ap.add_argument("--dialect", action="store_true", help="compare with dialect_fold (heavy-dialect films)")
    ap.add_argument("--keep-all", action="store_true",
                    help="a test set: every line kept whatever the listener heard (only song lines and pieces under 0.5 s out)")
    ap.add_argument("--max-piece", type=float, default=15.0, help="lines run together up to this long (s)")
    ap.add_argument("--no-clips", action="store_true",
                    help="write no clip files: each row points into the sound itself (offset, duration)")
    ap.add_argument("--shift", type=float, default=0.0, help="seconds added to every cue time (a track timed for another edit)")
    ap.add_argument("--interjections", action="store_true",
                    help="judge lines without their interjections; keep a short interjection-only line with voice in it")
    a = ap.parse_args()
    fold = dialect_fold if a.dialect else (lambda k: k)
    skip = nonword if a.interjections else (lambda k: False)
    t0 = time.time()
    out = Path(a.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)  # (stays empty with --no-clips)
    audio, sr = sf.read(a.sound, dtype="int16")  # 16-bit in memory; read_all scales each piece
    assert sr == SR and audio.ndim == 1
    ps = pieces([(s + a.shift, e + a.shift, t) for s, e, t in cues(a.vtt) if s + a.shift >= 0], hi=a.max_piece)
    spans = []
    for k, (s, e, t) in enumerate(ps):
        a0 = max(0.0, s - 0.15)
        a1 = e + 0.25
        if k + 1 < len(ps):
            a1 = min(a1, ps[k + 1][0] - 0.05)
        spans.append((int(a0 * SR), int(min(len(audio) / SR, a1) * SR)))
    texts = read_all(audio, spans, a.threads, a.listener, a.gain)
    level = [10 * np.log10(np.mean(audio[s:e].astype(np.float64) ** 2) + 1e-9) if e - s >= SR // 10 else -120.0
             for s, e in spans]
    typical = float(np.median(level)) if level else 0.0  # the film's usual speech level, dB
    kept, cers = 0.0, []
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for (s, e), (_, _, label), hyp, lv in zip(spans, ps, texts, level):
            said = re.sub(r"\{[^{}]*\}", "", label)  # a gloss's standard letters are not said
            ref_k = [k for w in said.split() for k in keys_of(w)]
            ref_s = " ".join(fold(k) for k in ref_k if not skip(k))
            hyp_s = " ".join(fold(k) for w in hyp.split() for k in keys_of(w) if not skip(k))
            only_inj = bool(ref_k) and not ref_s and a.interjections
            if not ref_s and not only_inj:
                continue
            if a.keep_all:
                if "♪" in label or "#" in label or (e - s) < 0.5 * SR:
                    continue
                cer = Levenshtein.normalized_distance(hyp_s, ref_s) if ref_s else 0.0
                if ref_s:
                    cers.append(cer)
            elif only_inj:  # "Hıh!", "Eeeh...", "Hahaha!": nothing to compare; short, and a voice there
                if not (0.4 * SR <= e - s <= 2.5 * SR and lv >= typical - 15):
                    continue
                cer = 0.0
            else:
                cer = Levenshtein.normalized_distance(hyp_s, ref_s)
                cers.append(cer)
                if cer > a.max_cer or (e - s) < 0.8 * SR or not edges_heard(label, [k for w in hyp.split() for k in keys_of(w)]):
                    continue
            if a.no_clips:
                path = str(Path(a.sound).resolve())
            else:
                ext = "ogg" if a.opus else "flac"
                name = f"{Path(a.sound).stem}_{s // 16:09d}_{e // 16:09d}.{ext}"
                if a.opus:
                    sf.write(out / "clips" / name, audio[s:e], SR, format="OGG", subtype="OPUS", compression_level=0.9)
                else:
                    sf.write(out / "clips" / name, audio[s:e], SR, format="FLAC")
                path = str((out / "clips" / name).resolve())
            kept += (e - s) / SR
            row = {"audio_filepath": path, "duration": round((e - s) / SR, 3),
                   "text_raw": label, "heard": hyp, "cer": round(cer, 4), "start": round(s / SR, 2)}
            if a.no_clips:
                row["offset"] = round(s / SR, 3)
            if only_inj:
                row["interjection_only"] = True
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dialogue = sum(e - s for s, e in spans) / SR
    c = np.array(cers)
    (out / "summary.json").write_text(json.dumps({"sound_h": round(len(audio) / SR / 3600, 3),
                                                 "dialogue_h": round(dialogue / 3600, 3), "kept_h": round(kept / 3600, 3)}))
    print(f"{len(audio) / SR / 3600:.2f} h of sound, {len(ps)} pieces, {dialogue / 3600:.2f} h under subtitles; "
          f"kept {kept / 3600:.2f} h ({kept / max(1, dialogue):.0%}); cer median {np.median(c):.3f}, "
          f"under 0.1 {np.mean(c < 0.1):.0%}, under 0.15 {np.mean(c < 0.15):.0%}, under 0.25 {np.mean(c < 0.25):.0%} "
          f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
