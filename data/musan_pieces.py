"""Clips with no speech, text "" (from MUSAN or any other licence-checked noise and music folder):

- test: --clips pieces of 10-20 s, every second one music without vocals, the others noise -> a no-speech test (a
  model that writes words there invents them: a credit line, a loop of numbers). The files it drew from are listed
  beside it (<out>.files.txt): give that list to the training lists (--exclude) and to moon_train.py --bg-exclude,
  so the test is never heard.
- train: --clips pieces of 3-15 s from the other files -> the no-speech training list (Moonshine's decoder invented
  words over music before it heard some).
- sung: --clips pieces of 3-15 s of music with singing (MUSAN's music/*/ANNOTATIONS, vocals "Y"), text "": the model
  learns not to write words over songs it cannot follow. Every fifth file is held out (<out>.held.txt).

The noise and music listing is a NeMo-style manifest of the source files (audio_filepath with /noise/ or /music/ in
the path, duration). Audio must be 16 kHz.

usage: python musan_pieces.py test <noise.json> <out.jsonl> [--clips 150]
       python musan_pieces.py train <noise.json> <out.jsonl> --exclude <test.files.txt> [--clips 4000]
       python musan_pieces.py sung <musan music dir> <out.jsonl> [--clips 1500]
The clips are written as FLAC beside <out.jsonl> (<out>/)."""
import argparse
import json
import random
from pathlib import Path

import soundfile as sf


def write(rows, out):
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{out}: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} h")


def cut(path, start, length):
    x, sr = sf.read(path, dtype="float32", start=int(start * 16000), frames=int(length * 16000))
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["test", "train", "sung"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--clips", type=int, default=0)
    ap.add_argument("--exclude", default="", help="lists of source files never to use, comma-separated")
    a = ap.parse_args()
    out = Path(a.out)
    folder = out.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    if a.kind == "test":
        files = [json.loads(line) for line in open(a.src, encoding="utf-8") if line.strip()]
        music = [f for f in files if "/music/" in f["audio_filepath"] and f["duration"] > 25]
        noise = [f for f in files if "/noise/" in f["audio_filepath"] and f["duration"] > 12]
        rng, used = random.Random(77), set()
        for k in range(a.clips or 150):
            f = rng.choice(music if k % 2 else noise)
            length = rng.uniform(10, 20)
            start = rng.uniform(0, max(0.0, f["duration"] - length))
            used.add(f["audio_filepath"])
            x, sr = cut(f["audio_filepath"], start, length)
            if sr != 16000 or len(x) < 16000 * 5:
                continue
            p = folder / f"nonspeech_{k:03d}.flac"
            sf.write(p, x, 16000, format="FLAC")
            rows.append({"audio_filepath": str(p.resolve()), "duration": round(len(x) / 16000, 3), "text": "",
                         "kind": "music" if k % 2 else "noise"})
        Path(str(out.with_suffix("")) + ".files.txt").write_text("".join(u + "\n" for u in sorted(used)))
    elif a.kind == "train":
        files = [json.loads(line) for line in open(a.src, encoding="utf-8") if line.strip()]
        used = set()
        for listing in filter(None, a.exclude.split(",")):
            used |= {s.strip() for s in open(listing, encoding="utf-8") if s.strip()}
        pool = [f for f in files if f["audio_filepath"] not in used and f["duration"] > 4]
        rng = random.Random(2027)
        for k in range(a.clips or 4000):
            f = rng.choice(pool)
            length = min(rng.uniform(3, 15), f["duration"] - 0.5)
            start = rng.uniform(0, f["duration"] - length)
            x, sr = cut(f["audio_filepath"], start, length)
            if sr != 16000 or len(x) < 16000 * 2:
                continue
            p = folder / f"ns_{k:05d}.flac"
            sf.write(p, x, 16000, format="FLAC")
            rows.append({"audio_filepath": str(p.resolve()), "duration": round(len(x) / 16000, 3), "text": ""})
    else:
        files = []
        for ann in sorted(Path(a.src).glob("*/ANNOTATIONS")):
            for line in open(ann, encoding="utf-8", errors="ignore"):
                parts = line.split()
                if len(parts) >= 3 and parts[2].upper() == "Y" and (ann.parent / f"{parts[0]}.wav").exists():
                    files.append(ann.parent / f"{parts[0]}.wav")
        train = [f for k, f in enumerate(files) if k % 5]
        Path(str(out.with_suffix("")) + ".held.txt").write_text(
            "".join(f"{f}\n" for k, f in enumerate(files) if not k % 5))
        rng = random.Random(2028)
        for k in range(a.clips or 1500):
            f = rng.choice(train)
            info = sf.info(str(f))
            length = min(rng.uniform(3, 15), info.duration - 0.5)
            start = rng.uniform(0, max(0.0, info.duration - length))
            x, sr = sf.read(str(f), dtype="float32", start=int(start * info.samplerate),
                            frames=int(length * info.samplerate))
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000 or len(x) < 16000 * 2:
                continue
            p = folder / f"song_{k:05d}.flac"
            sf.write(p, x, 16000, format="FLAC")
            rows.append({"audio_filepath": str(p.resolve()), "duration": round(len(x) / 16000, 3), "text": ""})
    write(rows, out)


if __name__ == "__main__":
    main()
