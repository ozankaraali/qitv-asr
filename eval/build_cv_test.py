"""A Common Voice test for scoring only: --clips random clips (fixed seed) of a Common Voice release's Turkish test
split, as 16 kHz FLAC, from a copy of the release you already have (its tr/ folder: test.tsv and clips/).
Short read lines from many voices and microphones, with question marks. Leave every sentence of the dev and test
splits out of training (data/build_mix.py's held-out text check does this when the lists are among --heldout).
The published number: Common Voice 27.0, 1,500 clips.

usage: python build_cv_test.py <cv tr dir> <out dir> [--clips 1500] [--name cv_test]
writes <out dir>/<name>.jsonl and the clips under <out dir>/<name>/."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tr_text import pc_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cv")
ap.add_argument("out")
ap.add_argument("--clips", type=int, default=1500)
ap.add_argument("--name", default="cv_test")
ap.add_argument("--split", default="test")
a = ap.parse_args()
torch.set_num_threads(4)
csv.field_size_limit(10**8)
cv, out = Path(a.cv), Path(a.out)
with open(cv / f"{a.split}.tsv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
random.Random(27).shuffle(rows)
out_dir = out / a.name
out_dir.mkdir(parents=True, exist_ok=True)
kept = []
for r in rows[: a.clips]:
    audio, sr = sf.read(cv / "clips" / r["path"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = AF.resample(torch.from_numpy(audio), sr, 16000).numpy()
    dst = out_dir / (Path(r["path"]).stem + ".flac")
    sf.write(dst, audio, 16000, format="FLAC")
    kept.append({"audio_filepath": str(dst.resolve()), "duration": round(len(audio) / 16000, 3),
                 "text": pc_text(r["sentence"])})
with open(out / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(kept), "clips", round(sum(r["duration"] for r in kept) / 3600, 2), "h;",
      sum("?" in r["text"] for r in kept), "with a question mark")
