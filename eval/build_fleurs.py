"""FLEURS Turkish dev and test (CC BY 4.0) as manifests, from a copy you already have of FLEURS's tr_tr folder (its
<split>.tsv and audio/<split>.tar.gz, or audio/<split>/ unpacked). The dev split checks training (it picks the
checkpoints); the test split is the final score. Neither is trained on.

Text: the raw transcription (cased, punctuated; column 3 of the tsv) through pc_text. Score with
eval/score.py --fix-thousands (FLEURS writes 40.000 as "40. 000").

usage: python build_fleurs.py <tr_tr dir> <out dir> [--splits dev,test]
writes <out dir>/fleurs_<split>.jsonl and the clips as 16 kHz FLAC under <out dir>/fleurs/<split>/."""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tr_text import pc_text  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--splits", default="dev,test")
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)


def audio_files(split):
    """(file name, readable) for every clip of the split, from the tar or the unpacked folder."""
    tar = src / "audio" / f"{split}.tar.gz"
    if tar.exists():
        with tarfile.open(tar) as t:
            for m in t:
                if m.isfile():
                    yield Path(m.name).name, io.BytesIO(t.extractfile(m).read())
    else:
        for p in sorted((src / "audio" / split).glob("*.wav")):
            yield p.name, p


for split in a.splits.split(","):
    texts = {}
    for line in (src / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if len(cols) > 2:
            texts[cols[1]] = cols[2]
    folder = out / "fleurs" / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, fh in audio_files(split):
        if name not in texts:
            continue
        audio, sr = sf.read(fh, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, sr
        dst = folder / (Path(name).stem + ".flac")
        sf.write(dst, audio, 16000, format="FLAC")
        rows.append({"audio_filepath": str(dst.resolve()), "duration": round(len(audio) / 16000, 3),
                     "text": pc_text(texts[name])})
    with open(out / f"fleurs_{split}.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["audio_filepath"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(split, len(rows), "clips", round(sum(r["duration"] for r in rows) / 3600, 1), "h", flush=True)
