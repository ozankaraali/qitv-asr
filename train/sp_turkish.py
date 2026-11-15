"""The Turkish SentencePiece for Moonshine: unigram, 8,192 pieces, trained on the training text (numbers as words),
apostrophes and Turkish letters kept as written (identity normalisation), digits split, with Moonshine's special
ids (pad 0, start 1, end 2, unknown 3). It cuts Turkish into about 1.7 pieces a word, where the English base
tokenizer needs about 3.7 and has no apostrophes.

The text comes from the "text" field of the training manifests given (up to --max-lines, shuffled with a fixed
seed). Leave dev and test lists out.

usage: python sp_turkish.py <out dir> <manifest.jsonl> [...] [--size 8192] [--max-lines 3000000]
writes <out dir>/tr.model, tr.vocab and the text it trained on (text.txt)."""
import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("manifests", nargs="+")
ap.add_argument("--size", type=int, default=8192)
ap.add_argument("--max-lines", type=int, default=3_000_000)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
lines = []
for f in sorted(a.manifests):
    lines += [t for t in (json.loads(line).get("text") or "" for line in open(f, encoding="utf-8") if line.strip()) if t]
random.Random(3).shuffle(lines)
(out / "text.txt").write_text("\n".join(lines[:a.max_lines]) + "\n", encoding="utf-8")
spm.SentencePieceTrainer.train(
    input=str(out / "text.txt"), model_prefix=str(out / "tr"), vocab_size=a.size, model_type="unigram",
    character_coverage=1.0, normalization_rule_name="identity", split_digits=True,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3, input_sentence_size=a.max_lines, shuffle_input_sentence=True)
sp = spm.SentencePieceProcessor(model_file=str(out / "tr.model"))
for t in ["Hepiniz esen kalın. Ne diyeceğiz? Çok güzel, öğretmenimiz şöyle dedi.", "İstanbul'da yaşıyorum, Iğdır'a gideceğim."]:
    ids = sp.encode(t)
    print(len(t.split()), "words ->", len(ids), "pieces; round trip ok:", sp.decode(ids) == t, "|", sp.encode(t, out_type=str)[:14])
