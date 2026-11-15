"""Text for the piece n-gram: every training line as the Moonshine tokenizer's piece ids, written as decimal words
(one line a line), the ids the decoder writes, so the n-gram scores exactly the decoder's tokens. Lines equal to a
held-out line (any text of the --heldout manifests, compared lower-cased without punctuation) are left out, and
each distinct line is written once.

Then build an ARPA file with KenLM's lmplz and turn it into the .qng file the decoder reads, for example:
    python lm_text.py tr_en.model train_lists/*.jsonl --heldout eval/*.jsonl > lm_text.txt
    lmplz -o 4 --prune 0 2 4 6 --discount_fallback < lm_text.txt > lm.arpa
    python qng.py lm.arpa lm.qng
(The released LM was a pruned 4-gram of 38 MB; see README.md for what is known of its build.)

usage: python lm_text.py <tokenizer.model> <manifest.jsonl> [...] [--heldout a.jsonl,b.jsonl]"""
import argparse
import json
import re
import sys

import sentencepiece as spm


def flat(text):
    return " ".join(re.sub(r"[^\w\s]", " ", text.replace("İ", "i").replace("I", "ı").lower()).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--heldout", default="", help="test and dev manifests, comma-separated")
    a = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    held = set()
    for p in filter(None, a.heldout.split(",")):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                for f in ("text", "text_raw", "text_spoken"):
                    if r.get(f):
                        held.add(flat(r[f]))
    seen, kept, out = set(), 0, 0
    for p in a.manifests:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            t = (json.loads(line).get("text") or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            if flat(t) in held:
                out += 1
                continue
            sys.stdout.write(" ".join(str(i) for i in sp.encode(t)) + "\n")
            kept += 1
    print(f"{kept} lines written, {out} held-out lines left out", file=sys.stderr)


if __name__ == "__main__":
    main()
