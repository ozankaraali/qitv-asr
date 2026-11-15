"""The cased, punctuated Turkish vocabulary for the FastConformer: SentencePiece unigram, 1,024 pieces, digits split,
laid out as NeMo's change_vocabulary(new_tokenizer_type="bpe") reads it (NeMo calls any SentencePiece model "bpe"):
<out dir>/tokenizer.model and vocab.txt.

The text is one training line a line (the "text" field of the training manifests, numbers as words).

usage: python sp_conformer.py <out dir> <text.txt | manifest.jsonl> [...]"""
import json
import sys
from pathlib import Path

import sentencepiece as spm

if len(sys.argv) < 3:
    sys.exit(__doc__)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
lines = []
for f in sys.argv[2:]:
    for line in open(f, encoding="utf-8"):
        if f.endswith(".jsonl"):
            t = json.loads(line).get("text") if line.strip() else ""
        else:
            t = line.rstrip("\n")
        if t:
            lines.append(t)
(out / "text_for_tokenizer.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
spm.SentencePieceTrainer.train(
    input=str(out / "text_for_tokenizer.txt"), model_prefix=str(out / "tokenizer"),
    vocab_size=1024, model_type="unigram", character_coverage=1.0, input_sentence_size=3_000_000,
    shuffle_input_sentence=True, split_digits=True, bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1)
sp = spm.SentencePieceProcessor(model_file=str(out / "tokenizer.model"))
(out / "vocab.txt").write_text("\n".join(sp.id_to_piece(i) for i in range(sp.get_piece_size())) + "\n", encoding="utf-8")
print("pieces", sp.get_piece_size(), sp.encode("İstanbul'da kostümleri ben hallederim, zaten. Ne dersin?", out_type=str))
