"""A Moonshine weights folder written by train/moon_train.py still carries the English base model's tokenizer.json
(32,000 pieces); 🤗 Transformers needs one built from our SentencePiece (tr.model) for processor.batch_decode to
read the model's ids right. This builds it, checks it against SentencePiece (ids and decoded text) on the texts of
the manifests given plus a few hard cases, and with --write puts tokenizer.json, tokenizer_config.json and
special_tokens_map.json into the folder (the old files are kept as *.base).

usage: python hf_tokenizer.py <weights dir> [manifest.jsonl ...] [--write]"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2
from tokenizers import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

sys.modules.setdefault("sentencepiece_model_pb2", sentencepiece_model_pb2)  # what tokenizers' from_spm imports
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    sys.exit(__doc__)
WRITE = "--write" in sys.argv
d = Path(args[0])
texts = []
for name in args[1:]:
    with open(name, encoding="utf-8") as fh:
        texts += [json.loads(line)["text"] for line in fh if line.strip()]
texts += ["Bana yalan söylüyorsun, değil mi?", "Hello world, 2026!", "çift  boşluk", "İSTANBUL'da ılık ŞUBAT"]
print(len(texts), "texts")

sp = spm.SentencePieceProcessor(model_file=str(d / "tr.model"))
tk = SentencePieceUnigramTokenizer.from_spm(str(d / "tr.model"))
fast = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                               pad_token="<pad>")
assert len(fast) == sp.get_piece_size(), (len(fast), sp.get_piece_size())
assert [fast.convert_ids_to_tokens(i) for i in range(4)] == ["<pad>", "<s>", "</s>", "<unk>"]
bad_ids = bad_txt = 0
for t in texts:
    a = sp.encode(t)
    b = fast.encode(t, add_special_tokens=False)
    bad_ids += a != b
    known = [i for i in a if i != sp.unk_id()]  # an unknown character: SentencePiece prints ' ⁇ ', HF drops it
    bad_txt += sp.decode(known) != fast.decode(known, skip_special_tokens=True)
print(d, f"{len(fast)} pieces; ids differ on", bad_ids, "texts; decoded text differs on", bad_txt)
if WRITE and bad_ids == 0 and bad_txt == 0:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        fast.save_pretrained(out)
        for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            if (d / f).exists() and not (d / f"{f}.base").exists():
                shutil.copy2(d / f, d / f"{f}.base")
        old_cfg = json.loads((d / "tokenizer_config.json").read_text()) if (d / "tokenizer_config.json").exists() else {}
        new_cfg = json.loads((out / "tokenizer_config.json").read_text())
        new_cfg["processor_class"] = old_cfg.get("processor_class", "MoonshineStreamingProcessor")
        new_cfg["model_max_length"] = old_cfg.get("model_max_length", 4096)
        for f in ("tokenizer.json", "tokenizer_config.json"):  # a new file each, never a write through a hard link
            (d / f).unlink(missing_ok=True)
        shutil.copy2(out / "tokenizer.json", d / "tokenizer.json")
        (d / "tokenizer_config.json").write_text(json.dumps(new_cfg, indent=2, ensure_ascii=False) + "\n")
        sm = out / "special_tokens_map.json"
        if sm.exists():
            (d / "special_tokens_map.json").unlink(missing_ok=True)
            shutil.copy2(sm, d / "special_tokens_map.json")
    print(d, "written:", sorted(p.name for p in d.iterdir()))
elif WRITE:
    sys.exit("not written: the tokenizers differ")
