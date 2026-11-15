"""Turkish + English pieces. The Turkish SentencePiece cuts English into about 2.6 pieces a word, so the model spells
English nearly letter by letter. English pieces are appended after the Turkish ones, so every Turkish id keeps its
learned weights, and their scores are lowered by one offset so Turkish text splits as before.

  build:  train an English unigram model (3,000 pieces) on the English training text, merge it at each offset
          tried, and keep the first offset at which at most 0.2 % of a Turkish sample splits differently (or the
          one given by --offset). The released 10,623-piece tokenizer used --offset -8: about half a percent of
          Turkish lines split differently, and English went from 2.61 to 1.85 pieces a word.
  extend: grow a fine-tuned Moonshine's decoder embeddings and output layer to the new size, each new piece
          starting from the mean of the Turkish pieces that spell it; the merged model is copied in as tr.model.
          (moon_train.py does the same swap by itself when it starts from the English base with --sp.)

usage: python sp_extend.py build --tr tr.model --out tr_en.model --english a.jsonl[,b.jsonl] --turkish c.jsonl[,...]
                                 [--work DIR] [--offset -8]
       python sp_extend.py extend <src model dir> <dst model dir> --tr tr.model --out tr_en.model"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb


def texts(path, field="text", n=None):
    out = []
    for k, line in enumerate(open(path, encoding="utf-8")):
        if n and k >= n:
            break
        t = json.loads(line).get(field) or ""
        if t:
            out.append(t)
    return out


def merged(tr_proto, en_proto, offset):
    m = pb.ModelProto()
    m.CopyFrom(tr_proto)
    have = {p.piece for p in m.pieces}
    for p in en_proto.pieces:
        if p.type != pb.ModelProto.SentencePiece.NORMAL or p.piece in have:
            continue
        q = m.pieces.add()
        q.piece, q.score, q.type = p.piece, p.score + offset, pb.ModelProto.SentencePiece.NORMAL
    return m


def build(a):
    tr_path, out = Path(a.tr), Path(a.out)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    en = [t for f in a.english.split(",") for t in texts(f)]
    (work / "en.txt").write_text("\n".join(en), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(work / "en.txt"), model_prefix=str(work / "en"), vocab_size=a.en_size,
                                   model_type="unigram", character_coverage=1.0, num_threads=8,
                                   input_sentence_size=400000, shuffle_input_sentence=True)
    tr_proto, en_proto = pb.ModelProto(), pb.ModelProto()
    tr_proto.ParseFromString(tr_path.read_bytes())
    en_proto.ParseFromString((work / "en.model").read_bytes())
    old = spm.SentencePieceProcessor(model_file=str(tr_path))
    turkish = [t for f in a.turkish.split(",") for t in texts(f, n=a.per_list)]
    eng = en[::50][:4000]
    base_tr = [old.encode(t) for t in turkish]
    best = None
    for offset in (0.0, -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -15.0, -20.0):
        m = merged(tr_proto, en_proto, offset)
        (work / "try.model").write_bytes(m.SerializeToString())
        sp = spm.SentencePieceProcessor(model_file=str(work / "try.model"))
        changed = sum(sp.encode(t) != b for t, b in zip(turkish, base_tr)) / len(turkish)
        per_word = sum(len(sp.encode(t)) for t in eng) / sum(len(t.split()) for t in eng)
        print(f"offset {offset:6.1f}: {len(m.pieces)} pieces; Turkish lines split differently {100 * changed:.2f}%; "
              f"English {per_word:.2f} pieces a word", flush=True)
        if best is None and (offset == a.offset if a.offset is not None else changed <= 0.002):
            best = (offset, m, per_word, changed)
    if best is None:
        sys.exit("no offset keeps Turkish unchanged")
    offset, m, per_word, changed = best
    out.write_bytes(m.SerializeToString())
    old_pw = sum(len(old.encode(t)) for t in eng) / sum(len(t.split()) for t in eng)
    print(f"chosen offset {offset}: {out} with {len(m.pieces)} pieces; Turkish changed {100 * changed:.2f}%; "
          f"English {old_pw:.2f} -> {per_word:.2f} pieces a word")
    sp = spm.SentencePieceProcessor(model_file=str(out))
    for t in eng[:3] + ["Yeni yönetim zirve için Antalya'ya geldi.", "Büyük indirimler başladı."]:
        print("  ", t, "->", sp.encode(t, out_type=str))


def extend(src, dst, tr_path, out_path):
    import torch
    from transformers import MoonshineStreamingForConditionalGeneration

    old = spm.SentencePieceProcessor(model_file=str(tr_path))
    new = spm.SentencePieceProcessor(model_file=str(out_path))
    n_old, n_new = old.get_piece_size(), new.get_piece_size()
    for i in range(n_old):  # every Turkish id must be the same piece
        assert old.id_to_piece(i) == new.id_to_piece(i), i
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(src)
    w_in = model.get_input_embeddings().weight.data.clone()
    w_out = model.get_output_embeddings().weight.data.clone()
    assert w_in.shape[0] == n_old, (w_in.shape, n_old)
    add_in, add_out = [], []
    for i in range(n_old, n_new):
        ids = [j for j in old.encode(new.id_to_piece(i).replace("▁", " ")) if j > 3] or [3]
        add_in.append(w_in[ids].mean(0))
        add_out.append(w_out[ids].mean(0))
    model.resize_token_embeddings(n_new)
    model.get_input_embeddings().weight.data[:n_old] = w_in
    model.get_input_embeddings().weight.data[n_old:] = torch.stack(add_in)
    model.get_output_embeddings().weight.data[:n_old] = w_out
    model.get_output_embeddings().weight.data[n_old:] = torch.stack(add_out)
    model.config.vocab_size = n_new
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "vocab_size"):
        model.config.decoder.vocab_size = n_new
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)  # processor and tokenizer files as the source
    model.save_pretrained(dst)
    shutil.copy(out_path, dst / "tr.model")
    print(f"extended {src} -> {dst}: {n_old} -> {n_new} pieces")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "extend"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--tr", required=True, help="the Turkish SentencePiece (sp_turkish.py)")
    ap.add_argument("--out", required=True, help="the merged Turkish + English SentencePiece")
    ap.add_argument("--english", default="", help="English training manifests, comma-separated (build)")
    ap.add_argument("--turkish", default="", help="Turkish manifests for the split check, comma-separated (build)")
    ap.add_argument("--per-list", type=int, default=6000, help="Turkish lines read from each list for the check")
    ap.add_argument("--en-size", type=int, default=3000)
    ap.add_argument("--offset", type=float, default=None, help="use this score offset (the released one: -8)")
    ap.add_argument("--work", default="sp_extend_work")
    args = ap.parse_args()
    if args.cmd == "build":
        if not args.english or not args.turkish:
            ap.error("build needs --english and --turkish")
        build(args)
    else:
        if not args.src or not args.dst:
            ap.error("extend needs <src model dir> <dst model dir>")
        extend(args.src, args.dst, args.tr, args.out)
