"""tokens.txt for sherpa-onnx's Moonshine v2 loader from our SentencePiece: one line per id,
"<base64 of the piece's bytes> <id>". sherpa joins the decoded bytes as they are, so the
SentencePiece word mark becomes a real space and byte pieces (<0xC3>) become that byte.
usage: python moon_tokens.py <tr.model> <tokens.txt>"""
import base64
import sys

import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if sp.is_control(i) or sp.is_unknown(i):
            raw = b""
        elif sp.is_byte(i):
            raw = bytes([int(piece[3:5], 16)])
        else:
            raw = piece.replace("▁", " ").encode("utf-8")
        f.write(f"{base64.b64encode(raw).decode('ascii') or 'AA=='} {i}\n")
print("tokens", sp.get_piece_size())
