"""A Moonshine v2 export (the sherpa-onnx 8-bit files) through onnxruntime's Python API, as QiTV's captions decoder
calls it: the merged decoder's first call ([BOS], empty past, use_cache_branch false) gives the cross-attention keys
and values once; each later call takes one token per hypothesis, the previous presents and the first call's encoder
ones (use_cache_branch true). Greedy decoding follows sherpa-onnx's rules (silence gives nothing, sherpa's token
cap, a loop stop); the beam search adds shallow fusion with an n-gram over the model's pieces:
score = log p_model + lam * log p_LM + beta * tokens."""
import base64
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

BOS, EOS = 1, 2
BANNED = (0, 1, 3)  # pad, bos, unk: never text
FRAME = 80
RATE = 16000
LN10 = math.log(10.0)
LOOP_MAX_N, LOOP_REPEATS_SHORT, LOOP_REPEATS = 8, 4, 3
_SILENT = float(np.float32(1e-6))


def token_cap(frames):
    return int(frames * 384 / 16000.0 * 15)


def encoder_frames(samples):
    if samples < FRAME:
        return 0
    return ((samples // FRAME - 1) // 2 + 1 - 1) // 2 + 1


def loop_tail(tokens):
    size = len(tokens)
    for n in range(1, LOOP_MAX_N + 1):
        repeats = LOOP_REPEATS_SHORT if n <= 2 else LOOP_REPEATS
        if size < n * repeats:
            break
        tail = tokens[size - n:]
        if all(tokens[size - n * k: size - n * (k - 1)] == tail for k in range(2, repeats + 1)):
            return n
    return 0


def read_pieces(path):
    table = {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if not parts:
            continue
        coded, idx = ("", parts[0]) if len(parts) == 1 else parts
        table[int(idx)] = base64.b64decode(coded)
    out = [b""] * (max(table) + 1)
    for k, v in table.items():
        out[k] = v if k > 3 else b""
    return out


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


class Moon:
    def __init__(self, folder, threads=1, encoder=True):
        folder = Path(folder)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(folder / "encoder_model_quantized.onnx"), so, providers=prov) if encoder else None
        self.dec = ort.InferenceSession(str(folder / "decoder_model_merged_quantized.onnx"), so, providers=prov)
        self.pieces = read_pieces(folder / "tokens.txt")
        names = [i.name for i in self.dec.get_inputs()]
        self.past = [n for n in names if n.startswith("past_key_values.")]
        self.self_past = [n for n in self.past if ".decoder." in n]
        self.cross_past = [n for n in self.past if ".decoder." not in n]
        pres = lambda n: "present." + n[len("past_key_values."):]
        self.first_out = ["logits"] + [pres(n) for n in self.past]
        self.step_out = ["logits"] + [pres(n) for n in self.self_past]
        shapes = {i.name: i.shape for i in self.dec.get_inputs()}
        self.empty = {n: np.zeros((1, shapes[n][1], 0, shapes[n][3]), np.float32) for n in self.past}
        self.false = np.array([False])
        self.true = np.array([True])

    # ---- model calls
    def encode(self, samples):
        """float32 16 kHz mono -> (hidden [1, T, 768] or None for silence, frames)."""
        x = np.asarray(samples, np.float32)
        pad = -len(x) % FRAME
        if pad:
            x = np.concatenate([x, np.zeros(pad, np.float32)])
        frames = encoder_frames(len(x))
        if not len(x) or np.abs(x).max() <= _SILENT:
            return None, frames
        (h,) = self.enc.run(["last_hidden_state"], {"input_values": x[None, :]})
        return h, h.shape[1]

    def first(self, hidden, start=BOS):
        feed = {"input_ids": np.array([[start]], np.int64), "encoder_hidden_states": hidden, "use_cache_branch": self.false}
        feed.update(self.empty)
        outs = self.dec.run(self.first_out, feed)
        pres = dict(zip(self.past, outs[1:]))
        return outs[0][:, -1, :], {n: pres[n] for n in self.self_past}, {n: pres[n] for n in self.cross_past}

    def step(self, tokens, own, cross, hidden):
        feed = {"input_ids": np.asarray(tokens, np.int64).reshape(-1, 1), "encoder_hidden_states": hidden,
                "use_cache_branch": self.true}
        feed.update(own)
        feed.update(cross)
        outs = self.dec.run(self.step_out, feed)
        return outs[0][:, -1, :], dict(zip(self.self_past, outs[1:]))

    def text(self, tokens):
        return b"".join(self.pieces[t] for t in tokens if 0 <= t < len(self.pieces)).decode("utf-8", "ignore").strip()

    # ---- greedy (sherpa-onnx's rules)
    def greedy(self, hidden, frames, start=BOS):
        if hidden is None:
            return []
        cap = token_cap(frames)
        if cap <= 0:
            return []
        logits, own, cross = self.first(hidden, start)
        out = []
        while True:
            row = logits[0]
            tok = int(np.argmax(row))
            if tok == EOS:
                break
            out.append(tok)
            if loop_tail(out) or len(out) >= cap:
                break
            logits, own = self.step([tok], own, cross, hidden)
        return out

    # ---- beam search with shallow fusion
    def beam(self, hidden, frames, k=4, lm=None, lam=0.0, beta=0.0, pre=None, batched=True, eos_gap=None,
             drop_loops=False, start=BOS):
        """lm: a qng.NgramLM or a kenlm.Model over piece ids written as decimal words (or None). pre: candidates
        per hypothesis
        taken by the model's score before the LM is asked (default max(2k, 8)). eos_gap: a hypothesis may end only
        where the model's log p(EOS) is within this of its best token (None: anywhere in its candidates).
        drop_loops: a hypothesis that starts looping is dropped (else kept as finished, as greedy stops)."""
        if hidden is None:
            return []
        cap = token_cap(frames)
        if cap <= 0:
            return []
        pre = pre or max(2 * k, 8)
        use_lm = lm is not None and lam != 0.0
        if use_lm and type(lm).__name__ == "NgramLM":  # qng.NgramLM (standard library only): states are id tuples
            st0 = (BOS,)
            cache = {}

            def lm_score(state, tok):
                key = (state, tok)
                got = cache.get(key)
                if got is None:
                    s, o = lm.score(state, tok)
                    got = cache[key] = (s * LN10, o)
                return got
        elif use_lm:
            import kenlm
            st0 = kenlm.State()
            lm.BeginSentenceWrite(st0)
            cache = {}

            def lm_score(state, tok):
                key = (state, tok)
                got = cache.get(key)
                if got is None:
                    o = kenlm.State()
                    s = lm.BaseScore(state, "</s>" if tok == EOS else str(tok), o) * LN10
                    got = cache[key] = (s, o)
                return got
        else:
            st0 = None
        logits, own, cross = self.first(hidden, start)
        # a hypothesis: (score, tokens, am, lmsum, state)
        live = [(0.0, [], 0.0, 0.0, st0)]
        finished = []
        hid_b, cross_b, nb = hidden, cross, 1
        rows = [logits[0]]
        own_list = None if batched else [own]
        for _ in range(cap + 1):
            lp = log_softmax(np.stack(rows))
            lp[:, BANNED] = -np.inf
            cands = []
            for b, (sc, toks, am, lms, st) in enumerate(live):
                row = lp[b]
                top = np.argpartition(-row, pre)[:pre]
                floor = row.max() - eos_gap if eos_gap is not None else -np.inf
                for t in top:
                    t = int(t)
                    if t == EOS and row[EOS] < floor:
                        continue
                    a = am + float(row[t])
                    if use_lm:
                        d, o = lm_score(st, t)
                        l = lms + d
                    else:
                        o, l = None, 0.0
                    n = len(toks) + (t != EOS)
                    cands.append((a + lam * l + beta * n, b, t, a, l, o))
            cands.sort(key=lambda c: -c[0])
            new = []
            for rank, (s, b, t, a, l, o) in enumerate(cands):
                if t == EOS:
                    if rank < k:
                        finished.append((s, live[b][1]))
                    continue
                toks = live[b][1] + [t]
                if loop_tail(toks):
                    if not drop_loops:
                        finished.append((s, toks))
                    continue
                if len(toks) >= cap:
                    finished.append((s, toks))
                    continue
                new.append((s, b, t, a, l, o, toks))
                if len(new) == k:
                    break
            if not new:
                break
            best_fin = max((f[0] for f in finished), default=-np.inf)
            if len(finished) >= k and new[0][0] < best_fin:
                break
            idx = [c[1] for c in new]
            toks_now = [c[2] for c in new]
            live = [(c[0], c[6], c[3], c[4], c[5]) for c in new]
            if batched:
                if len(idx) != nb:
                    nb = len(idx)
                    hid_b = np.ascontiguousarray(np.repeat(hidden, nb, axis=0))
                    cross_b = {n: np.ascontiguousarray(np.repeat(v, nb, axis=0)) for n, v in cross.items()}
                if idx != list(range(len(idx))):
                    own = {n: v[idx] for n, v in own.items()}
                logits, own = self.step(toks_now, own, cross_b, hid_b)
                rows = list(logits)
            else:
                prev = own_list
                own_list, rows = [], []
                for b, t in zip(idx, toks_now):
                    lg, o = self.step([t], prev[b], cross, hidden)
                    own_list.append(o)
                    rows.append(lg[0])
        if not finished:
            finished = [(h[0], h[1]) for h in live]
        return max(finished, key=lambda f: f[0])[1]
