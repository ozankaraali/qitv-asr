"""The scorers, the same for references and every model's text.

standard (the published numbers): numbers spoken as the references speak them (common.tr_numbers.spoken: ordinals
  "7. adası" -> "yedinci adası", "1967'de" -> "bin dokuz yüz altmış yedide", "%25" -> "yüzde yirmi beş"), bracketed
  sound notes out (pc_text), any digit left spelled out, Turkish I/İ, case folded, circumflex folded (â=a),
  apostrophes joined, other punctuation and symbols to spaces. Every model's text first gets the display step
  (shown_text).
strict: digits spelled one by one with no suffix or ordinal handling, apostrophes split words, no circumflex
  folding, model text as written. The training checks print it beside the standard number.
English (en_words): Unicode NFKC, HTML entities read, numbers spelled out by num2words on both sides (1st -> first,
  1100-2099 as a year, "1,000" -> one thousand, 2.5 -> two point five, 50% -> fifty percent, $5 -> five dollars),
  Mr/Mrs/Ms/Dr -> mister/missus/miss/doctor, "per cent" -> percent, OK -> okay, lower case with no Turkish I rule,
  apostrophes kept inside a word and dropped at its edges, other punctuation and symbols to spaces.

Words wrong = word edits summed over clips / reference words. Letters wrong (CER) = letter edits / reference letters
of the standard scorer's words joined by single spaces (spaces count). Interval: 95 % percentile bootstrap over
clips (BOOT resamples, seeded by the set's name). Paired: the same resamples for two models.
FLEURS writes thousands as "40. 000": fixed_ref reads them as 40.000 (FLEURS only: elsewhere "300. 300" may be a
phone number).
Punctuation (punct_counts): per class (e = sentence end: a lone full stop or a run of ? or !; c = comma;
q = question mark; i = a sentence end inside the line), true, extra and missed marks placed through a word alignment
of the two texts; a mark after an unpaired word moves back to the last aligned pair.
"""
import html
import re
import unicodedata
import zlib

import numpy as np

from common.tr_numbers import spelled, spoken
from common.tr_text import pc_text

BOOT = 2000  # bootstrap resamples
FOLD = str.maketrans("âîûÂÎÛ", "aiuaiu")
APOS = set("'’‘ʼ`´")
THOUSANDS = re.compile(r"(\d)\. (\d{3})(?!\d)")  # FLEURS writes 40.000 as "40. 000"
LONE_I = re.compile(r"(?<!\w)i (?=\w)")  # the display step: a lone "i" joins the next word
RAW_TOKEN = re.compile(r"(?:\d+(?:[.,:/]\d+)+|\w+)(?:['’‘ʼ`´]\w+)*|([^\w\s])\1*")
APOS_OUT = str.maketrans("", "", "".join(APOS))


def fixed_ref(text):
    """The FLEURS reference repair both scorers share: "7. 000" is 7.000."""
    return THOUSANDS.sub(r"\1.\2", text)


def shown_text(hyp):
    """A model's text as the captions show it: a lone "i" (a split-off question particle or suffix) joins the next
    word. Applied to every model's text before the standard scorer, never to references."""
    return LONE_I.sub("i", hyp or "")


def words(text):
    """The standard scorer's words (references and models' text alike)."""
    t = pc_text(spoken(unicodedata.normalize("NFC", text or "")))
    t = unicodedata.normalize("NFKC", spelled(t)).replace("İ", "i").replace("I", "ı").casefold().translate(FOLD)
    return "".join("" if c in APOS else " " if unicodedata.category(c)[0] in "PS" else c for c in t).split()


def strict_words(text):
    """The strict scorer's words: digits spelled one by one (no suffix or ordinal handling), apostrophes split
    words, no circumflex folding."""
    t = unicodedata.normalize("NFKC", spelled(text or "")).replace("İ", "i").replace("I", "ı").casefold()
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t).split()


EN_APOS = str.maketrans({c: "'" for c in "’‘ʼ`´"})
EN_MONEY = re.compile(r"([€$£])\s?(\d+(?:,\d{3})*(?:\.\d+)?)(\s?(?:thousand|million|billion|trillion)\b)?", re.I)
EN_PERCENT = re.compile(r"(\d+(?:,\d{3})*(?:\.\d+)?)\s?%")
EN_ORDINAL = re.compile(r"(\d+)(st|nd|rd|th)\b", re.I)
EN_NUMBER = re.compile(r"\d+(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+")
EN_CURRENCY = {"€": "euros", "$": "dollars", "£": "pounds"}
EN_SAME = {"mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor", "ok": "okay"}


def en_number(token, year=True):
    """A number as words (num2words): "1,000" one thousand, 2.5 two point five, 1100-2099 alone as a year."""
    from num2words import num2words

    t = token.replace(",", "")
    if "." in t:
        return num2words(float(t))
    n = int(t)
    if year and len(t) == 4 and 1100 <= n <= 2099:
        return num2words(n, to="year")
    return num2words(n)


def en_words(text):
    """The English scorer's words (references and models' text alike): numbers spelled out the same way on both
    sides, no Turkish I rule, apostrophes kept inside a word."""
    from num2words import num2words

    t = unicodedata.normalize("NFKC", html.unescape(text or "")).translate(EN_APOS)
    t = EN_MONEY.sub(lambda m: f" {en_number(m[2], False)}{(m[3] or '').lower()} {EN_CURRENCY[m[1]]} ", t)
    t = EN_PERCENT.sub(lambda m: f" {en_number(m[1], False)} percent ", t)
    t = EN_ORDINAL.sub(lambda m: f" {num2words(int(m[1]), to='ordinal')} ", t)
    t = EN_NUMBER.sub(lambda m: f" {en_number(m[0])} ", t)
    t = t.lower().replace("\u0307", "")
    t = "".join(c if c == "'" or unicodedata.category(c)[0] not in "PS" else " " for c in t)
    out = []
    for w in t.split():
        w = w.strip("'")
        if not w:
            continue
        w = EN_SAME.get(w, w)
        if w == "cent" and out and out[-1] == "per":
            out[-1] = "percent"
            continue
        out.append(w)
    return out


def edits(a, b):
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def align(a, b):
    """A minimum-edit alignment of two token lists and its edit count (the same count edits() gives): pairs (i, j)
    with None for the missing side; ties go to a pairing first, then a deletion, then an insertion."""
    n, m = len(a), len(b)
    if not n or not m:
        return [(i, None) for i in range(n)] + [(None, j) for j in range(m)], n + m
    rows = [list(range(m + 1))]
    prev = rows[0]
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        rows.append(cur)
        prev = cur
    out, i, j = [], n, m
    while i or j:
        if i and j and rows[i][j] == rows[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            i, j = i - 1, j - 1
            out.append((i, j))
        elif i and rows[i][j] == rows[i - 1][j] + 1:
            i -= 1
            out.append((i, None))
        else:
            j -= 1
            out.append((None, j))
    out.reverse()
    return out, rows[n][m]


def letter_edits(a, b):
    """The edit distance of two strings (Myers' bit-vector algorithm, Hyyrö's form: the number edits() gives)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    m = len(b)
    peq = {}
    for k, c in enumerate(b):
        peq[c] = peq.get(c, 0) | (1 << k)
    full, top = (1 << m) - 1, 1 << (m - 1)
    pv, mv, score = full, 0, m
    for c in a:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & full) ^ pv) | eq
        ph = mv | (~(xh | pv) & full)
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = mh | (~(xv | ph) & full)
        mv = ph & xv
    return score


def fold_word(w):
    """A word as the punctuation alignment compares it: Turkish lower case, â=a, apostrophes out."""
    return w.replace("İ", "i").replace("I", "ı").casefold().translate(FOLD).translate(APOS_OUT)


def raw_tokens(text):
    """The case-and-punctuation scorer's tokens: words as written, each punctuation run a token of its own."""
    t = unicodedata.normalize("NFC", text or "").replace("…", "...")
    return [m.group(0) for m in RAW_TOKEN.finditer(t)]


def marks(tokens):
    """Folded words and, for each, the classes of the marks written after it (before the next word):
    e = sentence end (a lone full stop or a run of ? or !), c = comma, q = question mark. Marks before the first word
    are left out."""
    ws, after = [], []
    for t in tokens:
        if t[0].isalnum() or t[0] == "_":
            ws.append(fold_word(t))
            after.append(set())
        elif after:
            c = t[0]
            if t == "." or c in "?!":
                after[-1].add("e")
            if c == ",":
                after[-1].add("c")
            if c == "?":
                after[-1].add("q")
    return ws, after


PCLASSES = ("e", "i", "c", "q")  # i = a sentence end inside the line (not at its last position)


def punct_counts(ref_tokens, hyp_tokens):
    """Per class: [true, extra, missed] marks, placed through the word alignment (a mark after an unpaired word moves
    back to the last aligned pair, or to the line's start)."""
    rw, ra = marks(ref_tokens)
    hw, ha = marks(hyp_tokens)
    cols, _ = align(rw, hw)
    slots_r, slots_h = [set()], [set()]  # slot 0 = before the first pair
    for i, j in cols:
        if i is not None and j is not None:
            slots_r.append(set(ra[i]))
            slots_h.append(set(ha[j]))
        elif i is not None:
            slots_r[-1] |= ra[i]
        else:
            slots_h[-1] |= ha[j]
    out = {k: [0, 0, 0] for k in PCLASSES}
    last = len(slots_r) - 1
    for k, (r, h) in enumerate(zip(slots_r, slots_h)):
        for cls in ("e", "c", "q") + (("i",) if k < last else ()):
            key = "e" if cls == "i" else cls
            x, y = key in r, key in h
            if x or y:
                out[cls][0 if x and y else 1 if y else 2] += 1
    return out


def wer(e, n):
    return 100.0 * e.sum() / max(1, n.sum())


def resamples(name, count):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.integers(0, count, size=(BOOT, count), dtype=np.int32)


def interval(e, n, idx):
    w = 100.0 * e[idx].sum(axis=1) / np.maximum(1, n[idx].sum(axis=1))
    return np.percentile(w, 2.5), np.percentile(w, 97.5)


def paired(ea, eb, n, idx):
    nn = np.maximum(1, n[idx].sum(axis=1))
    d = 100.0 * (ea[idx].sum(axis=1) - eb[idx].sum(axis=1)) / nn
    return wer(ea, n) - wer(eb, n), np.percentile(d, 2.5), np.percentile(d, 97.5)
