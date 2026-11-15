"""Turkish numbers as words.

- `turkish_number(2023)` -> "iki bin yirmi üç".
- `spelled(text)`: every run of digits spelled out on its own (no suffix or ordinal handling). The strict scorer
  uses it.
- `spoken(text)`: numbers as people say them, for training text and for the standard scorer:
  "1980'li" -> "bin dokuz yüz seksenli", "28,5" -> "yirmi sekiz virgül beş", "%25" -> "yüzde yirmi beş",
  "1.132" -> "bin yüz otuz iki", "12'nci" -> "on ikinci", "2013'te" -> "iki bin on üçte",
  "65. sayfa" -> "altmış beşinci sayfa" (a full stop before a lowercase word marks an ordinal).
  Everything else (capitals, punctuation, the apostrophes of names) stays as written.

usage: python -m common.tr_numbers --demo
       python -m common.tr_numbers <in.jsonl> <out.jsonl>   (adds "text_spoken" from text_std, text_raw or text)
"""
import json
import re
import sys

_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def turkish_number(n: int) -> str:
    if n == 0:
        return "sıfır"
    words = []
    for size, name in ((10**9, "milyar"), (10**6, "milyon"), (1000, "bin")):
        if n >= size:
            count, n = divmod(n, size)
            if not (size == 1000 and count == 1):  # bin, not bir bin
                words.append(turkish_number(count))
            words.append(name)
    if n >= 100:
        hundreds, n = divmod(n, 100)
        if hundreds > 1:
            words.append(_ONES[hundreds])  # yüz, not bir yüz
        words.append("yüz")
    if n >= 10:
        words.append(_TENS[n // 10])
        n %= 10
    if n:
        words.append(_ONES[n])
    return " ".join(words)


def spelled(text: str) -> str:
    return re.sub(r"\d+", lambda m: " " + turkish_number(int(m.group())) + " ", text)


NUM = re.compile(r"(?<![\w.,])(%?)(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?:['’](\w+))?(\.(?=\s+[a-zçğıöşü]))?(?![\w%])")


def ordinal(w):
    """"beş" -> "beşinci", "altı" -> "altıncı", "üç" -> "üçüncü", "dört" -> "dördüncü" (vowel harmony)."""
    if w == "dört":
        return "dördüncü"
    v = [c for c in w if c in "aeıioöuü"][-1]
    suf = {"a": "ıncı", "ı": "ıncı", "e": "inci", "i": "inci", "o": "uncu", "u": "uncu", "ö": "üncü", "ü": "üncü"}[v]
    return w + (suf[1:] if w[-1] in "aeıioöuü" else suf)


def spoken(text):
    def say(m):
        pct, whole, dec, suffix, odot = m.groups()
        if len(whole.replace(".", "")) > 12:
            return m.group(0)
        words = (["yüzde"] if pct else []) + turkish_number(int(whole.replace(".", ""))).split()
        if dec:
            words += ["virgül"] + turkish_number(int(dec)).split()
        if suffix:
            words[-1] += suffix
        elif odot and not dec and not pct:  # "65. sayfa": a full stop before a lowercase word marks an ordinal
            words[-1] = ordinal(words[-1])
        return " ".join(words)
    return NUM.sub(say, text)


def add_spoken(src, dst):
    """A manifest with a "text_spoken" field beside its written text (a TDK-spelling field first, when there)."""
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    changed = 0
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            written = r.get("text_std") or r.get("text_raw") or r.get("text") or ""
            r["text_spoken"] = spoken(written)
            changed += r["text_spoken"] != written
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{dst}: {len(rows)} rows, {changed} with numbers spoken out")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--demo"]:
        for t in ["1980'li yıllarda", "yaklaşık 28,5 milyar dolar", "%25 oranında artış", "1.132 hastane",
                  "12'nci Birleşim", "2013'te başladı", "3 Kasım 2026 Salı", "Şimdi 65. sayfayı açın.", "4. kattayız",
                  "Saat 5. Sonra gel.", "16 yaşındaydım."]:
            print(f"{t!r:34} -> {spoken(t)!r}")
    elif len(sys.argv) == 3:
        add_spoken(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
