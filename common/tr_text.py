"""Training text, Turkish: the cleaning every training list goes through, and the checks that decide whether a
line can be what was said in its clip.

- pc_text / pc: cased, punctuated text in the allowed alphabet (ALLOWED).
- fold, quotes: invisible marks, quotation marks and odd letters out, apostrophes of names kept.
- respell, respell_unquoted: glued spellings fixed.
- sentences: capitals after . ? ! (and full stops at subtitle line breaks made commas, when asked).
- usable, fits, fits_tv, fits_sung: letters-per-second checks (speech, short lines, listener-approved clips, songs).
- key, norm_words: forms for comparing lines (identical-text caps, held-out sentence checks).
- INTERJ, has_interjection: Turkish interjections and written laughter.
"""
import re
import unicodedata

ALLOWED = set("abcçdefgğhıijklmnoöprsştuüvyzqwxâîûABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZQWXÂÎÛ0123456789.,?!'- ")
ENDS = re.compile(r"[.?!]")


def pc_text(text):
    """Cased, punctuated training text: capitals, . , ? ! ' and a hyphen inside a word; other marks mapped or
    dropped; bracketed sound notes out; digits kept (spoken() turns them into words first)."""
    text = unicodedata.normalize("NFC", text).replace("i̇", "i")
    text = re.sub(r"[(\[][^)\]]*[)\]]", " ", text)  # (Müzik), [Alkış]: sounds, not speech
    text = text.replace("’", "'").replace("`", "'").replace("…", ".").replace(";", ",").replace(":", ",")
    text = re.sub(r"[\"“”«»()\[\]{}*_/\\|<>~^#&%$@=+]", " ", text)
    text = re.sub(r"(?<=\w)\s*-\s*(?=\s)|(?<=\s)-\s*", " ", text)  # a hyphen only inside a word
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    text = re.sub(r"([.,?!])(?=\w)", r"\1 ", text)
    text = re.sub(r"([.,?!])[.,?!]+", r"\1", text)
    text = re.sub(r"\s*'\s*", "'", text)
    return " ".join(text.split()).strip(" ,")


def usable(text, dur):
    """Only the allowed letters and marks, 0.8-30 s, and 3-22 letters a second (a line that cannot be what was
    said in that time is left out)."""
    if not text or not 0.8 <= dur <= 30.0 or any(c not in ALLOWED for c in text):
        return False
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and 3.0 <= letters / dur <= 22.0


# glued spellings (TDK writes them apart or differently); PREFIX also catches suffixed forms
PREFIX = {"herşey": "her şey", "birşey": "bir şey", "hiçbirşey": "hiçbir şey", "hoşçakal": "hoşça kal",
          "orjinal": "orijinal", "yanlız": "yalnız", "şuan": "şu an", "herkez": "herkes", "hiçkimse": "hiç kimse",
          "pekçok": "pek çok"}
WHOLE = {"demekki": "demek ki", "yada": "ya da", "malesef": "maalesef", "belkide": "belki de", "yinede": "yine de",
         "birsürü": "bir sürü", "herzaman": "her zaman", "birdaha": "bir daha"}
GLUED = re.compile(r"(?<!\w)(?:(" + "|".join(PREFIX) + r")(\w*)|(" + "|".join(WHOLE) + r")(?!\w))", re.IGNORECASE)


def respell(text):
    """Glued or plain misspellings fixed ("birşey" -> "bir şey", "yanlız" -> "yalnız"), capitals kept."""
    def fix(m):
        word = m.group(1) or m.group(3)
        new = PREFIX.get(word.lower()) or WHOLE[word.lower()]
        if word[0].isupper():
            new = new[0].upper() + new[1:]
        return new + (m.group(2) or "")
    return GLUED.sub(fix, text)


LINE_STOP = re.compile(r"\.(\s+)(?=[a-zçğıöşü])")
LOWER_START = re.compile(r"([.?!])(\s+)([a-zçğıöşü])")


def upper_tr(c):
    return {"i": "İ", "ı": "I"}.get(c, c.upper())


def sentences(text, joined_lines):
    """A word after . ? ! starts a sentence. Some subtitle sources put a full stop at every line break, also in
    mid-sentence ("Oğlum. emin misin"); for those (joined_lines) a full stop before a lowercase word becomes a
    comma. Elsewhere such words are mostly numbers spoken out at a sentence start ("istiyorum. yedi Ekimden"):
    they get their capital."""
    if joined_lines:
        text = LINE_STOP.sub(r",\1", text)
    return LOWER_START.sub(lambda m: m.group(1) + m.group(2) + upper_tr(m.group(3)), text)


# invisible marks out, odd forms of the circumflexed letters mapped, dashes to spaces
FOLD = (("​", ""), ("‌", ""), ("‍", ""), ("⁠", ""), ("﻿", ""), ("­", ""), (" ", " "),
        (" ", " "), ("Ȃ", "Â"), ("ȃ", "â"), ("ȋ", "î"), ("Ȋ", "Î"), ("ȗ", "û"), ("Ȗ", "Û"), ("ￚ", " "), ("–", " "),
        ("—", " "), ("‒", " "), ("―", " "), ("‐", "-"), ("‑", "-"))


def fold(text):
    """Unicode oddities folded, "%" before a spoken number made "yüzde", a lead-in ellipsis dropped, quotation
    marks out with apostrophes kept (quotes())."""
    text = unicodedata.normalize("NFC", text)
    for a, b in FOLD:
        text = text.replace(a, b)
    text = re.sub(r"%\s*(?=[^\W\d_])", "yüzde ", text)  # "% otuz": the sign before a number already spoken
    text = re.sub(r"(?<=\w) \+ (?=\w)", " artı ", text)
    text = re.sub(r"(^|(?<=[\"“‘'«(]))\s*(?:\.\.\.|…)\s*(?=\w)", "", text)  # “...söyleyeceklerimiz: a lead-in
    return quotes(text)


def pc(text):
    """pc_text, with no mark left at the start."""
    return pc_text(text).lstrip(" .,?!")


LOWER = "a-zçğıöşüâîû"
OPEN_D, CLOSE_D = "“«„", "”»"
SUFFIX = re.compile(r"(?:[yn]?[ıiuü]n?|[yn]?[ae]|[dt][ae]n?|[dt][ae]ki|y?l[ae]|n[dt][ae]n?|[dt][ıiuü]r|s[ıiuü]n?)"
                    r"(?![\wçğıöşü])")
QUOTED = re.compile(r"[\"“«](.*?)[\"”»]")  # double quotes only: ' is the suffix apostrophe (Türkiye'de)


def respell_unquoted(text):
    """Plain misspellings fixed outside quotes only: inside them a speaker may be talking about the spelling
    itself."""
    out, last = [], 0
    for m in QUOTED.finditer(text):
        out.append(respell(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(respell(text[last:]))
    return "".join(out)


def capital_before(out):
    words = re.findall(r"\w+", "".join(out[-60:]))
    return bool(words) and words[-1][0].isupper()


def quotes(text):
    """Quotation marks out without gluing words (bize 'bir kilo' -> bize bir kilo); a quote closing straight
    before a suffix becomes the apostrophe after a capitalised word ("Ali"nin -> Ali'nin) and nothing after a
    lowercase one ("hayır"a -> hayıra); curly apostrophes straight. pc_text glues every ' to its neighbours, so a
    single quote that opens or closes a quotation has to go before it."""
    out, n = [], len(text)
    open_d = open_s = False
    skip = 0
    for i, c in enumerate(text):
        if skip:
            skip -= 1
            continue
        prev = text[i - 1] if i else " "
        nxt = text[i + 1] if i + 1 < n else " "
        if c in OPEN_D + CLOSE_D + "\"":
            closing = c in CLOSE_D or (c == "\"" and open_d)
            open_d = not closing
            if closing and re.match(f"[{LOWER}]", nxt):  # “Kitap”ın, “hayır”a
                out.append("'" if capital_before(out) else "")
            else:
                out.append(" ")
            continue
        if c in "‘’ʼ´′'`":
            before = prev.isalnum() or prev == "+"
            after = nxt.isalpha()
            if before and after:  # Türkiye’de, Ahmet'i, ‘zona fasulyesi’yle
                if c != "‘" and open_s and re.match(f"[{LOWER}]", nxt):
                    out.append("'" if capital_before(out) else "")
                    open_s = False
                else:
                    out.append("'")
                continue
            if c == "’" and prev.isspace() and re.match(f"[{LOWER}]", nxt) and not open_s:  # Kılavuzu ’nu
                out.append("'")
                continue
            if before and nxt == " " and not open_s and SUFFIX.match(text, i + 2):  # Ayşe' le
                out.append("'")
                skip = 1
                continue
            if after and not before:  # an opening quote
                open_s = True
                out.append(" " if prev.isalnum() else "")
                continue
            open_s = False  # a closing one: after a letter or punctuation, before a space, punctuation or the end
            out.append(" " if nxt.isalnum() else "")
            continue
        out.append(c)
    return "".join(out)


INTERJ = re.compile(r"^(e+h*|a+h+|aa+|ı+h+|ıı+|o+h+|oo+|u+h+|uu+|h[ıi]+h*|hı+m+|hm+|ö+f+|ü+f+|u+f+|of+|vay+|hey+|ah+|"
                    r"oh+|eh+|ee+h*|hah+|heh+|ihh+|ii+h*|ayy+|aah+|öhö+|cık+|a?(ha)+h?|(he)+h?|(hı)+h?|(hi)+h?|"
                    r"(ho)+h?|kı(kı)+|ki(ki)+)$")
LABEL = re.compile(r"(?:^|(?<=[.!?…]\s))[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ ]{1,30}:\s*")  # "DIŞ SES:" speaker labels
SHORT_S = 3.0  # a short line (Ah!, Hıh!, Peki!): its own letters/s floor and no repeat cap
LISTENER_GATE = 0.15  # subs_align's gate (letter error of the listener); a clip it approved needs only TV_FLOOR letters/s
TV_FLOOR = 1.0
# a video's outro (subscribe, like, thanks for watching) and a known subtitle credit: never training text
OUTRO = re.compile(r"abone ol|abone olma|abone olun|beğenmeyi unutma|beğenip abone|izlediğiniz için teşekkür|"
                   r"bir sonraki videoda|kanal(?:ım|ımız)a abone|bildirimleri aç|zil(?:i|ine) bas|altyaz[ıi] m\.?k",
                   re.IGNORECASE)


def low_tr(text):
    return text.replace("İ", "i").replace("I", "ı").lower()


def words_tr(text):
    return re.sub(r"[^\wçğıöşüâîû ]", " ", low_tr(text)).split()


def has_interjection(text):
    return any(INTERJ.match(w) for w in words_tr(text))


def fits(text, dur):
    """usable(), with a floor of 1 letter/s for short lines (one "Ah!" of 1.7 s is speech too)."""
    if usable(text, dur):
        return True
    if not text or not 0.5 <= dur <= SHORT_S:
        return False
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and letters / dur >= 1.0 and usable(text, max(dur, letters / 21.0))


def fits_tv(text, dur):
    """A clip the listener gate approved: as usable(), from TV_FLOOR letters/s (pauses and music under long clips)."""
    if not text or not 0.8 <= dur <= 30.0 or any(c not in ALLOWED for c in text):
        return False
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and TV_FLOOR <= letters / dur <= 22.0


def fits_sung(text, dur):
    """Singing holds its notes: from 0.8 letters/s (a held "Gel güzelim" is ~2.6), else as usable()."""
    if not text or not 0.8 <= dur <= 25.0 or any(c not in ALLOWED for c in text):
        return False
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and 0.8 <= letters / dur <= 22.0


def key(text):
    return re.sub(r"[^\wçğıöşüâîû]", "", low_tr(text).translate(str.maketrans("âîû", "aiu")))


def norm_words(text):
    """Words compared for containment: lowercased, circumflexes and apostrophes aside (Türkiye'de = türkiyede)."""
    t = low_tr(unicodedata.normalize("NFC", text)).translate(str.maketrans("âîû", "aiu"))
    return re.sub(r"[^\wçğıöşü]+", " ", re.sub(r"['’‘`´]", "", t)).split()
