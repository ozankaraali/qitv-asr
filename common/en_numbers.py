"""English numbers as spoken words in the training text. The training text writes every number as words (an app
turns them into digits for display), so no English training line may keep a digit or one of $ % & # @.

A number is spelled out (num2words) only when the words are not in doubt:
- whole numbers under 1,000, also with %, $, euro or pound signs ("5%" -> "five percent", "$10" -> "ten dollars");
- ordinals under 1,000 ("21st" -> "twenty-first");
- decimals under 1,000 with one or two places, not before million/billion ("2.5" -> "two point five");
- years 1100-1999 as years ("1989" -> "nineteen eighty-nine") and their decades ("1990s" -> "nineteen nineties").

The line is dropped instead when how it was said cannot be known from the text: years from 2000 on ("twenty
nineteen" or "two thousand nineteen"), other numbers of 1,000 or more ("a thousand" or "one thousand"), codes
and mixed letters and digits ("CO2", "G20"), times, dates, fractions, ranges, URLs, and @ # &.

    say(text) -> the line with its numbers in words, or None (drop it)
usage: python -m common.en_numbers LIST.jsonl [...]   rewrites each list in place (backup <list>.before_numbers
       first), and prints how many lines were spelled out and how many dropped."""

import json
import re
import shutil
import sys
from pathlib import Path

from num2words import num2words

BAD = re.compile(r"https?://|www\.|[@#&]|\d[:/]\d|\d\s*[-–]\s*\d|[A-Za-z]\d|\d(?!(?:st|nd|rd|th|s)\b)[A-Za-z]")
NUM = re.compile(r"([$€£]?)(\d[\d,]*(?:\.\d+)?)(\s?%|st\b|nd\b|rd\b|th\b|s\b)?")
CUR = {"$": ("dollar", "dollars"), "€": ("euro", "euros"), "£": ("pound", "pounds")}
BIG = re.compile(r"\s*(million|billion|trillion|thousand)\b", re.I)


class Drop(Exception):
    pass


def words(m, after):
    cur, num, tail = m.group(1), m.group(2), (m.group(3) or "").strip()
    if "," in num:  # "1,500": 1,000 or more, never a year
        raise Drop
    if "." in num:
        whole, frac = num.split(".")
        if int(whole) >= 1000 or len(frac) > 2 or cur or tail not in ("", "%") or BIG.match(after):
            raise Drop
        out = num2words(float(num))
        return out + (" percent" if tail == "%" else "")
    n = int(num)
    if tail in ("st", "nd", "rd", "th"):
        if n >= 1000 or cur:
            raise Drop
        return num2words(n, to="ordinal")
    if tail == "s":  # a decade: "1990s"
        if cur or not (1100 <= n <= 1999 and n % 10 == 0):
            raise Drop
        y = num2words(n, to="year")
        return y[:-1] + "ies" if y.endswith("y") else y + "s"
    if len(num) == 4 and not cur and tail != "%" and not BIG.match(after):
        if 1100 <= n <= 1999:
            return num2words(n, to="year")
        raise Drop  # 2000 on: "twenty nineteen" or "two thousand nineteen"
    if n >= 1000:
        raise Drop  # "a thousand" or "one thousand", "fifteen hundred" or "one thousand five hundred"
    w = num2words(n)
    if cur:
        one, many = CUR[cur]
        return f"{w} {one if n == 1 else many}"
    return w + (" percent" if tail == "%" else "")


def say(text):
    if re.search(r"https?://|www\.|\w\.(com|org|net|eu|tr)\b", text):
        return None  # a web address
    if not re.search(r"[\d$%&#@€£]", text):
        return text
    if BAD.search(text):
        return None
    try:
        out = NUM.sub(lambda m: words(m, text[m.end():]), text)
    except Drop:
        return None
    if re.search(r"[\d$%&#@€£]", out):
        return None
    out = " ".join(out.split())
    return out[:1].upper() + out[1:] if text[:1].isdigit() else out


def rewrite(path):
    path = Path(path)
    backup = path.with_name(path.name + ".before_numbers")
    if not backup.exists():
        shutil.copy2(path, backup)
    keep, spelled, dropped = [], 0, 0
    for line in open(backup, encoding="utf-8"):
        r = json.loads(line)
        t = say(r["text"])
        if t is None:
            dropped += 1
            continue
        if t != r["text"]:
            spelled += 1
            r = dict(r, text=t)
        keep.append(r)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    tmp.replace(path)
    left = sum(1 for r in keep if re.search(r"[\d$%&#@]", r["text"]))
    print(f"{path}: {len(keep)} lines kept, {spelled} with numbers spelled out, {dropped} dropped; "
          f"lines with a digit or $ % & # @ left: {left} (backup {backup.name})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        rewrite(p)
