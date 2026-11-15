"""The training mix: one cleaned manifest per source and mix.json (each source's sampling weight), from source
manifests described in a JSON config (see examples/mix_config.json). The trainers sample clips by weight, so a
source's weight is its share of the hours heard divided by its mean clip length.

Shares. Sources sit in groups, each with a share of the hours heard (the released models: drama 0.30, parliament
0.13, read and formal speech 0.18, broad speech 0.25, films and sung lyrics 0.08, English 0.09, no speech 0.01).
Inside a group every source gets the same number of passes, none over its cap (max_passes: 6 for any source, 3 for
a group that sets it, lower for a source that sets it; share_max caps a source's share); what a group cannot take,
and whatever the shares leave, goes to the group marked "takes_rest" (drama: the hardest test and the most useful
data). The mix is sized for --hours-heard (or --train-hours x --heard-per-gpu-hour: about 480 h of sound per GPU
hour for Moonshine Medium on one RTX 5080, 800 for Small). A share of the drama group (interjections.share_of_group)
goes to a list of the drama lines that hold an interjection, so they are heard more often.

Text, per source ("text" in the config):
- subtitles: hand-written text (subtitles, minutes, transcripts): Unicode folded, quotation marks out with
  apostrophes kept, speaker labels out, misspellings fixed outside quotes, cased and punctuated (pc_text),
  capitals after . ? ! ("joined_lines": full stops at subtitle line breaks become commas).
- clean: text that was already proofread (books, proofread transcripts): folded and pc_text only
  ("sentences": capitals after . ? ! too).
- read: lists whose text is final (Common Voice, FLEURS train; numbers already words): misspellings fixed, capitals
  after . ? !; a line with a digit is left out; none of the row checks below.
- sung: sung lyrics: as clean, first letter capitalised, a floor of 0.8 letters a second.
- english: English text as it is (numbers already words: common/en_numbers.py), clips of 0.8-30 s.
- none: clips with no speech (text "").
Numbers are spoken (common/tr_numbers.spoken) unless the row has a text_spoken field already; a row that still
holds a digit is left out where "drop_digits" is set.

Row checks (every source but read, english and none):
- not a video outro or a subtitle credit; not an episode listed in "exclude_episodes";
- the clip can be what the text says in its length (usable / fits: 3-22 letters a second, short lines from
  1 a second; for a drama row whose listener letter error "cer" is at most 0.15, from 1 a second: pauses and music
  under long clips);
- held out (the leak filters, against every text of the "heldout" lists: test and dev manifests): no line equal to
  a held-out line of six words or more, none containing one, none sharing eight words in a row with one (the same
  speech cut differently);
- held out by field: a row whose field (series, episode, session, ...) is in a held-out list ("held_out":
  {"series": "file.txt"}), or whose episode is <held series>-<number>, is out;
- held out by recording text ("held_out_by_text": true, for a source whose recording ids cannot be matched to the
  tests', e.g. hashed video ids): a recording of the source (its video_id or episode field, else the clip name's
  <recording>) whose rows together hold 20 % or more of a held-out recording's distinct five-word runs (10 runs or
  more) is taken for that recording and left out whole (the final round's YODAS v3 spans);
- no doubles ("dedup"): a row overlapping a kept row of the same episode (its "episode", "start", "end" fields, or
  a clip name <episode>_<start ms>_<end ms>) by more than 0.3 s or 10 % of its length is out. "seed" rows are kept
  as they are and only seed the check; "check" rows are checked against them and every row taken before.
Identical text appears at most --cap times (lines of 3 s or less are not capped). Clips longer than --max-s are
never heard by the trainer, so they are left out and the passes stay true.

After filtering, every guard runs again on what is about to be written (digits, alphabet, held-out text, held-out
fields on the clip paths, doubles, no clip of a test or dev list) and stops the script on any hit, before anything
is written.

usage: python build_mix.py <config.json> <out dir> [--hours-heard H | --train-hours 28 --heard-per-gpu-hour 480]
       [--cap 5] [--max-s 25]
writes <out dir>/<source>.jsonl, val_<check>.jsonl for the config's "checks", mix.json, and mix_fc.json (the
FastConformer's mix: the sources named in "fc_drop" out, weights renormalised)."""

import argparse
import bisect
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.tr_numbers import spoken  # noqa: E402
from common.tr_text import (ALLOWED, LABEL, LISTENER_GATE, OUTRO, SHORT_S, fits, fits_sung, fits_tv,  # noqa: E402
                            fold, has_interjection, key, norm_words, pc, respell, respell_unquoted, sentences,
                            upper_tr)

PASSES_MAX = 6  # any source
RUN = 8  # words in a row shared with a held-out text
OVERLAP_S, OVERLAP_FRAC = 0.3, 0.10  # a new drama row may touch a kept one by this much (the larger), no more
CLIP_ID = re.compile(r"^(.+?)_\d{6,}_\d{6,}")  # <recording>_<start>_<end>...: the clip's source
SPAN = re.compile(r"^(.+?)_(\d{6,})_(\d{6,})")  # <episode or recording>_<start ms>_<end ms>
CHECKED = ("subtitles", "clean", "sung")  # the text modes whose rows go through the row checks


def name_span(path):
    """(source, start s, end s) from a clip name, or None."""
    m = SPAN.match(Path(path).stem)
    return (m.group(1), int(m.group(2)) / 1000, int(m.group(3)) / 1000) if m else None


def row_span(r):
    """A row's (episode, start, end): the clip name's ms times (exact), else the row's own fields."""
    s = name_span(r["audio_filepath"])
    ep = r.get("episode") or (s[0] if s else None)
    if s and s[2] > s[1]:
        return ep, s[1], s[2]
    if ep and r.get("start") is not None:
        start = float(r["start"])
        return ep, start, float(r["end"]) if r.get("end") is not None else start + float(r["duration"])
    return None


class Spans:
    """Kept clips per episode (or recording), for the no-doubles rule."""

    def __init__(self):
        self.by = defaultdict(list)

    def add(self, k, s, e):
        bisect.insort(self.by[k], (s, e))

    def overlap(self, k, s, e):
        iv = self.by.get(k)
        if not iv:
            return 0.0
        best = 0.0
        i = bisect.bisect_left(iv, (e, float("inf")))
        while i > 0:
            i -= 1
            s2, e2 = iv[i]
            if s2 < s - 120:  # no clip is that long
                break
            best = max(best, min(e, e2) - max(s, s2))
        return best


def read_list(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def lines(path):
    if not path or not os.path.exists(path):
        return set()
    return {s.strip() for s in Path(path).read_text(encoding="utf-8").split("\n") if s.strip() and not s.startswith("#")}


def clip_ids(path):
    """Every name a clip path may carry its episode or recording under."""
    p = Path(path)
    ids = set(p.parts)
    m = CLIP_ID.match(p.stem)
    if m:
        ids.add(m.group(1))
    return ids


class HeldOut:
    """Every held-out test and dev text of six words or more: equal lines, contained ones, and eight-word runs shared
    in either direction (the same speech cut differently); and every held-out clip path."""

    def __init__(self, paths):
        self.keys, self.first, self.runs, self.clips = set(), defaultdict(set), set(), set()
        for p in paths:
            for r in read_list(p):
                self.clips.add(os.path.splitext(r["audio_filepath"])[0])
                for f in ("text", "text_raw", "text_spoken", "text_std"):
                    t = r.get(f)
                    if not t:
                        continue
                    for v in {t, spoken(t)}:
                        w = norm_words(v)
                        if len(w) < 6:  # shorter ones are set phrases ("Sayın Başkan, değerli milletvekilleri")
                            continue
                        self.keys.add(key(v))
                        self.first[tuple(w[:4])].add(tuple(w))
                        for i in range(len(w) - RUN + 1):
                            self.runs.add(tuple(w[i:i + RUN]))

    def equal(self, text):
        return key(text) in self.keys

    def contains(self, w):
        for i in range(len(w) - 5):
            for seq in self.first.get(tuple(w[i:i + 4]), ()):
                if tuple(w[i:i + len(seq)]) == seq:
                    return True
        return False

    def shares_run(self, w):
        return any(tuple(w[i:i + RUN]) in self.runs for i in range(len(w) - RUN + 1))


def recording_of(r):
    """A row's recording: its video_id or episode field, else the clip name's <recording>."""
    if r.get("video_id") or r.get("episode"):
        return r.get("video_id") or r.get("episode")
    m = CLIP_ID.match(Path(r["audio_filepath"]).stem)
    return m.group(1) if m else Path(r["audio_filepath"]).stem


class HeldRecordings:
    """Each held-out recording's distinct five-word runs (its rows' text, and the text with numbers spoken), for a
    source whose recordings cannot be matched to the tests' by id."""
    RUN, COVER, MIN_RUNS = 5, 0.20, 10

    def __init__(self, paths):
        self.runs = defaultdict(set)
        for p in paths:
            for r in read_list(p):
                for f in ("text", "text_raw", "text_spoken"):
                    for t in {r.get(f) or "", spoken(r.get(f) or "")}:
                        self.runs[(p, recording_of(r))] |= self.grams(norm_words(t))
        self.owner = defaultdict(set)
        for k, gs in self.runs.items():
            for g in gs:
                self.owner[g].add(k)

    def grams(self, w):
        return {tuple(w[i:i + self.RUN]) for i in range(len(w) - self.RUN + 1)}

    def matched(self, rows):
        """The source's recordings whose text holds >= 20 % of a held-out recording's runs (>= 10 runs)."""
        mine = defaultdict(set)
        for r in rows:
            t = r.get("text_spoken") or spoken(r.get("text_std") or r.get("text_raw") or r.get("text") or "")
            mine[recording_of(r)] |= self.grams(norm_words(t))
        out = set()
        for rec, gs in mine.items():
            hits = Counter(k for g in gs & self.owner.keys() for k in self.owner[g])
            if any(n >= self.MIN_RUNS and n >= self.COVER * len(self.runs[k]) for k, n in hits.items()):
                out.add(rec)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--hours-heard", type=float, default=0.0, help="expected hours heard over the run")
    ap.add_argument("--train-hours", type=float, default=0.0, help="GPU hours of training the mix is sized for")
    ap.add_argument("--heard-per-gpu-hour", type=float, default=480.0)
    ap.add_argument("--cap", type=int, default=5, help="identical text at most this many times")
    ap.add_argument("--max-s", type=float, default=25.0, help="the trainer's longest clip (moon_train --max-s)")
    ap.add_argument("--seed", type=int, default=44)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    here = Path(a.config).resolve().parent

    def path_of(p):
        p = Path(os.path.expandvars(p))
        return p if p.is_absolute() else here / p

    hours_heard = a.hours_heard or cfg.get("hours_heard") or a.train_hours * a.heard_per_gpu_hour
    if not hours_heard:
        sys.exit("give --hours-heard (or --train-hours, or hours_heard in the config)")
    print(f"sized for {hours_heard:.0f} h heard")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = cfg["groups"]
    group_by_name = {g["name"]: g for g in groups}
    rest = [g["name"] for g in groups if g.get("takes_rest")]
    if len(rest) != 1:
        sys.exit("exactly one group takes the rest (\"takes_rest\": true)")
    rest = rest[0]
    sources = cfg["sources"]
    for s in sources:
        if s["group"] not in group_by_name:
            sys.exit(f"{s['name']}: no group {s['group']}")
    heldout_paths = sorted({p for pat in cfg.get("heldout", []) for p in glob.glob(str(path_of(pat)))})
    held = HeldOut(heldout_paths)
    print(f"held-out texts: {len(held.keys)} lines of six words or more from {len(heldout_paths)} lists")
    excluded = set()
    for f in cfg.get("exclude_episodes", []):
        excluded |= lines(path_of(f))
    src = defaultdict(list)
    count = Counter()
    order = iter(range(10**12))
    dedup = Spans()
    held_fields = {s["name"]: {f: lines(path_of(p)) for f, p in (s.get("held_out") or {}).items()} for s in sources}
    held_ids = {k: set().union(*v.values()) for k, v in held_fields.items() if v}  # for the path check

    def held_by_field(s, r):
        for field, ids in held_fields[s["name"]].items():
            if r.get(field) in ids:
                return True
            ep = r.get("episode") or ""
            if field == "series" and any(ep.startswith(x + "-") for x in ids):
                return True
        return False

    def raw(r, s):
        text = r.get("text_spoken")
        if not text:
            written = r.get("text_std") or r.get("text_raw") or r.get("text") or ""
            text = spoken(written) if s.get("spoken", True) else written
        return LABEL.sub("", fold(text))

    def add(s, r, text):
        name = s["name"]
        if OUTRO.search(text):
            count[f"{name}: a video outro or subtitle credit"] += 1
            return
        sp = row_span(r) if (s.get("dedup") or excluded) else None
        if excluded and sp and sp[0] in excluded:
            count[f"{name}: an excluded episode"] += 1
            return
        if s.get("drop_digits") and re.search(r"\d", text):
            count[f"{name}: a digit still in the text"] += 1
            return
        if held.equal(text):
            count[f"{name}: a held-out sentence (equal)"] += 1
            return
        w = norm_words(text)
        if held.contains(w):
            count[f"{name}: a held-out sentence inside it"] += 1
            return
        if held.shares_run(w):
            count[f"{name}: {RUN} words in a row shared with a held-out text"] += 1
            return
        dur = r["duration"]
        ok = (fits_sung if s["text"] == "sung" else fits)(text, dur)
        if not ok and s["group"] == rest and isinstance(r.get("cer"), (int, float)) and r["cer"] <= LISTENER_GATE \
                and fits_tv(text, dur):  # the listener approved it: from 1 letter a second
            ok = True
        if not ok:
            count[f"{name}: text and length do not fit"] += 1
            return
        if s.get("dedup") == "check":
            if sp is None:
                count[f"{name}: no episode and times to check for doubles"] += 1
                return
            ov = dedup.overlap(*sp)
            if ov > max(OVERLAP_S, OVERLAP_FRAC * (sp[2] - sp[1])):
                count[f"{name}: doubles a kept row"] += 1
                return
        if s.get("dedup") and sp:
            dedup.add(*sp)
        row = {"audio_filepath": r["audio_filepath"], "duration": dur, "text": text, "_n": next(order)}
        if sp:
            row["_span"] = sp
        src[name].append(row)

    held_runs = HeldRecordings(heldout_paths) if any(s.get("held_out_by_text") for s in sources) else None
    print("found:")
    for s in sources:
        rows = read_list(path_of(s["path"]))
        print(f"  {s['name']:<16} {len(rows):>7} rows {sum(r.get('duration', 0) for r in rows) / 3600:7.1f} h  {s['path']}")
        if s.get("held_out_by_text"):
            gone = held_runs.matched(rows)
            if gone:
                before = len(rows)
                rows = [r for r in rows if recording_of(r) not in gone]
                count[f"{s['name']}: a recording taken for a held-out one by its text"] += before - len(rows)
                print(f"    {len(gone)} recordings taken for held-out ones by their text, e.g. {sorted(gone)[:3]}")
        mode = s["text"]
        for r in rows:
            if held_by_field(s, r):
                count[f"{s['name']}: held out by field"] += 1
                continue
            if mode == "subtitles":
                add(s, r, sentences(pc(respell_unquoted(raw(r, s))), s.get("joined_lines", False)))
            elif mode == "clean":
                t = pc(raw(r, s))
                add(s, r, sentences(t, False) if s.get("sentences") else t)
            elif mode == "sung":
                t = pc(raw(r, s))
                add(s, r, upper_tr(t[:1]) + t[1:])
            elif mode == "read":
                t = r.get("text") or ""
                if t and not re.search(r"\d", t):
                    src[s["name"]].append(dict(r, text=sentences(respell(t), False)))
            elif mode == "english":
                if r.get("text") and 0.8 <= r["duration"] <= 30:
                    src[s["name"]].append({"audio_filepath": r["audio_filepath"], "duration": r["duration"],
                                           "text": r["text"]})
            elif mode == "none":
                src[s["name"]].append({"audio_filepath": r["audio_filepath"], "duration": r["duration"], "text": ""})
            else:
                sys.exit(f"{s['name']}: unknown text mode {mode}")

    no_text = {s["name"] for s in sources if s["text"] == "none"}
    copies, capped = Counter(), 0
    rng = random.Random(a.seed)
    for name in src:
        rng.shuffle(src[name])
        if name in no_text:
            continue
        keep = []
        for r in src[name]:
            same = re.sub(r"[^\w ]", "", r["text"].lower())
            copies[same] += 1
            if copies[same] > a.cap and r["duration"] > SHORT_S:
                capped += 1
                continue
            keep.append(r)
        src[name] = keep

    group_of = {s["name"]: s["group"] for s in sources}
    inter = cfg.get("interjections")
    if inter:
        iname = inter.get("name", "drama_interj")
        src[iname] = [r for s in sources if s["group"] == rest for r in src[s["name"]] if has_interjection(r["text"])]
        group_of[iname] = rest
    mode_of = {s["name"]: s["text"] for s in sources}
    turkish = [k for k in src if mode_of.get(k, "subtitles") not in ("english", "none")]

    # ---- guards on what is written: any hit stops the script
    def stop(msg):
        sys.exit(f"LEAK, nothing written: {msg}")

    for k in turkish:
        bad = sum(bool(re.search(r"\d", r["text"])) for r in src[k])
        if src[k] and bad / len(src[k]) > 0.001:
            stop(f"{k}: {bad} of {len(src[k])} rows still carry a digit (numbers must be spoken)")
        odd = Counter(c for r in src[k] for c in r["text"] if c not in ALLOWED)
        if odd and mode_of.get(k) == "read":  # final lists, taken as they are
            print(f"  note: {k} keeps characters outside the alphabet: {dict(odd.most_common(10))}")
        elif odd:
            stop(f"{k}: characters outside the alphabet left in the text: {dict(odd.most_common(10))}")
        for r in src[k]:
            if held.equal(r["text"]):
                stop(f"{k}: a held-out test/dev sentence: {r['text']!r} ({r['audio_filepath']})")
            if mode_of.get(k, "subtitles") in CHECKED:
                w = norm_words(r["text"])
                if held.contains(w) or held.shares_run(w):
                    stop(f"{k}: a row holding a held-out sentence: {r['text']!r} ({r['audio_filepath']})")
    for k, ids in held_ids.items():  # a second, independent check on the clip paths
        for r in src[k]:
            hit = clip_ids(r["audio_filepath"]) & ids
            if hit:
                stop(f"a held-out id ({', '.join(sorted(hit))}) is in {k}: {r['audio_filepath']}")
    check = Spans()  # no doubles, re-checked in the order the rows were taken
    for s in sources:
        if s.get("dedup") == "seed":
            for r in src[s["name"]]:
                if r.get("_span"):
                    check.add(*r["_span"])
    for r in sorted((r for s in sources if s.get("dedup") == "check" for r in src[s["name"]]), key=lambda r: r["_n"]):
        sp = r["_span"]
        ov = check.overlap(*sp)
        if ov > max(OVERLAP_S, OVERLAP_FRAC * (sp[2] - sp[1])):
            stop(f"a row doubles a kept row by {ov:.2f} s: {r['audio_filepath']}")
        check.add(*sp)
    for k in src:  # no training row may name a clip of a test or dev list (any extension)
        for r in src[k]:
            if os.path.splitext(r["audio_filepath"])[0] in held.clips:
                stop(f"{k} names a clip of a test or dev list: {r['audio_filepath']}")
    gone = [r["audio_filepath"] for k in src for r in src[k] if not os.path.exists(r["audio_filepath"])]
    if gone:
        print(f"  warning: {len(gone)} clips of the lists are missing on disk, e.g. {gone[:3]}")

    long_clips = Counter()
    for name in src:
        keep = [r for r in src[name] if r["duration"] <= a.max_s]
        long_clips[name] = len(src[name]) - len(keep)
        src[name] = keep
    src = {k: v for k, v in src.items() if v}
    hours = {name: sum(r["duration"] for r in rows) / 3600 for name, rows in src.items()}
    max_passes = {s["name"]: s.get("max_passes", PASSES_MAX) for s in sources}
    share_max = {s["name"]: s.get("share_max", 1.0) for s in sources}
    if inter:
        max_passes[iname] = inter.get("max_passes", PASSES_MAX)

    def cap_of(k):
        g = group_by_name[group_of[k]]
        passes = min(max_passes.get(k, PASSES_MAX), g.get("max_passes", PASSES_MAX), PASSES_MAX)
        return min(hours[k] * passes / hours_heard, share_max.get(k, 1.0))

    def fill(members, total):
        """Equal passes over members, capped (water-filling); returns the shares and what is left over."""
        share, free, left = {}, [k for k in members if hours.get(k)], total
        while free and left > 1e-12:
            h = sum(hours[k] for k in free)
            over = [k for k in free if left * hours[k] / h >= cap_of(k)]
            if not over:
                share.update({k: left * hours[k] / h for k in free})
                return share, 0.0
            for k in over:
                share[k] = cap_of(k)
                left -= share[k]
                free.remove(k)
        return share, max(left, 0.0)

    share, spare = {}, 1.0 - sum(g["share"] for g in groups)
    for g in groups:
        if g["name"] != rest:
            got, left = fill([k for k in src if group_of.get(k) == g["name"]], g["share"])
            share.update(got)
            spare += left
    rest_part = group_by_name[rest]["share"] + spare
    if inter:
        got, _ = fill([iname], rest_part * inter["share_of_group"])
        share.update(got)
    rest_members = [k for k in src if group_of.get(k) == rest and k != (inter or {}).get("name")]
    got, left2 = fill(rest_members, rest_part - share.get((inter or {}).get("name", ""), 0.0))
    share.update(got)
    if left2 > 1e-9:
        print(f"  warning: {left2:.4f} of the hours heard found no source under its cap")

    # ---- written only now, after every guard: the checks (in the trainer's format, text handled as in training;
    # not filtered, they are held out), the lists, then the mix
    for c in cfg.get("checks", []):
        rows = read_list(path_of(c["path"]))
        with open(out / f"val_{c['name']}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                text = sentences(pc(respell_unquoted(raw(r, c))), False)
                if text:
                    f.write(json.dumps({"audio_filepath": r["audio_filepath"], "duration": r["duration"],
                                        "text": text}, ensure_ascii=False) + "\n")
        print(f"check set val_{c['name']}.jsonl: {len(rows)} clips")
    for name, rows in src.items():
        with open(out / f"{name}.jsonl.tmp", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
        os.replace(out / f"{name}.jsonl.tmp", out / f"{name}.jsonl")
    mix = []
    for name, s in share.items():
        if not s:
            continue
        rows = src[name]
        mean = sum(r["duration"] for r in rows) / len(rows)
        mix.append({"type": "nemo", "manifest_filepath": str((out / f"{name}.jsonl").resolve()), "weight": s / mean,
                    "share_of_hours": round(s, 4), "hours": round(hours[name], 1),
                    "repeats": round(s * hours_heard / max(hours[name], 0.01), 1)})
    total = sum(m["weight"] for m in mix)
    for m in mix:
        m["weight"] = round(m["weight"] / total, 6)
    json.dump(mix, open(out / "mix.json", "w"), indent=1)
    drop = tuple(cfg.get("fc_drop", ["english", "nonspeech"]))
    fc = [dict(m) for m in mix if not any(d in Path(m["manifest_filepath"]).name for d in drop)]
    w = sum(m["weight"] for m in fc) or 1.0
    for m in fc:
        m["weight"] = round(m["weight"] / w, 6)
    json.dump(fc, open(out / "mix_fc.json", "w"), indent=1)

    print(f"repeats over the cap left out: {capped}")
    print(f"left out: {dict(count)}")
    print(f"over --max-s {a.max_s:g} s (never heard), left out: {dict(+long_clips)}")
    print(f"mix over {hours_heard:.0f} h heard (the groups' shares add up to {sum(g['share'] for g in groups):.2f}: "
          f"the rest, and what a group cannot fill, goes to {rest}):")
    gsum = defaultdict(float)
    for m in mix:
        gsum[group_of.get(Path(m["manifest_filepath"]).stem, "?")] += m["share_of_hours"]
    for g in groups:
        print(f" {g['name']:<14} share {gsum.get(g['name'], 0):.4f} (aim {g['share']:.2f})  "
              f"heard {gsum.get(g['name'], 0) * hours_heard:7.0f} h")
        for m in sorted((m for m in mix if group_of.get(Path(m["manifest_filepath"]).stem) == g["name"]),
                        key=lambda m: -m["share_of_hours"]):
            print(f"   {Path(m['manifest_filepath']).stem:<24} {m['hours']:7.1f} h  share {m['share_of_hours']:.4f}  "
                  f"heard {m['share_of_hours'] * hours_heard:7.0f} h  passes {m['repeats']:5.1f}  "
                  f"weight {m['weight']:.6f}")
    print(f"mix_fc.json: {len(fc)} of {len(mix)} sources")


if __name__ == "__main__":
    main()
