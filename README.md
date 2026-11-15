# QiTV ASR: training and evaluation code

The scripts behind three Turkish speech recognizers built for live captions in [QiTV](https://github.com/ozankaraali/QiTV):

- **[qitv-asr-tr-moonshine-mid](https://huggingface.co/ozankaraali/qitv-asr-tr-moonshine-mid):** Moonshine Streaming Medium, trained further on Turkish (it also writes English);
- **[qitv-asr-tr-moonshine-small](https://huggingface.co/ozankaraali/qitv-asr-tr-moonshine-small):** the same for Moonshine Streaming Small;
- **[qitv-asr-tr-conformer](https://huggingface.co/ozankaraali/qitv-asr-tr-conformer):** NVIDIA's cache-aware streaming FastConformer (hybrid RNNT + CTC), trained further on
  Turkish.

The models, their results and their licence are on the model pages. This repository is the method behind them.

## What is here

Everything from clip lists to scores:

- tokenizers and the training mix;
- both trainers, the final cool-down (a last stretch at a falling learning rate) and checkpoint averaging;
- the ONNX export for sherpa-onnx;
- the scorer and its text normaliser;
- beam search with an n-gram language model;
- the test-set builders.

## What is not here

- **No data:** no audio, no transcripts, no manifests (clip lists), no lists of what was trained on or held out. The
  training audio and text are not redistributed.
- **No download code:** every script reads files you already have.

Use it to train on data you have the rights to, and to build your own tests (for example on TV drama, talk shows or
live streams you may use).

## Layout

```
common/   tr_numbers.py   Turkish numbers as words (training text and the scorer)
          tr_numbers_written.py  the models' number words as digits, as QiTV shows them (for display)
          tr_text.py      training-text cleaning and the letters-per-second checks
          en_numbers.py   English numbers as words, or the line dropped when the reading is unclear
          scoring.py      the scorers (standard, strict, English), WER/CER, bootstrap, punctuation counts
data/     subs_align.py   a recording + its subtitles -> clips (training lines, or a test set with --keep-all)
          long_align.py   a long recording + its written text (minutes, transcript) -> clips
          mms_check.py    the forced-aligner check: keeps a clip only where MMS finds its text in the audio,
                          start and end included
          musan_pieces.py no-speech and sung-without-text clips (training) and the no-speech test
          build_mix.py    source manifests -> cleaned per-source lists + mix.json (weights) + mix_fc.json
train/    sp_turkish.py   the Turkish SentencePiece for Moonshine (8,192 pieces)
          sp_extend.py    + English pieces appended (10,623 in all); grows a model's embeddings to match
          sp_conformer.py the FastConformer's SentencePiece (1,024 pieces)
          moon_train.py   the Moonshine trainer (vocabulary swap, augmentation, wsd/decay schedules, resume)
          tempo.py        WSOLA time-stretch (pitch kept), used by the trainer's augmentation
          moon_average.py the mean of a run's last snapshots
          fc_train.py     the FastConformer trainer (NeMo + Lightning)
export/   moon_export.sh  Moonshine -> sherpa-onnx int8 files (with WorkmindAI/stt-models)
          moon_tokens.py  the base64 tokens.txt sherpa-onnx's Moonshine loader reads
          hf_tokenizer.py tokenizer.json for the 🤗 Transformers weights folder, checked against SentencePiece
          fc_export.py/sh FastConformer .nemo -> the released sherpa-onnx streaming transducer files (int8)
eval/     read.py         one model reads every clip of a test manifest, as it is deployed
          score.py        WER/CER with 95 % intervals, paired differences, punctuation F1
          build_fleurs.py, build_cv_test.py, build_subtitle_test.py   test-set builders
decode/   moonbeam.py     Moonshine through onnxruntime: greedy and beam search with shallow fusion
          beam_run.py     decoder settings compared on a test manifest
          lm_text.py      training text as piece ids, for KenLM
          qng.py          the compact n-gram file the decoder reads (ARPA -> .qng)
examples/ invented example rows, a mix config, a subtitle file and a subtitle-test item list
```

Run the scripts from the repository's root (they find `common/` themselves). `python <script> --help` gives every
option.

## Requirements

- **Software:** Python 3.12 and the versions in `requirements.txt` (PyTorch 2.11 with CUDA 12.8, Transformers 5.22.1,
  NeMo 3.0.0, sherpa-onnx 2.0.1). On an RTX 50xx, NeMo's RNNT loss needs `numba-cuda` 0.30.4.
- **Hardware used for the released models:** one consumer GPU at a time. An RTX 5080 (16 GB) trained most of it; an
  RTX 4090 (24 GB) trained part of the FastConformer and of the small Moonshine. 32 CPU cores and 62 GB of RAM fed the data loaders.
- **Speed:** on the 5080, Moonshine Medium went through about 480 hours of audio per GPU hour (batches of 240 s,
  gradient checkpointing), Small about 800.
- **Time:** Medium took about three days of training in all; its final cool-down took 7 hours.

## Terms used below

- **Manifest:** a list of clips with their text, one JSON object per line (the format is in the next section).
- **Test set and dev set:** a test set is only scored. A dev (development) set is scored during training, to check
  progress and pick checkpoints.
- **Held out:** kept out of training.
- **Listener:** a speech recognizer that the clip cutters use to place text in the audio and to check clips. Any
  offline sherpa-onnx NeMo transducer will do.
- **Forced aligner:** a tool that is given the text and finds where it sits in the audio. It never writes a
  transcript of its own. These scripts use MMS's.
- **Mix:** the training lists together with a sampling weight for each source.
- **Pass:** one time through all of a source's clips.
- **Greedy (plain) decoding and beam search:** greedy decoding writes the model's most likely next piece (a word fragment) at each
  step; a beam search keeps several candidate texts as it goes.

## The manifest format

Every list is JSON lines, one clip a line, 16 kHz mono audio (WAV, FLAC or Ogg Opus):

```json
{"audio_filepath": "/data/clips/drama_a/ep01_000012340_000015870.ogg", "duration": 3.53, "text": "Ne zaman geleceksin?"}
```

- `audio_filepath`: the clip; `duration`: seconds.
- `text`: what the trainer learns: cased, punctuated Turkish (`. , ? !`, apostrophes of names, a hyphen inside a
  word), **numbers as words** ("on yedi", not "17"). `data/build_mix.py` writes it from the fields below.
- Optional fields, as the builders write them:
  - `text_raw`: the text as written, digits and all;
  - `text_spoken`: numbers as words (`python -m common.tr_numbers in.jsonl out.jsonl` adds it);
  - `text_std`: a spelling-normalised version that wins over `text_raw` when present;
  - `episode`, `series`, `session`, `start`, `end`: where the clip sits in its recording. The checks that keep
    held-out material and overlapping copies out of training use them;
  - `cer` and `heard`: the listener's letter error and its transcript, from the aligners.
- A clip name of the form `<recording>_<start ms>_<end ms>.<ext>` (the aligners write them) also tells the checks
  where the clip sits.
- Test manifests use the same format; the scorer reads `text_spoken`, else `text`, else `text_raw`.

See `examples/manifest_example.jsonl`. Every row there is invented.

## The pipeline, step by step

Paths below are examples. `$Q` is any folder of yours.

| Step | What it does |
|---|---|
| [1. Test sets first](#1-test-sets-first) | builds the test and dev sets |
| [2. Training lists](#2-training-lists) | cuts recordings and their text into checked clips |
| [3. The mix](#3-the-mix) | cleans the lists, keeps test material out, sets the weights |
| [4. Tokenizers](#4-tokenizers) | builds the vocabularies |
| [5. Moonshine training](#5-moonshine-training) | the long run |
| [6. The cool-down and averaging](#6-the-cool-down-and-averaging) | the last stretch of Moonshine training |
| [7. FastConformer training](#7-fastconformer-training) | NVIDIA's English FastConformer to Turkish, and fine-tuning |
| [8. Export](#8-export) | writes the sherpa-onnx files |
| [9. Evaluation](#9-evaluation) | transcribes the test sets and scores them |
| [10. Beam search with an n-gram](#10-beam-search-with-an-n-gram-optional) | optional decoding for Moonshine |

### 1. Test sets first

Build the test and dev sets before any training list, so the mix builder can keep their sentences, recordings and
speakers out.

The subtitle test below has a second, checked version. Uploaded subtitles often shorten or rephrase what is said, so
`data/mms_check.py` keeps only the clips whose text the forced aligner finds in the audio, start and end included.
It works in three commands:

- `controls` makes control clips from the list itself: clips given text from another recording, and clips whose text
  fits, with a word at the start or end taken out or put in.
- `bars` sets the pass marks (the bars) from those controls, before any clip is checked.
- `check` keeps the clips that pass.

No recognizer reads anything in this check.

```bash
# FLEURS Turkish dev (it picks checkpoints) and test, from your copy of FLEURS's tr_tr folder
python eval/build_fleurs.py /data/fleurs/tr_tr $Q/eval
# Common Voice: 1,500 random clips of a release's Turkish test split, from your copy of the release
python eval/build_cv_test.py /data/cv-corpus-27.0/tr $Q/eval --clips 1500
# A test from recordings and subtitles you may use (talk, vlogs, drama, streams): cut on the subtitle cues, checked
# for timing by a voice detector only; see examples/subtitle_items_example.jsonl
python eval/build_subtitle_test.py items.jsonl $Q/eval/subs --vad silero_vad.onnx --name talk_test
# Its checked version: controls, then bars, then the check (see above)
python data/mms_check.py controls $Q/eval/subs/talk_test.jsonl $Q/eval/subs/talk.controls.jsonl --mode fixed \
    --foreign other --field text
python data/mms_check.py bars $Q/eval/subs/talk.controls.jsonl $Q/eval/subs/talk.bars.json --placed-controls
python data/mms_check.py check $Q/eval/subs/talk_test.jsonl $Q/eval/subs/talk.bars.json \
    $Q/eval/subs/talk_test_checked.jsonl --field text --min-words 1
# The same subtitles run together into caption-sized pieces of up to 10 s, every line kept (how the model cards'
# Drama A test is cut)
python data/subs_align.py episode.wav episode.tr.vtt $Q/eval/drama_pieces --listener <listener dir> --keep-all \
    --max-piece 10 --interjections
# A no-speech test (noise and music without vocals); its source files go to --bg-exclude later
python data/musan_pieces.py test /data/musan/noise.json $Q/eval/nonspeech_test.jsonl
```

MediaSpeech, VoxPopuli and others: write their clips and transcripts into the same manifest format.

### 2. Training lists

**What the released models learned from:** subtitled conversational speech and films, parliamentary sessions with
their minutes, speeches, audiobooks, podcasts, sung lyrics, online videos (YODAS), Common Voice and FLEURS train,
synthetic speech, and English. The final training set held about 7,700 hours of Turkish (unique audio) and 650 of
English. Text came from people wherever possible. Machine transcripts were used only after a language model
proofread them or, for the final training set's YODAS v3 spans, where the forced aligner confirmed that the text
fits the audio.

There are two ways to choose which clips stay:

- **The listener gate** (`subs_align.py`, `long_align.py`): the listener transcribes each piece, and the piece is
  kept where its transcript and the text differ by at most 15 % (subtitles) or 10 % (minutes) of their letters.
- **The forced-aligner check** (`data/mms_check.py`): a listener or the subtitle times only place the text roughly.
  The forced aligner then decides. Its pass marks come from controls (text from another recording, a word at the
  start or end taken out or put in), never from a model's score.

In the released models' training data, the forced-aligner check chose the podcast clips, part of the parliament
clips, the newest TV subtitle lines and the YODAS v3 spans; the listener gate chose the rest. When two lists cover
the same audio, the mix builder keeps them from overlapping.

```bash
# The listener gate. A recording and its hand-made subtitles -> clips whose text is the subtitler's, kept where the
# listener hears the same words within 15 % of letters
python data/subs_align.py episode.wav episode.tr.vtt $Q/clips/ep01 --listener <listener dir> --opus --interjections
# A long recording and its written text (official minutes, a transcript), aligned word by word, kept within 10 %
python data/long_align.py session.ogg session.txt $Q/clips/session01 --listener <listener dir> --opus

# The forced-aligner check instead, window mode (subtitle lines, minutes, transcripts). First cut every piece the
# text is placed on. The aligner then takes the piece's words +-4 words in order, free to start and end at any word,
# and the words it covers become the clip's text. Calibrate once per source, on a sample of its recordings, before
# checking
python data/subs_align.py episode.wav episode.tr.vtt $Q/pieces/ep01 --listener <listener dir> --opus --keep-all
python data/long_align.py session.ogg session.txt $Q/pieces/session01 --listener <listener dir> --opus --placed
cat $Q/pieces/ep*/manifest.jsonl > $Q/pieces/tv.jsonl
python data/mms_check.py controls $Q/pieces/tv.jsonl $Q/pieces/tv.controls.jsonl --mode window \
    --recordings 45 --calib-max-cer 0.15
python data/mms_check.py bars $Q/pieces/tv.controls.jsonl $Q/pieces/tv.bars.json
python data/mms_check.py check $Q/pieces/tv.jsonl $Q/pieces/tv.bars.json $Q/sources/tv_lines.jsonl
# Fixed mode (a span list with its own transcripts): rows {audio_filepath, duration, text, video_id, start, end}.
# Rows of the same recordings that a text filter left out still go in, with "context": true: their words are the
# neighbouring text. --need-marks leaves out a transcript with no punctuation mark at all, as the final training
# set did
python data/mms_check.py controls spans.jsonl spans.controls.jsonl --mode fixed --field text --need-marks \
    --recordings 40 --min-clips 30
python data/mms_check.py bars spans.controls.jsonl spans.bars.json
python data/mms_check.py check spans.jsonl spans.bars.json $Q/sources/video_spans.jsonl --field text --need-marks

# Numbers as words beside the written text
python -m common.tr_numbers $Q/clips/ep01/manifest.jsonl $Q/sources/drama_a.jsonl
python -m common.en_numbers $Q/sources/english.jsonl
# No-speech clips and songs without text (the files of the no-speech test excluded)
python data/musan_pieces.py train /data/musan/noise.json $Q/sources/nonspeech.jsonl --exclude $Q/eval/nonspeech_test.files.txt
python data/musan_pieces.py sung /data/musan/music $Q/sources/songs_no_text.jsonl
```

### 3. The mix

`data/build_mix.py` turns the source lists into the training mix. It:

- cleans every source's text;
- keeps test material out, checking against every test and dev list: no line equal to a held-out line of six words
  or more, none containing one, none sharing eight words in a row with one, no held-out episode, series, session or
  clip;
- caps identical lines at 5;
- leaves out overlapping copies of the same speech;
- sets each source's sampling weight from its group's share of the hours the trainer goes through.

It stops before writing anything if one of its checks fails. See its docstring and `examples/mix_config.json`.

```bash
python data/build_mix.py examples/mix_config.json $Q/mix --train-hours 70 --heard-per-gpu-hour 480
# -> $Q/mix/<source>.jsonl, val_<check>.jsonl, mix.json (Moonshine), mix_fc.json (FastConformer: no English or
#    no-speech lists)
```

**Source names matter.** The trainer reads a source's kind from its list's file name:

- `nonspeech` and `songs`: the no-speech lists (no augmentation, no tempo change);
- names starting with `drama` and `films`: few added backgrounds, because they carry their own music;
- `parliament`, `speeches`, `cv`, `fleurs`, `books` and `english`: each has its own background share;
- any other name: a background share of 0.4.

**The released models' group shares:**

| Group | Share of the hours |
|---|---|
| Drama | 0.30 (it also takes whatever the others cannot fill) |
| Parliament | 0.13 |
| Read and formal speech | 0.18 |
| Broad speech | 0.25 (at most 3 passes per source) |
| Films and sung lyrics | 0.08 |
| English | 0.09 |
| No speech | 0.01 |

Any source gets at most 6 passes; synthetic speech at most 1.5 passes and 5 %.

### 4. Tokenizers

```bash
# Moonshine: a Turkish unigram SentencePiece (8,192 pieces, identity normalisation, apostrophes kept), then English
# pieces appended at a score offset of -8, so every Turkish piece keeps its id: 10,623 pieces
python train/sp_turkish.py $Q/sp $Q/mix/drama_a.jsonl $Q/mix/parliament.jsonl ...
python train/sp_extend.py build --tr $Q/sp/tr.model --out $Q/sp/tr_en.model --english $Q/mix/english.jsonl \
    --turkish $Q/mix/drama_a.jsonl,$Q/mix/parliament.jsonl --offset -8
# FastConformer: a cased, punctuated unigram SentencePiece of 1,024 pieces
python train/sp_conformer.py $Q/sp_fc $Q/mix/*.jsonl
```

### 5. Moonshine training

The trainer swaps the vocabulary in when it starts from the English base (`--base` with `--sp`, no `--init`): each
new piece's embedding starts as the mean of the English pieces that spell it. Every evaluation saves `<out>/last`
and a resumable `<out>/state.pt`.

A long run, warm-up then a constant learning rate (`--schedule wsd`, called wsd below; it can be stopped at any evaluation):

```bash
python train/moon_train.py $Q/exp/qitv-asr-tr-moonshine-mid --base moonshine-streaming-medium --sp $Q/sp/tr_en.model \
    --mix $Q/mix/mix.json --val $Q/eval/fleurs_dev.jsonl,$Q/mix/val_drama_dev.jsonl \
    --musan-dir /data/musan --rirs /data/rirs.jsonl.gz --bg-exclude $Q/eval/nonspeech_test.files.txt \
    --bucket 4 --carry 1 --no-repeat 1 --max-s 25 --aug2 1 --specaug 0.5 --dec-dropout 0.1 \
    --batch-s 240 --checkpointing 1 --lr 1e-4 --warmup 2000 --schedule wsd --decay-frac 0.1 --final-lr 2e-6 \
    --hours 70 --eval-every 4000 --snapshots 3 --best-by drama --workers 8
# a stopped run goes on with --resume (same arguments)
```

### 6. The cool-down and averaging

When the long run stops improving, cool it down on the final mix in the same folder (copy `last/` and `state.pt`
aside first if you want to keep the long run's state): `--schedule decay` runs the learning rate from `--lr` down to `--final-lr` over `--hours` from now,
keeping the optimizer state.

```bash
python train/moon_train.py $Q/exp/qitv-asr-tr-moonshine-mid --resume --base moonshine-streaming-medium --sp $Q/sp/tr_en.model \
    --mix $Q/mix_final/mix.json --val ... (as above) --batch-s 240 --checkpointing 1 \
    --schedule decay --lr 5e-5 --final-lr 2e-6 --hours 7 --eval-every 2000 --snapshots 3 --best-by drama
python train/moon_average.py $Q/exp/qitv-asr-tr-moonshine-mid $Q/exp/qitv-asr-tr-moonshine-mid/avg 3  # optional: the mean of the last 3 snapshots
```

The released Moonshine weights are the **last** state of the cool-down (`<out>/last`).

| | Medium (released) | Small (released) |
|---|---|---|
| Start | English Moonshine Streaming Medium, vocabulary swapped | English Moonshine Streaming Small, vocabulary swapped |
| Long run | wsd, peak 1e-4 then 5e-5, batches of 240 s, gradient checkpointing; 384,000 steps | wsd, peak 8e-5, decay over its last 20 %, batches of 120 s x 2, no checkpointing |
| Cool-down | 7 h, 5e-5 -> 2e-6 | 3 h, re-warmed to 3e-5 -> 1e-6 |
| Both | `--bucket 4 --carry 1 --no-repeat 1 --max-s 25 --aug2 1 --specaug 0.5 --dec-dropout 0.1` | |


**Fine-tuning a released Moonshine model.** Start from a download of its model repository with `--init` (its Turkish and English
pieces are already in, so nothing is swapped) and give its `tr.model` as `--sp`, for example a short decaying run on
your own mix:

```bash
python train/moon_train.py $Q/exp/my_medium --init qitv-asr-tr-moonshine-mid \
    --base moonshine-streaming-medium --sp qitv-asr-tr-moonshine-mid/tr.model \
    --mix $Q/mix/mix.json --val ... (as above) --batch-s 240 --checkpointing 1 \
    --schedule decay --lr 3e-5 --final-lr 1e-6 --hours 3 --eval-every 2000 --snapshots 3 --best-by drama
```

### 7. FastConformer training

One run turns NVIDIA's English streaming FastConformer into a Turkish one: the Turkish vocabulary goes in, then it
trains on your mix with a warm-up, a cosine decay and an exponential moving average (EMA) of the weights. The first
argument names the run; its model is written to `exp/<name>/checkpoints/<name>.nemo`.

```bash
python train/fc_train.py qitv-asr-tr-conformer 105000 --init stt_en_fastconformer_hybrid_large_streaming_multi.nemo \
    --tokenizer-dir $Q/sp_fc --mix $Q/mix/mix_fc.json --val $Q/eval/fleurs_dev.jsonl,$Q/mix/val_drama_dev.jsonl \
    --noise /data/musan/noise.json --rirs /data/rirs.jsonl.gz --lr 1e-3 --warmup 3000 --min-lr 1e-5
```

**Fine-tuning the released conformer.** Start from its `qitv-asr-tr-conformer.nemo` (no `--tokenizer-dir`: the
Turkish vocabulary is already in) with a low learning rate, as its own last training did:

```bash
python train/fc_train.py my-conformer 6000 --init qitv-asr-tr-conformer.nemo --mix $Q/mix/mix_fc.json \
    --val ... (as above) --lr 5e-5 --warmup 300 --min-lr 5e-6 --batch-s 90 --accumulate 5 --max-dur 25 \
    --val-every 2000
```

The released sherpa-onnx files were exported from the released .nemo with the command below.

### Showing numbers as digits

The models write every number in words, as they were trained ("saat on beş otuzda"). QiTV shows them as Turkish
subtitles write them; `common/tr_numbers_written.py` does the same, with the standard library only:

```python
from common.tr_numbers_written import written
print(written("saat on beş otuzda yüzde beş arttı ve iki bin yedide on dört kişi geldi"))
# saat 15.30'da %5 arttı ve 2007'de 14 kişi geldi
```

### 8. Export

```bash
# Moonshine -> sherpa-onnx (int8 encoder and merged decoder, base64 tokens.txt)
git clone https://github.com/WorkmindAI/stt-models  # MIT; the released files used commit 91d5726
STT_MODELS=$PWD/stt-models bash export/moon_export.sh $Q/exp/qitv-asr-tr-moonshine-mid/last $Q/export/qitv-asr-tr-moonshine-mid
# the 🤗 Transformers folder needs a tokenizer.json built from tr.model (the trainer leaves the base model's)
python export/hf_tokenizer.py $Q/exp/qitv-asr-tr-moonshine-mid/last $Q/eval/fleurs_test.jsonl --write
# FastConformer -> the released sherpa-onnx streaming transducer files: look-ahead pinned at 1,040 ms; NeMo's own
# features and NeMo's first streaming chunk built into the encoder; int8 (the encoder and joiner with per-channel
# weights, the encoder's subsampling front end in float32); with a quick check
PYTHON=python CHECK=$Q/eval/fleurs_dev.jsonl bash export/fc_export.sh exp/qitv-asr-tr-conformer/checkpoints/qitv-asr-tr-conformer.nemo $Q/export/qitv-asr-tr-conformer
```

**Running the conformer's files:**

- set `normalize_samples=False` (left on, the encoder writes nothing);
- feed 0.16 s of silence before the speech (or a first word can be lost);
- for a whole clip, feed about 3 s of silence after it.

**Reproducibility:** from the released .nemo, `export/fc_export.py` writes the released int8 files byte for
byte (NeMo 3.0.0, onnx 1.23.2, onnxruntime 1.31.1, on the CPU).

### 9. Evaluation

Every model transcribes every clip, as it is deployed; one scorer is used for every model and reference.

```bash
python eval/read.py moonshine $Q/export/qitv-asr-tr-moonshine-mid/onnx $Q/eval/fleurs_test.jsonl $Q/hyps/mid_fleurs.jsonl --workers 8 --threads 1
python eval/read.py sherpa-stream $Q/export/qitv-asr-tr-conformer $Q/eval/fleurs_test.jsonl $Q/hyps/fc_fleurs.jsonl \
    --int8 --raw --lead-s 0.16 --tail-s 3 --paths 4 --blank-penalty 0.5 --workers 8 --threads 1
# the conformer's .nemo in NeMo (the model cards give these numbers too)
python eval/read.py nemo qitv-asr-tr-conformer.nemo $Q/eval/fleurs_test.jsonl $Q/hyps/fc_nemo_fleurs.jsonl \
    --paths 4 --blank-penalty 0.5 --gpu-gb 6
python eval/score.py $Q/eval/fleurs_test.jsonl mid=$Q/hyps/mid_fleurs.jsonl conformer=$Q/hyps/fc_fleurs.jsonl \
    --fix-thousands --punct
python eval/score.py $Q/eval/en_test.jsonl mid=$Q/hyps/mid_en.jsonl --scorer en
```

**The standard scorer** gives the published numbers. Before comparing, it:

- writes numbers as spoken words on both sides (ordinals and suffixes included);
- takes out bracketed sound notes and punctuation;
- folds Turkish I/İ, case and circumflexes, and joins apostrophes;
- applies QiTV's display step to the model's text first (a lone "i" joins the next word).

WER (words wrong) = word edits summed over clips / reference words, with a 95 % bootstrap interval over clips. CER
(letters wrong) is counted on the same words joined by spaces.

**How each released model was run:**

- **The Moonshine models:** their int8 files in sherpa-onnx, plain (greedy) decoding.
- **The FastConformer:** its int8 files in sherpa-onnx's streaming recognizer, as in the commands above: whole clips
  with 0.16 s of silence before and 3 s after, fed in 0.1 s pieces, normalize_samples off, beam search over 4 paths,
  the blank's score lowered by 0.5. Its .nemo was also read in NeMo at a 1,040 ms look-ahead with the same beam and
  penalty.
- **Threads:** one thread a worker, because onnxruntime's int8 sums can differ by thread count.
- **The comparison models:** the ones in `eval/read.py`'s docstring were read as their model cards set them up.

### 10. Beam search with an n-gram (optional)

QiTV can decode Moonshine with a 4-path beam and an n-gram language model over the model's pieces (weight 0.1, bonus
0.5 a piece). Adding a language model's score during the beam search is called shallow fusion. It gave 0.5-1.5 WER
points on the public sets. Plain decoding often reads more naturally, and the model cards' main numbers are greedy.
The model cards' beam numbers are the 4-path beam alone, no n-gram (`--config b4=4` below).

```bash
python decode/lm_text.py $Q/sp/tr_en.model $Q/mix_final/*.jsonl --heldout $Q/eval/fleurs_test.jsonl,... > lm_text.txt
lmplz -o 4 --prune 0 2 4 6 --discount_fallback < lm_text.txt > lm.arpa   # KenLM
python decode/qng.py lm.arpa lm.qng
cd decode && python beam_run.py $Q/export/qitv-asr-tr-moonshine-mid/onnx $Q/eval/fleurs_test.jsonl $Q/beam \
    --config greedy=1 --config b4=4 --config b4lm=4,lm.qng,0.1,0.5
python ../eval/score.py $Q/eval/fleurs_test.jsonl greedy=$Q/beam/greedy.jsonl b4lm=$Q/beam/b4lm.jsonl --fix-thousands
```

The released language model was a pruned 4-gram of 38 MB over piece ids. The exact command that built it was not
kept with the code; the lines above follow its file name (order 4, pruning 0 2 4 6) and are a reconstruction.

## Credits

- **Models and runtimes:** [Moonshine Streaming](https://huggingface.co/moonshine-ai/moonshine-streaming-medium) by
  Useful Sensors (Moonshine AI), MIT; NVIDIA's
  [streaming FastConformer](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi)
  (CC BY 4.0) and [NeMo](https://github.com/NVIDIA/NeMo) (Apache 2.0);
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) by the k2-fsa team (Apache 2.0).
- **Tools:** [WorkmindAI/stt-models](https://github.com/WorkmindAI/stt-models) (MIT) for the Moonshine ONNX export;
  [KenLM](https://github.com/kpu/kenlm) for the n-gram; [Silero VAD](https://github.com/snakers4/silero-vad);
  [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)'s forced aligner, through
  [torchaudio's MMS_FA pipeline](https://docs.pytorch.org/audio/stable/pipelines.html), for the clip check.
- **Text normalization:** [Damga](https://pypi.org/project/damga/) by mihu.ai (Apache 2.0): its text normalizer
  wrote the numbers as spoken words in part of the training text, in earlier stages of training. The number code in
  this repository (`common/tr_numbers.py`, `common/tr_numbers_written.py`) is our own, not a fork of it.
- **Data:** [MUSAN](https://www.openslr.org/17/) and [room impulse responses](https://www.openslr.org/28/) (OpenSLR)
  for augmentation; [YODAS](https://huggingface.co/datasets/espnet/yodas) and
  [YODAS v3](https://huggingface.co/datasets/espnet/yodas3) (ESPnet) for speech from online videos.
- **Test sets:** the [FLEURS](https://huggingface.co/datasets/google/fleurs),
  [Common Voice](https://commonvoice.mozilla.org), [MediaSpeech](https://github.com/NTRLab/MediaSpeech) and
  [VoxPopuli](https://github.com/facebookresearch/voxpopuli) teams, for public test sets.

## Licence

The code: MIT (see LICENSE). The models have their own licence (CC BY 4.0) and notices on their model pages.
