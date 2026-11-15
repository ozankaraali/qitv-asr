"""NVIDIA's English cache-aware streaming FastConformer (hybrid RNNT + CTC, CC BY 4.0) trained further on Turkish,
with NeMo and Lightning.

- The first run starts from NVIDIA's .nemo (stt_en_fastconformer_hybrid_large_streaming_multi, downloaded by you
  from its model page) and swaps in a cased, punctuated Turkish SentencePiece (--tokenizer-dir: tokenizer.model and
  vocab.txt from train/sp_conformer.py), laid out as NeMo's change_vocabulary(new_tokenizer_type="bpe") reads it.
  Later rounds go on from the previous round's .nemo (--init, no --tokenizer-dir).
- Data: one manifest (--train) or a weighted mix (--mix: data/build_mix.py's mix_fc.json, one lhotse input per
  source with its weight), lhotse bucketing, clips up to --max-dur seconds.
- Augmentation (NeMo's own): noise and music beds (--noise, a NeMo noise manifest) at 5-20 dB SNR on 40 % of clips,
  room responses (--rirs, a lhotse recordings .jsonl.gz) on 30 %, speed perturbation.
- AdamW (betas 0.9/0.98, weight decay 1e-3), cosine schedule with warm-up to --min-lr, bf16, gradient clipping 1.0,
  EMA weights (decay 0.999). Checkpoints: the best three by the first validation manifest's WER, each also as .nemo.
- A batch that does not fit in GPU memory is skipped instead of ending the run.

usage: python fc_train.py <run name> <max steps> --init <model.nemo> (--train <manifest> | --mix <mix.json>)
       --val <dev1.jsonl,...> [--tokenizer-dir DIR] [--lr 1e-3] [--warmup 3000] [--min-lr 1e-5] [--max-hours H]
       [--batch-s 200] [--accumulate 2] [--max-dur 30] [--noise noise.json] [--rirs rirs.jsonl.gz] [--exp-dir exp]
The run (checkpoints, logs) is written under <exp-dir>/<run name>; a run of that name that exists is resumed."""

import argparse
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("steps", type=int)
ap.add_argument("--dry", action="store_true", help="tiny batches, two validation batches")
ap.add_argument("--init", required=True, help="the .nemo to start from (NVIDIA's base, or a previous round)")
ap.add_argument("--tokenizer-dir", default="", help="swap in this SentencePiece (tokenizer.model + vocab.txt)")
ap.add_argument("--train", default="", help="one training manifest")
ap.add_argument("--mix", default="", help="a weighted mix ([{manifest_filepath, weight}, ...]) instead of --train")
ap.add_argument("--val", required=True, help="validation manifests, comma-separated (the first decides checkpoints)")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--warmup", type=int, default=3000)
ap.add_argument("--min-lr", type=float, default=1e-5)
ap.add_argument("--max-hours", type=float, default=0.0, help="stop after this many hours (0: by steps only)")
ap.add_argument("--batch-s", type=float, default=200.0, help="seconds of audio a batch")
ap.add_argument("--accumulate", type=int, default=2, help="batches a step")
ap.add_argument("--max-dur", type=float, default=30.0, help="longest training clip, seconds")
ap.add_argument("--noise", default="", help="a NeMo noise manifest for noise/music beds ('' = none)")
ap.add_argument("--rirs", default="", help="room impulse responses, a lhotse recordings .jsonl.gz ('' = none)")
ap.add_argument("--workers", type=int, default=8, help="loader workers")
ap.add_argument("--val-every", type=int, default=4000, help="updates between checks")
ap.add_argument("--exp-dir", default="exp")
A = ap.parse_args()
if bool(A.train) == bool(A.mix):
    ap.error("give one of --train or --mix")
NAME, STEPS, DRY = A.name, A.steps, A.dry
BATCH_S, ACCUMULATE = A.batch_s, A.accumulate
torch.set_float32_matmul_precision("high")

# Resuming loads our own checkpoints; NeMo's EMA callback calls torch.load with PyTorch >= 2.6's
# weights_only default, which refuses the OmegaConf config inside them. These files are ours.
_torch_load = torch.load


def _load_own_checkpoint(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _load_own_checkpoint


def main():
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(A.init, map_location="cpu")
    if A.tokenizer_dir:  # the first run from the English base: the Turkish vocabulary
        model.change_vocabulary(new_tokenizer_dir=A.tokenizer_dir, new_tokenizer_type="bpe")
    cfg = model.cfg
    with open_dict(cfg):
        train = cfg.train_ds
        if A.mix:  # one manifest per source, each sampled by its weight (lhotse repeats each as needed)
            train.input_cfg = [
                {"type": "nemo", "manifest_filepath": m["manifest_filepath"], "weight": float(m["weight"])}
                for m in json.load(open(A.mix))
            ]
            train.manifest_filepath = None
        else:
            train.manifest_filepath = A.train
        train.is_tarred, train.tarred_audio_filepaths = False, None
        train.use_lhotse, train.batch_size = True, None
        train.batch_duration = BATCH_S if not DRY else 60
        train.use_bucketing, train.num_buckets = True, 20
        train.bucket_buffer_size = train.shuffle_buffer_size = 10000
        # lhotse's background bucket filler dies silently on any error (an unreadable clip) and training then
        # waits forever: fill buckets in the main loop so an error stops the run with its cause
        train.concurrent_bucketing = False
        train.max_duration, train.min_duration = A.max_dur, 0.8
        train.num_workers, train.pin_memory, train.shuffle = (A.workers if not DRY else 4), True, True
        # augmentation: noise and music beds, room responses, speed
        if A.noise:
            train.noise_path = A.noise
            train.noise_snr, train.noise_mix_prob = [5.0, 20.0], 0.4
        train.perturb_speed = True
        if A.rirs:
            train.rir_enabled, train.rir_path, train.rir_prob = True, A.rirs, 0.3
        val = cfg.validation_ds
        val.manifest_filepath = A.val.split(",")  # the first decides the checkpoints
        val.use_lhotse, val.batch_size, val.num_workers, val.is_tarred = True, 16, 4, False
        if "name" in val and val.name is None:  # NeMo's multi-set loader cannot take name: null;
            with open_dict(val):  # names then come from the files
                val.pop("name")
        cfg.optim.name = "adamw"
        cfg.optim.lr = A.lr
        cfg.optim.betas = [0.9, 0.98]
        cfg.optim.weight_decay = 1e-3
        cfg.optim.sched = {"name": "CosineAnnealing", "warmup_steps": min(A.warmup, max(10, STEPS // 10)),
                           "min_lr": A.min_lr}
    model.setup_training_data(cfg.train_ds)
    model.setup_multiple_validation_data(cfg.validation_ds)
    every = min(A.val_every, STEPS)
    trainer = pl.Trainer(
        devices=1, accelerator="gpu", precision="bf16-mixed", max_steps=STEPS,
        val_check_interval=every * ACCUMULATE, check_val_every_n_epoch=None,
        limit_val_batches=1.0 if not DRY else 2, log_every_n_steps=50 if not DRY else 1,
        gradient_clip_val=1.0, accumulate_grad_batches=ACCUMULATE, enable_checkpointing=False,
        logger=False, use_distributed_sampler=False, num_sanity_val_steps=0,
        **({"max_time": {"hours": A.max_hours}} if A.max_hours else {}))
    exp_manager(trainer, {
        "exp_dir": str(Path(A.exp_dir).resolve()), "name": NAME, "create_tensorboard_logger": True,
        "create_checkpoint_callback": True,
        # the first validation set (a cased, punctuated read-speech dev) decides
        "checkpoint_callback_params": {"monitor": "val_wer", "mode": "min", "save_top_k": 3, "always_save_nemo": True},
        "ema": {"enable": True, "decay": 0.999, "validate_original_weights": False},
        "resume_if_exists": True, "resume_ignore_no_checkpoint": True})
    model.set_trainer(trainer)
    model.setup_optimization(cfg.optim)
    # an unusually dense clip can run the joint out of memory: skip that batch (Lightning skips a step that
    # returns None) instead of losing the run since its last check
    step_of = model.training_step

    loss_of = getattr(model.joint, "loss", None)
    reduction = getattr(loss_of, "reduction", None)

    def training_step(batch, batch_idx=0):  # named batch_idx: Lightning passes it only under that name
        try:
            return step_of(batch, batch_idx)
        except torch.OutOfMemoryError:
            pass  # leave the except block first so the half-built graph is freed
        if loss_of is not None:  # the fused joint sets reduction None mid-batch and only restores it at the end
            loss_of.reduction = reduction
        torch.cuda.empty_cache()
        print(f"OOM: a batch at {trainer.global_step} did not fit; skipped", flush=True)
        return None

    model.training_step = training_step
    trainer.fit(model)


if __name__ == "__main__":
    main()
    os._exit(0)  # the loader workers must not keep the process (and the GPU) after the run
