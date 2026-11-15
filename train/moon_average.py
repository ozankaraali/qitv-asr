"""The mean of a run's last check snapshots (moon_train.py --snapshots N), as a model folder like `last`: averaged
weights are usually better and steadier than any one check. The released Moonshine models are the last state of
their cool-down; the average of the last three snapshots was made and scored beside it.
usage: python moon_average.py <run dir> <out dir> [how many, default all]"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

run, out = Path(sys.argv[1]), Path(sys.argv[2])
snaps = sorted(run.glob("snap-*"))
if len(sys.argv) > 3:
    snaps = snaps[-int(sys.argv[3]):]
if not snaps:
    sys.exit(f"no snapshots in {run}")
total = None
for s in snaps:
    w = load_file(str(s / "model.safetensors"))
    if total is None:
        total = {k: v.float().clone() for k, v in w.items()}
    else:
        for k, v in w.items():
            total[k] += v.float()
dtypes = {k: v.dtype for k, v in load_file(str(snaps[-1] / "model.safetensors")).items()}
avg = {k: (v / len(snaps)).to(dtypes[k]) for k, v in total.items()}
if out.exists():
    shutil.rmtree(out)
shutil.copytree(run / "last", out)  # config, processor, tokenizer and tr.model as `last`
save_file(avg, str(out / "model.safetensors"), metadata={"format": "pt"})
print(f"averaged {len(snaps)} snapshots ({snaps[0].name} .. {snaps[-1].name}) into {out}")
