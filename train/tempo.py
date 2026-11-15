"""Faster (or slower) speech with its pitch kept: WSOLA, waveform-similarity overlap-add (Verhelst and Roelands
1993), the family VLC's own scaletempo (its 2x playback) and sox's tempo belong to; numpy only.

The sound is cut into 32 ms Hann-windowed frames laid 16 ms apart in the output (they sum to one), each taken
from the input at `factor` times that pace; within +-12 ms of its place (one pitch period of a low voice), a
frame starts where it best continues the frame before it (normalised cross-correlation with what would have
followed that frame), so voiced sound joins in phase and no pitch is changed. factor 1.5: 1.5 times as fast,
2/3 as long. moon_train.py --tempo (training clips) and its pitch shift read through this function.
"""
import math

import numpy as np

FRAME, TOL = 512, 192  # 16 kHz: 32 ms frames (hop 16 ms), 12 ms search
_WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)).astype(np.float32)  # periodic Hann: sums to 1 at 50 %


def wsola(x, factor):
    """`x` (mono float, 16 kHz) played `factor` times as fast, pitch kept: round(len(x) / factor) samples."""
    x = np.asarray(x, dtype=np.float32)
    n_out = int(round(len(x) / factor))
    if factor == 1.0 or len(x) < FRAME or n_out < 1:
        return x if factor == 1.0 else _resampled(x, n_out)
    hop = FRAME // 2
    left = hop + TOL  # input index p sits at xp[left + p]
    frames = int(math.ceil(n_out / hop)) + 1  # frame k is centred on output sample k * hop
    right = int(math.ceil(frames * hop * factor)) - len(x) + 2 * hop + 2 * TOL + 1
    xp = np.concatenate([np.zeros(left, np.float32), x, np.zeros(max(right, 0), np.float32)])
    y = np.zeros((frames + 1) * hop, np.float32)  # y[hop + i] = output sample i (frame 0 starts at -hop)
    width = 2 * hop
    prev = 0
    for k in range(frames):
        nominal = int(round(k * hop * factor))
        if k == 0:
            centre = 0
        else:
            target = xp[left + prev: left + prev + width]  # what would have followed the previous frame
            lo = left + nominal - TOL - hop
            region = xp[lo: lo + width + 2 * TOL]
            corr = np.correlate(region, target, mode="valid")  # 2 * TOL + 1 candidate starts
            power = np.cumsum(np.concatenate([[0.0], region.astype(np.float64) ** 2]))
            energy = power[width:] - power[:-width]
            centre = nominal - TOL + int(np.argmax(corr / np.sqrt(energy + 1e-9)))
        y[k * hop: k * hop + width] += _WIN * xp[left + centre - hop: left + centre + hop]
        prev = centre
    return y[hop: hop + n_out]


def _resampled(x, n):
    """A clip too short for a frame: plain resampling (pitch moves, a few ms of sound)."""
    if n < 1:
        return x[:0]
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
