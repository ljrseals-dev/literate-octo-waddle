"""endless_beats: a seeded, numpy-only beat grammar and drum renderer (the 'Endless Engine' idea in miniature).
Euclidean rhythms, swing, humanize, half-time and density-halving, and three synthesized drums with stereo width."""
from __future__ import annotations
import numpy as np


def euclid(k: int, n: int):
    """Euclidean rhythm: k onsets spread as evenly as possible over n steps (equal to Bjorklund's result up to rotation)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n and n > 0")
    return [1 if (i * k) % n < k else 0 for i in range(n)]


def rotate_pattern(p, r):
    r %= len(p)
    return p[-r:] + p[:-r] if r else list(p)


def half_time(p, factor=2):
    """Play the same pattern `factor` times slower: every step is followed by factor-1 rests (the pattern now spans factor bars)."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    out = []
    for v in p:
        out.append(v)
        out.extend([0] * (factor - 1))
    return out


def halve_density(p):
    """Keep every other onset (1st, 3rd, ...), drop the rest. Length unchanged."""
    out, seen = [], 0
    for v in p:
        if v:
            out.append(1 if seen % 2 == 0 else 0)
            seen += 1
        else:
            out.append(0)
    return out


def _kick(sr):
    t = np.arange(int(0.5 * sr)) / sr
    f = 45 + (150 - 45) * np.exp(-t / 0.04)
    y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.18)
    return y, y


def _snare(sr, rng):
    t = np.arange(int(0.35 * sr)) / sr
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06) * 0.5
    nl, nr = rng.standard_normal(len(t)), rng.standard_normal(len(t))
    mix = 0.35                                          # partly shared noise so the snare has a centre and some width
    shared = rng.standard_normal(len(t))
    l = tone + 0.5 * (mix * shared + (1 - mix) * nl) * np.exp(-t / 0.09)
    r = tone + 0.5 * (mix * shared + (1 - mix) * nr) * np.exp(-t / 0.09)
    return l, r


def _hat(sr, rng):
    t = np.arange(int(0.12 * sr)) / sr
    env = np.exp(-t / 0.03)
    l, r = np.diff(rng.standard_normal(len(t) + 1)) * env * 0.35, np.diff(rng.standard_normal(len(t) + 1)) * env * 0.35   # decorrelated = wide
    return l, r


def render(patterns: dict, bpm=92.0, bars=4, steps_per_bar=16, swing=0.0, humanize_ms=0.0, sr=44100, seed=0, gains=None):
    """patterns: {'kick': [0/1 per step], 'snare': [...], 'hat': [...]}, each of length steps_per_bar (repeats each bar) or bars*steps_per_bar.
    swing: fraction of a step that off-beat steps (odd indices) are delayed by (0..0.5). Returns (n, 2) float32 in [-1, 1]."""
    if not 20 <= bpm <= 300 or not 0 <= swing <= 0.5 or bars < 1:
        raise ValueError("bpm 20-300, swing 0-0.5, bars >= 1")
    rng = np.random.default_rng(seed)
    step = 60.0 / bpm * 4.0 / steps_per_bar
    total = int((bars * steps_per_bar * step + 1.0) * sr)
    out = np.zeros((total, 2))
    gains = gains or {"kick": 0.9, "snare": 0.7, "hat": 0.5}
    voices = {"kick": lambda: _kick(sr), "snare": lambda: _snare(sr, rng), "hat": lambda: _hat(sr, rng)}
    for name, pat in patterns.items():
        if name not in voices:
            raise ValueError(f"unknown voice {name!r}")
        full = pat if len(pat) == bars * steps_per_bar else (list(pat) * bars if len(pat) == steps_per_bar else None)
        if full is None:
            raise ValueError(f"{name}: pattern length must be {steps_per_bar} or {bars * steps_per_bar}")
        for i, hit in enumerate(full):
            if not hit:
                continue
            t0 = i * step + (swing * step if i % 2 else 0.0) + rng.normal(0, humanize_ms / 1000.0)
            s0 = max(0, int(t0 * sr))
            l, r = voices[name]()
            m = min(len(l), total - s0)
            if m > 0:
                out[s0:s0 + m, 0] += l[:m] * gains[name]
                out[s0:s0 + m, 1] += r[:m] * gains[name]
    pk = np.max(np.abs(out))
    return (out / pk * 0.89 if pk > 0 else out).astype(np.float32)


def default_patterns():
    return {"kick": euclid(5, 16), "snare": [1 if i in (4, 12) else 0 for i in range(16)], "hat": euclid(11, 16)}
