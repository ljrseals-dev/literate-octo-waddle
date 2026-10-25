"""stereo_cat: a small, testable stereo-field algebra (numpy only).

T = P o R o C, applied per frequency band:
  C  cut     : scale the Mid (M) and Difference (D) components      (halving the side = diff 0.5, collapsing to mono = diff 0)
  R  rotate  : rotate the (M, D) plane by an angle                  (energy-preserving change of width / polarity)
  P  place   : balance the band between left and right              (constant-power law, 0 = unchanged)
Band split F: complementary raised-cosine crossovers in log-frequency; the weights sum to exactly 1, so an identity recipe
returns the input.

NOTE: this is a reconstruction from a one-line summary of the "Cutting the Stereo Cat" algebra (M/D/F and three operators).
Replace the definitions with the paper's wherever they differ; the tests then tell you what changed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ------------------------------------------------------------------ M/D representation
def to_md(x):
    """(n, 2) L/R -> (mid, difference). L = M + D, R = M - D."""
    return (x[:, 0] + x[:, 1]) / 2.0, (x[:, 0] - x[:, 1]) / 2.0


def from_md(m, d):
    return np.stack([m + d, m - d], axis=1)


# ------------------------------------------------------------------ band split (F)
def band_weights(freqs, edges, width_oct=0.5):
    """Weights w_b(f), shape (len(edges)+1, len(freqs)), with sum_b w_b(f) == 1 for every f.
    Each crossover is a raised-cosine ramp `width_oct` octaves wide, centred on the edge."""
    edges = [float(e) for e in edges]
    if any(e <= 0 for e in edges) or edges != sorted(edges):
        raise ValueError("edges must be positive and ascending")
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi / lo < 2.0 ** width_oct:
            raise ValueError("crossovers overlap: edges must be at least width_oct octaves apart")
    f = np.asarray(freqs, np.float64)
    safe = np.where(f > 0, f, 1e-12)
    steps = []
    for e in edges:
        t = np.clip(np.log2(safe / e) / width_oct + 0.5, 0.0, 1.0)
        steps.append((1.0 - np.cos(np.pi * t)) / 2.0)
    W = np.empty((len(edges) + 1, len(f)))
    prev = np.ones_like(f)
    for b, s in enumerate(steps):
        W[b] = prev - s
        prev = s
    W[len(edges)] = prev
    return W


# ------------------------------------------------------------------ operators
@dataclass(frozen=True)
class BandOp:
    mid: float = 1.0          # C: gain on M
    diff: float = 1.0         # C: gain on D  (0 = mono, 0.5 = half the width)
    rotate_deg: float = 0.0   # R: rotation of the (M, D) plane
    pan: float = 0.0          # P: -1 left ... +1 right

    def validate(self):
        if not (0.0 <= self.mid <= 4.0 and 0.0 <= self.diff <= 4.0):
            raise ValueError("mid and diff gains must be in [0, 4]")
        if not -180.0 <= self.rotate_deg <= 180.0:
            raise ValueError("rotate_deg must be in [-180, 180]")
        if not -1.0 <= self.pan <= 1.0:
            raise ValueError("pan must be in [-1, 1]")


def cut(m, d, gm, gd):
    return m * gm, d * gd


def rotate(m, d, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return c * m - s * d, s * m + c * d


def place(lr, pan):
    a = (pan + 1.0) * np.pi / 4.0                       # constant-power balance; pan 0 -> gains (1, 1)
    return lr * np.array([np.sqrt(2.0) * np.cos(a), np.sqrt(2.0) * np.sin(a)])


@dataclass(frozen=True)
class Recipe:
    name: str
    edges: tuple
    ops: tuple
    width_oct: float = 0.5

    def validate(self):
        if len(self.ops) != len(self.edges) + 1:
            raise ValueError("need exactly len(edges)+1 band operations")
        for op in self.ops:
            op.validate()
        if not 0.1 <= self.width_oct <= 2.0:
            raise ValueError("width_oct must be in [0.1, 2]")


def _spectrum(x, sr):
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 16))))
    return np.fft.rfft(x, nfft, axis=0), nfft, np.fft.rfftfreq(nfft, 1.0 / sr)


def split_bands(x, sr, edges, width_oct=0.5):
    """-> list of (n, 2) band signals whose sum equals x."""
    x = _check(x)
    X, nfft, freqs = _spectrum(x, sr)
    W = band_weights(freqs, edges, width_oct)
    return [np.fft.irfft(X * W[b][:, None], nfft, axis=0)[:len(x)] for b in range(len(W))]


def _check(x):
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2:
        raise ValueError("x must be a stereo array of shape (n, 2)")
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or Inf")
    return x


def apply(x, sr, recipe: Recipe):
    """T = P o R o C in every band, then sum the bands."""
    recipe.validate()
    bands = split_bands(x, sr, recipe.edges, recipe.width_oct)
    out = np.zeros_like(bands[0])
    for xb, op in zip(bands, recipe.ops):
        m, d = to_md(xb)
        m, d = cut(m, d, op.mid, op.diff)               # C
        m, d = rotate(m, d, op.rotate_deg)              # R
        out += place(from_md(m, d), op.pan)             # P
    return out


def band_report(x, sr, edges, width_oct=0.5):
    """Per band: level (dB re full scale RMS), width = rms(D)/rms(M), and L/R correlation. Lower width = closer to mono."""
    rows = []
    for b, xb in enumerate(split_bands(x, sr, edges, width_oct)):
        m, d = to_md(xb)
        rm, rd = np.sqrt(np.mean(m ** 2)) + 1e-30, np.sqrt(np.mean(d ** 2))
        l, r = xb[:, 0], xb[:, 1]
        den = np.sqrt(np.sum(l ** 2) * np.sum(r ** 2)) + 1e-30
        lo = 0 if b == 0 else edges[b - 1]
        hi = edges[b] if b < len(edges) else sr / 2
        rows.append({"band_hz": (lo, hi), "level_db": 20 * np.log10(np.sqrt(np.mean(xb ** 2)) + 1e-30),
                     "width": rd / rm, "correlation": float(np.sum(l * r) / den)})
    return rows


# ------------------------------------------------------------------ example recipes (NOT the paper's seven)
def example_recipes():
    O = BandOp
    return {
        "mono_bass": Recipe("mono_bass", (120.0,), (O(diff=0.0), O())),
        "halve_side": Recipe("halve_side", (), (O(diff=0.5),)),
        "widen_air": Recipe("widen_air", (4000.0,), (O(), O(diff=1.6))),
        "rotate_image": Recipe("rotate_image", (), (O(rotate_deg=25.0),)),
        "cat_split": Recipe("cat_split", (250.0, 2500.0), (O(diff=0.0), O(rotate_deg=15.0), O(diff=1.4, pan=0.2))),
        "hard_collapse": Recipe("hard_collapse", (), (O(diff=0.0),)),
        "tilt_place": Recipe("tilt_place", (200.0, 2000.0), (O(pan=-0.3), O(), O(pan=0.3))),
    }
