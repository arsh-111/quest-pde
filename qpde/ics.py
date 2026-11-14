"""Initial-condition generators (used for the OOD test set and by the app's IC controls)."""
from __future__ import annotations

import numpy as np


def grid(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def constant(n, level=0.1, **_):
    return np.full(n, float(level))


def smooth(n, seed=0, base=0.1, amplitude=0.05, corr_len=0.1, **_):
    """base + amplitude * smooth Gaussian random field (squared-exponential spectrum), max-normalised."""
    rng = np.random.default_rng(seed)
    x = grid(n)
    k = np.fft.rfftfreq(n, d=1.0 / n)
    spec = np.exp(-0.5 * (2 * np.pi * k * corr_len) ** 2)
    coef = (rng.standard_normal(k.size) + 1j * rng.standard_normal(k.size)) * np.sqrt(spec)
    coef[0] = 0
    f = np.fft.irfft(coef, n=n)
    f = f / (np.abs(f).max() + 1e-12)
    return np.clip(base + amplitude * f, 0.0, None) + 0 * x


def noise(n, seed=0, base=0.1, noise_level=0.05, **_):
    rng = np.random.default_rng(seed)
    return np.clip(base + noise_level * rng.uniform(-1, 1, n), 0.0, None)


def pulse(n, base=0.05, position=0.5, width=0.05, height=0.5, **_):
    x = grid(n)
    return base + height * np.exp(-0.5 * ((x - position) / width) ** 2)


def step(n, base=0.05, position=0.5, height=0.3, **_):
    x = grid(n)
    return np.where(x < position, base + height, base)


GENERATORS = {"constant": constant, "smooth": smooth, "noise": noise, "pulse": pulse, "step": step}


def make(kind: str, n: int, **kw) -> np.ndarray:
    return GENERATORS[kind](n, **kw).astype(np.float64)
