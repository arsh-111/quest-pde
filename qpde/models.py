"""Compact 1D models mapping [u0(x), x] -> u(x, t_k), k = 1..K (direct IC-to-trajectory)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """rfft -> keep the lowest `modes` frequencies -> complex channel mixing -> irfft."""

    def __init__(self, cin: int, cout: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w = nn.Parameter(scale * torch.randn(cin, cout, modes, 2))

    def forward(self, x):
        n = x.shape[-1]
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xf.shape[-1])
        w = torch.view_as_complex(self.w[:, :, :m])
        out = torch.zeros(x.shape[0], w.shape[1], xf.shape[-1], dtype=xf.dtype, device=x.device)
        out[..., :m] = torch.einsum("bim,iom->bom", xf[..., :m], w)
        return torch.fft.irfft(out, n=n, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, n_out: int = 100, width: int = 32, modes: int = 16, layers: int = 4,
                 pad_frac: float = 0.0):
        super().__init__()
        self.pad_frac = pad_frac
        self.lift = nn.Conv1d(2, width, 1)
        self.spec = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, n_out, 1))

    def forward(self, x):  # x: (B, 2, N)
        n = x.shape[-1]
        h = self.lift(x)
        p = int(round(self.pad_frac * n))
        if p > 0:
            # zero padding on both ends: the domain stops looking periodic to the FFT
            h = F.pad(h, (p, p))
        for i, (s, w) in enumerate(zip(self.spec, self.pw)):
            h = s(h) + w(h)
            if i < len(self.spec) - 1:
                h = F.gelu(h)
        if p > 0:
            h = h[..., p:p + n]
        return self.proj(h)


class CNN1d(nn.Module):
    """Small encoder-decoder (U-Net style) baseline with one skip per level."""

    def __init__(self, n_out: int = 100, width: int = 32):
        super().__init__()
        c1, c2, c3 = width, 2 * width, 4 * width
        def block(a, b):
            return nn.Sequential(nn.Conv1d(a, b, 5, padding=2), nn.GELU(), nn.Conv1d(b, b, 5, padding=2), nn.GELU())
        self.e1, self.e2, self.e3 = block(2, c1), block(c1, c2), block(c2, c3)
        self.u2, self.d2 = nn.ConvTranspose1d(c3, c2, 2, stride=2), block(2 * c2, c2)
        self.u1, self.d1 = nn.ConvTranspose1d(c2, c1, 2, stride=2), block(2 * c1, c1)
        self.out = nn.Conv1d(c1, n_out, 1)

    def forward(self, x):  # N must be divisible by 4
        h1 = self.e1(x)
        h2 = self.e2(F.max_pool1d(h1, 2))
        h3 = self.e3(F.max_pool1d(h2, 2))
        h = self.d2(torch.cat([self.u2(h3), h2], 1))
        h = self.d1(torch.cat([self.u1(h), h1], 1))
        return self.out(h)


def build(kind: str, **kw) -> nn.Module:
    if kind == "fno":
        return FNO1d(**kw)
    if kind == "cnn":
        return CNN1d(**kw)
    raise ValueError(kind)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
