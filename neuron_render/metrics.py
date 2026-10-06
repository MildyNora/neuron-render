"""Image-similarity metrics on display-referred float images in [0, 1]."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def ssim(a, b, sigma=1.5, size=11):
    """Mean SSIM (Wang et al.), Gaussian window, averaged over channels."""
    x = torch.from_numpy(np.ascontiguousarray(a, np.float32)).permute(2, 0, 1)[None]
    y = torch.from_numpy(np.ascontiguousarray(b, np.float32)).permute(2, 0, 1)[None]
    g = torch.exp(-((torch.arange(size) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)
    blur = lambda t: F.conv2d(t, w, groups=3)
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(s.mean())
