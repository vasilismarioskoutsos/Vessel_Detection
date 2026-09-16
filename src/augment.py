from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import affine_transform

@dataclass
class AugParams:
    hflip: float = 0.5
    vflip: float = 0.5
    affine_prob: float = 0.5
    max_rotate_deg: float = 5.0
    scale_range: tuple[float, float] = (0.9, 1.1)
    brightness_db: float = 1.5
    contrast_range: tuple[float, float] = (0.9, 1.1)
    noise_db_std: float = 0.5

def photometric_db(vv_db: np.ndarray, vh_db: np.ndarray, p: AugParams, rng: np.random.Generator):
    """Brightness, contrast, noise in dB, NaNs stay Nan, no clipping"""
    shift = rng.uniform(-p.brightness_db, p.brightness_db)
    gain = rng.uniform(*p.contrast_range)
    out = []
    for x in (vv_db, vh_db):
        ok = np.isfinite(x)
        y = x.copy()

        if ok.any():
            mu = np.nanmean(x)
            y[ok] = (x[ok] - mu) * gain + mu + shift
            if p.noise_db_std > 0:
                y[ok] += rng.normal(0.0, p.noise_db_std, ok.sum()).astype(np.float32)

        out.append(y)
    return out[0], out[1]

def geometric(img: np.ndarray, col: np.ndarray, row: np.ndarray, length_px: np.ndarray, p: AugParams, rng: np.random.Generator, fill_value: float = 0.0):
    """flips and a small similarity transform about the patch center"""

    C, H, W = img.shape
    col, row, length_px = col.copy(), row.copy(), length_px.copy()

    if rng.random() < p.hflip:
        img = img[:, :, ::-1]
        col = (W - 1) - col

    if rng.random() < p.vflip:
        img = img[:, ::-1, :]
        row = (H - 1) - row

    if rng.random() < p.affine_prob:
        theta = np.deg2rad(rng.uniform(-p.max_rotate_deg, p.max_rotate_deg))
        s = rng.uniform(*p.scale_range)
        c, si = np.cos(theta), np.sin(theta)

        R = np.array([[c, -si], [si, c]]) * s
        center = np.array([(W - 1) / 2.0, (H - 1) / 2.0])

        Rinv = np.linalg.inv(R)
        M_rc = Rinv[::-1, ::-1]
        center_rc = center[::-1]
        offset = center_rc - M_rc @ center_rc
        img = np.stack([affine_transform(ch, M_rc, offset=offset, order=1, mode="constant", cval=fill_value) for ch in img], 0)
        pts = np.stack([col, row], 1) - center
        pts = pts @ R.T + center
        col, row = pts[:, 0], pts[:, 1]
        length_px = length_px * s

    return np.ascontiguousarray(img, dtype=np.float32), col.astype(np.float32), row.astype(np.float32), length_px.astype(np.float32)