from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

def decode_maps(pred: np.ndarray, stride: int = 2, peak_kernel: int = 3, objectness_threshold: float = 0.3, log_length: bool = True, max_detections: int = 20000) -> pd.DataFrame:

    heat = pred[0]
    local_max = maximum_filter(heat, size=peak_kernel, mode="nearest")
    peaks = (heat >= local_max) & (heat > objectness_threshold)
    ys, xs = np.nonzero(peaks)

    if len(ys) == 0:
        return _empty()

    score = heat[ys, xs]
    if len(score) > max_detections:
        keep = np.argsort(-score)[:max_detections]
        ys, xs, score = ys[keep], xs[keep], score[keep]

    col = (xs + pred[1, ys, xs]) * stride
    row = (ys + pred[2, ys, xs]) * stride
    length_raw = pred[3, ys, xs]
    length_px = np.expm1(length_raw) if log_length else length_raw
    length_px = np.clip(length_px, 0, None)
    vessel_p = pred[4, ys, xs]

    return pd.DataFrame(dict(col=col.astype(np.float32), row=row.astype(np.float32), score=score.astype(np.float32), length_px=length_px.astype(np.float32), vessel_p=vessel_p.astype(np.float32)))

def _empty() -> pd.DataFrame:
    return pd.DataFrame({k: np.zeros(0, np.float32) for k in ["col", "row", "score", "length_px", "vessel_p"]})