from __future__ import annotations
import numpy as np
from .scenes import SceneObjects

def _disc_weights(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(x * x + y * y)
    w = np.clip(1.0 - d / (r + 1.0), 0.0, 1.0)
    return w.astype(np.float32)

def _gaussian_peak(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    d2 = x * x + y * y
    sigma = r / 2.0
    g = np.exp(-d2 / (2 * sigma * sigma))
    g[d2 > r * r] = 0.0
    return g.astype(np.float32)

def _paste_max(canvas: np.ndarray, patch: np.ndarray, cy: int, cx: int) -> None:
    r = patch.shape[0] // 2
    H, W = canvas.shape
    y0, y1 = max(cy - r, 0), min(cy + r + 1, H)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, W)

    if y1 <= y0 or x1 <= x0:
        return

    py0, px0 = y0 - (cy - r), x0 - (cx - r)
    sub = patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)

def _paste_weighted_value(value_map, weight_map, patch_w, value, cy, cx) -> None:
    r = patch_w.shape[0] // 2
    H, W = weight_map.shape
    y0, y1 = max(cy - r, 0), min(cy + r + 1, H)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, W)

    if y1 <= y0 or x1 <= x0:
        return

    py0, px0 = y0 - (cy - r), x0 - (cx - r)
    sub = patch_w[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)]
    region_w = weight_map[y0:y1, x0:x1]
    take = sub > region_w
    value_map[y0:y1, x0:x1][take] = value
    region_w[take] = sub[take]

def build_targets(objs: SceneObjects, patch_h: int, patch_w: int, stride: int = 2, heat_radius_px: int = 3, cls_radius_px: int = 2, log_length: bool = True) -> dict:
    Ho, Wo = patch_h // stride, patch_w // stride
    heat = np.zeros((Ho, Wo), np.float32)
    offset = np.zeros((2, Ho, Wo), np.float32)
    offset_mask = np.zeros((Ho, Wo), np.float32)
    length = np.zeros((Ho, Wo), np.float32)
    length_mask = np.zeros((Ho, Wo), np.float32)
    vessel = np.zeros((Ho, Wo), np.float32)
    vessel_weight = np.zeros((Ho, Wo), np.float32)
    vessel_unknown = np.zeros((Ho, Wo), np.float32)

    r_heat = max(1, int(round(heat_radius_px / stride)))
    r_cls = max(1, int(round(cls_radius_px / stride)))
    peak = _gaussian_peak(r_heat)
    disc = _disc_weights(r_cls)

    for i in range(len(objs)):
        cx_f, cy_f = objs.col[i] / stride, objs.row[i] / stride
        if not (0 <= cx_f < Wo and 0 <= cy_f < Ho):
            continue
        cx, cy = int(np.floor(cx_f)), int(np.floor(cy_f))

        _paste_max(heat, peak, cy, cx)
        heat[cy, cx] = 1.0 

        offset[0, cy, cx] = cx_f - cx
        offset[1, cy, cx] = cy_f - cy
        offset_mask[cy, cx] = 1.0

        L = objs.length_px[i]
        if np.isfinite(L) and L > 0:
            val = np.log1p(L) if log_length else L
            _paste_weighted_value(length, length_mask, disc, val, cy, cx)

        v = objs.is_vessel[i]
        if np.isfinite(v):
            _paste_weighted_value(vessel, vessel_weight, disc, float(v), cy, cx)
        else:
            _paste_max(vessel_unknown, disc, cy, cx)

    vessel_unknown[vessel_weight > 0] = 0.0

    return dict(heat=heat[None], offset=offset, offset_mask=offset_mask[None],
                length=length[None], length_mask=length_mask[None],
                vessel=vessel[None], vessel_weight=vessel_weight[None],
                vessel_unknown=vessel_unknown[None])

def stack_prediction_like_target(t: dict) -> np.ndarray:

    return np.concatenate([t["heat"], t["offset"], t["length"], t["vessel"]], axis=0)
