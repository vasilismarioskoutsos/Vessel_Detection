from __future__ import annotations
import numpy as np
from scipy.special import expit

def db_to_unit(x_db: np.ndarray, center: float, scale: float, fill_value: float = 0.0) -> np.ndarray:
    out = expit((x_db - center) / scale).astype(np.float32)
    out[~np.isfinite(x_db)] = fill_value
    return out

def normalize_pair(vv_db: np.ndarray, vh_db: np.ndarray, norm_cfg) -> np.ndarray:
    vv = db_to_unit(vv_db, norm_cfg.vv.center, norm_cfg.vv.scale, norm_cfg.fill_value)
    vh = db_to_unit(vh_db, norm_cfg.vh.center, norm_cfg.vh.scale, norm_cfg.fill_value)
    return np.stack([vv, vh], axis=0)

def raw_to_db(arr: np.ndarray, nodata_value: float | None) -> np.ndarray:
    x = arr.astype(np.float32, copy=True)
    if nodata_value is not None:
        x[arr == nodata_value] = np.nan
    x[~np.isfinite(x)] = np.nan
    
    return x
