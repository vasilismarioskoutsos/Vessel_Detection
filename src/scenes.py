from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .normalize import raw_to_db
import rasterio
from rasterio.windows import Window

CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

@dataclass
class SceneObjects:

    col: np.ndarray
    row: np.ndarray
    length_px: np.ndarray
    is_vessel: np.ndarray
    near_shore: np.ndarray

    def __len__(self) -> int:
        return len(self.col)

    def subset(self, idx) -> "SceneObjects":
        return SceneObjects(self.col[idx], self.row[idx], self.length_px[idx], self.is_vessel[idx], self.near_shore[idx])

def load_objects_table(objects_file: str | Path) -> pd.DataFrame:
    p = Path(objects_file)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

def objects_for_scene(df: pd.DataFrame, scene_id: str, pixel_size_m: float, min_confidence: str = "LOW", near_shore_km: float = 2.0) -> SceneObjects:
    d = df[df["scene_id"] == scene_id]
    if "confidence" in d.columns:
        keep = d["confidence"].map(CONFIDENCE_RANK).fillna(0) >= CONFIDENCE_RANK[min_confidence]
        d = d[keep]

    col = d["detect_scene_column"].to_numpy(dtype=np.float32)
    row = d["detect_scene_row"].to_numpy(dtype=np.float32)
    length_px = (d["vessel_length_m"].to_numpy(dtype=np.float32) / pixel_size_m if "vessel_length_m" in d.columns else np.full(len(d), np.nan, np.float32))
    is_vessel = _to_float_bool(d["is_vessel"]) if "is_vessel" in d.columns else np.full(len(d), np.nan, np.float32)

    if "distance_from_shore_km" in d.columns:
        near = d["distance_from_shore_km"].to_numpy(dtype=np.float32) <= near_shore_km
    else:
        near = np.zeros(len(d), bool)
    return SceneObjects(col, row, length_px, is_vessel, near)

def _to_float_bool(s: pd.Series) -> np.ndarray:
    out = np.full(len(s), np.nan, np.float32)
    v = s.to_numpy()

    for i, x in enumerate(v):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl in ("true", "1", "yes"):
                out[i] = 1.0
            elif xl in ("false", "0", "no"):
                out[i] = 0.0
        else:
            out[i] = 1.0 if bool(x) else 0.0
    return out

class SceneReader:
    def __init__(self, vv_path: str | Path, vh_path: str | Path, nodata_value: float | None):
        self.vv_path, self.vh_path = str(vv_path), str(vh_path)
        self.nodata_value = nodata_value
        self._vv = self._vh = None

    def _open(self):
        if self._vv is None:
            self._vv = rasterio.open(self.vv_path)
            self._vh = rasterio.open(self.vh_path)
            assert self._vv.shape == self._vh.shape, "VV and VH rasters must have identical shape"

    @property
    def shape(self) -> tuple[int, int]:
        self._open()
        return self._vv.shape

    @property
    def transform(self):
        self._open()
        return self._vv.transform

    @property
    def crs(self):
        self._open()
        return self._vv.crs

    def read_window(self, row0: int, col0: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        self._open()
        H, W = self.shape
        r0, c0 = max(row0, 0), max(col0, 0)
        r1, c1 = min(row0 + h, H), min(col0 + w, W)
        vv = np.full((h, w), np.nan, np.float32)
        vh = np.full((h, w), np.nan, np.float32)

        if r1 > r0 and c1 > c0:
            win = Window(c0, r0, c1 - c0, r1 - r0)
            a = raw_to_db(self._vv.read(1, window=win), self.nodata_value)
            b = raw_to_db(self._vh.read(1, window=win), self.nodata_value)
            vv[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = a
            vh[r0 - row0:r1 - row0, c0 - col0:c1 - col0] = b

        return vv, vh

    def close(self):
        for h in (self._vv, self._vh):
            if h is not None:
                h.close()

        self._vv = self._vh = None