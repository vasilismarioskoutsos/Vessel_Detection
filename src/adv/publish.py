from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import geopandas as gpd
from shapely.geometry import box

IDENTITY_COLUMNS = ("mmsi", "MMSI", "imo", "IMO", "name", "vessel_name", "shipname", "callsign", "call_sign", "destination")

def sea_relative_stretch(db: np.ndarray, below: float = 4.0, above: float = 20.0) -> np.ndarray:
    finite = np.isfinite(db)
    if not finite.any():
        return np.zeros(db.shape, np.uint8)
    
    med = float(np.median(db[finite]))
    lo, hi = med - below, med + above
    out = np.clip((db - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out[~finite] = 0.0
    
    return (out * 255.0).astype(np.uint8)

def percentile_stretch(db: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:

    finite = np.isfinite(db)
    if not finite.any():
        return np.zeros(db.shape, np.uint8)

    lo, hi = np.percentile(db[finite], [lo_pct, hi_pct])

    if hi <= lo:
        hi = lo + 1.0

    out = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return (out * 255.0).astype(np.uint8)

def save_png(u8: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8, mode="L").save(p, optimize=True)

def extract_chip(reader, col: float, row: float, window_m: float, pixel_size_m: float):
    n = max(8, int(round(window_m / pixel_size_m)))
    r0 = int(round(row)) - n // 2
    c0 = int(round(col)) - n // 2

    return reader.read_window(r0, c0, n, n)

def downsample_db(db: np.ndarray, max_px: int) -> np.ndarray:
    h, w = db.shape
    step = max(1, int(np.ceil(max(h, w) / max_px)))
    
    if step == 1:
        return db
    
    h2, w2 = (h // step) * step, (w // step) * step
    blocks = db[:h2, :w2].reshape(h2 // step, step, w2 // step, step)
    
    with np.errstate(invalid="ignore"):
        return np.nanmean(blocks, axis=(1, 3))

class HeldBack(Exception):
    """when a pass is still inside the delay window"""

def strip_identities(df):
    cols = [c for c in df.columns if c in IDENTITY_COLUMNS]
    return df.drop(columns=cols) if cols else df

def load_exclusion_masks(path: str | Path | None):    
    p = Path(path)
    g = gpd.read_file(p)

    return g.to_crs("EPSG:4326") if g.crs is not None else g.set_crs("EPSG:4326")


def drop_inside_masks(gdf, masks):
    if masks is None or len(masks) == 0 or len(gdf) == 0:
        return gdf, 0
    
    hit = gpd.sjoin(gdf[["geometry"]], masks[["geometry"]], how="inner", predicate="within")

    if len(hit) == 0:
        return gdf, 0
    bad = set(hit.index)
    
    return gdf[~gdf.index.isin(bad)].copy(), len(bad)

def aggregate_to_cells(gdf, cell_km: float, bbox):

    if len(gdf) == 0:
        return gpd.GeoDataFrame({"n": []}, geometry=[], crs="EPSG:4326")
    
    lon0, lat0, lon1, lat1 = bbox
    dlat = cell_km / 111.0
    dlon = cell_km / (111.0 * np.cos(np.deg2rad((lat0 + lat1) / 2)))
    lon = gdf.geometry.x.to_numpy()
    lat = gdf.geometry.y.to_numpy()
    ix = np.floor((lon - lon0) / dlon).astype(int)
    iy = np.floor((lat - lat0) / dlat).astype(int)
    counts: dict[tuple[int, int], int] = {}
    
    for a, b in zip(ix, iy):
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    geoms, vals = [], []
    
    for (a, b), n in counts.items():
        x0, y0 = lon0 + a * dlon, lat0 + b * dlat
        geoms.append(box(x0, y0, x0 + dlon, y0 + dlat))
        vals.append(n)
    
    return gpd.GeoDataFrame({"n": vals}, geometry=geoms, crs="EPSG:4326")

def apply_publication_rules(gdf, acquired: datetime, cfg, now: datetime | None = None, masks=None, verbose: bool = True):
    now = now or datetime.now(timezone.utc)
    if acquired.tzinfo is None:
        acquired = acquired.replace(tzinfo=timezone.utc)
    
    age = now - acquired
    delay = timedelta(hours=float(cfg["delay_hours"]))
    if age < delay:
        raise HeldBack(
            f"acquired {acquired:%Y-%m-%d %H:%M}Z is {age.total_seconds() / 3600:.1f} h old; "
            f"the delay is {cfg['delay_hours']} h. Publish it after "
            f"{(acquired + delay):%Y-%m-%d %H:%M}Z."
        )

    n_in = len(gdf)
    gdf, n_masked = drop_inside_masks(gdf, masks)

    floor = float(cfg["size_floor_m"])
    if "len_m" in gdf.columns:
        len_m = gdf["len_m"].fillna(0.0)
    else:
        len_m = np.zeros(len(gdf))

    is_structure = gdf["cls"].eq("structure") if "cls" in gdf.columns else np.zeros(len(gdf), bool)
    keep = (len_m >= floor) | is_structure
    individual = strip_identities(gdf[keep].copy())
    aggregated = aggregate_to_cells(gdf[~keep], float(cfg["aggregate_cell_km"]), tuple(cfg["bbox"]))

    report = {
        "in": n_in, "masked_out": n_masked,
        "published": len(individual), "aggregated_points": int((~keep).sum()),
        "aggregated_cells": len(aggregated),
        "age_hours": round(age.total_seconds() / 3600, 1),
    }
    
    return individual, aggregated, report