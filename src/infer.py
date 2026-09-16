from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .config import load_config
from .decode import decode_maps
from .model import CircleNet, activate
from .normalize import normalize_pair
from .scenes import SceneReader
from pyproj import Transformer
from rasterio.transform import xy
import geopandas as gpd
from shapely import STRtree, points

def load_model(cfg, checkpoint: str, device) -> CircleNet:
    m = cfg.model
    model = CircleNet(m.encoder, pretrained=False, in_chans=2, decoder_channels=tuple(m.decoder_channels), head_channels=m.head_channels, groupnorm_groups=m.groupnorm_groups)
    ck = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ck["model"])

    return model.to(device).eval()

def tile_origins(size: int, tile: int, step: int) -> list[int]:
    if size <= tile:
        return [0]
    o = list(range(0, size - tile, step))
    o.append(size - tile) # last tile flush with the edge

    return o

@torch.no_grad()
def predict_scene(model, reader: SceneReader, cfg, device) -> np.ndarray:
    H, W = reader.shape
    s = cfg.targets.stride
    T, S = cfg.inference.tile_size, cfg.inference.tile_step
    acc = np.zeros((5, (H + s - 1) // s, (W + s - 1) // s), np.float32)
    cnt = np.zeros(acc.shape[1:], np.float32)

    t = T // s
    ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).clip(max=64) / 64.0
    taper = np.outer(ramp, ramp).astype(np.float32)

    for r0 in tile_origins(H, T, S):
        for c0 in tile_origins(W, T, S):
            vv, vh = reader.read_window(r0, c0, T, T)

            if not np.isfinite(vv).any():
                continue # missing data
            x = torch.from_numpy(normalize_pair(vv, vh, cfg.normalization))[None].to(device)

            with torch.autocast(device_type=device.type, enabled=cfg.train.amp and device.type == "cuda"):
                out = activate(model(x).float())
                if cfg.inference.flip_tta:
                    f = activate(model(torch.flip(x, dims=[3])).float())
                    f = torch.flip(f, dims=[3])
                    f[:, 1] = 1.0 - f[:, 1] # horizontal offset changes sign under a horizontal flip
                    out = 0.5 * (out + f)

            o = out[0].cpu().numpy()
            rr, cc = r0 // s, c0 // s
            hh, ww = min(t, acc.shape[1] - rr), min(t, acc.shape[2] - cc)
            acc[:, rr:rr + hh, cc:cc + ww] += o[:, :hh, :ww] * taper[:hh, :ww]
            cnt[rr:rr + hh, cc:cc + ww] += taper[:hh, :ww]

    cnt[cnt == 0] = 1.0
    return acc / cnt

def pixels_to_lonlat(reader: SceneReader, cols: np.ndarray, rows: np.ndarray):
    xs, ys = xy(reader.transform, rows, cols, offset="center")
    xs, ys = np.asarray(xs), np.asarray(ys)

    if reader.crs is not None and not reader.crs.is_geographic:
        tr = Transformer.from_crs(reader.crs, "EPSG:4326", always_xy=True)
        xs, ys = tr.transform(xs, ys)

    return xs, ys

def drop_on_land(df: pd.DataFrame, land_path: str, buffer_m: float = 300.0) -> pd.DataFrame:
    land = gpd.read_file(land_path).to_crs("EPSG:3035")
    geoms = land.geometry.buffer(buffer_m).values
    tree = STRtree(geoms)
    pts = gpd.GeoSeries(points(df["lon"].to_numpy(), df["lat"].to_numpy()), crs="EPSG:4326").to_crs("EPSG:3035")
    hit = tree.query(pts.values, predicate="intersects")
    on_land = np.zeros(len(df), bool)
    on_land[np.unique(hit[0])] = True

    return df[~on_land].reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vv", required=True)
    ap.add_argument("--vh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--land", default=None)
    ap.add_argument("--scene-id", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, a.checkpoint, device)
    reader = SceneReader(a.vv, a.vh, cfg.data.nodata_value)

    maps = predict_scene(model, reader, cfg, device)
    det = decode_maps(maps, cfg.targets.stride, cfg.decode.peak_kernel, cfg.decode.objectness_threshold, cfg.targets.log_length, cfg.decode.max_detections)
    det["length_m"] = det["length_px"] * cfg.data.pixel_size_m
    det["is_vessel"] = det["vessel_p"] >= cfg.decode.vessel_threshold
    lon, lat = pixels_to_lonlat(reader, det["col"].to_numpy(), det["row"].to_numpy())
    det["lon"], det["lat"] = lon, lat
    det["scene_id"] = a.scene_id or Path(a.vv).parent.name

    if a.land:
        det = drop_on_land(det, a.land)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(out.with_suffix(".csv"), index=False)

    gpd.GeoDataFrame(det, geometry=gpd.points_from_xy(det.lon, det.lat), crs="EPSG:4326").to_file(out, driver="GeoJSON")
    print(f"{len(det)} detections -> {out}")

if __name__ == "__main__":
    main()