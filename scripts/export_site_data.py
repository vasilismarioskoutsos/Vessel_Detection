from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adv.config import load_confi
from adv.publish import (HeldBack, apply_publication_rules, downsample_db, extract_chip,load_exclusion_masks, percentile_stretch, save_png, sea_relative_stretch)
from adv.scenes import SceneReader

SAT = {"S1A": "Sentinel-1A", "S1B": "Sentinel-1B", "S1C": "Sentinel-1C", "S1D": "Sentinel-1D"}

def parse_product_name(name: str):
    """S1C_IW_GRDH_1SDV_20260918T064211_.. to (datetime, 'Sentinel-1C')"""
    m = re.search(r"(S1[ABCD])_\w+_\w+_\w+_(\d{8}T\d{6})", str(name))
    if not m:
        return None, None
    
    return (datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc), SAT.get(m.group(1), m.group(1)))

def acquisition_for(scene_id: str, mapping: pd.DataFrame | None, override: str | None):
    if override:
        return datetime.fromisoformat(override.replace("Z", "+00:00")), "unknown"
    
    if mapping is not None:
        row = mapping[mapping.iloc[:, 0].astype(str) == scene_id]
        if not row.empty:
            for val in row.iloc[0].tolist()[1:]:
                t, sat = parse_product_name(val)
                if t:
                    return t, sat
                
    raise SystemExit(f"no acquisition time for scene {scene_id}")

def scene_footprint_and_overview(vv_path, vh_path, nodata, want_overview, max_px, pct):
    with rasterio.open(vv_path) as src:
        b = src.bounds
        wgs = transform_bounds(src.crs, "EPSG:4326", *b, densify_pts=21) if src.crs else tuple(b)

    footprint = box(*wgs)
    overview = None

    if want_overview:
        reader = SceneReader(vv_path, vh_path, nodata)
        h, w = reader.shape
        vv, _ = reader.read_window(0, 0, h, w)
        reader.close()
        small = downsample_db(vv, max_px)
        overview = (percentile_stretch(small, pct[0], pct[1]), wgs)

    return footprint, overview


def load_detections(path: str, no_ais: bool) -> pd.DataFrame:
    d = pd.read_csv(path)

    if "cls" not in d.columns:
        if no_ais or "matched" not in d.columns:
            d["cls"] = "detection"
        else:
            d["cls"] = np.where(d["matched"].astype(bool), "matched", "unmatched")

    if "len_m" not in d.columns:
        d["len_m"] = d["length_m"] if "length_m" in d.columns else np.nan
    if "conf" not in d.columns:
        d["conf"] = d["score"] if "score" in d.columns else np.nan
    if "id" not in d.columns:
        d["id"] = [f"det_{i:06d}" for i in range(len(d))]

    return d

def reasons_for(row, no_ais: bool) -> list[dict]:
    if no_ais:
        return [{"k": "info", "t": "AIS comparison is not running yet — this is a radar detection only"}]
    out = []
    if row.get("cls") == "unmatched":
        d = row.get("nearest_ais_m")
        out.append({"k": "warn", "t": (f"Nearest AIS vessel is {d:.0f} m from the Doppler-corrected "
                                       "position, outside the gate" if pd.notna(d)
                                       else "No AIS vessel within the search gate")})
        cov = row.get("ais_msgs_per_day")

        if pd.notna(cov):
            good = cov >= 200
            out.append({"k": "ok" if good else "warn",
                        "t": (f"Receiver coverage here: {cov:,.0f} messages/day "
                              f"({'good' if good else 'sparse — treat with caution'})")})

        up = row.get("recorder_uptime")
        if pd.notna(up):
            out.append({"k": "ok" if up >= 0.99 else "warn",
                        "t": f"Recorder was online for {up * 100:.0f}% of the matching window"})

    elif row.get("cls") == "matched":
        res = row.get("residual_m")
        if pd.notna(res):
            out.append({"k": "ok", "t": f"Matched within {res:.0f} m after Doppler correction"})
        ais_len, rad_len = row.get("ais_len_m"), row.get("len_m")
        if pd.notna(ais_len) and pd.notna(rad_len) and ais_len > 0 and rad_len / ais_len > 2.0:
            out.append({"k": "warn", "t": (f"Radar footprint ({rad_len:.0f} m) is {rad_len / ais_len:.1f}x "
                                           f"the reported length ({ais_len:.0f} m)")})
    else:
        n = row.get("seen_in_passes")
        out.append({"k": "info", "t": (f"Detected at this spot in {int(n)} separate passes — treated as a "
                                       "fixed structure" if pd.notna(n)
                                       else "Repeats across passes; treated as a fixed structure")})
    return out

def to_geojson(df: pd.DataFrame, no_ais: bool) -> dict:
    feats = []

    for _, r in df.iterrows():
        props = {"id": str(r["id"]), "cls": r["cls"]}
        for k, src in (("len_m", "len_m"), ("conf", "conf"), ("vessel_p", "vessel_p"),
                       ("residual_m", "residual_m"), ("doppler_m", "doppler_m")):
            v = r.get(src)
            if pd.notna(v):
                props[k] = round(float(v), 3)
        for k in ("chip_vh", "chip_vv"):
            if pd.notna(r.get(k)):
                props[k] = r[k]
        if pd.notna(r.get("chip_px")):
            props["chip_px"] = int(r["chip_px"])
        ais = {}
        for k, src in (("len_m", "ais_len_m"), ("sog", "ais_sog"), ("cog", "ais_cog"),
                       ("gap_s", "ais_gap_s"), ("sigma_m", "ais_sigma_m")):
            v = r.get(src)
            if pd.notna(v):
                ais[k] = round(float(v), 2)
        if ais:
            props["ais"] = ais
        props["reasons"] = reasons_for(r, no_ais)
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point",
                                   "coordinates": [round(float(r["lon"]), 6), round(float(r["lat"]), 6)]}})

    return {"type": "FeatureCollection", "features": feats}

def rebuild_manifest(out_dir: Path, cfg, sample: bool = False) -> dict:
    passes = []

    for meta in sorted((out_dir / "passes").glob("*.meta.json")):
        passes.append(json.loads(meta.read_text()))

    passes.sort(key=lambda p: p["acquired"], reverse=True)
    has_ais = any(p.get("counts", {}).get("matched", 0) or p.get("counts", {}).get("unmatched", 0) for p in passes)
   
    man = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample": bool(sample),
        "title": cfg.site.title,
        "bbox": list(cfg.site.bbox),
        "ais": bool(has_ais),
        "rules": {
            "delay_hours": cfg.rules.delay_hours,
            "size_floor_m": cfg.rules.size_floor_m,
            "aggregate_cell_km": cfg.rules.aggregate_cell_km,
            "masked": bool(cfg.rules.masks),
        },
        "coastline": {"hi": "coastline.geojson", "lo": "coastline-low.geojson"},
        "structures": "structures.geojson" if (out_dir / "structures.geojson").exists() else None,
        "passes": passes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"manifest: {len(passes)} passes, ais={has_ais}, sample={sample}")

    return man

def apply_retention(out_dir: Path, keep_days) -> None:
    if not keep_days:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=float(keep_days))
    removed = 0

    for meta in (out_dir / "passes").glob("*.meta.json"):
        m = json.loads(meta.read_text())
        when = datetime.fromisoformat(m["acquired"].replace("Z", "+00:00"))
        if when >= cutoff:
            continue
        for key in ("detections", "cells", "footprint"):
            if m.get(key):
                (out_dir / m[key]).unlink(missing_ok=True)
        if m.get("overview", {}).get("url"):
            (out_dir / m["overview"]["url"]).unlink(missing_ok=True)
        for f in (out_dir / "chips").glob(f"{m['id']}*"):
            f.unlink(missing_ok=True)
        meta.unlink()
        removed += 1

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-config", default="configs/publish.yaml")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--detections", nargs="*", default=[])
    ap.add_argument("--mapping", default=None, help="ESA_xView3_sceneName_mapping.csv")
    ap.add_argument("--acquired", default=None, help="ISO time, overrides --mapping")
    ap.add_argument("--no-ais", action="store_true", help="detector-only stage")
    ap.add_argument("--skip-chips", action="store_true")
    ap.add_argument("--skip-overview", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--now", default=None, help="ISO time, for testing the delay rule")
    a = ap.parse_args()

    cfg = load_config(a.publish_config)
    out_dir = Path(cfg.site.out_dir)
    (out_dir / "passes").mkdir(parents=True, exist_ok=True)
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)

    if a.manifest_only:
        apply_retention(out_dir, cfg.retention.keep_days)
        rebuild_manifest(out_dir, cfg)
        return

    if not a.detections or not a.scenes:
        ap.error("--detections and --scenes are required unless --manifest-only")

    masks = load_exclusion_masks(cfg.rules.masks)
    if masks is None:
        print("exclusion mask not configured\n")

    mapping = pd.read_csv(a.mapping) if a.mapping else None
    scenes = pd.read_csv(a.scenes)
    rules = dict(cfg.rules.to_dict(), bbox=list(cfg.site.bbox))

    for det_path in a.detections:
        df = load_detections(det_path, a.no_ais)
        scene_id = str(df["scene_id"].iloc[0]) if "scene_id" in df.columns else Path(det_path).stem
        srow = scenes[scenes.scene_id == scene_id]

        if srow.empty:
            print(f"! {scene_id} not in scenes.csv, skipping")
            continue

        srow = srow.iloc[0]
        acquired, satellite = acquisition_for(scene_id, mapping, a.acquired)
        pass_id = acquired.strftime("%Y%m%dT%H%M")

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

        try:
            individual, cells, report = apply_publication_rules(gdf, acquired, rules, now, masks)
        except HeldBack as e:
            print(f"  HELD BACK: {e}")
            continue

        if not a.skip_chips and len(individual):
            ch = cfg.chips
            reader = SceneReader(srow.vv_path, srow.vh_path, cfg_nodata(srow))
            chip_dir = out_dir / "chips"
            n_chips = 0

            for i, r in individual.iterrows():
                vv_db, vh_db = extract_chip(reader, r["col"], r["row"], ch.window_m, 10.0)
                if not np.isfinite(vh_db).any():
                    continue
                for pol, db in (("vh", vh_db), ("vv", vv_db)):
                    if pol.upper() not in [p.upper() for p in ch.polarizations]:
                        continue
                    u8 = (sea_relative_stretch(db, ch.sea_relative_below_db, ch.sea_relative_above_db)
                          if ch.stretch == "sea_relative" else percentile_stretch(db))
                    name = f"{pass_id}_{r['id']}_{pol}.png"
                    save_png(u8, chip_dir / name)
                    individual.loc[i, f"chip_{pol}"] = f"chips/{name}"
                individual.loc[i, "chip_px"] = int(round(ch.window_m / 10.0))
                n_chips += 1

            reader.close()
            print(f"  chips: {n_chips}")

        det_file = f"passes/{pass_id}.geojson"
        (out_dir / det_file).write_text(json.dumps(to_geojson(individual, a.no_ais)))
        cells_file = None
        if len(cells):
            cells_file = f"passes/{pass_id}.cells.geojson"
            cells.to_file(out_dir / cells_file, driver="GeoJSON")

        footprint, overview = scene_footprint_and_overview(
            srow.vv_path, srow.vh_path, cfg_nodata(srow),
            cfg.overview.enabled and not a.skip_overview,
            int(cfg.overview.max_px), list(cfg.overview.percentile))
        fp_file = f"passes/{pass_id}.footprint.geojson"
        gpd.GeoDataFrame(geometry=[footprint], crs="EPSG:4326").to_file(out_dir / fp_file, driver="GeoJSON")

        ov = None
        if overview is not None:
            u8, wgs = overview
            name = f"overviews/{pass_id}_{scene_id[:8]}.png"
            save_png(u8, out_dir / name)
            ov = {"url": name, "bounds": [[wgs[0], wgs[1]], [wgs[2], wgs[3]]],
                  "px": [int(u8.shape[1]), int(u8.shape[0])]}
            print(f"  overview: {u8.shape[1]}x{u8.shape[0]} px, "
                  f"{(out_dir / name).stat().st_size / 1024:.0f} KB")

        counts = individual["cls"].value_counts().to_dict() if len(individual) else {}
        meta = {
            "id": pass_id,
            "acquired": acquired.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "satellite": satellite,
            "scenes": [scene_id],
            "detections": det_file,
            "cells": cells_file,
            "footprint": fp_file,
            "overview": ov,
            "counts": {"total": int(len(individual)), **{k: int(v) for k, v in counts.items()}},
            "aggregated_points": report["aggregated_points"],
            "masked_out": report["masked_out"],
        }
        (out_dir / "passes" / f"{pass_id}.meta.json").write_text(json.dumps(meta, indent=1))

    apply_retention(out_dir, cfg.retention.keep_days)
    rebuild_manifest(out_dir, cfg)

def cfg_nodata(srow) -> float:
    v = srow.get("nodata")
    return float(v) if v is not None and not pd.isna(v) else -32768.0

if __name__ == "__main__":
    main()