from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from shapely.geometry import Point, Polygon
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adv.config import load_config
from adv.publish import save_png

# aegean outline pixels
PX_PER_LON, PX_PER_LAT, LON0, LAT0 = 158.86, 136.77, 22.0, 40.8

MAINLAND = [(0,0),(165,0),(178,34),(196,26),(206,52),(232,44),(224,76),(246,70),(238,104),(262,118),
            (256,158),(278,186),(266,226),(288,262),(274,300),(292,330),(276,368),(288,392),(262,404),
            (240,392),(212,404),(196,436),(206,462),(178,476),(150,458),(124,474),(132,436),(104,420),
            (82,392),(96,356),(70,330),(52,286),(28,256),(14,206),(0,168)]
TURKEY = [(1112,0),(1112,700),(1040,690),(1010,648),(1012,600),(980,566),(964,520),(930,486),(906,440),
          (872,410),(844,372),(850,330),(820,300),(832,262),(806,232),(822,196),(850,170),(878,120),
          (900,74),(940,40),(980,16),(1010,0)]
CRETE = [(228,742),(300,726),(380,734),(460,722),(540,730),(620,742),(688,752),(716,768),(690,786),
         (610,778),(520,784),(430,776),(340,782),(262,772),(226,760)]
EVIA = [(276,250),(292,262),(304,300),(312,340),(318,372),(308,392),(292,386),(282,350),(274,306),(268,272)]
# cx, cy, rx, ry
ISLANDS = [(432,20,17,12),(502,42,15,9),(520,122,22,13),(672,218,26,18),(640,327,12,22),(768,415,21,10),
           (712,430,16,7),(880,520,19,7),(975,596,16,26),(880,690,8,25),(400,260,10,13),(500,420,8,18),
           (512,442,10,7),(540,458,9,6),(516,472,8,10),(536,508,11,10),(562,504,16,14),(470,580,13,8),
           (548,600,8,11)]

LANES = [[(548,72),(918,532)], [(312,406),(676,660)], [(246,86),(416,348)], [(320,820),(720,818)]]
STRUCTURES = [(462,44),(472,52),(456,58),(478,38),(330,300),(336,308),(330,316),(338,324),(331,332),
              (339,340),(690,466),(698,472),(690,478),(698,484),(690,490)]

def px2ll(x: float, y: float) -> tuple[float, float]:
    return round(LON0 + x / PX_PER_LON, 5), round(LAT0 - y / PX_PER_LAT, 5)

def ring(points) -> list:
    r = [list(px2ll(x, y)) for x, y in points]
    if r[0] != r[-1]:
        r.append(r[0])
    return r

def ellipse_ring(cx, cy, rx, ry, n=14) -> list:
    pts = [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    return ring(pts)

def land_polygons() -> list:
    polys = [ring(MAINLAND), ring(TURKEY), ring(CRETE), ring(EVIA)]
    polys += [ellipse_ring(*i) for i in ISLANDS]
    return polys

def synthetic_chip(n: int, length_px: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.gamma(4.5, 5.0, size=(n, n))
    yy, xx = np.mgrid[0:n, 0:n]
    cy = cx = n / 2

    ang = rng.uniform(0, np.pi)
    dx = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    dy = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    sx = max(1.2, length_px / 3.0)
    a += 250.0 * np.exp(-(dx ** 2) / (2 * sx ** 2) - (dy ** 2) / (2 * 1.6 ** 2))

    return np.clip(a, 0, 255).astype(np.uint8)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-config", default="configs/publish.yaml")
    ap.add_argument("--passes", type=int, default=6)
    ap.add_argument("--no-ais", action="store_true", help="label everything 'detection' (v1 stage)")
    a = ap.parse_args()

    cfg = load_config(a.publish_config)
    out = Path(cfg.site.out_dir)
    (out / "passes").mkdir(parents=True, exist_ok=True)
    (out / "chips").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260918)

    polys = land_polygons()
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [p]}}
        for p in polys]}
    (out / "coastline.geojson").write_text(json.dumps(fc))
    (out / "coastline-low.geojson").write_text(json.dumps(fc))
    (out / "bbox.json").write_text(json.dumps({"bbox": list(cfg.site.bbox)}))
    print(f"coastline: {len(polys)} coarse polygons (replace with build_coastline.py)")

    land = [Polygon(p) for p in polys]

    def at_sea(lon, lat) -> bool:
        p = Point(lon, lat)
        return not any(g.contains(p) for g in land)

    floor = float(cfg.rules.size_floor_m)
    chip_px = int(round(cfg.chips.window_m / 10.0))
    base = datetime.now(timezone.utc).replace(minute=42, second=11, microsecond=0) - timedelta(days=4)
    total_bytes = 0

    for k in range(a.passes):
        acquired = base - timedelta(days=6 * k, hours=int(rng.integers(0, 3)))
        pid = acquired.strftime("%Y%m%dT%H%M")
        feats, counts = [], {}
        j = 0

        for (x0, y0), (x1, y1) in LANES:
            for t in np.linspace(0.02, 0.98, 14):
                x = x0 + (x1 - x0) * t + rng.normal(0, 11)
                y = y0 + (y1 - y0) * t + rng.normal(0, 11)
                lon, lat = px2ll(x, y)

                if not at_sea(lon, lat):
                    continue

                unmatched = rng.random() < 0.16
                cls = "detection" if a.no_ais else ("unmatched" if unmatched else "matched")
                length = float(rng.uniform(16, 120) if unmatched else rng.uniform(34, 290))
                conf = float(rng.uniform(0.45, 0.99))
                did = f"{pid}_{j:04d}"
                j += 1

                if length < floor:
                    continue

                if rng.random() < 0.30:
                    save_png(synthetic_chip(chip_px, length / 10.0, j + k * 1000),
                             out / "chips" / f"{pid}_{did}_vh.png")
                    chip = f"chips/{pid}_{did}_vh.png"

                else:
                    chip = None
                props = {"id": did, "cls": cls, "len_m": round(length, 1), "conf": round(conf, 2),
                         "vessel_p": round(float(rng.uniform(0.8, 0.99)), 2)}
                
                if chip:
                    props["chip_vh"] = chip
                    props["chip_px"] = chip_px

                if cls == "unmatched":
                    props["reasons"] = [
                        {"k": "warn", "t": f"Nearest AIS vessel is {rng.integers(260, 900)} m from the "
                                           "Doppler-corrected position, outside the gate"},
                        {"k": "ok", "t": f"Receiver coverage here: {rng.integers(400, 2400):,} messages/day (good)"},
                        {"k": "ok", "t": "Recorder was online for 100% of the matching window"}]
                    
                elif cls == "matched":
                    props["residual_m"] = int(rng.integers(20, 180))
                    props["doppler_m"] = int(rng.integers(120, 700))
                    props["ais"] = {"len_m": round(length * float(rng.uniform(0.85, 1.15)), 1),
                                    "sog": round(float(rng.uniform(4, 18)), 1),
                                    "cog": int(rng.integers(0, 360)), "gap_s": int(rng.integers(2, 40)),
                                    "sigma_m": int(rng.integers(90, 320))}
                    props["reasons"] = [{"k": "ok", "t": f"Matched within {props['residual_m']} m " "after Doppler correction"}]
                else:
                    props["reasons"] = [{"k": "info", "t": "AIS comparison is not running yet — " "this is a radar detection only"}]
                counts[cls] = counts.get(cls, 0) + 1
                feats.append({"type": "Feature", "properties": props,
                              "geometry": {"type": "Point", "coordinates": [lon, lat]}})

        for i, (x, y) in enumerate(STRUCTURES):
            lon, lat = px2ll(x + rng.normal(0, 1.2), y + rng.normal(0, 1.2))
            counts["structure"] = counts.get("structure", 0) + 1
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": {"id": f"{pid}_str_{i:03d}", "cls": "structure",
                                         "len_m": round(float(rng.uniform(30, 90)), 1),
                                         "conf": round(float(rng.uniform(0.6, 0.95)), 2),
                                         "reasons": [{"k": "info", "t": f"Detected at this spot in "
                                                                        f"{rng.integers(7, 22)} separate passes — "
                                                                        "treated as a fixed structure"}]}})

        det = f"passes/{pid}.geojson"
        (out / det).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

        lon0, lat0, lon1, lat1 = cfg.site.bbox
        fp = f"passes/{pid}.footprint.geojson"
        (out / fp).write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
                [lon0 + 0.6, lat0 + 0.4], [lon0 + 3.9, lat0 + 0.4],
                [lon0 + 3.9, lat1 - 0.3], [lon0 + 0.6, lat1 - 0.3], [lon0 + 0.6, lat0 + 0.4]]]}},
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
                [lon0 + 3.3, lat0 + 0.9], [lon1 - 0.4, lat0 + 0.9],
                [lon1 - 0.4, lat1 - 0.8], [lon0 + 3.3, lat1 - 0.8], [lon0 + 3.3, lat0 + 0.9]]]}}]}))

        meta = {"id": pid, "acquired": acquired.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "satellite": "Sentinel-1C" if k % 2 == 0 else "Sentinel-1D",
                "scenes": [f"sample{k:02d}"], "detections": det, "cells": None, "footprint": fp,
                "overview": None,
                "counts": {"total": len(feats), **counts},
                "aggregated_points": int(rng.integers(20, 70)), "masked_out": 0}
        (out / "passes" / f"{pid}.meta.json").write_text(json.dumps(meta, indent=1))
        print(f"  pass {pid}: {len(feats)} detections {counts}")

    man = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample": True,
        "title": cfg.site.title,
        "bbox": list(cfg.site.bbox),
        "ais": not a.no_ais,
        "rules": {"delay_hours": cfg.rules.delay_hours, "size_floor_m": cfg.rules.size_floor_m,
                  "aggregate_cell_km": cfg.rules.aggregate_cell_km, "masked": False},
        "coastline": {"hi": "coastline.geojson", "lo": "coastline-low.geojson"},
        "structures": None,
        "passes": sorted([json.loads(p.read_text()) for p in (out / "passes").glob("*.meta.json")],
                         key=lambda p: p["acquired"], reverse=True),
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    total_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nsample site data: {total_bytes / 1e6:.1f} MB in {out}")
    print("serve it with:  python -m http.server 8000 --directory web")

if __name__ == "__main__":
    main()