from __future__ import annotations
import geopandas as gpd
from shapely.geometry import box
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adv.config import load_config

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-config", default="configs/publish.yaml")
    ap.add_argument("--source", default=None, help="override coastline.source")
    a = ap.parse_args()

    cfg = load_config(a.publish_config)
    src = Path(a.source or cfg.coastline.source)

    lon0, lat0, lon1, lat1 = cfg.site.bbox
    clip = box(lon0, lat0, lon1, lat1)

    land = gpd.read_file(src, bbox=(lon0, lat0, lon1, lat1))
    if land.crs is None:
        land = land.set_crs("EPSG:4326")
    land = land.to_crs("EPSG:4326")
    print(f"{len(land)} polygons intersect the box")

    land = gpd.GeoDataFrame(geometry=land.geometry.intersection(clip), crs="EPSG:4326")
    land = land[~land.geometry.is_empty & land.geometry.notna()]

    min_area = float(cfg.coastline.min_area_deg2)
    before = len(land)
    land = land[land.geometry.area >= min_area]
    out_dir = Path(cfg.site.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hi_tol, lo_tol = (float(t) for t in cfg.coastline.simplify_deg)
    for tol, name in ((lo_tol, "coastline.geojson"), (hi_tol, "coastline-low.geojson")):
        g = gpd.GeoDataFrame(geometry=land.geometry.simplify(tol, preserve_topology=True), crs="EPSG:4326")
        g = g[~g.geometry.is_empty & g.geometry.notna()]
        path = out_dir / name
        g[["geometry"]].to_file(path, driver="GeoJSON")
        print(f"{name}: {len(g)} polygons, {path.stat().st_size / 1024:.0f} KB (tolerance {tol} deg)")

    (out_dir / "bbox.json").write_text(json.dumps({"bbox": [lon0, lat0, lon1, lat1]}))

if __name__ == "__main__":
    main()