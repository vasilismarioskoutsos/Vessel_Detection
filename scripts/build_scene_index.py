from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

def find_scene_dirs(root: Path, vv_name: str) -> dict[str, Path]:
    return {p.parent.name: p.parent for p in root.rglob(vv_name)}

def scene_record(scene_id: str, d: Path, vv_name: str, vh_name: str) -> dict:
    vv, vh = d / vv_name, d / vh_name

    with rasterio.open(vv) as src:
        h, w = src.shape
        b = src.bounds
        crs = src.crs
        cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2

        if crs is not None and not crs.is_geographic:
            cx, cy = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(cx, cy)

        nodata = src.nodata
    assert vh.exists(), f"missing {vh}"
    return dict(scene_id=scene_id, vv_path=str(vv), vh_path=str(vh), height=h, width=w, crs=str(crs), nodata=nodata, lon_center=cx, lat_center=cy)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", action="append", required=True, help="label CSV(s); repeatable")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--vv-name", default="VV_dB.tif")
    ap.add_argument("--vh-name", default="VH_dB.tif")
    a = ap.parse_args()

    dirs = find_scene_dirs(Path(a.root), a.vv_name)
    print(f"found {len(dirs)} scene folders")
    labels = []
    for f in a.labels:
        df = pd.read_csv(f)
        df["label_source"] = Path(f).stem
        labels.append(df)

    labels = pd.concat(labels, ignore_index=True)
    labels = labels[labels.scene_id.isin(dirs)]
    print(f"{len(labels)} label rows for scenes on disk ({labels.scene_id.nunique()} scenes)")

    recs = []
    for sid, d in sorted(dirs.items()):
        r = scene_record(sid, d, a.vv_name, a.vh_name)
        sub = labels[labels.scene_id == sid]
        r["n_objects"] = len(sub)
        r["label_source"] = ",".join(sorted(sub.label_source.unique())) if len(sub) else ""

        # labels must fall inside the raster
        if len(sub):
            bad = ((sub.detect_scene_row < 0) | (sub.detect_scene_row >= r["height"]) |
                   (sub.detect_scene_column < 0) | (sub.detect_scene_column >= r["width"])).sum()
            if bad:
                print(f"{sid}: {bad} labels outside raster bounds")

        recs.append(r)
    scenes = pd.DataFrame(recs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(out / "scenes.csv", index=False)
    labels.to_parquet(out / "objects.parquet", index=False)
    print(scenes[["scene_id", "height", "width", "n_objects", "label_source"]].to_string(index=False))
    print(f"wrote {out / 'scenes.csv'} and {out / 'objects.parquet'}")
    nod = scenes.nodata.dropna().unique()
    print(f"nodata values seen in rasters: {nod}  (config data.nodata_value must match)")

if __name__ == "__main__":
    main()