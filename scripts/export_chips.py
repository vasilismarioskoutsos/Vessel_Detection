from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adv.config import load_config  # noqa: E402
from adv.publish import extract_chip, percentile_stretch, save_png, sea_relative_stretch  # noqa: E402
from adv.scenes import SceneReader

def points_from_labels(objects_file: str, scene_id: str, pixel_size_m: float) -> pd.DataFrame:
    df = pd.read_parquet(objects_file) if objects_file.endswith(".parquet") else pd.read_csv(objects_file)

    d = df[df["scene_id"] == scene_id].copy()
    out = pd.DataFrame({"col": d["detect_scene_column"].astype(float), "row": d["detect_scene_row"].astype(float)})
    out["len_m"] = d["vessel_length_m"].astype(float).to_numpy() if "vessel_length_m" in d else np.nan
    out["id"] = [f"{scene_id[:12]}_lbl_{i:05d}" for i in range(len(out))]

    return out.reset_index(drop=True)

def points_from_detections(path: str, scene_id: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    if "scene_id" in d.columns:
        d = d[d["scene_id"] == scene_id]

    out = d[["col", "row"]].astype(float).copy()
    out["len_m"] = d["length_m"].astype(float).to_numpy() if "length_m" in d else np.nan
    out["id"] = (d["id"].astype(str).to_numpy() if "id" in d else [f"{scene_id[:12]}_det_{i:05d}" for i in range(len(d))])
  
    return out.reset_index(drop=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish-config", default="configs/publish.yaml")
    ap.add_argument("--scenes", required=True, help="data/processed/scenes.csv")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--objects", default=None, help="label table (no model needed)")
    ap.add_argument("--detections", default=None, help="model output CSV from adv.infer")
    ap.add_argument("--out", default=None, help="default: <site.out_dir>/chips")
    ap.add_argument("--limit", type=int, default=0, help="0 = all; useful for a quick look")
    ap.add_argument("--nodata", type=float, default=-32768.0)
    ap.add_argument("--pixel-size", type=float, default=10.0)
    a = ap.parse_args()

    if not (a.objects or a.detections):
        ap.error("give either --objects (labels) or --detections (model output)")

    cfg = load_config(a.publish_config)
    ch = cfg.chips
    out_dir = Path(a.out or (Path(cfg.site.out_dir) / "chips"))
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = pd.read_csv(a.scenes)
    row = scenes[scenes.scene_id == a.scene]
    if row.empty:
        raise SystemExit(f"scene {a.scene} is not in {a.scenes}")
    row = row.iloc[0]

    pts = (points_from_labels(a.objects, a.scene, a.pixel_size) if a.objects
           else points_from_detections(a.detections, a.scene))
    if a.limit:
        pts = pts.head(a.limit)

    if pts.empty:
        raise SystemExit("no points for this scene")

    reader = SceneReader(row.vv_path, row.vh_path, a.nodata)
    records = []

    for _, p in pts.iterrows():
        vv_db, vh_db = extract_chip(reader, p.col, p.row, ch.window_m, a.pixel_size)
        if not np.isfinite(vh_db).any():
            continue

        rec = {"id": p.id, "col": p.col, "row": p.row, "len_m": p.len_m}
        
        for pol, db in (("vh", vh_db), ("vv", vv_db)):
            if pol.upper() not in [x.upper() for x in ch.polarizations]:
                continue
            u8 = (sea_relative_stretch(db, ch.sea_relative_below_db, ch.sea_relative_above_db)
                  if ch.stretch == "sea_relative" else percentile_stretch(db))
            name = f"{p.id}_{pol}.png"
            save_png(u8, out_dir / name)
            rec[f"chip_{pol}"] = f"chips/{name}"
        rec["chip_px"] = int(round(ch.window_m / a.pixel_size))
        records.append(rec)

    reader.close()

    df = pd.DataFrame(records)
    index = out_dir / "chips.csv"
    if index.exists():
        df = pd.concat([pd.read_csv(index), df], ignore_index=True).drop_duplicates("id", keep="last")
    df.to_csv(index, index=False)

    total_kb = sum(f.stat().st_size for f in out_dir.glob("*.png")) / 1024

if __name__ == "__main__":
    main()