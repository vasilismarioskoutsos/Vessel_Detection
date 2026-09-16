from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decode import decode_maps
from metrics import detection_metrics
from scenes import SceneObjects, load_objects_table, objects_for_scene
from targets import build_targets, stack_prediction_like_target

def synthetic_objects(n=400, h=2048, w=2048, seed=0, crowded_frac=0.3) -> SceneObjects:
    rng = np.random.default_rng(seed)
    col = rng.uniform(5, w - 5, n).astype(np.float32)
    row = rng.uniform(5, h - 5, n).astype(np.float32)

    # make a few objects sit very close to another
    k = int(n * crowded_frac)
    idx = rng.choice(n, k, replace=False)
    col[idx] = np.clip(col[(idx + 1) % n] + rng.uniform(3, 12, k), 0, w - 1)
    row[idx] = np.clip(row[(idx + 1) % n] + rng.uniform(-6, 6, k), 0, h - 1)
    length = rng.uniform(1.0, 40.0, n).astype(np.float32)
    length[rng.random(n) < 0.3] = np.nan
    vessel = (rng.random(n) < 0.7).astype(np.float32)
    vessel[rng.random(n) < 0.2] = np.nan

    return SceneObjects(col, row, length, vessel, np.zeros(n, bool))

def run(objs: SceneObjects, h: int, w: int, strides=(1, 2, 4, 8), log_length=True, max_dist=20.0):
    rows = []
    for s in strides:
        t = build_targets(objs, h, w, stride=s, log_length=log_length)
        pred = stack_prediction_like_target(t)
        det = decode_maps(pred, stride=s, objectness_threshold=0.5, log_length=log_length)
        m = detection_metrics(det[["col", "row"]].to_numpy(), np.stack([objs.col, objs.row], 1), max_dist, det["length_px"].to_numpy(), objs.length_px, det["vessel_p"].to_numpy(), objs.is_vessel)

        rows.append(dict(stride=s, n_gt=len(objs), n_det=len(det), f1=m["f1"], precision=m["precision"], recall=m["recall"], length_mae_px=m.get("length_mae_px", np.nan), vessel_acc=m.get("vessel_acc", np.nan)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--raw-length", action="store_true")
    a = ap.parse_args()

    if a.objects:
        df = load_objects_table(a.objects)
        objs = objects_for_scene(df, a.scene, pixel_size_m=10.0)
        h = int(np.ceil(objs.row.max())) + 8
        w = int(np.ceil(objs.col.max())) + 8
    else:
        objs = synthetic_objects(h=a.size, w=a.size)
        h = w = a.size

    table = run(objs, h, w, log_length=not a.raw_length)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))