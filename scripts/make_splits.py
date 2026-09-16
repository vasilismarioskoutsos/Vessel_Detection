from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps-deg", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--leaky", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    s = pd.read_csv(a.scenes)
    s = s[s.n_objects > 0].reset_index(drop=True)
    xy = s[["lon_center", "lat_center"]].to_numpy()
    s["location"] = DBSCAN(eps=a.eps_deg, min_samples=1).fit_predict(xy) if not a.leaky else np.arange(len(s))
    rng = np.random.default_rng(a.seed)
    locs = s.location.unique()
    rng.shuffle(locs)
    n = len(locs)
    n_test = max(1, int(round(n * a.test_frac)))
    n_val = max(1, int(round(n * a.val_frac)))
    test_locs, val_locs, train_locs = set(locs[:n_test]), set(locs[n_test:n_test + n_val]), set(locs[n_test + n_val:])

    if not train_locs: # tiny datasets
        train_locs, val_locs = val_locs, set()

    val_scenes = s[s.location.isin(val_locs)].scene_id.tolist()
    tune = val_scenes[: max(1, len(val_scenes) // 3)] if len(val_scenes) > 1 else val_scenes
    out = dict(train=s[s.location.isin(train_locs)].scene_id.tolist(),
               val=val_scenes, tune=tune,
               test=s[s.location.isin(test_locs)].scene_id.tolist(),
               location={r.scene_id: int(r.location) for r in s.itertuples()},
               leaky=a.leaky, eps_deg=a.eps_deg)
    
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{n} locations from {len(s)} scenes -> train {len(out['train'])} / val {len(out['val'])} / " f"test {len(out['test'])} scenes  (leaky={a.leaky})")
    print(s.groupby("location").scene_id.count().rename("scenes_per_location").to_string())

if __name__ == "__main__":
    main()