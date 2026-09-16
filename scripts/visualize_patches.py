from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import load_config
from dataset import PatchDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--val", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    scenes = pd.read_csv(cfg.data.scene_index)
    objects = pd.read_parquet(cfg.data.objects_file)
    splits = json.load(open(cfg.data.splits_file))
    sub = scenes[scenes.scene_id.isin(splits["val" if a.val else "train"])]
    ds = PatchDataset(sub, objects, cfg, train=not a.val, fixed_patches=a.n)
    cols = 4
    rows = int(np.ceil(a.n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    s = cfg.targets.stride

    for i, ax in enumerate(axes.ravel()):
        if i >= a.n:
            ax.axis("off"); continue

        item = ds[i]
        img = item["image"].numpy()
        ax.imshow(img[1], cmap="gray", vmin=0, vmax=1)
        ys, xs = np.nonzero(item["heat"].numpy()[0] >= 1.0)
        ax.scatter(xs * s, ys * s, s=40, facecolors="none", edgecolors="lime", linewidths=0.8)
        ax.set_title(f"{len(xs)} objects", fontsize=8)
        ax.axis("off")

    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig("figures/patches.png", dpi=100)
    print("saved figures/patches.png")

if __name__ == "__main__":
    main()