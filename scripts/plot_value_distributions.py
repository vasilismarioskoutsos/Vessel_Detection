from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import load_config
from normalize import db_to_unit, raw_to_db
from scenes import SceneReader, objects_for_scene

FIG = Path("figures")

def sample_scene(reader: SceneReader, n_windows=40, size=512, rng=None):
    rng = rng or np.random.default_rng(0)
    H, W = reader.shape
    vv, vh = [], []
    for _ in range(n_windows):
        r0, c0 = rng.integers(0, H - size), rng.integers(0, W - size)
        a, b = reader.read_window(r0, c0, size, size)
        vv.append(a.ravel()); vh.append(b.ravel())

    vv, vh = np.concatenate(vv), np.concatenate(vh)
    ok = np.isfinite(vv) & np.isfinite(vh)

    return vv[ok], vh[ok]

def ship_values(reader: SceneReader, objs, max_n=2000):
    vv, vh = [], []
    for i in range(min(len(objs), max_n)):
        a, b = reader.read_window(int(objs.row[i]) - 1, int(objs.col[i]) - 1, 3, 3)
        if np.isfinite(a).any():
            vv.append(np.nanmax(a)); vh.append(np.nanmax(b))

    return np.array(vv), np.array(vh)

def plot_channel(ax, sea, ships, name, center, scale):
    bins = np.linspace(-40, 15, 111)
    ax.hist(sea, bins, density=True, alpha=0.5, label="random pixels (mostly sea)")

    if len(ships):
        ax.hist(ships, bins, density=True, alpha=0.5, label="labelled object centres (3x3 max)")

    ax2 = ax.twinx()
    x = np.linspace(-40, 15, 400)
    ax2.plot(x, db_to_unit(x, center, scale), "k--", label=f"sigmoid c={center} s={scale}")
    ax2.set_ylim(0, 1); ax2.set_ylabel("normalized value")
    ax.set_title(name); ax.set_xlabel("dB"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes"); ap.add_argument("--objects"); ap.add_argument("--scene")
    ap.add_argument("--compare-vv", nargs=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    FIG.mkdir(parents=True, exist_ok=True)

    if a.scene:
        s = pd.read_csv(a.scenes); r = s[s.scene_id == a.scene].iloc[0]
        reader = SceneReader(r.vv_path, r.vh_path, cfg.data.nodata_value)
        objs = objects_for_scene(pd.read_parquet(a.objects), a.scene, cfg.data.pixel_size_m)
        sea_vv, sea_vh = sample_scene(reader)
        sh_vv, sh_vh = ship_values(reader, objs)
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        plot_channel(ax[0], sea_vv, sh_vv, "VV", cfg.normalization.vv.center, cfg.normalization.vv.scale)
        plot_channel(ax[1], sea_vh, sh_vh, "VH", cfg.normalization.vh.center, cfg.normalization.vh.scale)
        fig.tight_layout(); fig.savefig(FIG / f"values_{a.scene}.png", dpi=120)

        for name, sea, sh in (("VV", sea_vv, sh_vv), ("VH", sea_vh, sh_vh)):
            print(f"{name}: sea p5/p50/p95 = {np.percentile(sea, [5, 50, 95]).round(1)}   "
                  f"ships p5/p50/p95 = {np.percentile(sh, [5, 50, 95]).round(1) if len(sh) else 'n/a'}")
        print(f"saved {FIG / f'values_{a.scene}.png'}")

    if a.compare_vv:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(-40, 15, 111)
        for p in a.compare_vv:
            rd = SceneReader(p, p, cfg.data.nodata_value)
            vv, _ = sample_scene(rd)
            ax.hist(vv, bins, density=True, alpha=0.5, label=Path(p).parent.name)
            print(f"{p}: p5/p50/p95 = {np.percentile(vv, [5, 50, 95]).round(1)}")

        ax.set_xlabel("VV dB"); ax.legend(); ax.set_title("Preprocessing consistency check")
        fig.savefig(FIG / "compare_vv.png", dpi=120)
        print(f"saved {FIG / 'compare_vv.png'}")

if __name__ == "__main__":
    main()