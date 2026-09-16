from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .augment import AugParams, geometric, photometric_db
from .normalize import normalize_pair
from .scenes import SceneObjects, SceneReader, objects_for_scene
from .targets import build_targets

class SceneEntry:
    def __init__(self, scene_row: pd.Series, objs: SceneObjects, cfg):
        self.scene_id = scene_row["scene_id"]
        self.vv_path, self.vh_path = scene_row["vv_path"], scene_row["vh_path"]
        self.height, self.width = int(scene_row["height"]), int(scene_row["width"])
        self.objs = objs
        self.anchor_weights = self._anchor_weights(objs, cfg)
        self._reader = None

    @staticmethod
    def _anchor_weights(objs: SceneObjects, cfg) -> np.ndarray:
        n = len(objs)

        if n == 0:
            return np.zeros(0)
        
        w = np.ones(n)
        is_nv = objs.is_vessel == 0.0
        w[is_nv] *= cfg.sampling.weight_non_vessel
        w[objs.near_shore] *= cfg.sampling.weight_near_shore

        if n > 1:
            tree = cKDTree(np.stack([objs.col, objs.row], 1))
            counts = np.array([len(c) for c in tree.query_ball_point(np.stack([objs.col, objs.row], 1), r=cfg.sampling.crowd_radius_px)])
            w /= np.sqrt(counts) # object in a cluster of 100 gets 1/10 the weight

        return w / w.sum()

    def reader(self, nodata) -> SceneReader:
        if self._reader is None:
            self._reader = SceneReader(self.vv_path, self.vh_path, nodata)
        return self._reader

class PatchDataset(Dataset):
    def __init__(self, scenes_df: pd.DataFrame, objects_df: pd.DataFrame, cfg, train: bool = True, fixed_patches: int | None = None, seed: int = 0):
        self.cfg = cfg
        self.train = train
        self.P = cfg.sampling.patch_size
        self.entries = [
            SceneEntry(r, objects_for_scene(objects_df, r["scene_id"], cfg.data.pixel_size_m, cfg.targets.min_confidence, cfg.sampling.near_shore_km), cfg)
            for _, r in scenes_df.iterrows()]
        
        n_obj = np.array([len(e.objs) for e in self.entries], float)
        self.scene_weights = (n_obj + 1.0) / (n_obj + 1.0).sum() # scenes with more objects sampled more
        self.aug = AugParams(**cfg.augmentation.to_dict()) if train else None
        self.epoch_length = fixed_patches or cfg.sampling.epoch_length
        self._base_seed = seed

        self.fixed = None
        if not train:
            rng = np.random.default_rng(seed)
            self.fixed = [self._choose_window(rng) for _ in range(self.epoch_length)]

    def __len__(self) -> int:
        return self.epoch_length

    def _choose_window(self, rng: np.random.Generator):
        si = rng.choice(len(self.entries), p=self.scene_weights)
        e = self.entries[si]
        P = self.P

        if len(e.objs) and rng.random() < self.cfg.sampling.object_centered_fraction:
            oi = rng.choice(len(e.objs), p=e.anchor_weights)
            j = self.cfg.sampling.jitter_px
            cx = e.objs.col[oi] + rng.uniform(-j, j)
            cy = e.objs.row[oi] + rng.uniform(-j, j)
            row0, col0 = int(cy - P / 2), int(cx - P / 2)
        else:
            row0 = int(rng.uniform(-P // 4, e.height - 3 * P // 4))
            col0 = int(rng.uniform(-P // 4, e.width - 3 * P // 4))
        return si, row0, col0

    def __getitem__(self, idx: int) -> dict:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self._base_seed, wid, idx, int(torch.initial_seed()) % (2**31)])
        si, row0, col0 = self.fixed[idx] if self.fixed is not None else self._choose_window(rng)
        e = self.entries[si]
        P = self.P

        vv, vh = e.reader(self.cfg.data.nodata_value).read_window(row0, col0, P, P)
        if self.train:
            vv, vh = photometric_db(vv, vh, self.aug, rng)
        img = normalize_pair(vv, vh, self.cfg.normalization)

        o = e.objs
        inside = (o.col >= col0 - 8) & (o.col < col0 + P + 8) & (o.row >= row0 - 8) & (o.row < row0 + P + 8)
        objs = SceneObjects(o.col[inside] - col0, o.row[inside] - row0, o.length_px[inside], o.is_vessel[inside], o.near_shore[inside])

        if self.train:
            img, c, r, L = geometric(img, objs.col, objs.row, objs.length_px, self.aug, rng, self.cfg.normalization.fill_value)
            objs = SceneObjects(c, r, L, objs.is_vessel, objs.near_shore)

        t = build_targets(objs, P, P, stride=self.cfg.targets.stride,
                          heat_radius_px=self.cfg.targets.heat_radius_px,
                          cls_radius_px=self.cfg.targets.cls_radius_px,
                          log_length=self.cfg.targets.log_length)
        
        out = {k: torch.from_numpy(v) for k, v in t.items()}
        out["image"] = torch.from_numpy(img)

        # keep labels for metric computation
        out["gt_col"], out["gt_row"] = torch.from_numpy(objs.col), torch.from_numpy(objs.row)
        out["gt_length_px"], out["gt_vessel"] = torch.from_numpy(objs.length_px), torch.from_numpy(objs.is_vessel)
        return out

def collate(batch: list[dict]) -> dict:
    out = {}
    for k in batch[0]:

        if k.startswith("gt_"):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch], 0)
    return out
