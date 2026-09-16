from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import load_config
from dataset import PatchDataset, collate
from losses import circlenet_loss
from metrics import detection_metrics
from model import activate, decode_torch
from train import build_model
from torchinfo import summary

def shapes(cfg, device):
    model = build_model(cfg).to(device)
    P = cfg.sampling.patch_size
    x = torch.zeros(1, 2, P, P, device=device)

    with torch.no_grad():
        y = model(x)

    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}")
    assert y.shape == (1, 5, P // cfg.targets.stride, P // cfg.targets.stride), "unexpected output shape"
    summary(model, input_size=(1, 2, P, P), depth=2, col_names=("output_size", "num_params"))

    print("shape check OK")

def overfit(cfg, device, steps):
    scenes = pd.read_csv(cfg.data.scene_index)
    objects = pd.read_parquet(cfg.data.objects_file)
    splits = json.load(open(cfg.data.splits_file))

    ds = PatchDataset(scenes[scenes.scene_id.isin(splits["train"])], objects, cfg, train=False, fixed_patches=2)
    batch = collate([ds[0], ds[1]])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    n_gt = sum(int(((c >= 0) & (c < cfg.sampling.patch_size)).sum()) for c in batch["gt_col"])

    print(f"overfitting 2 patches containing {n_gt} objects")
    model = build_model(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = None

    for step in range(steps):
        raw = model(batch["image"])
        loss, parts = circlenet_loss(raw, batch, cfg)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        first = first or parts["total"]
        if step % 25 == 0 or step == steps - 1:
            print(f"step {step:4d} " + " ".join(f"{k}={v:.4f}" for k, v in parts.items()))

    model.eval()
    with torch.no_grad():
        dets = decode_torch(activate(model(batch["image"])), cfg.targets.stride, cfg.decode.peak_kernel, cfg.decode.objectness_threshold, cfg.targets.log_length)
    md = cfg.evaluation.match_distance_m / cfg.data.pixel_size_m

    for b, d in enumerate(dets):
        gt = torch.stack([batch["gt_col"][b], batch["gt_row"][b]], 1).cpu().numpy()
        P = cfg.sampling.patch_size
        inside = (gt[:, 0] >= 0) & (gt[:, 0] < P) & (gt[:, 1] >= 0) & (gt[:, 1] < P)
        m = detection_metrics(torch.stack([d["col"], d["row"]], 1).cpu().numpy(), gt[inside], md, d["length_px"].cpu().numpy(), batch["gt_length_px"][b].cpu().numpy()[inside])
        print(f"patch {b}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))

    print(f"loss {first:.3f} -> {parts['total']:.3f}  ({first / max(parts['total'], 1e-6):.0f}x)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shapes-only", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    cfg = load_config(a.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shapes(cfg, device)

    if not a.shapes_only:
        overfit(cfg, device, a.steps)