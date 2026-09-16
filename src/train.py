from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .config import load_config
from .dataset import PatchDataset, collate
from .losses import circlenet_loss
from .metrics import detection_metrics
from .model import CircleNet, activate, decode_torch
import wandb

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_loaders(cfg):
    scenes = pd.read_csv(cfg.data.scene_index)
    objects = pd.read_parquet(cfg.data.objects_file)
    splits = json.load(open(cfg.data.splits_file))
    tr = scenes[scenes.scene_id.isin(splits["train"])]
    va = scenes[scenes.scene_id.isin(splits["val"])]

    assert len(tr) and len(va), "empty train or val split"
    ds_tr = PatchDataset(tr, objects, cfg, train=True, seed=cfg.train.seed)
    ds_va = PatchDataset(va, objects, cfg, train=False, fixed_patches=cfg.train.eval_patches, seed=cfg.train.seed)
    kw = dict(num_workers=cfg.train.num_workers, pin_memory=True, collate_fn=collate, persistent_workers=cfg.train.num_workers > 0)

    return (DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **kw), DataLoader(ds_va, batch_size=cfg.train.batch_size, shuffle=False, **kw))

def build_model(cfg) -> CircleNet:
    m = cfg.model
    return CircleNet(m.encoder, m.pretrained, 2, tuple(m.decoder_channels), m.head_channels, m.groupnorm_groups, m.objectness_prior)

def build_optimizer(model, cfg, steps_per_epoch: int):
    t = cfg.train
    opt = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()), "lr": t.lr_encoder},
        {"params": model.decoder_parameters(), "lr": t.lr_decoder},
    ], weight_decay=t.weight_decay)

    total = t.epochs * steps_per_epoch
    warm = max(1, int(t.warmup_epochs * steps_per_epoch))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        
        prog = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched

@torch.no_grad()
def evaluate(model, loader, cfg, device) -> dict:
    model.eval()
    tot = dict(tp=0, fp=0, fn=0)
    len_err, ves_hits, ves_n, loss_sum, n_batches = [], 0, 0, 0.0, 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=cfg.train.amp):
            raw = model(img)

        raw = raw.float()
        tb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, _ = circlenet_loss(raw, tb, cfg)
        loss_sum += loss.item(); n_batches += 1
        dets = decode_torch(activate(raw), cfg.targets.stride, cfg.decode.peak_kernel, cfg.decode.objectness_threshold, cfg.targets.log_length, cfg.decode.max_detections)

        max_dist = cfg.evaluation.match_distance_m / cfg.data.pixel_size_m
        for b, d in enumerate(dets):
            pred_xy = torch.stack([d["col"], d["row"]], 1).cpu().numpy()
            gt_xy = torch.stack([batch["gt_col"][b], batch["gt_row"][b]], 1).numpy()
            P = cfg.sampling.patch_size
            inside = (gt_xy[:, 0] >= 0) & (gt_xy[:, 0] < P) & (gt_xy[:, 1] >= 0) & (gt_xy[:, 1] < P)

            m = detection_metrics(pred_xy, gt_xy[inside], max_dist, d["length_px"].cpu().numpy(),
                                  batch["gt_length_px"][b].numpy()[inside], d["vessel_p"].cpu().numpy(),
                                  batch["gt_vessel"][b].numpy()[inside], cfg.decode.vessel_threshold)
            
            for k in tot:
                tot[k] += m[k]
            if "length_mae_px" in m:
                len_err.append((m["length_mae_px"], m["length_n"]))
            if "vessel_acc" in m:
                ves_hits += m["vessel_acc"] * m["vessel_n"]; ves_n += m["vessel_n"]

    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(val_loss=loss_sum / max(1, n_batches), precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)

    if len_err:
        w = sum(n for _, n in len_err)
        out["length_mae_px"] = sum(e * n for e, n in len_err) / w

    if ves_n:
        out["vessel_acc"] = ves_hits / ves_n
    model.train()

    return out

def save_ckpt(path: Path, model, opt, sched, scaler, epoch, best_f1, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best_f1=best_f1, config=cfg.to_dict()), path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)
    steps_per_epoch = len(train_loader) // cfg.train.accumulate_grad_batches
    opt, sched = build_optimizer(model, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.train.amp and device.type == "cuda")
    ckdir = Path(cfg.train.checkpoint_dir)
    start_epoch, best_f1 = 0, -1.0

    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_f1 = ck["epoch"] + 1, ck["best_f1"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    wb = None
    if cfg.train.wandb_project:
        try:
            wb = wandb.init(project=cfg.train.wandb_project, config=cfg.to_dict(), resume="allow")
        except Exception as e:
            print(f"wandb disabled: {e}")

    model.train()
    for epoch in range(start_epoch, cfg.train.epochs):
        t0, agg, n = time.time(), {}, 0
        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            
            with torch.autocast(device_type=device.type, enabled=cfg.train.amp):
                raw = model(batch["image"])

            loss, parts = circlenet_loss(raw.float(), batch, cfg)
            scaler.scale(loss / cfg.train.accumulate_grad_batches).backward()

            if (it + 1) % cfg.train.accumulate_grad_batches == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

            if it % 50 == 0:
                print(f"ep {epoch} it {it}/{len(train_loader)} " + " ".join(f"{k}={v:.4f}" for k, v in parts.items()))
        train_log = {f"train/{k}": v / n for k, v in agg.items()}
        train_log["lr"] = sched.get_last_lr()[-1]

        if (epoch + 1) % cfg.train.eval_every == 0:
            ev = evaluate(model, val_loader, cfg, device)
            print(f"[epoch {epoch}] {time.time() - t0:.0f}s " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
            if ev["f1"] > best_f1:
                best_f1 = ev["f1"]
                save_ckpt(ckdir / "best.pt", model, opt, sched, scaler, epoch, best_f1, cfg)
                print(f"  new best f1={best_f1:.4f} -> {ckdir / 'best.pt'}")
            train_log.update({f"val/{k}": v for k, v in ev.items()})
        save_ckpt(ckdir / "last.pt", model, opt, sched, scaler, epoch, best_f1, cfg)

        if wb:
            wb.log(train_log, step=epoch)

if __name__ == "__main__":
    main()