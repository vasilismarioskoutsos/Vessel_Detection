from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

@dataclass
class MatchResult:
    pairs: list[tuple[int, int]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)

def match_points(pred_xy: np.ndarray, gt_xy: np.ndarray, max_dist: float) -> MatchResult:
    n, m = len(pred_xy), len(gt_xy)

    if n == 0 or m == 0:
        return MatchResult([], list(range(n)), list(range(m)))

    tree = cKDTree(gt_xy)
    cand = tree.query_ball_point(pred_xy, r=max_dist)
    pred_with_cand = [i for i, c in enumerate(cand) if c]

    if not pred_with_cand:
        return MatchResult([], list(range(n)), list(range(m)))

    gt_with_cand = sorted({j for i in pred_with_cand for j in cand[i]})
    gi = {j: k for k, j in enumerate(gt_with_cand)}
    BIG = 1e6
    cost = np.full((len(pred_with_cand), len(gt_with_cand)), BIG, np.float64)

    for a, i in enumerate(pred_with_cand):
        for j in cand[i]:
            cost[a, gi[j]] = np.linalg.norm(pred_xy[i] - gt_xy[j])

    rows, cols = linear_sum_assignment(cost)
    pairs = [(pred_with_cand[r], gt_with_cand[c]) for r, c in zip(rows, cols) if cost[r, c] < BIG]
    mp = {p for p, _ in pairs}
    mg = {g for _, g in pairs}

    return MatchResult(pairs, [i for i in range(n) if i not in mp], [j for j in range(m) if j not in mg])

def detection_metrics(pred_xy, gt_xy, max_dist, pred_length=None, gt_length=None, pred_vessel_p=None, gt_vessel=None, vessel_threshold=0.5) -> dict:
    mr = match_points(np.asarray(pred_xy, float), np.asarray(gt_xy, float), max_dist)
    tp, fp, fn = len(mr.pairs), len(mr.unmatched_pred), len(mr.unmatched_gt)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)

    if mr.pairs and pred_length is not None and gt_length is not None:
        p = np.array([pred_length[i] for i, _ in mr.pairs], float)
        g = np.array([gt_length[j] for _, j in mr.pairs], float)
        ok = np.isfinite(g) & (g > 0)
        if ok.any():
            out["length_mae_px"] = float(np.mean(np.abs(p[ok] - g[ok])))
            out["length_rel_err"] = float(np.mean(np.abs(p[ok] - g[ok]) / g[ok]))
            out["length_n"] = int(ok.sum())

    if mr.pairs and pred_vessel_p is not None and gt_vessel is not None:
        p = np.array([pred_vessel_p[i] for i, _ in mr.pairs], float) >= vessel_threshold
        g = np.array([gt_vessel[j] for _, j in mr.pairs], float)
        ok = np.isfinite(g)
        if ok.any():
            out["vessel_acc"] = float(np.mean(p[ok] == (g[ok] >= 0.5)))
            out["vessel_n"] = int(ok.sum())
            
    return out
