
from __future__ import annotations
import torch
import torch.nn.functional as F

def penalty_reduced_focal(logit: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logit).clamp(eps, 1 - eps)
    pos = (gt >= 1.0).float()
    neg = 1.0 - pos
    pos_loss = -torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = -torch.log(1 - p) * p ** alpha * (1 - gt) ** beta * neg
    n_pos = pos.sum().clamp(min=1.0)

    return (pos_loss.sum() + neg_loss.sum()) / n_pos

def masked_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    l = F.smooth_l1_loss(pred, target, reduction="none") * weight
    return l.sum() / weight.sum().clamp(min=1.0)

def smoothed_bce(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, smoothing: float = 0.05) -> torch.Tensor:

    t = target * (1 - smoothing) + 0.5 * smoothing
    l = F.binary_cross_entropy_with_logits(logit, t, reduction="none") * weight
    return l.sum() / weight.sum().clamp(min=1.0)

def binary_entropy(logit: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logit).clamp(eps, 1 - eps)
    h = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return (h * weight).sum() / weight.sum().clamp(min=1.0)

def circlenet_loss(raw: torch.Tensor, batch: dict, cfg) -> tuple[torch.Tensor, dict]:
    l_obj = penalty_reduced_focal(raw[:, 0:1], batch["heat"], cfg.loss.focal_alpha, cfg.loss.focal_beta)
    l_off = masked_l1(raw[:, 1:3], batch["offset"], batch["offset_mask"])
    l_len = masked_l1(raw[:, 3:4], batch["length"], batch["length_mask"])
    l_ves = smoothed_bce(raw[:, 4:5], batch["vessel"], batch["vessel_weight"], cfg.loss.label_smoothing)
    l_ent = binary_entropy(raw[:, 4:5], batch["vessel_unknown"])

    total = (cfg.loss.w_objectness * l_obj + cfg.loss.w_offset * l_off + cfg.loss.w_length * l_len + cfg.loss.w_vessel * l_ves + cfg.loss.w_entropy * l_ent)
    parts = dict(objectness=l_obj.item(), offset=l_off.item(), length=l_len.item(), vessel=l_ves.item(), entropy=l_ent.item(), total=total.item())

    return total, parts