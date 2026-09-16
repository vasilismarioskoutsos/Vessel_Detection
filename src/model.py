from __future__ import annotations
import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_CHANNELS = 5

def conv_gn_act(cin: int, cout: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout),
        nn.SiLU(inplace=True))

class DecoderBlock(nn.Module):
    """Upsample x2, concatenate the skip, two conv blocks"""

    def __init__(self, cin: int, cskip: int, cout: int, groups: int):
        super().__init__()
        self.conv1 = conv_gn_act(cin + cskip, cout, groups)
        self.conv2 = conv_gn_act(cout, cout, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))

class Head(nn.Module):

    def __init__(self, cin: int, mid: int, cout: int, groups: int):
        super().__init__()
        self.body = conv_gn_act(cin, mid, groups)
        self.out = nn.Conv2d(mid, cout * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.cout = cout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.out(self.body(x)))

    def set_output_bias(self, value: float) -> None:
        with torch.no_grad():
            self.out.bias.fill_(value)

class CircleNet(nn.Module):
    def __init__(self, encoder: str = "tf_efficientnetv2_s.in21k_ft_in1k", pretrained: bool = True, in_chans: int = 2, decoder_channels=(256, 128, 64, 48), head_channels: int = 64, groupnorm_groups: int = 8, objectness_prior: float = 0.01):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, in_chans=in_chans, out_indices=(0, 1, 2, 3, 4))
        chs = self.encoder.feature_info.channels()
        reds = self.encoder.feature_info.reduction()
        assert reds[-1] == 32 and reds[1] == 4, f"unexpected encoder strides {reds}"

        d = list(decoder_channels)[:3]
        self.dec16 = DecoderBlock(chs[4], chs[3], d[0], groupnorm_groups)
        self.dec8 = DecoderBlock(d[0], chs[2], d[1], groupnorm_groups)
        self.dec4 = DecoderBlock(d[1], chs[1], d[2], groupnorm_groups)

        self.head_obj = Head(d[2], head_channels, 1, groupnorm_groups)
        self.head_off = Head(d[2], head_channels, 2, groupnorm_groups)
        self.head_len = Head(d[2], head_channels, 1, groupnorm_groups)
        self.head_ves = Head(d[2], head_channels, 1, groupnorm_groups)

        self.head_obj.set_output_bias(math.log(objectness_prior / (1 - objectness_prior)))
        self.head_off.set_output_bias(0.5)
        self.head_len.set_output_bias(math.log1p(8.0))
        self.head_ves.set_output_bias(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f4, f8, f16, f32 = self.encoder(x)
        y = self.dec16(f32, f16)
        y = self.dec8(y, f8)
        y = self.dec4(y, f4)
        return torch.cat([self.head_obj(y), self.head_off(y), self.head_len(y), self.head_ves(y)], dim=1)

    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self):
        enc_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

def activate(raw: torch.Tensor) -> torch.Tensor:
    out = raw.clone()
    out[:, 0] = torch.sigmoid(raw[:, 0])
    out[:, 4] = torch.sigmoid(raw[:, 4])
    return out

@torch.no_grad()
def decode_torch(act: torch.Tensor, stride: int, peak_kernel: int, objectness_threshold: float, log_length: bool, max_detections: int = 20000) -> list[dict]:
    heat = act[:, 0:1]
    pooled = F.max_pool2d(heat, peak_kernel, stride=1, padding=peak_kernel // 2)
    peaks = (heat >= pooled) & (heat > objectness_threshold)

    results = []
    for b in range(act.shape[0]):
        ys, xs = torch.nonzero(peaks[b, 0], as_tuple=True)
        score = heat[b, 0, ys, xs]
        if len(score) > max_detections:
            top = torch.topk(score, max_detections).indices
            ys, xs, score = ys[top], xs[top], score[top]

        col = (xs.float() + act[b, 1, ys, xs]) * stride
        row = (ys.float() + act[b, 2, ys, xs]) * stride
        L = act[b, 3, ys, xs]
        L = torch.expm1(L) if log_length else L
        results.append(dict(col=col, row=row, score=score, length_px=L.clamp(min=0), vessel_p=act[b, 4, ys, xs]))

    return results