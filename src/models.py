"""Model zoo: a from-scratch U-Net plus pretrained-encoder variants (smp)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2, optionally followed by spatial dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Transposed-conv upsampling, skip concatenation, DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Classic U-Net (Ronneberger et al., 2015) with BatchNorm and 4 levels."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32):
        super().__init__()
        c = [base_channels * 2 ** i for i in range(5)]  # e.g. 32, 64, 128, 256, 512
        self.inc = DoubleConv(in_channels, c[0])
        self.downs = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(c[i], c[i + 1], dropout=0.1 if i == 3 else 0.0))
            for i in range(4)
        ])
        self.ups = nn.ModuleList([Up(c[i + 1], c[i], c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()  # bottleneck
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)  # raw logits


def build_model(model_cfg: dict) -> nn.Module:
    name = model_cfg["name"].lower()
    if name == "unet_scratch":
        return UNet(in_channels=3, num_classes=1, base_channels=int(model_cfg.get("base_channels", 32)))

    import segmentation_models_pytorch as smp

    architectures = {"smp_unet": smp.Unet, "smp_unetplusplus": smp.UnetPlusPlus, "smp_fpn": smp.FPN}
    if name not in architectures:
        raise ValueError(f"Unknown model '{name}'. Choose from unet_scratch, {', '.join(architectures)}")
    return architectures[name](
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=3,
        classes=1,
    )


def get_param_groups(model: nn.Module, lr: float, encoder_lr_mult: float = 1.0) -> list[dict]:
    """Optionally give a pretrained encoder a smaller learning rate than the decoder."""
    if encoder_lr_mult != 1.0 and hasattr(model, "encoder"):
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        encoder = [p for p in model.parameters() if id(p) in encoder_ids]
        rest = [p for p in model.parameters() if id(p) not in encoder_ids]
        return [
            {"params": encoder, "lr": lr * encoder_lr_mult, "name": "encoder"},
            {"params": rest, "lr": lr, "name": "decoder_head"},
        ]
    return [{"params": list(model.parameters()), "lr": lr, "name": "all"}]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
