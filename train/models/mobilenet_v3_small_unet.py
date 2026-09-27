"""Компактная сегментация на MobileNetV3-Small с лёгким skip decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from . import register_model
from .base import BaseSegmenter


class _FuseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


@register_model("mobilenet_v3_small_unet")
class MobileNetV3SmallUNet(BaseSegmenter):
    """MobileNetV3-Small encoder и узкий U-Net-подобный декодер.

    Вход — нормализованный RGB-тензор `[N,3,H,W]`, выход — одноканальные logits.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        pretrained: bool = True,
        decoder_channels: tuple[int, int, int, int] = (32, 24, 16, 12),
    ):
        super().__init__()
        if n_channels != 3:
            raise ValueError("MobileNetV3-Small pretrained encoder requires 3 channels")
        if n_classes != 1:
            raise ValueError("This segmenter supports binary output (n_classes=1)")

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features

        # torchvision MobileNetV3-Small outputs at feature indices 0/3/8/12:
        # 16@stride2, 24@stride8, 48@stride16, 576@stride32.
        c2, c8, c16, c32 = decoder_channels
        self.lateral2 = nn.Sequential(
            nn.Conv2d(16, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True)
        )
        self.lateral8 = nn.Sequential(
            nn.Conv2d(24, c8, 1, bias=False), nn.BatchNorm2d(c8), nn.ReLU(inplace=True)
        )
        self.lateral16 = nn.Sequential(
            nn.Conv2d(48, c16, 1, bias=False),
            nn.BatchNorm2d(c16),
            nn.ReLU(inplace=True),
        )
        self.lateral32 = nn.Sequential(
            nn.Conv2d(576, c32, 1, bias=False),
            nn.BatchNorm2d(c32),
            nn.ReLU(inplace=True),
        )
        self.fuse16 = _FuseBlock(c32 + c16, c16)
        self.fuse8 = _FuseBlock(c16 + c8, c8)
        self.fuse2 = _FuseBlock(c8 + c2, c2)
        self.refine = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(c2, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x2 = x8 = x16 = None
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index == 0:
                x2 = x
            elif index == 3:
                x8 = x
            elif index == 8:
                x16 = x

        x = self.lateral32(x)
        x = self.fuse16(x, self.lateral16(x16))
        x = self.fuse8(x, self.lateral8(x8))
        x = self.fuse2(x, self.lateral2(x2))
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.outc(self.refine(x))
