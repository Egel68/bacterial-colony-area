import torch.nn as nn

from . import register_model
from .base import BaseSegmenter
from .unet import DoubleConv, Down, Up


@register_model("unet_small")
class UNetSmall(BaseSegmenter):
    """Уменьшенная U-Net (контрольный вариант для сравнения архитектур).

    Та же структура down/up блоков, что в `UNet`, но с уменьшенным числом
    каналов по умолчанию — быстрее обучается и требует меньше памяти на GPU.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        features: tuple = (32, 64, 128, 256),
    ):
        super().__init__()
        self.inc = DoubleConv(n_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        self.down4 = Down(features[3], features[3])
        self.up1 = Up(features[3] + features[3], features[2])
        self.up2 = Up(features[2] + features[2], features[1])
        self.up3 = Up(features[1] + features[1], features[0])
        self.up4 = Up(features[0] + features[0], features[0])
        self.outc = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)