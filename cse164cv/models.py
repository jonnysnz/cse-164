"""From-scratch segmentation models.

No torchvision model constructors, pretrained weights, or public checkpoints
are used here. The U-Net below is initialized only with PyTorch random
initializers and trained on the competition data.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .constants import NUM_CLASSES, NUM_SEG_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class SmallUNet(nn.Module):
    """A compact U-Net baseline with 301 segmentation output channels."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_SEG_CLASSES,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_conv = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.seg_head = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def decode(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x1, x2, x3, x4, x5 = features
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.seg_head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class MultiTaskUNet(SmallUNet):
    """Shared scratch U-Net encoder with segmentation and classification heads."""

    def __init__(
        self,
        in_channels: int = 3,
        num_seg_classes: int = NUM_SEG_CLASSES,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        classification_dropout: float = 0.2,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_classes=num_seg_classes,
            base_channels=base_channels,
        )
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_dropout = nn.Dropout(classification_dropout)
        self.cls_head = nn.Linear(base_channels * 16, num_classes)
        self.cls_head.apply(self._init_weights)

    def classify_features(self, bottleneck: torch.Tensor) -> torch.Tensor:
        pooled = self.cls_pool(bottleneck).flatten(1)
        return self.cls_head(self.cls_dropout(pooled))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decode(features), self.classify_features(features[-1])

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the shared encoder and classification head."""
        return self.classify_features(self.encode(x)[-1])


def build_model_from_config(config: dict[str, object]) -> nn.Module:
    """Construct a randomly initialized model matching a saved checkpoint config."""
    model_type = str(config.get("model_type", "small_unet"))
    base_channels = int(config.get("base_channels", 32))
    num_seg_classes = int(config.get("num_seg_classes", NUM_SEG_CLASSES))
    if model_type == "small_unet":
        return SmallUNet(num_classes=num_seg_classes, base_channels=base_channels)
    if model_type == "multitask_unet":
        return MultiTaskUNet(
            num_seg_classes=num_seg_classes,
            num_classes=int(config.get("num_classes", NUM_CLASSES)),
            base_channels=base_channels,
            classification_dropout=float(config.get("classification_dropout", 0.2)),
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def split_model_output(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return segmentation logits and optional classification logits."""
    if isinstance(output, tuple):
        return output
    return output, None
