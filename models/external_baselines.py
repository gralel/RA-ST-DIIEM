"""External receiver-time denoising baselines for the fixed 11x869 TEM benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TEM1DFormer869(nn.Module):
    """Temporal-transformer baseline adapted from the previous TEM1DFormer869 code."""

    def __init__(
        self,
        input_channels: int = 11,
        embed_size: int = 96,
        num_heads: int = 4,
        transformer_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.embed_size = int(embed_size)
        self.num_heads = int(num_heads)
        self.transformer_depth = int(transformer_depth)
        self.dropout = float(dropout)

        self.conv1 = ConvBlock1D(self.input_channels, self.embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_size,
            nhead=self.num_heads,
            dim_feedforward=self.embed_size * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_depth)
        self.residual_blocks = nn.Sequential(
            ResidualBlock1D(self.embed_size),
            ResidualBlock1D(self.embed_size),
            ResidualBlock1D(self.embed_size),
            ResidualBlock1D(self.embed_size),
        )
        self.head = nn.Sequential(
            nn.Conv1d(self.embed_size, self.embed_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.embed_size, self.input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.transformer(h.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.residual_blocks(h)
        return self.head(h)


class DnCNN2DReceiverTime(nn.Module):
    """Generic 2D residual denoiser for receiver-time matrices."""

    def __init__(
        self,
        hidden_channels: int = 64,
        depth: int = 8,
        kernel_size: int = 3,
        residual_prediction: bool = True,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.residual_prediction = bool(residual_prediction)
        padding = self.kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(1, self.hidden_channels, kernel_size=self.kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(self.depth - 2, 0)):
            layers.extend(
                [
                    nn.Conv2d(
                        self.hidden_channels,
                        self.hidden_channels,
                        kernel_size=self.kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm2d(self.hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(self.hidden_channels, 1, kernel_size=self.kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, R, T]
        correction = self.net(x.unsqueeze(1)).squeeze(1)
        if self.residual_prediction:
            return x + correction
        return correction


def build_external_baseline(name: str, model_config: dict | None = None) -> nn.Module:
    cfg = dict(model_config or {})
    if name == "tem1dformer869":
        return TEM1DFormer869(
            input_channels=int(cfg.get("input_channels", cfg.get("num_receivers", 11))),
            embed_size=int(cfg.get("embed_size", 96)),
            num_heads=int(cfg.get("num_heads", 4)),
            transformer_depth=int(cfg.get("transformer_depth", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
    if name == "dncnn2d":
        return DnCNN2DReceiverTime(
            hidden_channels=int(cfg.get("hidden_channels", 64)),
            depth=int(cfg.get("depth", 8)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            residual_prediction=bool(cfg.get("residual_prediction", True)),
        )
    raise KeyError(f"Unknown external baseline: {name}")
