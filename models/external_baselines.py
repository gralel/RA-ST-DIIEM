"""External receiver-time denoising baselines for the fixed 11x869 TEM benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .single_receiver_pcbilstm import IndependentReceiverPCBiLSTM
from .spatial_receiver_pcbilstm import SpatialReceiverPCBiLSTM


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


class TEMDnetStyle2D(nn.Module):
    """Signal-to-image TEMDnet-style adaptation for a receiver-time tensor.

    Each receiver trace is zero-padded to a square image, denoised by a shared
    residual 2-D CNN, and mapped back to the original time samples.  The model
    is intentionally described as an adaptation, rather than an exact
    reproduction, because the public TEMDnet demo uses a different data layout.
    """

    def __init__(self, seq_len: int = 869, hidden_channels: int = 64, depth: int = 8):
        super().__init__()
        self.seq_len = int(seq_len)
        self.side = int(self.seq_len**0.5)
        if self.side * self.side < self.seq_len:
            self.side += 1
        layers: list[nn.Module] = [nn.Conv2d(1, hidden_channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(max(int(depth) - 2, 0)):
            layers.extend([nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(hidden_channels, 1, 3, padding=1))
        self.noise_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, receivers, length = x.shape
        if length != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {length}")
        flat = x.reshape(batch * receivers, 1, length)
        padded = F.pad(flat, (0, self.side * self.side - length))
        image = padded.reshape(batch * receivers, 1, self.side, self.side)
        clean_image = image - self.noise_net(image)
        return clean_image.reshape(batch, receivers, -1)[..., :length]


class TEMNLnetStyle(nn.Module):
    """Supervised TEM-NLnet-inspired noise-prediction comparison.

    The network learns the additive disturbance in receiver-time space and
    subtracts it from the observation. It does not reproduce the published
    method's GAN-based measured-noise learning stage; it uses the same paired
    synthetic training protocol as the other benchmark methods.
    """

    def __init__(self, hidden_channels: int = 64, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, hidden_channels, 3, padding=1), nn.LeakyReLU(0.1, inplace=True)]
        for _ in range(max(int(depth) - 2, 0)):
            layers.extend([nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.InstanceNorm2d(hidden_channels, affine=True), nn.LeakyReLU(0.1, inplace=True)])
        layers.append(nn.Conv2d(hidden_channels, 1, 3, padding=1))
        self.noise_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        predicted_noise = self.noise_net(x.unsqueeze(1)).squeeze(1)
        return x - predicted_noise


class JointGatedMoEPCBiLSTM(nn.Module):
    """Jointly trained single/spatial PC-BiLSTM experts with a learned gate."""

    def __init__(
        self,
        seq_len: int = 869,
        num_receivers: int = 11,
        filters: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        num_heads: int = 4,
        spatial_layers: int = 2,
        dropout: float = 0.1,
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.independent_expert = IndependentReceiverPCBiLSTM(
            seq_len=seq_len, num_receivers=num_receivers, filters=filters,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, num_heads=num_heads,
        )
        self.spatial_expert = SpatialReceiverPCBiLSTM(
            seq_len=seq_len, num_receivers=num_receivers, filters=filters,
            lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, num_heads=num_heads,
            spatial_layers=spatial_layers, dropout=dropout,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(6, gate_hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(gate_hidden, gate_hidden, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(gate_hidden, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        independent = self.independent_expert(x)
        spatial = self.spatial_expert(x)
        gate_input = torch.stack(
            (x, independent, spatial, independent - spatial, (x - independent).abs(), (x - spatial).abs()),
            dim=2,
        )
        batch, receivers, channels, length = gate_input.shape
        weight = self.gate(gate_input.reshape(batch * receivers, channels, length))
        weight = weight.reshape(batch, receivers, 1)
        return spatial + weight * (independent - spatial)


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
    if name == "temdnet_style":
        return TEMDnetStyle2D(
            seq_len=int(cfg.get("seq_len", 869)),
            hidden_channels=int(cfg.get("hidden_channels", 64)),
            depth=int(cfg.get("depth", 8)),
        )
    if name == "temnlnet_style":
        return TEMNLnetStyle(
            hidden_channels=int(cfg.get("hidden_channels", 64)),
            depth=int(cfg.get("depth", 8)),
        )
    if name == "joint_gated_moe":
        return JointGatedMoEPCBiLSTM(
            seq_len=int(cfg.get("seq_len", 869)),
            num_receivers=int(cfg.get("num_receivers", 11)),
            filters=int(cfg.get("filters", 16)),
            lstm_hidden=int(cfg.get("lstm_hidden", 32)),
            lstm_layers=int(cfg.get("lstm_layers", 2)),
            num_heads=int(cfg.get("num_heads", 4)),
            spatial_layers=int(cfg.get("spatial_layers", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            gate_hidden=int(cfg.get("gate_hidden", 32)),
        )
    raise KeyError(f"Unknown external baseline: {name}")
