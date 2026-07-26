"""
Spatial Receiver PC-BiLSTM for multi-receiver TEM denoising.

Input:
    x: [B, R, T] where R=11, T=869
Output:
    y_hat: [B, R, T]

This model is a spatial extension of MultiReceiver PC-BiLSTM:
1. Shared per-receiver parallel temporal convolution extracts temporal features.
2. Receiver-channel attention explicitly exchanges information among receivers.
3. Shared BiLSTM decodes temporal features for each receiver.
4. Residual prediction is used by default in log10 amplitude space.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def build_chain_adjacency(num_receivers: int, sigma: float = 2.0) -> torch.Tensor:
    """
    Build a default receiver relation matrix from receiver index distance.

    A_ij = exp(-(i-j)^2 / (2*sigma^2))

    Use this when receiver offsets/coordinates are not available.
    If you have real receiver offsets or a clean-data correlation matrix,
    pass it through graph_matrix in SpatialReceiverPCBiLSTM instead.
    """
    idx = torch.arange(num_receivers, dtype=torch.float32)
    dist2 = (idx[:, None] - idx[None, :]).pow(2)
    A = torch.exp(-dist2 / (2.0 * sigma * sigma))
    A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
    return A


class ParallelTemporalConv(nn.Module):
    """
    Shared parallel temporal convolution for each receiver.

    Input:
        x: [B*R, 1, T]
    Output:
        h: [B*R, C, T]
    """

    def __init__(
        self,
        in_channels: int = 1,
        filters: int = 16,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 11),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(filters),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                )
            )

        self.out_channels = filters * len(kernel_sizes)

        self.fuse = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm1d(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        h = torch.cat(feats, dim=1)
        return self.fuse(h)


class GraphBiasedReceiverAttention(nn.Module):
    """
    Receiver-channel self-attention with optional graph/adjacency bias.

    Input:
        h: [B, T, R, D]
    Output:
        h: [B, T, R, D]

    Attention is computed over receiver dimension R for each time step.
    Optional graph_matrix A [R, R] is injected as:
        score_ij = q_i k_j / sqrt(d) + beta * log(A_ij + eps)
    """

    def __init__(
        self,
        dim: int,
        num_receivers: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1,
        graph_matrix: Optional[torch.Tensor] = None,
        use_graph_bias: bool = True,
        learnable_graph_scale: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_receivers = num_receivers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_graph_bias = use_graph_bias

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        if graph_matrix is None:
            graph_matrix = build_chain_adjacency(num_receivers)
        graph_matrix = torch.as_tensor(graph_matrix, dtype=torch.float32)
        if graph_matrix.shape != (num_receivers, num_receivers):
            raise ValueError(
                f"graph_matrix must have shape {(num_receivers, num_receivers)}, "
                f"got {tuple(graph_matrix.shape)}"
            )

        graph_matrix = graph_matrix.clamp_min(1e-6)
        graph_matrix = graph_matrix / (graph_matrix.sum(dim=-1, keepdim=True) + 1e-8)
        self.register_buffer("graph_log_bias", torch.log(graph_matrix + 1e-6))

        if learnable_graph_scale:
            self.graph_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.register_buffer("graph_scale", torch.tensor(0.5, dtype=torch.float32))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, R, D]
        B, T, R, D = h.shape
        if R != self.num_receivers:
            raise ValueError(f"Expected R={self.num_receivers}, got {R}")
        if D != self.dim:
            raise ValueError(f"Expected D={self.dim}, got {D}")

        x = self.norm1(h)
        x = x.reshape(B * T, R, D)

        qkv = self.qkv(x)
        qkv = qkv.reshape(B * T, R, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*T, H, R, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.use_graph_bias:
            scores = scores + self.graph_scale * self.graph_log_bias[None, None, :, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B*T, H, R, Hd]
        out = out.transpose(1, 2).reshape(B * T, R, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.reshape(B, T, R, D)

        h = h + out
        h = h + self.ffn(self.norm2(h))
        return h


class SpatialReceiverPCBiLSTM(nn.Module):
    """
    Spatial PC-BiLSTM for joint multi-receiver TEM denoising.

    Input:
        x: [B, R, T]
    Output:
        y_hat: [B, R, T]
    """

    def __init__(
        self,
        seq_len: int = 869,
        num_receivers: int = 11,
        filters: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        num_heads: int = 4,
        spatial_layers: int = 2,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 11),
        dropout: float = 0.1,
        graph_matrix: Optional[torch.Tensor] = None,
        use_graph_bias: bool = True,
        residual: bool = True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_receivers = num_receivers
        self.residual = residual

        self.temporal_pc = ParallelTemporalConv(
            in_channels=1,
            filters=filters,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )
        feat_dim = self.temporal_pc.out_channels

        self.receiver_embedding = nn.Parameter(
            torch.zeros(1, 1, num_receivers, feat_dim)
        )
        nn.init.normal_(self.receiver_embedding, mean=0.0, std=0.02)

        self.spatial_blocks = nn.ModuleList(
            [
                GraphBiasedReceiverAttention(
                    dim=feat_dim,
                    num_receivers=num_receivers,
                    num_heads=num_heads,
                    dropout=dropout,
                    graph_matrix=graph_matrix,
                    use_graph_bias=use_graph_bias,
                    learnable_graph_scale=True,
                )
                for _ in range(spatial_layers)
            ]
        )

        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, R, T]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [B, R, T], got {tuple(x.shape)}")

        B, R, T = x.shape
        if R != self.num_receivers:
            raise ValueError(f"Expected {self.num_receivers} receivers, got {R}")
        if T != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {T}")

        identity = x

        # Shared per-receiver PC feature extraction.
        # [B, R, T] -> [B*R, 1, T]
        h = x.reshape(B * R, 1, T)
        h = self.temporal_pc(h)  # [B*R, D, T]
        D = h.shape[1]

        # [B*R, D, T] -> [B, T, R, D]
        h = h.reshape(B, R, D, T).permute(0, 3, 1, 2).contiguous()

        # Add receiver identity.
        h = h + self.receiver_embedding

        # Explicit receiver-channel spatial modeling.
        for block in self.spatial_blocks:
            h = block(h)  # [B, T, R, D]

        # Shared temporal BiLSTM for each receiver after spatial aggregation.
        # [B, T, R, D] -> [B*R, T, D]
        h = h.permute(0, 2, 1, 3).contiguous().reshape(B * R, T, D)
        h, _ = self.bilstm(h)  # [B*R, T, 2H]

        delta = self.head(h).squeeze(-1)  # [B*R, T]
        delta = delta.reshape(B, R, T)

        if self.residual:
            return identity + delta
        return delta


if __name__ == "__main__":
    model = SpatialReceiverPCBiLSTM(
        seq_len=869,
        num_receivers=11,
        filters=16,
        lstm_hidden=32,
        lstm_layers=2,
        num_heads=4,
        spatial_layers=2,
    )
    x = torch.randn(2, 11, 869)
    y = model(x)
    print("input:", tuple(x.shape), "output:", tuple(y.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
