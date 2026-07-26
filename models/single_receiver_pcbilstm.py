"""Single-receiver PC-BiLSTM baseline adapted for independent 11x869 prediction.

This module keeps the original single-trace PC-BiLSTM structure and wraps it so
an input with shape [B, 11, 869] is processed as 11 independent traces. There is
no spatial interaction between receivers; outputs are simply stacked back to
[B, 11, 869].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class ConvolutionalModule(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size, padding=padding)
        self.attn1 = AttentionBlock(filters)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(filters, 2 * filters, kernel_size, padding=padding)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(2 * filters, 4 * filters, kernel_size, padding=padding)
        self.pool3 = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        x = self.attn1(x)
        x = x.permute(0, 2, 1)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        return x


class PCBiLSTMTraceNet(nn.Module):
    """Original single-trace PC-BiLSTM with configurable sequence length."""

    def __init__(
        self,
        input_length: int = 869,
        in_channels: int = 1,
        filters: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.in_channels = int(in_channels)
        self.filters = int(filters)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.num_heads = int(num_heads)

        self.branch3 = ConvolutionalModule(in_channels, filters, kernel_size=3)
        self.branch5 = ConvolutionalModule(in_channels, filters, kernel_size=5)
        self.branch7 = ConvolutionalModule(in_channels, filters, kernel_size=7)

        merged_dim = 4 * filters * 3
        self.attn_merge = AttentionBlock(merged_dim, num_heads=num_heads)
        self.bilstm = nn.LSTM(
            input_size=merged_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * lstm_hidden, input_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"PCBiLSTMTraceNet expects [B, 1, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"PCBiLSTMTraceNet expects {self.in_channels} input channel(s), got {x.shape[1]}"
            )
        if x.shape[2] != self.input_length:
            raise ValueError(
                f"PCBiLSTMTraceNet expects sequence length {self.input_length}, got {x.shape[2]}"
            )

        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        merged = torch.cat([b3, b5, b7], dim=1)
        merged = merged.permute(0, 2, 1)
        merged = self.attn_merge(merged)
        lstm_out, _ = self.bilstm(merged)
        out = self.fc(lstm_out[:, -1, :])
        return out.unsqueeze(1)


@dataclass(frozen=True)
class IndependentForwardInfo:
    batch_size: int
    num_receivers: int
    seq_len: int


class IndependentReceiverPCBiLSTM(nn.Module):
    """Apply the original single-receiver PC-BiLSTM independently to each receiver.

    Input:  [B, 11, 869]
    Output: [B, 11, 869]

    The same single-trace network is shared across all receivers. Receiver outputs
    are concatenated back in their original order, with no cross-receiver mixing.
    """

    def __init__(
        self,
        seq_len: int = 869,
        num_receivers: int | None = 11,
        filters: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_receivers = None if num_receivers is None else int(num_receivers)
        self.trace_model = PCBiLSTMTraceNet(
            input_length=seq_len,
            in_channels=1,
            filters=filters,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            num_heads=num_heads,
        )

    def _flatten_receivers(self, x: torch.Tensor) -> tuple[torch.Tensor, IndependentForwardInfo]:
        if x.ndim != 3:
            raise ValueError(
                f"IndependentReceiverPCBiLSTM expects [B, R, T], got {tuple(x.shape)}"
            )
        batch_size, num_receivers, seq_len = x.shape
        if self.num_receivers is not None and num_receivers != self.num_receivers:
            raise ValueError(
                f"Expected {self.num_receivers} receivers, but input has {num_receivers}"
            )
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, but input has {seq_len}")
        flattened = x.reshape(batch_size * num_receivers, 1, seq_len)
        return flattened, IndependentForwardInfo(batch_size, num_receivers, seq_len)

    @staticmethod
    def _restore_receivers(pred_flat: torch.Tensor, info: IndependentForwardInfo) -> torch.Tensor:
        if pred_flat.ndim != 2:
            raise ValueError(f"Expected flattened prediction [B*R, T], got {tuple(pred_flat.shape)}")
        return pred_flat.reshape(info.batch_size, info.num_receivers, info.seq_len)

    def forward_single_traces(self, traces: torch.Tensor) -> torch.Tensor:
        """Run the shared single-receiver network on [B*R, 1, T]."""
        return self.trace_model(traces)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat_x, info = self._flatten_receivers(x)
        flat_pred = self.forward_single_traces(flat_x).squeeze(1)
        return self._restore_receivers(flat_pred, info)
