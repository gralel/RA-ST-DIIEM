"""Multi-receiver PC-BiLSTM baseline for direct 11x869 prediction.

This variant keeps the original PC-BiLSTM backbone idea, but feeds all 11
receivers into the network jointly as input channels. Unlike the independent
single-receiver baseline, this model can learn cross-receiver correlations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .single_receiver_pcbilstm import AttentionBlock, ConvolutionalModule


class MultiReceiverPCBiLSTM(nn.Module):
    """Joint multi-receiver PC-BiLSTM.

    Input:  [B, 11, 869]
    Output: [B, 11, 869]
    """

    def __init__(
        self,
        seq_len: int = 869,
        num_receivers: int = 11,
        filters: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_receivers = int(num_receivers)
        self.filters = int(filters)
        self.lstm_hidden = int(lstm_hidden)
        self.lstm_layers = int(lstm_layers)
        self.num_heads = int(num_heads)

        self.branch3 = ConvolutionalModule(self.num_receivers, self.filters, kernel_size=3)
        self.branch5 = ConvolutionalModule(self.num_receivers, self.filters, kernel_size=5)
        self.branch7 = ConvolutionalModule(self.num_receivers, self.filters, kernel_size=7)

        merged_dim = 4 * self.filters * 3
        self.attn_merge = AttentionBlock(merged_dim, num_heads=self.num_heads)
        self.bilstm = nn.LSTM(
            input_size=merged_dim,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * self.lstm_hidden, self.num_receivers * self.seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"MultiReceiverPCBiLSTM expects [B, R, T], got {tuple(x.shape)}")
        batch_size, num_receivers, seq_len = x.shape
        if num_receivers != self.num_receivers:
            raise ValueError(
                f"Expected {self.num_receivers} receivers, but input has {num_receivers}"
            )
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, but input has {seq_len}")

        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        merged = torch.cat([b3, b5, b7], dim=1)
        merged = merged.permute(0, 2, 1)
        merged = self.attn_merge(merged)
        lstm_out, _ = self.bilstm(merged)
        out = self.fc(lstm_out[:, -1, :])
        return out.reshape(batch_size, self.num_receivers, self.seq_len)
