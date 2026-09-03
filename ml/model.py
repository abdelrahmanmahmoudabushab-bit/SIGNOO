"""Sign classifier: a small Transformer encoder over landmark sequences.

Why this architecture:

A sign is a short sequence where the informative moment can be anywhere — the
handshape at the start, a direction change in the middle, a final contact. Self-
attention lets every frame consult every other frame directly, which suits that
better than a recurrent model compressing the clip left to right.

It is deliberately small (~1-3 M parameters). Sign datasets are small; a large model
memorises signers instead of learning signs. If accuracy on held-out signers is much
worse than on held-out repetitions, the answer is more signers, not more parameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positions, so the model knows frame order after resampling."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SignTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 192,
        nhead: int = 6,
        layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # more stable at this depth
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """x: (batch, frames, features) → logits (batch, classes)."""
        h = self.project(self.input_norm(x))
        h = self.encoder(self.pos(h))
        return self.head(h.mean(dim=1))       # average over time


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
