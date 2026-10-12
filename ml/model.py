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


class AttentionPool(nn.Module):
    """Learned weighted average over frames, instead of a plain mean.

    A sign's informative moment is a few frames (the handshape at the start, the contact at the
    end); a plain mean dilutes them with transitions and idle frames. The weights are learned.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):
        weights = torch.softmax(self.score(h), dim=1)
        return (weights * h).sum(dim=1)


class SignNet(nn.Module):
    """Deeper sign classifier: conv stem for short motions, Transformer, attention pooling.

    Over `SignTransformer` it adds (1) a residual 1-D convolution before attention, which reads
    local motion (a flick, a tap) that attention otherwise has to assemble from single frames;
    (2) attention pooling; (3) `embed()`, the clip embedding that the supervised contrastive loss
    and the pretraining head use; (4) an optional text-embedding head: with `text_dim` set, class
    scores are cosine similarities between the clip embedding and fixed per-word text embeddings
    (stored as the `class_text` buffer), so words with few clips borrow structure from
    semantically related words and new words can be ranked without a new output unit.
    """

    def __init__(self, input_dim: int, num_classes: int, d_model: int = 192, nhead: int = 6,
                 layers: int = 4, dim_feedforward: int = 384, dropout: float = 0.3,
                 conv_kernel: int = 5, text_dim: int = 0):
        super().__init__()
        self.d_model, self.text_dim = d_model, text_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, d_model)
        self.stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2),
        )
        self.stem_norm = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.pool = AttentionPool(d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        if text_dim:
            self.to_text = nn.Linear(d_model, text_dim)
            self.register_buffer("class_text", torch.zeros(num_classes, text_dim))
            self.logit_scale = nn.Parameter(torch.tensor(10.0))
        else:
            self.head = nn.Linear(d_model, num_classes)

    def encode(self, x):
        """(batch, frames, features) -> per-frame states (batch, frames, d_model)."""
        h = self.project(self.input_norm(x))
        h = h + self.stem(h.transpose(1, 2)).transpose(1, 2)
        return self.encoder(self.pos(self.stem_norm(h)))

    def embed(self, x):
        return self.embed_norm(self.pool(self.encode(x)))

    def classify(self, e):
        e = self.drop(e)
        if self.text_dim:
            z = nn.functional.normalize(self.to_text(e), dim=-1)
            t = nn.functional.normalize(self.class_text, dim=-1)
            return self.logit_scale * z @ t.T
        return self.head(e)

    def forward(self, x):
        return self.classify(self.embed(x))


class Ensemble(nn.Module):
    """Averages the members' probabilities. Returns log-probabilities, so a softmax over the
    output gives back the averaged distribution and every consumer of `model(x)` still works."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x):
        probs = torch.stack([torch.softmax(m(x), dim=-1) for m in self.members]).mean(dim=0)
        return torch.log(probs.clamp_min(1e-9))


ARCHITECTURES = {"transformer": SignTransformer, "signnet": SignNet}


def build_model(ckpt: dict):
    """Rebuilds the exact network a checkpoint was saved from, weights loaded, in eval mode.

    The one place that turns a checkpoint into a model: training, evaluation, calibration, the
    server's shadow scoring and the kiosk all use it, so none can build a different network from
    the one that was trained. Old checkpoints have no `arch` and are the original transformer.
    """
    if "members" in ckpt:
        return Ensemble([build_model(m) for m in ckpt["members"]]).eval()
    arch = ckpt.get("arch", "transformer")
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; known: {sorted(ARCHITECTURES)}")
    net = ARCHITECTURES[arch](input_dim=ckpt["input_dim"], num_classes=len(ckpt["labels"]),
                              **ckpt.get("config", {}))
    net.load_state_dict(ckpt["model"])
    return net.eval()
