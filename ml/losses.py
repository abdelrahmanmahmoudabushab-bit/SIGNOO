"""Training losses and samplers beyond plain cross-entropy.

Supervised contrastive loss pulls clips of the same word together in embedding space and pushes
different words apart, *across signers in the same batch*. With a few dozen clips per word the
standard failure is memorising who signed rather than what was signed; here the positives for a
clip are, by construction, other signers' clips of the same word.
"""
from __future__ import annotations

from collections import Counter

import torch
from torch.nn import functional as F


def supervised_contrastive(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Khosla et al., "Supervised Contrastive Learning". Zero when no clip has a same-word partner."""
    z = F.normalize(embeddings, dim=-1)
    n = len(z)
    logits = z @ z.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    has_partner = positives.any(dim=1)
    if not has_partner.any():
        return embeddings.sum() * 0.0
    per_anchor = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1)) / positives.sum(dim=1).clamp_min(1)
    return per_anchor[has_partner].mean()


def class_balanced_weights(labels: list[int], power: float = 0.5) -> list[float]:
    """Per-clip sampling weights, 1 / count**power: 0 = natural frequency, 1 = fully balanced.

    Half-way (the default) keeps a sign with 140 clips from drowning one with 8 without making the
    8-clip sign repeat so often that it is memorised.
    """
    counts = Counter(labels)
    return [1.0 / counts[label] ** power for label in labels]
