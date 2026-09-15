"""Shared code between the kiosk runtime and the training pipeline.

Everything here is imported by both `kiosk/` and `ml/`, and nowhere else. The reason
this package exists at all: a sign-recognition model is only as good as the promise
that training-time and inference-time features are computed identically. Divergence
between the two is a classic, silent, expensive bug in this class of system — putting
the feature code in one place instead of two makes that divergence impossible instead
of merely unlikely.
"""
