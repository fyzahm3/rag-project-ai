"""Torch device auto-detection for local mode (mps on Apple Silicon, cuda if available, else cpu)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> str:
    """Resolve 'auto'/'cpu'/'cuda'/'mps' to a concrete torch device string.

    Falls back to 'cpu' if torch isn't importable yet (device is only actually used
    once a model is loaded, at which point torch is guaranteed to be present).
    """
    if preference != "auto":
        return preference
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
