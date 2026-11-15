"""One wrapper per modality, all sharing the ModalityScore interface."""

from .base import ModalityScore, artifacts_dir

__all__ = ["ModalityScore", "artifacts_dir"]
