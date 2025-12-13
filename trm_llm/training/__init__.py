"""Training components for TRM-LLM"""

from .trainer import TRMTrainer
from .loss import compute_trm_loss

__all__ = ['TRMTrainer', 'compute_trm_loss']
