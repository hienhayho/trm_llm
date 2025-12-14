"""Training components for TRM-LLM"""

from trm_llm.training.trainer import TRMTrainer
from trm_llm.training.loss import compute_trm_loss

__all__ = ['TRMTrainer', 'compute_trm_loss']
