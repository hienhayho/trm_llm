"""Pretrained weights loader for TRM-LLM

Loads pretrained embeddings and LM head from models like Qwen, LLaMA, etc.
Freezes these components so only TRM-specific modules are trained.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from trm_llm.utils.logger import log


def load_pretrained_weights(
    model_name_or_path: str,
    device: str = "cpu",
    trust_remote_code: bool = True,
    new_vocab_size: Optional[int] = None
) -> Tuple[nn.Embedding, nn.Linear, int, int]:
    """Load pretrained token embeddings and LM head from a causal LM

    Args:
        model_name_or_path: HuggingFace model name or path (e.g., "Qwen/Qwen2.5-3B")
        device: Device to load weights to
        trust_remote_code: Whether to trust remote code for custom models
        new_vocab_size: If provided, resize embeddings to this size (for added special tokens)

    Returns:
        token_embedding: Pretrained token embedding layer
        lm_head: Pretrained LM head (output projection)
        vocab_size: Vocabulary size (original or new_vocab_size if provided)
        hidden_dim: Hidden dimension of the pretrained model
    """
    # Load pretrained model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,  # Use float32 for training
        device_map=device if device != "cpu" else None,
    )

    # Extract embeddings and LM head
    # Different models have different attribute names
    if hasattr(pretrained_model, 'model'):
        # Qwen, LLaMA style
        embed_tokens = pretrained_model.model.embed_tokens
    elif hasattr(pretrained_model, 'transformer'):
        # GPT-2 style
        embed_tokens = pretrained_model.transformer.wte
    else:
        raise ValueError(f"Unknown model architecture: {type(pretrained_model)}")

    if hasattr(pretrained_model, 'lm_head'):
        lm_head = pretrained_model.lm_head
    elif hasattr(pretrained_model, 'transformer') and hasattr(pretrained_model, 'lm_head'):
        lm_head = pretrained_model.lm_head
    else:
        raise ValueError(f"Cannot find lm_head in model: {type(pretrained_model)}")

    original_vocab_size = embed_tokens.num_embeddings
    hidden_dim = embed_tokens.embedding_dim

    # Determine final vocab size
    final_vocab_size = new_vocab_size if new_vocab_size else original_vocab_size
    resizing = new_vocab_size and new_vocab_size != original_vocab_size

    # Clone the weights (don't share memory with original model)
    token_embedding = nn.Embedding(final_vocab_size, hidden_dim)
    # Copy original weights
    copy_size = min(original_vocab_size, final_vocab_size)
    token_embedding.weight.data[:copy_size] = embed_tokens.weight.data[:copy_size].clone().to(device)
    # Initialize new tokens with mean of existing embeddings if vocab expanded
    if final_vocab_size > original_vocab_size:
        mean_embed = embed_tokens.weight.data.mean(dim=0)
        for i in range(original_vocab_size, final_vocab_size):
            token_embedding.weight.data[i] = mean_embed + torch.randn_like(mean_embed) * 0.02

    output_lm_head = nn.Linear(hidden_dim, final_vocab_size, bias=False)
    # Copy original weights
    output_lm_head.weight.data[:copy_size] = lm_head.weight.data[:copy_size].clone().to(device)
    # Initialize new tokens with small random values if vocab expanded
    if final_vocab_size > original_vocab_size:
        for i in range(original_vocab_size, final_vocab_size):
            output_lm_head.weight.data[i] = torch.randn(hidden_dim) * 0.02

    # Clean up
    del pretrained_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    load_info = {
        "model": model_name_or_path,
        "original_vocab_size": original_vocab_size,
        "final_vocab_size": final_vocab_size,
        "hidden_dim": hidden_dim,
    }
    if resizing:
        load_info["resized"] = True
    log("Pretrained weights loaded", **load_info)

    return token_embedding, output_lm_head, final_vocab_size, hidden_dim


def get_pretrained_tokenizer(model_name_or_path: str, trust_remote_code: bool = True):
    """Get tokenizer from pretrained model

    Args:
        model_name_or_path: HuggingFace model name or path

    Returns:
        tokenizer: HuggingFace tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code
    )
    return tokenizer


def freeze_module(module: nn.Module):
    """Freeze all parameters in a module

    Args:
        module: Module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """Unfreeze all parameters in a module

    Args:
        module: Module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and frozen parameters

    Args:
        model: Model to count parameters

    Returns:
        dict with total, trainable, frozen counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
        "frozen_M": frozen / 1e6,
    }
