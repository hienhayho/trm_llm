"""Basic Transformer building blocks for TRM-LLM"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    More efficient than LayerNorm, used in modern LLMs
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        return self.weight * x_normalized


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward

    Standard transformer block used in encoder
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Pre-attention norm
        self.norm1 = RMSNorm(hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # Expect (batch, seq, dim)
        )

        # Pre-FFN norm
        self.norm2 = RMSNorm(hidden_dim)

        # Feed-forward network with SwiGLU activation
        self.ff = SwiGLU(hidden_dim, ff_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, is_causal=False):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len) or (seq_len, seq_len)
            is_causal: whether to use causal masking

        Returns:
            (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )
        x = residual + self.dropout(attn_output)

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))

        return x


class SwiGLU(nn.Module):
    """SwiGLU activation function

    GLU variant with Swish activation, used in modern transformers
    Formula: SwiGLU(x) = Swish(W1 @ x) ⊙ (W2 @ x)
    """

    def __init__(self, hidden_dim: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: swish(W1 @ x) * (W2 @ x), then project back
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerEncoder(nn.Module):
    """Stack of Transformer blocks

    Standard transformer encoder used for encoding input sequences
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(hidden_dim)

    def forward(self, x, attention_mask=None, is_causal=False):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            attention_mask: optional mask
            is_causal: whether to use causal masking

        Returns:
            (batch_size, seq_len, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, is_causal=is_causal)

        return self.final_norm(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encodings

    Alternative to learned positional embeddings
    """

    def __init__(self, hidden_dim: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))

        pe = torch.zeros(1, max_seq_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TinyTransformer(nn.Module):
    """Small 2-layer transformer for recursive modules

    Used in RecursiveReasoningModule and ActionStateModule
    Following TRM principle: small recursive networks instead of huge ones
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert num_layers <= 4, "TinyTransformer should have <= 4 layers (TRM principle)"

        ff_dim = dim * 4  # Standard 4x expansion

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, dim) or (batch_size, dim)

        Returns:
            same shape as input
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_output = True
        else:
            squeeze_output = False

        for layer in self.layers:
            x = layer(x, is_causal=False)

        x = self.norm(x)

        if squeeze_output:
            x = x.squeeze(1)

        return x
