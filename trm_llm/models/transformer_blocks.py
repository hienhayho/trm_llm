"""Basic Transformer building blocks for TRM-LLM"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDPAttention(nn.Module):
    """Multi-head attention using PyTorch 2.0 Scaled Dot Product Attention

    This automatically uses Flash Attention when available on CUDA,
    providing significant speedup and memory savings for long sequences.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout

        # QKV projection
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len) or (seq_len, seq_len) attention mask
            is_causal: whether to apply causal masking

        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * hidden)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch, heads, seq, head_dim)

        # Use PyTorch's scaled_dot_product_attention (enables Flash Attention)
        # This automatically selects the best backend (Flash, Memory-Efficient, or Math)
        dropout_p = self.dropout if self.training else 0.0

        # Convert attention_mask if provided
        attn_mask = None
        if attention_mask is not None and not is_causal:
            # If mask is (batch, seq), convert to (batch, 1, 1, seq) for broadcasting
            if attention_mask.dim() == 2:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
                attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
            else:
                attn_mask = attention_mask

        # SDPA with automatic backend selection
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )  # (batch, heads, seq, head_dim)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


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

    Standard transformer block used in encoder.
    Uses SDPAttention for automatic Flash Attention support on PyTorch 2.0+.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1,
                 use_sdp_attention: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_sdp_attention = use_sdp_attention

        # Pre-attention norm
        self.norm1 = RMSNorm(hidden_dim)

        # Multi-head self-attention (SDP or standard)
        if use_sdp_attention:
            self.attention = SDPAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
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

        if self.use_sdp_attention:
            attn_output = self.attention(x, attention_mask=attention_mask, is_causal=is_causal)
        else:
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

    Standard transformer encoder used for encoding input sequences.
    Uses SDPAttention by default for Flash Attention support.
    """

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, use_sdp_attention: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout, use_sdp_attention)
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
    Following TRM principle: small recursive networks instead of huge ones.
    Uses SDPAttention by default for Flash Attention support.
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1,
                 use_sdp_attention: bool = True):
        super().__init__()
        assert num_layers <= 4, "TinyTransformer should have <= 4 layers (TRM principle)"

        ff_dim = dim * 4  # Standard 4x expansion

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim, dropout, use_sdp_attention)
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
