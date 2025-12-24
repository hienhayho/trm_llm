"""Configuration for TRM-LLM"""

from dataclasses import dataclass


@dataclass
class TRMLLMConfig:
    """Configuration for TRM-LLM model

    This config defines the architecture for a ~150M parameter model
    using recursive reasoning and deep supervision from TRM paper.
    """

    # ===== Model Architecture =====
    # Vocabulary and embeddings
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    hidden_dim: int = 768
    max_seq_len: int = 20480

    # Encoder (standard transformer)
    num_layers: int = 12
    num_heads: int = 12
    ff_dim: int = 3072  # 4 * hidden_dim
    dropout: float = 0.1

    # ===== TRM-Specific Components =====
    # Recursive reasoning module
    reasoning_dim: int = 512  # Dimension for z (reasoning state)
    action_dim: int = 256  # Dimension for y (action state)
    num_recursions: int = 3  # Number of recursive refinement steps (n) - latent reasoning iterations

    # Deep recursion (T parameter from TRM paper)
    # Within each supervision step: T iterations of latent_recursion
    # T-1 iterations without gradients, 1 iteration with gradients
    deep_recursion_steps: int = 3  # T parameter (default: 3, set to 1 to disable)
    use_original_trm_grad: bool = False  # Use original TRM gradient flow (T-1 no_grad + 1 with grad)

    # Deep supervision
    max_supervision_steps: int = 8  # Maximum supervision iterations during training

    # Output heads
    num_action_types: int = 2  # [direct_answer, tool_call]
    max_tools: int = 50  # Maximum number of tools in context
    max_parallel_calls: int = 5  # Maximum number of parallel tool calls

    # Unified generation head (generates either tool call JSON or direct answer)
    max_generation_len: int = 512  # Maximum generation sequence length

    # ===== Training Hyperparameters =====
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 50
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0

    # ===== Staged Training =====
    # -1 = standard (all params, all losses)
    # 0 = backbone stage (freeze generation_head, train encoder/reasoning/action/output_heads)
    # 1 = generation stage (freeze all except generation_head)
    # 2 = finetune stage (all params, all losses, typically with smaller dataset)
    training_stage: int = -1

    # ===== Adaptive Computation Time (ACT) =====
    q_threshold: float = 0.5  # Threshold for early stopping (Q > threshold means model thinks it's correct)
    q_loss_weight: float = 0.5  # Weight for Q loss (correctness prediction, TRM paper)

    # ===== Architecture Options =====
    use_causal_encoder: bool = False  # Use causal attention in encoder (for pure LLM training)
    detach_between_steps: bool = True  # Detach states between supervision steps (original TRM behavior)
    use_flash_attention: bool = True  # Use PyTorch 2.0 SDPA for Flash Attention

    # ===== Step Embeddings (connects latent spaces between supervision steps) =====
    # PE_in: Added BEFORE recursive loop - tells model "processing at step N"
    # PE_out: Added AFTER recursive loop - labels output "produced at step N"
    use_step_embedding_in: bool = False  # Add step embedding to y,z BEFORE reasoning (input context)
    use_step_embedding_out: bool = False  # Add step embedding to y,z AFTER action (output labeling)

    # ===== Loss Weights =====
    action_loss_weight: float = 2.0  # Weight for action classification loss (tool_call vs direct_answer)
    action_class_weights: tuple = None  # Class weights for action loss [direct_answer, tool_call], None = auto-compute from batch
    use_focal_loss: bool = True  # Use Focal Loss for action classification (better for class imbalance)
    focal_gamma: float = 1.5  # Focal Loss gamma parameter (1.5 for stability with mixed precision, higher can cause instability)
    num_calls_loss_weight: float = 1.0  # Weight for num_calls loss (set to 0 if dataset has no parallel calls)
    tool_call_gen_weight: float = 2.0  # Weight for tool call JSON generation loss (higher = focus more on tool calls)
    direct_answer_gen_weight: float = 1.0  # Weight for direct answer generation loss
    special_token_weight: float = 5.0  # Extra weight for special tokens like <tool_call>, </tool_call> (improves structure)
    label_smoothing: float = 0.1  # Label smoothing for generation loss (0.0 = no smoothing, 0.1 = recommended)

    # ===== Data Processing =====
    pad_token_id: int = 50256  # GPT-2 pad token

    def __post_init__(self):
        """Validate configuration"""
        assert (
            self.hidden_dim % self.num_heads == 0
        ), f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert (
            self.ff_dim >= self.hidden_dim
        ), f"ff_dim ({self.ff_dim}) should be >= hidden_dim ({self.hidden_dim})"
        assert 0 <= self.dropout <= 1, f"dropout must be between 0 and 1, got {self.dropout}"
        assert (
            self.num_recursions > 0
        ), f"num_recursions must be positive, got {self.num_recursions}"
        assert (
            self.max_supervision_steps > 0
        ), f"max_supervision_steps must be positive, got {self.max_supervision_steps}"

    def estimate_parameters(self) -> dict:
        """Estimate number of parameters in millions"""
        # Token embeddings
        token_embed = self.vocab_size * self.hidden_dim

        # Position embeddings
        pos_embed = self.max_seq_len * self.hidden_dim

        # Encoder: each layer has ~4 * hidden_dim^2 (attention + FFN)
        # Attention: 4 * hidden_dim^2 (Q, K, V, O projections)
        # FFN: 2 * hidden_dim * ff_dim
        encoder_per_layer = 4 * self.hidden_dim**2 + 2 * self.hidden_dim * self.ff_dim
        encoder_total = self.num_layers * encoder_per_layer

        # Reasoning module (2 layers, reasoning_dim)
        reasoning = 2 * (
            4 * self.reasoning_dim**2 + 2 * self.reasoning_dim * (self.reasoning_dim * 4)
        )

        # Action module (2 layers, action_dim)
        action = 2 * (4 * self.action_dim**2 + 2 * self.action_dim * (self.action_dim * 4))

        # Output heads (action, num_calls, q_head - no tool_head since tool selection is via generation)
        action_head = self.action_dim * self.num_action_types
        num_calls_head = self.action_dim * self.max_parallel_calls
        q_head = self.action_dim * 1
        output_heads = action_head + num_calls_head + q_head

        total = token_embed + pos_embed + encoder_total + reasoning + action + output_heads

        return {
            "token_embeddings_M": token_embed / 1e6,
            "position_embeddings_M": pos_embed / 1e6,
            "encoder_M": encoder_total / 1e6,
            "reasoning_module_M": reasoning / 1e6,
            "action_module_M": action / 1e6,
            "output_heads_M": output_heads / 1e6,
            "total_M": total / 1e6,
        }

    def __repr__(self):
        params = self.estimate_parameters()
        return (
            f"TRMLLMConfig(\n"
            f"  Architecture: {self.num_layers}L-{self.hidden_dim}H-{self.num_heads}A\n"
            f"  TRM: reasoning_dim={self.reasoning_dim}, action_dim={self.action_dim}, "
            f"recursions={self.num_recursions}\n"
            f"  Estimated params: {params['total_M']:.1f}M\n"
            f"  Training: lr={self.learning_rate}, batch_size={self.batch_size}, "
            f"max_epochs={self.max_epochs}\n"
            f")"
        )
