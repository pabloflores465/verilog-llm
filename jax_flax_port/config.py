"""Centralized configuration for JAX/Flax Verilog training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Qwen2 architecture config."""
    # Model identifiers
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    hub_model_id: str = "Pablo-Flores-Mollinedo/verilog-qwen-14b-sota"

    # Architecture (populated from HF config.json)
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    use_sliding_window: bool = False
    sliding_window: int = 131072
    tie_word_embeddings: bool = False

    # Precision
    dtype: str = "bfloat16"  # TPU native

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class LoRAConfig:
    """LoRA hyperparameters."""
    r: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    output_dir: str = "/kaggle/working/checkpoints"
    max_seq_length: int = 2048
    batch_size: int = 1
    grad_accum: int = 4
    learning_rate: float = 2e-4
    min_lr: float = 0.0
    warmup_steps: int = 50
    num_epochs: int = 3
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    seed: int = 42

    # Data
    train_path: str = "/kaggle/input/verilog-curated-dataset/train.jsonl"
    eval_path: str = "/kaggle/input/verilog-curated-dataset/eval.jsonl"
    working_dataset_dir: str = "/kaggle/working/verilog-curated-dataset"

    # Checkpointing
    save_steps: int = 100
    save_total_limit: int = 3

    # Logging
    log_every: int = 10


# Presets
MODEL_7B = ModelConfig(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    hidden_size=3584,
    intermediate_size=18944,
    num_hidden_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,
)

MODEL_14B = ModelConfig(
    model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
    hidden_size=5120,
    intermediate_size=13824,
    num_hidden_layers=48,
    num_attention_heads=40,
    num_key_value_heads=8,
)

# Preset for 14B TPU training with filtering
TRAIN_14B = TrainConfig(
    max_seq_length=2048,
    batch_size=1,
    grad_accum=4,
    learning_rate=2e-4,
    num_epochs=3,
    save_steps=100,
)

# Optimal config: 7B, full dataset (19K), 3 epochs, data parallelism on 8 TPU cores
TRAIN_OPTIMAL = TrainConfig(
    max_seq_length=2048,
    batch_size=1,
    grad_accum=4,
    learning_rate=2e-4,
    num_epochs=3,
    save_steps=100,
    log_every=10,
)
