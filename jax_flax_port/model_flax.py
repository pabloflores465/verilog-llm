"""
Flax implementation of Qwen2 architecture with manual LoRA.
Compatible with PyTorch Qwen2.5 weights via weight conversion.
"""

import math
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass

from config import ModelConfig, LoRAConfig

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def precompute_rope_frequencies(
    head_dim: int,
    max_position_embeddings: int = 32768,
    theta: float = 1_000_000.0,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute rotary frequency tensors."""
    dim = head_dim
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_position_embeddings, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)  # [max_len, dim//2]
    emb = jnp.concatenate([freqs, freqs], axis=-1)  # [max_len, dim]
    cos = jnp.cos(emb).astype(dtype)
    sin = jnp.sin(emb).astype(dtype)
    return cos, sin


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotate half the hidden dims of the input."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(
    q: jnp.ndarray,
    k: jnp.ndarray,
    cos: jnp.ndarray,
    sin: jnp.ndarray,
    position_ids: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply rotary embeddings to q and k.
    q, k: [batch, heads, seq_len, head_dim]
    cos, sin: [max_len, head_dim]
    """
    seq_len = q.shape[2]
    if position_ids is None:
        position_ids = jnp.arange(seq_len)
    cos = cos[position_ids][None, None, :, :]  # [1,1,seq_len,head_dim]
    sin = sin[position_ids][None, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    eps: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        weight = weight.astype(self.dtype)
        x = x.astype(jnp.float32)
        variance = jnp.mean(x ** 2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return (x * weight).astype(self.dtype)


# ---------------------------------------------------------------------------
# LoRA wrapper
# ---------------------------------------------------------------------------

class LoRADense(nn.Module):
    """Dense layer with optional LoRA adaptation.
    Mimics nn.Dense API but adds low-rank deltas to target modules.
    """
    features: int
    use_lora: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: Any = None
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # Base dense (frozen or trainable depending on caller)
        dense = nn.Dense(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            use_bias=False,
            name="base",
        )
        out = dense(inputs)

        if self.use_lora:
            # LoRA path: W + (alpha/r) * B @ A
            # A: [in_features, r], B: [r, out_features]
            in_dim = inputs.shape[-1]
            scale = self.lora_alpha / self.lora_r

            lora_a = self.param(
                "lora_a",
                nn.initializers.normal(stddev=0.02),
                (in_dim, self.lora_r),
            )
            lora_b = self.param(
                "lora_b",
                nn.initializers.zeros,
                (self.lora_r, self.features),
            )
            lora_a = lora_a.astype(self.dtype)
            lora_b = lora_b.astype(self.dtype)

            if not deterministic and self.lora_dropout > 0.0:
                keep_prob = 1.0 - self.lora_dropout
                mask = jax.random.bernoulli(
                    self.make_rng("dropout"), keep_prob, lora_a.shape
                )
                lora_a = lora_a * mask / keep_prob

            lora_out = jnp.dot(inputs, lora_a)
            lora_out = jnp.dot(lora_out, lora_b)
            out = out + (lora_out * scale)

        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class FlaxQwen2Attention(nn.Module):
    """Grouped Query Attention with RoPE."""
    config: ModelConfig
    lora_config: Optional[LoRAConfig] = None
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config
        head_dim = cfg.head_dim
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads

        # Determine which projections get LoRA
        def use_lora(name: str) -> bool:
            if self.lora_config is None:
                return False
            return name in self.lora_config.target_modules

        # Projections
        q_proj = LoRADense(
            num_heads * head_dim,
            use_lora=use_lora("q_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="q_proj",
        )
        k_proj = LoRADense(
            num_kv_heads * head_dim,
            use_lora=use_lora("k_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="k_proj",
        )
        v_proj = LoRADense(
            num_kv_heads * head_dim,
            use_lora=use_lora("v_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="v_proj",
        )
        o_proj = LoRADense(
            cfg.hidden_size,
            use_lora=use_lora("o_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="o_proj",
        )

        batch, seq_len, _ = hidden_states.shape

        query_states = q_proj(hidden_states)
        key_states = k_proj(hidden_states)
        value_states = v_proj(hidden_states)

        # Reshape to [batch, heads, seq_len, head_dim]
        query_states = query_states.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(batch, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Repeat k/v heads for GQA
        if num_kv_heads < num_heads:
            n_rep = num_heads // num_kv_heads
            key_states = jnp.repeat(key_states, n_rep, axis=1)
            value_states = jnp.repeat(value_states, n_rep, axis=1)

        # Attention: [batch, heads, q_len, head_dim] @ [batch, heads, head_dim, kv_len]
        attn_weights = jnp.einsum("...qhd,...khd->...hqk", query_states, key_states)
        attn_weights = attn_weights / math.sqrt(head_dim)

        if attention_mask is not None:
            attn_weights = jnp.where(attention_mask, attn_weights, jnp.finfo(self.dtype).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(self.dtype)
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        attn_output = o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------

class FlaxQwen2MLP(nn.Module):
    """SwiGLU MLP."""
    config: ModelConfig
    lora_config: Optional[LoRAConfig] = None
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.config

        def use_lora(name: str) -> bool:
            if self.lora_config is None:
                return False
            return name in self.lora_config.target_modules

        gate_proj = LoRADense(
            cfg.intermediate_size,
            use_lora=use_lora("gate_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="gate_proj",
        )
        up_proj = LoRADense(
            cfg.intermediate_size,
            use_lora=use_lora("up_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="up_proj",
        )
        down_proj = LoRADense(
            cfg.hidden_size,
            use_lora=use_lora("down_proj"),
            lora_r=self.lora_config.r if self.lora_config else 0,
            lora_alpha=self.lora_config.alpha if self.lora_config else 0,
            lora_dropout=self.lora_config.dropout if self.lora_config else 0.0,
            dtype=self.dtype,
            name="down_proj",
        )

        # SwiGLU: silu(gate) * up
        hidden = jax.nn.silu(gate_proj(x)) * up_proj(x)
        return down_proj(hidden)


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class FlaxQwen2DecoderLayer(nn.Module):
    config: ModelConfig
    lora_config: Optional[LoRAConfig] = None
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cos: jnp.ndarray,
        sin: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        cfg = self.config

        # Self-attention + residual
        residual = hidden_states
        hidden_states = RMSNorm(eps=cfg.rms_norm_eps, dtype=self.dtype, name="input_layernorm")(hidden_states)
        hidden_states = FlaxQwen2Attention(
            config=cfg,
            lora_config=self.lora_config,
            dtype=self.dtype,
            name="self_attn",
        )(hidden_states, cos, sin, attention_mask, position_ids, deterministic)
        hidden_states = residual + hidden_states

        # MLP + residual
        residual = hidden_states
        hidden_states = RMSNorm(eps=cfg.rms_norm_eps, dtype=self.dtype, name="post_attention_layernorm")(hidden_states)
        hidden_states = FlaxQwen2MLP(
            config=cfg,
            lora_config=self.lora_config,
            dtype=self.dtype,
            name="mlp",
        )(hidden_states, deterministic)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class FlaxQwen2ForCausalLM(nn.Module):
    """Complete Qwen2 Causal LM in Flax."""
    config: ModelConfig
    lora_config: Optional[LoRAConfig] = None
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.config
        batch, seq_len = input_ids.shape

        # Embeddings
        embed = nn.Embed(
            num_embeddings=cfg.vocab_size,
            features=cfg.hidden_size,
            dtype=self.dtype,
            name="embed_tokens",
        )
        hidden_states = embed(input_ids)

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_frequencies(
            head_dim=cfg.head_dim,
            max_position_embeddings=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
            dtype=jnp.float32,
        )

        # Causal mask
        if attention_mask is not None:
            # Expand to [batch, 1, seq_len, seq_len]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            causal_mask = causal_mask[None, None, :, :]
            attention_mask = attention_mask[:, None, None, :]  # [batch,1,1,seq_len]
            attention_mask = jnp.logical_and(attention_mask, causal_mask)
        else:
            attention_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            attention_mask = attention_mask[None, None, :, :]

        # Decoder layers
        for i in range(cfg.num_hidden_layers):
            hidden_states = FlaxQwen2DecoderLayer(
                config=cfg,
                lora_config=self.lora_config,
                dtype=self.dtype,
                name=f"layers_{i}",
            )(hidden_states, cos, sin, attention_mask, position_ids, deterministic)

        # Final norm
        hidden_states = RMSNorm(
            eps=cfg.rms_norm_eps,
            dtype=self.dtype,
            name="norm",
        )(hidden_states)

        # LM head
        if cfg.tie_word_embeddings:
            lm_head_kernel = embed.variables["params"]["embedding"]
            logits = jnp.dot(hidden_states, lm_head_kernel.T)
        else:
            lm_head = nn.Dense(
                cfg.vocab_size,
                use_bias=False,
                dtype=self.dtype,
                name="lm_head",
            )
            logits = lm_head(hidden_states)

        return logits, hidden_states


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(
    params: Any,
    model: FlaxQwen2ForCausalLM,
    batch_input_ids: jnp.ndarray,
    batch_labels: jnp.ndarray,
    dropout_rng: Optional[jax.random.PRNGKey] = None,
) -> jnp.ndarray:
    """Compute cross-entropy loss for causal LM."""
    if dropout_rng is not None:
        logits, _ = model.apply(
            {"params": params},
            input_ids=batch_input_ids,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
    else:
        logits, _ = model.apply(
            {"params": params},
            input_ids=batch_input_ids,
            deterministic=True,
        )

    # Shift logits and labels for next-token prediction
    logits = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
    labels = batch_labels[:, 1:]  # [batch, seq_len-1]

    # Mask padded positions (label == -100)
    mask = labels >= 0
    labels = jnp.where(mask, labels, 0)

    # Cross-entropy
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_token_loss = -jnp.take_along_axis(log_probs, labels[:, :, None], axis=-1).squeeze(-1)
    per_token_loss = jnp.where(mask, per_token_loss, 0.0)

    loss = jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss
