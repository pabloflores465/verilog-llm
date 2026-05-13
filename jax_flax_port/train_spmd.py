"""
14B Qwen2 training on Kaggle TPU v3-8 using JAX/Flax + SPMD.
Model parallelism (tensor parallelism) across 8 TPU cores with pjit.

This shards linear layers along the feature dimension so that 14B fits.
All dims are divisible by 8:
  hidden_size=5120, intermediate_size=13824, heads=40, kv_heads=8

Usage:
    python train_spmd.py --model 14b --epochs 1
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
import optax
from flax.training import train_state
from flax import struct
from transformers import AutoTokenizer
from huggingface_hub import login

from config import ModelConfig, LoRAConfig, TrainConfig, MODEL_14B
from model_flax import FlaxQwen2ForCausalLM, compute_loss
from data import build_tokenized_dataset
from checkpoint import save_checkpoint, load_checkpoint, upload_to_hub


# ---------------------------------------------------------------------------
# Sharding utilities
# ---------------------------------------------------------------------------

def create_mesh(devices):
    """Create 1D mesh for tensor parallelism."""
    return Mesh(mesh_utils.create_device_mesh((len(devices),)), ("tp",))


def get_sharding_specs(param_tree):
    """Assign PartitionSpec to each parameter for tensor parallelism.
    Linear kernels: [in_features, out_features] -> shard out_features.
    Embeddings & norms: replicate.
    """
    def spec_for_path(path, arr):
        name = "/".join(str(p) for p in path)
        # Embeddings replicated
        if "embed_tokens" in name or "lm_head" in name:
            return P()
        # RMSNorm scale replicated
        if "layernorm" in name or "norm" in name:
            return P()
        # LoRA parameters replicated (small)
        if "lora" in name:
            return P()
        # Linear kernels: shard output dimension (last axis)
        if "kernel" in name and arr.ndim == 2:
            return P(None, "tp")
        # Bias (if any): replicated
        return P()

    return jax.tree_util.tree_map_with_path(spec_for_path, param_tree)


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    dropout_rng: jax.Array
    step_counter: int = struct.field(pytree_node=False)


# ---------------------------------------------------------------------------
# pjit training step
# ---------------------------------------------------------------------------

def make_pjit_train_step(model: FlaxQwen2ForCausalLM, grad_accum: int, mesh: Mesh):
    """Build pjit training step with tensor parallelism."""

    def train_step(state: TrainState, batch_input_ids: jnp.ndarray, batch_labels: jnp.ndarray):
        """Single step with gradient accumulation if configured."""
        dropout_rng = state.dropout_rng

        if grad_accum <= 1:
            dropout_rng, step_rng = random.split(dropout_rng)

            def loss_fn(params):
                return compute_loss(params, model, batch_input_ids, batch_labels, step_rng)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
        else:
            # Gradient accumulation (single-device microbatches)
            accum_grads = jax.tree_map(jnp.zeros_like, state.params)
            total_loss = 0.0
            micro_size = batch_input_ids.shape[0] // grad_accum

            for i in range(grad_accum):
                start = i * micro_size
                end = start + micro_size
                dropout_rng, step_rng = random.split(dropout_rng)

                def loss_fn(params):
                    return compute_loss(
                        params, model,
                        batch_input_ids[start:end],
                        batch_labels[start:end],
                        step_rng,
                    )

                loss, grads = jax.value_and_grad(loss_fn)(state.params)
                accum_grads = jax.tree_map(lambda a, g: a + g, accum_grads, grads)
                total_loss += loss

            loss = total_loss / grad_accum
            grads = jax.tree_map(lambda g: g / grad_accum, accum_grads)

        state = state.replace(dropout_rng=dropout_rng)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Shardings
    # Input replicated across mesh (batch is small)
    in_shardings = (
        NamedSharding(mesh, P()),  # state replicated
        NamedSharding(mesh, P()),  # input_ids replicated
        NamedSharding(mesh, P()),  # labels replicated
    )
    out_shardings = (
        NamedSharding(mesh, P()),  # state replicated
        NamedSharding(mesh, P()),  # loss scalar replicated
    )

    return jax.jit(
        train_step,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        donate_argnums=(0,),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="14b", choices=["14b"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--from_jax_weights", default="/kaggle/working/qwen_jax_weights.npz")
    args = parser.parse_args()

    print("=" * 60)
    print("JAX/Flax Qwen2 14B SPMD Training on TPU")
    print("=" * 60)

    # TPU setup
    try:
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
    except ImportError:
        pass

    devices = jax.devices('tpu')
    n_devices = len(devices)
    print(f"TPU devices: {n_devices}")
    for d in devices:
        print(f"  {d}")

    if n_devices < 8:
        print("⚠️ 14B SPMD requires 8 TPU cores. Falling back to single-device.")

    # Mesh for tensor parallelism
    mesh = create_mesh(devices)
    print(f"Mesh: {mesh}")

    # Config
    model_cfg = MODEL_14B
    train_cfg = TrainConfig(
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        save_steps=args.save_steps,
    )
    lora_cfg = LoRAConfig()

    print(f"\nModel: {model_cfg.model_name}")
    print(f"  Hidden: {model_cfg.hidden_size}, Intermediate: {model_cfg.intermediate_size}")
    print(f"  Layers: {model_cfg.num_hidden_layers}, Heads: {model_cfg.num_attention_heads}")
    print(f"  TP sharding: 8-way tensor parallelism")
    print(f"\nTraining: seq={train_cfg.max_seq_length}, lr={train_cfg.learning_rate}")

    # Auth
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    if hf_token:
        login(token=hf_token)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Data
    train_ds, eval_ds, steps_per_epoch = build_tokenized_dataset(
        train_cfg, tokenizer, train_cfg.max_seq_length, skip_overlength=True, num_devices=n_devices
    )
    total_steps = steps_per_epoch * train_cfg.num_epochs
    print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
    train_iter = iter(train_ds.as_numpy_iterator())

    # Model
    dtype = jnp.bfloat16
    model = FlaxQwen2ForCausalLM(config=model_cfg, lora_config=lora_cfg, dtype=dtype)

    # Load weights
    jax_weights_path = Path(args.from_jax_weights)
    if not jax_weights_path.exists():
        print("Converting PyTorch weights...")
        from convert_weights import load_pytorch_state_dict, convert_to_flax_params, save_jax_checkpoint
        state_dict = load_pytorch_state_dict(model_cfg.model_name)
        params_wrapped = convert_to_flax_params(state_dict, model_cfg)
        save_jax_checkpoint(params_wrapped, str(jax_weights_path))
        loaded_params = params_wrapped["params"]
    else:
        from checkpoint import load_jax_checkpoint
        loaded_params = load_jax_checkpoint(str(jax_weights_path))

    # Init model to get full param tree (base + LoRA init)
    rng = random.PRNGKey(train_cfg.seed)
    dummy_ids = jnp.ones((1, train_cfg.max_seq_length), dtype=jnp.int32)
    init_variables = model.init(rng, dummy_ids, deterministic=True)
    init_params = init_variables["params"]

    # Merge: override base params with loaded weights, keep LoRA init
    def merge_params(init_p, loaded_p):
        if isinstance(init_p, dict) and isinstance(loaded_p, dict):
            result = {}
            for k in init_p:
                if k in loaded_p:
                    result[k] = merge_params(init_p[k], loaded_p[k])
                else:
                    result[k] = init_p[k]
            return result
        return loaded_p

    params = merge_params(init_params, loaded_params)
    print("✓ Base weights loaded + LoRA params initialized")

    # Apply sharding to params
    param_specs = get_sharding_specs(init_params)
    shardings = jax.tree_map(lambda spec: NamedSharding(mesh, spec), param_specs)

    # Put params on devices with sharding
    params = jax.tree_map(lambda p, s: jax.device_put(p, s), params, shardings)

    # Build trainable mask: True only for LoRA parameters
    def is_trainable(path, val):
        return 'lora' in '/'.join(str(p) for p in path)

    trainable_mask = jax.tree_util.tree_map_with_path(is_trainable, params)
    trainable_count = sum(int(x) for x in jax.tree_util.tree_leaves(trainable_mask))
    print(f"  Trainable param arrays: {trainable_count}")

    # Optimizer (only LoRA params updated)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=total_steps,
        end_value=train_cfg.min_lr,
    )
    inner_opt = optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=train_cfg.weight_decay)
    optimizer = optax.chain(
        optax.clip_by_global_norm(train_cfg.max_grad_norm) if train_cfg.max_grad_norm > 0 else optax.identity(),
        optax.masked(inner_opt, trainable_mask),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=rng,
        step_counter=0,
    )

    # Shard optimizer state to match params
    # (optax state is small, replicate is fine)

    print("\nCompiling SPMD train step (this may take 2-5 minutes)...")
    pjit_step = make_pjit_train_step(model, train_cfg.grad_accum, mesh)

    # Warmup
    dummy_batch_ids = jnp.ones((train_cfg.batch_size, train_cfg.max_seq_length), dtype=jnp.int32)
    dummy_batch_labels = jnp.ones((train_cfg.batch_size, train_cfg.max_seq_length), dtype=jnp.int32)
    _ = pjit_step(state, dummy_batch_ids, dummy_batch_labels)
    print("✓ Compilation done")

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING SPMD TRAINING")
    print("=" * 60 + "\n")

    start_time = time.time()
    global_step = 0
    losses = []

    try:
        while global_step < total_steps:
            try:
                batch_input_ids, batch_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_ds.as_numpy_iterator())
                batch_input_ids, batch_labels = next(train_iter)

            t0 = time.time()
            state, loss = pjit_step(state, batch_input_ids, batch_labels)
            loss = float(loss)
            step_time = time.time() - t0

            global_step += 1
            losses.append(loss)

            if global_step % train_cfg.log_every == 0:
                avg = sum(losses[-train_cfg.log_every:]) / min(len(losses), train_cfg.log_every)
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                eta = (total_steps - global_step) / sps if sps > 0 else 0
                print(
                    f"[Step {global_step:05d}/{total_steps}] "
                    f"loss={avg:.4f} | step={step_time:.2f}s | sps={sps:.2f} | eta={eta/3600:.1f}h"
                )

            if global_step % train_cfg.save_steps == 0:
                print(f"\n💾 Saving checkpoint {global_step}...")
                ckpt = save_checkpoint(state, train_cfg.output_dir, global_step)
                if hf_token:
                    upload_to_hub(ckpt, model_cfg.hub_model_id, hf_token, step=global_step)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving...")
        save_checkpoint(state, train_cfg.output_dir, global_step, name="interrupted")

    print("\n💾 Final save...")
    final = save_checkpoint(state, train_cfg.output_dir, global_step, name="final")
    if hf_token:
        upload_to_hub(final, model_cfg.hub_model_id, hf_token, name="final")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
