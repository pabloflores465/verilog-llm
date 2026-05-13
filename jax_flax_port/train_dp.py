"""
7B Qwen2 training on Kaggle TPU v3-8 using JAX/Flax.
Data parallelism across 8 TPU cores with pmap.

Usage:
    python train_dp.py --model 7b --epochs 3
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state
from flax import struct
from transformers import AutoTokenizer
from huggingface_hub import login, HfApi

from config import ModelConfig, LoRAConfig, TrainConfig, MODEL_7B, MODEL_14B
from model_flax import FlaxQwen2ForCausalLM, compute_loss
from data import build_tokenized_dataset
from checkpoint import save_checkpoint, load_checkpoint, upload_to_hub


# ---------------------------------------------------------------------------
# Distributed training state
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    dropout_rng: jax.Array
    step_counter: int = struct.field(pytree_node=False)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def create_train_state(
    rng: jax.Array,
    model: FlaxQwen2ForCausalLM,
    config: ModelConfig,
    train_cfg: TrainConfig,
    total_steps: int,
    params: Dict = None,
) -> TrainState:
    """Initialize or restore training state."""
    # Dummy input for shape inference
    dummy_ids = jnp.ones((1, train_cfg.max_seq_length), dtype=jnp.int32)

    if params is None:
        # Initialize from scratch (not recommended for LLMs)
        variables = model.init(rng, dummy_ids, deterministic=True)
        params = variables["params"]
    else:
        # Verify shapes match
        variables = model.init(rng, dummy_ids, deterministic=True)
        expected = variables["params"]
        # We trust the provided params match the architecture

    # Build trainable mask: True only for LoRA parameters
    def is_trainable(path, val):
        return 'lora' in '/'.join(str(p) for p in path)

    trainable_mask = jax.tree_util.tree_map_with_path(is_trainable, params)
    trainable_count = sum(int(x) for x in jax.tree_util.tree_leaves(trainable_mask))
    total_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Trainable params: {trainable_count} arrays")

    # LR schedule: warmup + cosine decay
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=total_steps,
        end_value=train_cfg.min_lr,
    )

    # AdamW optimizer wrapped with mask (only LoRA params updated)
    inner_opt = optax.adamw(
        learning_rate=schedule,
        b1=0.9,
        b2=0.95,
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
    )
    optimizer = optax.masked(inner_opt, trainable_mask)

    # Gradient clipping
    if train_cfg.max_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(train_cfg.max_grad_norm),
            optimizer,
        )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dropout_rng=rng,
        step_counter=0,
    )


# ---------------------------------------------------------------------------
# Training step (single device)
# ---------------------------------------------------------------------------

def make_train_step(model: FlaxQwen2ForCausalLM, grad_accum: int):
    """Build training step with optional gradient accumulation."""

    def train_step(
        state: TrainState,
        batch_input_ids: jnp.ndarray,
        batch_labels: jnp.ndarray,
    ) -> tuple:
        """Single training step (no grad accum)."""
        dropout_rng, new_rng = random.split(state.dropout_rng)

        def loss_fn(params):
            return compute_loss(params, model, batch_input_ids, batch_labels, dropout_rng)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.replace(dropout_rng=new_rng)
        return state, loss, grads

    if grad_accum <= 1:
        return train_step

    # Gradient accumulation version
    def accum_train_step(
        state: TrainState,
        batch_input_ids: jnp.ndarray,
        batch_labels: jnp.ndarray,
    ) -> tuple:
        """Training step with gradient accumulation."""
        dropout_rng = state.dropout_rng
        accum_grads = jax.tree_map(jnp.zeros_like, state.params)
        total_loss = 0.0

        # Split batch into microbatches
        batch_size = batch_input_ids.shape[0]
        microbatch_size = batch_size // grad_accum

        for i in range(grad_accum):
            start = i * microbatch_size
            end = start + microbatch_size
            micro_ids = batch_input_ids[start:end]
            micro_labels = batch_labels[start:end]

            dropout_rng, step_rng = random.split(dropout_rng)

            def loss_fn(params):
                return compute_loss(params, model, micro_ids, micro_labels, step_rng)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            accum_grads = jax.tree_map(lambda a, g: a + g, accum_grads, grads)
            total_loss += loss

        # Average gradients
        accum_grads = jax.tree_map(lambda g: g / grad_accum, accum_grads)
        avg_loss = total_loss / grad_accum
        state = state.replace(dropout_rng=dropout_rng)
        return state, avg_loss, accum_grads

    return accum_train_step


# ---------------------------------------------------------------------------
# pmap setup
# ---------------------------------------------------------------------------

def setup_pmap_train(model: FlaxQwen2ForCausalLM, grad_accum: int):
    """Return pmapped train step and update function."""
    train_step_fn = make_train_step(model, grad_accum)

    def step_and_update(state, batch_input_ids, batch_labels):
        state, loss, grads = train_step_fn(state, batch_input_ids, batch_labels)
        # pmean across devices
        loss = jax.lax.pmean(loss, axis_name='batch')
        grads = jax.lax.pmean(grads, axis_name='batch')
        state = state.apply_gradients(grads=grads)
        return state, loss

    pmapped_step = jax.pmap(
        step_and_update,
        axis_name='batch',
        donate_argnums=(0,),
    )
    return pmapped_step


def replicate_state(state: TrainState, devices) -> TrainState:
    """Replicate state across TPU cores."""
    return jax.device_put_replicated(state, devices)


def shard_batch(batch_input_ids: np.ndarray, batch_labels: np.ndarray, devices):
    """Split batch evenly across devices."""
    n_devices = len(devices)
    # Ensure batch is divisible by num devices
    batch_size = batch_input_ids.shape[0]
    if batch_size % n_devices != 0:
        # Trim to multiple
        trim = batch_size - (batch_size % n_devices)
        batch_input_ids = batch_input_ids[:trim]
        batch_labels = batch_labels[:trim]

    ids_shards = np.split(batch_input_ids, n_devices)
    labels_shards = np.split(batch_labels, n_devices)

    ids_pmapped = jax.device_put_sharded(list(ids_shards), devices)
    labels_pmapped = jax.device_put_sharded(list(labels_shards), devices)
    return ids_pmapped, labels_pmapped


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="7b", choices=["7b", "14b"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--from_jax_weights", default="/kaggle/working/qwen_jax_weights.npz")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # TPU Setup
    # ------------------------------------------------------------------
    print("=" * 60)
    print("JAX/Flax Qwen2 Training on TPU")
    print("=" * 60)

    try:
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
    except ImportError:
        pass

    devices = jax.devices('tpu')
    n_devices = len(devices)
    print(f"TPU devices detected: {n_devices}")
    for d in devices:
        print(f"  {d}")

    if n_devices == 0:
        print("⚠️ No TPU detected. Falling back to CPU/GPU.")
        devices = jax.devices()
        n_devices = len(devices)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    model_cfg = MODEL_7B if args.model == "7b" else MODEL_14B
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
    print(f"  Hidden size: {model_cfg.hidden_size}")
    print(f"  Layers: {model_cfg.num_hidden_layers}")
    print(f"  Heads: {model_cfg.num_attention_heads} / KV: {model_cfg.num_key_value_heads}")
    print(f"\nTraining:")
    print(f"  Seq length: {train_cfg.max_seq_length}")
    print(f"  Batch per device: {train_cfg.batch_size}")
    print(f"  Global batch: {train_cfg.batch_size * n_devices}")
    print(f"  Grad accum: {train_cfg.grad_accum}")
    print(f"  Effective batch: {train_cfg.batch_size * n_devices * train_cfg.grad_accum}")

    # ------------------------------------------------------------------
    # HF Auth
    # ------------------------------------------------------------------
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            print("\n✓ HF_TOKEN from Kaggle Secrets")
        except Exception:
            pass
    if not hf_token:
        print("\n⚠️ HF_TOKEN not set. Upload to Hub will be disabled.")
    else:
        login(token=hf_token)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\nBuilding dataset...")
    train_ds, eval_ds, steps_per_epoch = build_tokenized_dataset(
        train_cfg,
        tokenizer,
        max_seq_length=train_cfg.max_seq_length,
        skip_overlength=True,
        num_devices=n_devices,
    )
    total_steps = steps_per_epoch * train_cfg.num_epochs
    print(f"Total steps: {total_steps}")

    # Convert tf.data to numpy iterator
    train_iter = iter(train_ds.as_numpy_iterator())

    # ------------------------------------------------------------------
    # Model & Weights
    # ------------------------------------------------------------------
    print("\nInitializing model...")
    dtype = jnp.bfloat16
    model = FlaxQwen2ForCausalLM(
        config=model_cfg,
        lora_config=lora_cfg,
        dtype=dtype,
    )

    # Load or convert weights
    jax_weights_path = Path(args.from_jax_weights)
    if jax_weights_path.exists():
        print(f"Loading JAX weights from {jax_weights_path}...")
        import checkpoint
        loaded_params = checkpoint.load_jax_checkpoint(str(jax_weights_path))
    else:
        print("JAX weights not found. Converting from PyTorch...")
        from convert_weights import load_pytorch_state_dict, convert_to_flax_params, save_jax_checkpoint
        state_dict = load_pytorch_state_dict(model_cfg.model_name)
        params_wrapped = convert_to_flax_params(state_dict, model_cfg)
        save_jax_checkpoint(params_wrapped, str(jax_weights_path))
        loaded_params = params_wrapped["params"]  # unwrap

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
                    result[k] = init_p[k]  # LoRA params not in checkpoint
            return result
        return loaded_p

    params = merge_params(init_params, loaded_params)
    print("✓ Base weights loaded + LoRA params initialized")

    # Initialize training state
    state = create_train_state(
        rng=rng,
        model=model,
        config=model_cfg,
        train_cfg=train_cfg,
        total_steps=total_steps,
        params=params,
    )

    # Print trainable params
    def count_params(pytree):
        return sum(x.size for x in jax.tree_util.tree_leaves(pytree))
    total = count_params(state.params)
    # Count LoRA params
    def is_lora(path, val):
        return 'lora' in '/'.join(str(p) for p in path)
    lora_params = jax.tree_util.tree_map_with_path(
        lambda p, x: x.size if is_lora(p, x) else 0, state.params
    )
    lora_count = sum(jax.tree_util.tree_leaves(lora_params))
    print(f"\nTotal params: {total:,} ({total/1e9:.2f}B)")
    print(f"LoRA params: {lora_count:,} ({lora_count/1e6:.2f}M) [{100*lora_count/total:.4f}%]")

    # ------------------------------------------------------------------
    # Replicate across TPU cores
    # ------------------------------------------------------------------
    print(f"\nReplicating state to {n_devices} devices...")
    state = replicate_state(state, devices)
    # Give each device a unique dropout RNG
    base_rng = state.dropout_rng[0] if state.dropout_rng.shape else state.dropout_rng
    dropout_rngs = random.split(base_rng, n_devices)
    state = state.replace(
        dropout_rng=jax.device_put_sharded(list(dropout_rngs), devices)
    )

    # ------------------------------------------------------------------
    # Pmap setup
    # ------------------------------------------------------------------
    print("Compiling pmap train step...")
    pmapped_step = setup_pmap_train(model, train_cfg.grad_accum)

    # Warmup compilation with dummy batch
    dummy_ids = jnp.ones((n_devices, train_cfg.batch_size, train_cfg.max_seq_length), dtype=jnp.int32)
    dummy_labels = jnp.ones((n_devices, train_cfg.batch_size, train_cfg.max_seq_length), dtype=jnp.int32)
    _ = pmapped_step(state, dummy_ids, dummy_labels)
    print("✓ Compilation done")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    start_time = time.time()
    global_step = 0
    epoch = 0
    step_in_epoch = 0
    losses = []

    try:
        while global_step < total_steps:
            if step_in_epoch >= steps_per_epoch:
                epoch += 1
                step_in_epoch = 0
                print(f"\n--- Epoch {epoch + 1}/{train_cfg.num_epochs} ---")

            # Fetch batch
            try:
                batch_input_ids, batch_labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_ds.as_numpy_iterator())
                batch_input_ids, batch_labels = next(train_iter)

            # Shard across devices
            ids_shard, labels_shard = shard_batch(
                batch_input_ids, batch_labels, devices
            )

            # Train step
            t0 = time.time()
            state, loss = pmapped_step(state, ids_shard, labels_shard)
            loss = float(loss[0])  # All devices have same loss after pmean
            step_time = time.time() - t0

            losses.append(loss)
            global_step += 1
            step_in_epoch += 1

            # Logging
            if global_step % train_cfg.log_every == 0:
                avg_loss = sum(losses[-train_cfg.log_every:]) / min(len(losses), train_cfg.log_every)
                elapsed = time.time() - start_time
                sps = global_step / elapsed
                eta = (total_steps - global_step) / sps if sps > 0 else 0
                print(
                    f"[Step {global_step:05d}/{total_steps}] "
                    f"loss={avg_loss:.4f} | "
                    f"step_time={step_time:.2f}s | "
                    f"sps={sps:.2f} | "
                    f"eta={eta/3600:.1f}h"
                )

            # Checkpointing
            if global_step % train_cfg.save_steps == 0:
                print(f"\n💾 Saving checkpoint at step {global_step}...")
                # Unreplicate before saving
                single_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
                ckpt_path = save_checkpoint(
                    single_state,
                    train_cfg.output_dir,
                    global_step,
                )
                print(f"✓ Saved: {ckpt_path}")

                if hf_token:
                    print("📤 Uploading to HF Hub...")
                    upload_to_hub(
                        ckpt_path,
                        model_cfg.hub_model_id,
                        hf_token,
                        step=global_step,
                    )

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Saving checkpoint...")
        single_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        save_checkpoint(single_state, train_cfg.output_dir, global_step, name="interrupted")

    # Final save
    print("\n💾 Saving final checkpoint...")
    single_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    final_path = save_checkpoint(single_state, train_cfg.output_dir, global_step, name="final")

    if hf_token:
        upload_to_hub(final_path, model_cfg.hub_model_id, hf_token, name="final")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final checkpoint: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
