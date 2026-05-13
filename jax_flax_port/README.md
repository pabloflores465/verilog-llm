# JAX/Flax Port for Verilog Qwen2 Training on TPU

Complete reimplementation of Qwen2 architecture in **Flax** for training on **Kaggle TPU v3-8** (8 cores, 128GB HBM total).

## Why JAX/Flax?

Kaggle TPU VMs have a broken `torch_xla` (ABI mismatch with pre-installed PyTorch). However, **JAX works natively** on Kaggle TPU because:
- Kaggle pre-installs `jax[tpu]` compatible with the VM image
- `pmap` / `pjit` provide first-class multi-core training
- No CUDA/PTX dependency issues

## Architecture

This port implements from scratch:
- **RMSNorm** (Qwen2-style)
- **RoPE** (Rotary Position Embeddings)
- **Grouped Query Attention** (GQA)
- **SwiGLU MLP**
- **Manual LoRA** (r=64, alpha=128) on all linear projections

Model compatibility:
- Loads PyTorch weights from `Qwen/Qwen2.5-Coder-7B-Instruct` or `14B-Instruct`
- Converts to Flax params via name mapping + transpose
- Supports resuming from Flax checkpoints

## Files

| File | Purpose |
|------|---------|
| `config.py` | Model, LoRA, and training hyperparameters |
| `model_flax.py` | Full Qwen2 Flax model + LoRA layers + loss function |
| `convert_weights.py` | One-time PyTorch → Flax weight converter |
| `data.py` | `tf.data` pipeline: JSONL → tokens, filtering, batching |
| `train_dp.py` | **7B training** with `pmap` data parallelism on 8 TPU cores |
| `train_spmd.py` | **14B training** with `pjit` tensor parallelism on 8 TPU cores |
| `checkpoint.py` | Save/load NPZ checkpoints + HF Hub upload |
| `requirements.txt` | Dependencies |
| `kaggle_notebook.ipynb` | Ready-to-run Kaggle notebook |

## Training Modes

### 7B Data Parallelism (`train_dp.py`)

Each TPU core holds a full replica of the 7B model (~14GB in bfloat16).
- Global batch = `batch_size_per_device (1) × 8 cores = 8`
- With `grad_accum=4`, effective batch = 32
- Each step processes 8 sequences in parallel

```bash
python jax_flax_port/train_dp.py --model 7b --epochs 3 --max_seq_length 2048
```

### 14B SPMD Tensor Parallelism (`train_spmd.py`)

14B model does not fit in a single TPU core (28GB > 16GB). We shard every linear layer across 8 cores via `pjit`:
- `hidden_size=5120` → 640 per core ✅
- `intermediate_size=13824` → 1728 per core ✅
- `heads=40` → 5 per core ✅

```bash
python jax_flax_port/train_spmd.py --model 14b --epochs 1 --max_seq_length 1024
```

## Kaggle Quickstart

1. **Create a new Kaggle notebook**
2. **Accelerator**: Select `TPU VM v3-8`
3. **Upload** the `jax_flax_port/` folder as a dataset, OR paste cells from `kaggle_notebook.ipynb`
4. **Add secret**: `HF_TOKEN` in Kaggle notebook settings
5. **Run All**

The notebook will:
1. Install dependencies (`flax`, `optax`, `transformers`)
2. Download and convert PyTorch weights to Flax (~5 min)
3. Start training on all 8 TPU cores

## Expected Performance

| Config | Step time | Total time (3 epochs) |
|--------|-----------|----------------------|
| 14B, seq 2048, batch 1, grad accum 4 (SPMD) | ~10-15s | ~2-3 days |
| 7B, seq 2048, batch 1×8, grad accum 4 (DP) | ~3-5s | ~6-10h |

Compare to GPU (2×T4): ~75s/step → **15-25× speedup**

**Default**: 14B, 3 epochs, seq 2048, filter overlength examples

## Checkpoint Format

Checkpoints are saved as NPZ files (portable, no pickle):
- `params.npz` — flattened model parameters
- `opt_state.npz` — flattened optimizer state
- `metadata.json` — step count, RNG seed

To resume:
```python
from checkpoint import load_checkpoint
state = load_checkpoint("checkpoints/checkpoint-100", state_template)
```

## Converting Weights Manually

If you want to pre-convert weights (saves time on re-runs):
```bash
python jax_flax_port/convert_weights.py --model 7b --output /kaggle/working/qwen_7b_jax.npz
```

Then train with:
```bash
python jax_flax_port/train_dp.py --from_jax_weights /kaggle/working/qwen_7b_jax.npz
```

## Memory Notes

- 7B in bfloat16: ~14GB params + ~2GB activations + ~0.5GB LoRA → fits comfortably in 16GB/core
- 14B in bfloat16: ~28GB params, sharded to ~3.5GB/core + activations → fits with TP
- AdamW optimizer state is replicated in `train_dp.py`; for 14B SPMD it is also sharded

## Limitations

- This port implements Qwen2 architecture manually. It matches PyTorch weights but may diverge on edge cases (e.g., very long context beyond 32k).
- `train_spmd.py` uses basic tensor parallelism. For production-scale 14B training, consider adding pipeline parallelism or FSDP.
- Generation/sampling code is not included (only training). Add `model.apply(..., method=model.generate)` if needed.

## References

- [Flax documentation](https://flax.readthedocs.io/)
- [JAX multi-device training](https://jax.readthedocs.io/en/latest/multi_process.html)
- [Qwen2 architecture](https://huggingface.co/docs/transformers/model_doc/qwen2)
