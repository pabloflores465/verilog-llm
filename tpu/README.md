# TPU Training Port

## Quick Start

```bash
cd verilog-llm/tpu
kaggle kernels push -p .
```

## Requirements

- Kaggle account with TPU access (phone verification required)
- Set accelerator to **TPU VM v3-8** in notebook settings

## Differences from GPU Version

| Aspect | GPU (2× T4) | TPU (v3-8) |
|--------|-------------|------------|
| Quantization | 4-bit NF4 (bitsandbytes) | **bfloat16** (native) |
| Model size | 14B possible | 7B recommended (14B needs FSDP) |
| Speed | ~75s/step | **~10-15s/step** (est.) |
| Memory | 32GB total | **128GB HBM total** (16GB/core) |
| Multi-core | DataParallel via accelerate | Single-core (simple) or FSDP (complex) |

## TPU Limitations

- `bitsandbytes` is **CUDA-only** — not available on TPU
- `bnb_4bit_compute_type` → use `torch.bfloat16` instead
- `paged_adamw_8bit` → use standard `adamw_torch`
- No `prepare_model_for_kbit_training` needed (no quantization)

## Architecture

```
Single-core mode (default):
  - Uses 1 TPU core (16GB HBM)
  - 7B model in bfloat16 = ~14GB ✅ fits
  - Simple, stable, no multiprocessing issues

Multi-core mode (future):
  - 14B model sharded across 2+ cores via FSDP
  - Requires torch_xla.distributed.xla_multiprocessing
  - More complex, higher throughput
```

## Expected Performance

| Config | Steps | Est. Time |
|--------|-------|-----------|
| 7B, 1 epoch, seq 2048 | ~4,750 | **~15-20 hours** |
| 7B, 3 epochs, seq 2048 | ~14,250 | **~2-3 days** |
| 14B + FSDP (future) | ~14,250 | ~1-2 days |
