# TPU Training Port (SPMD FSDP)

## Quick Start

```bash
cd verilog-llm/tpu
kaggle kernels push -p .
```

**NOTE:** TPU on Kaggle requires **notebook format** (`.ipynb`), not script (`.py`).
The kernel uses `kaggle_tpu_fsdp.ipynb` with `"kernel_type": "notebook"`.

## Requirements

- Kaggle account with **phone verification** (required for TPU access)
- Set accelerator to **TPU VM v3-8** in notebook settings
- Add `HF_TOKEN` to Kaggle Secrets

## Architecture: SPMD FSDP

Este notebook usa **SPMD (Single Program Multiple Data)** con `torch_xla.experimental.xla_sharding` para shard el modelo Qwen 14B a través de los 8 cores del TPU v3-8.

```
Mesh shape: (1, 8, 1)  -> (dp=1, fsdp=8, mp=1)
- 14B params / 8 cores = ~3.5GB por core
- Activaciones seq 2048 = ~3-4GB por core
- Total por core = ~7-8GB ✅ (dentro de 16GB)
```

## Differences from GPU Version

| Aspect | GPU (2× T4) | TPU (v3-8 SPMD) |
|--------|-------------|-----------------|
| Quantization | 4-bit NF4 (bitsandbytes) | **bfloat16** (native) |
| Model | 14B en 4-bit | **14B en bfloat16** |
| Speed | ~75s/step | **~3-5s/step** (est.) |
| Memory | 32GB CUDA | **128GB HBM** total |
| Approach | QLoRA | **LoRA + SPMD sharding** |
| Epochs 3 | ~12 días | **~3-4 horas** |

## Key Files

| File | Purpose |
|------|---------|
| `kaggle_tpu_fsdp.ipynb` | **Kaggle notebook** (required for TPU) — contains all training code |
| `kaggle_tpu_fsdp_script.py` | Source Python script (for reference) |
| `spmd_util.py` | Model sharding rules for Qwen/LLaMA/Mistral/etc |
| `kernel-metadata.json` | Kaggle kernel metadata (points to `.ipynb`) |

## Dataset Filtering

**IMPORTANTE**: Ejemplos que excedan `MAX_SEQ_LENGTH` (2048 tokens) se **ELIMINAN**, no truncan.

```python
def tokenize_and_filter(text, max_len):
    tokens = tokenizer(text, truncation=False)  # NO truncar
    if len(tokens["input_ids"]) > max_len:
        return None  # Eliminar
    return tokens
```

## Expected Performance

| Config | Steps | Est. Time | Fits TPU quota? |
|--------|-------|-----------|-----------------|
| 14B, 3 epochs, seq 2048 | ~1,782 | **~3-4 horas** | ✅ Sí (20h/week) |
| 14B, 1 epoch, seq 2048 | ~594 | **~1-1.5 horas** | ✅ Sí |

## SPMD Sharding Rules

El archivo `spmd_util.py` define cómo se particionan las capas del modelo:

```python
QWEN_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),   # Embeddings: mp×fsdp
    ("self_attn\\.(q|k|v)_proj", ("fsdp", "mp")), # Q/K/V: fsdp×mp
    ("self_attn\\.o_proj", ("mp", "fsdp")),      # Output: mp×fsdp
    ("mlp\\.gate_proj", ("fsdp", "mp")),         # MLP up: fsdp×mp
    ("mlp\\.down_proj", ("mp", "fsdp")),         # MLP down: mp×fsdp
    ("lm_head", ("fsdp", "mp")),                  # Head: fsdp×mp
)
```

## Troubleshooting

### "Unsupported model to partitioning"
Asegúrate de que el modelo sea Qwen2, Llama, Mistral, o alguno de los soportados en `spmd_util.py`.

### OOM en TPU
Reduce `MAX_SEQ_LENGTH` o aumenta `GRAD_ACCUM`.

### Dataset no se monta
El script intenta:
1. `/kaggle/input/verilog-curated-dataset/`
2. Cualquier dataset con "verilog" en el nombre
3. Descarga con `kagglehub`
