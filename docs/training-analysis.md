# Verilog QLoRA Training Analysis

## Run Summary

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-Coder-14B-Instruct |
| GPUs | 2× Tesla T4 (16GB each) |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA params | 275,251,200 (~1.83% of total) |
| Dataset | 19,000 train + 1,000 eval |
| Batch size | 1 per device |
| Gradient accumulation | 4 |
| Effective batch | 8 (2 GPUs × 1 × 4) |
| Seq length | 2048 |
| Learning rate | 2e-4 |
| Epochs configured | 3 |

## Training Metrics (First 500 Steps)

| Step | Time (s) | Loss | Token Accuracy | Epoch |
|------|----------|------|----------------|-------|
| 10 | 540 | 1.881 | 68.3% | 0.0002 |
| 100 | 1,207 | 1.383 | 72.5% | 0.0021 |
| 200 | 1,968 | 0.705 | 82.7% | 0.0042 |
| 300 | 2,708 | 0.312 | 91.4% | 0.0063 |
| 400 | 3,436 | 0.123 | 96.3% | 0.0084 |
| 500 | 4,176 | 0.075 | 97.8% | 0.0105 |
| 600 | 4,941 | 0.052 | 98.5% | 0.0126 |
| 700 | 5,696 | 0.035 | 99.0% | 0.0147 |
| 800 | 6,458 | 0.025 | 99.3% | 0.0168 |
| 900 | 7,183 | 0.027 | 99.1% | 0.0190 |
| 1000 | 7,943 | 0.026 | 99.3% | 0.0211 |
| ... | ... | ... | ... | ... |
| 1500 | ~11,000 | ~0.017 | ~99.5% | ~0.03 |
| 2000 | ~14,500 | ~0.014 | ~99.5% | ~0.04 |
| 2500 | ~18,000 | ~0.012 | ~99.6% | ~0.05 |
| 3000 | ~21,500 | ~0.013 | ~99.5% | ~0.06 |
| 3500 | ~25,000 | ~0.014 | ~99.5% | ~0.07 |
| 4000 | ~28,500 | ~0.013 | ~99.6% | ~0.08 |
| 4500 | ~32,000 | ~0.015 | ~99.5% | ~0.09 |
| 5000 | ~35,500 | ~0.012 | ~99.6% | ~0.10 |

**Average step time: ~75 seconds**

## Bottleneck Analysis

The training is **functionally correct** but **prohibitively slow**:

1. **Model size**: 14B parameters in 4-bit = ~7GB base + 4.8GB (GPU 0) + 9.5GB (GPU 1) with LoRA overhead
2. **Sequence length**: 2048 tokens × batch 1 × grad accum 4 = high memory pressure per step
3. **T4 bandwidth**: Tesla T4 has limited memory bandwidth (320 GB/s) vs Ampere A100 (1,935 GB/s)
4. **Kaggle GPU limit**: 30 hours/week free tier

## Time Projection

| Scenario | Steps | Time | Fits in Kaggle free? |
|----------|-------|------|---------------------|
| 3 epochs (current config) | ~14,250 | **~12-13 days** | ❌ No |
| 1 epoch | ~4,750 | **~4 days** | ❌ No |
| 1 epoch, seq 1024 | ~4,750 | **~2.5 days** | ❌ Marginal |
| 7B model, 1 epoch, seq 1024 | ~4,750 | **~8-10 hours** | ✅ Yes |
| 7B model, 3 epochs, seq 2048 | ~14,250 | **~2-3 days** | ❌ No |

## Decision Options

### Option A: Continue with 14B, 3 epochs
- **Pros**: Best possible model quality
- **Cons**: ~12 days, exceeds Kaggle free tier, costs ~$50-100 on paid Kaggle or cloud
- **Verdict**: ❌ Not viable for free tier

### Option B: Reduce to 1 epoch, keep 14B
- **Pros**: Saves 2/3 of time, still 14B quality
- **Cons**: Still ~4 days, exceeds Kaggle free tier
- **Verdict**: ❌ Not viable for free tier

### Option C: Switch to 7B model, 3 epochs, seq 2048
- **Pros**: 7B is still very capable; 3 epochs for full convergence
- **Cons**: ~2-3 days, still exceeds Kaggle free tier
- **Verdict**: ❌ Marginal

### Option D: Switch to 7B model, 1 epoch, seq 1024 ⭐ RECOMMENDED
- **Pros**: 
  - Fits in **8-10 hours** (well within Kaggle free 30h/week)
  - 7B is sufficient for Verilog code generation
  - seq 1024 covers most Verilog modules
  - Can run to completion in one session
- **Cons**: 
  - Less capacity than 14B
  - Shorter context window
- **Verdict**: ✅ **Best option for free tier**

### Option E: Use Google Colab (A100/L4)
- **Pros**: A100 is 6× faster than T4; could finish 14B in ~2 days
- **Cons**: A100 costs ~$1.20/hour on Colab Pro; L4 is ~$0.80/hour
- **Verdict**: 💰 Viable if willing to pay ~$20-30

### Option F: Switch to Kaggle TPU v3-8 ⭐ NEW
- **Pros**:
  - **20 hours/week TPU quota** (separate from GPU quota)
  - **~10-15s/step** (5-7× faster than T4)
  - 128GB HBM total = no quantization needed, native bfloat16
  - 7B model in bfloat16 fits in **1 core** (16GB) with room to spare
  - 1 epoch, seq 2048, 7B = **~15-20 hours** ✅ fits in TPU quota
- **Cons**:
  - Requires porting code (no bitsandbytes, use bfloat16)
  - TPU v3-8 needs phone-verified Kaggle account
  - 14B model requires FSDP multi-core (more complex)
- **Verdict**: ✅ **Best speed/cost for free tier**

## Recommendation

**Primary: Option F** — Use Kaggle TPU v3-8 with 7B model, bfloat16, 1-3 epochs.
- Script: `tpu/kaggle_tpu_script.py`
- Push: `cd tpu && kaggle kernels push -p .`
- TPU quota is separate from GPU quota (20h/week)
- ~5-7× faster than T4

**Fallback: Option D** — If TPU is unavailable or has issues, use GPU with 7B + 1 epoch + seq 1024.

See `tpu/README.md` for TPU-specific setup and limitations.

## Checkpoints Uploaded

| Checkpoint | HF Hub Path | Upload Time |
|-----------|-------------|-------------|
| checkpoint-100 | `Pablo-Flores-Mollinedo/verilog-qwen-14b-sota/checkpoint-100` | ✓ |
| checkpoint-200 | `Pablo-Flores-Mollinedo/verilog-qwen-14b-sota/checkpoint-200` | ✓ |
| checkpoint-300 | `Pablo-Flores-Mollinedo/verilog-qwen-14b-sota/checkpoint-300` | ✓ |
| checkpoint-400 | `Pablo-Flores-Mollinedo/verilog-qwen-14b-sota/checkpoint-400` | ✓ |
| checkpoint-500 | `Pablo-Flores-Mollinedo/verilog-qwen-14b-sota/checkpoint-500` | ✓ |

These checkpoints are usable but training was interrupted at ~10 hours due to Kaggle session timeout.
