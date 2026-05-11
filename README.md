# Verilog LLM — Fine-tuning Qwen2.5-Coder-14B for HDL Generation

Fine-tuning project for State-of-the-Art Verilog code generation using Qwen2.5-Coder-14B with QLoRA on Kaggle (2x T4).

## Hardware Requirements

| Component | Spec | Notes |
|---|---|---|
| **Model** | Qwen2.5-Coder-14B | 14B params, code-specific |
| **Quantization** | 4-bit (NF4) | ~7.5 GB VRAM |
| **LoRA adapters** | r=64, α=128 | ~0.3 GB |
| **Total VRAM (1x T4)** | ~10-11 GB | Fits in 16GB T4 comfortably |
| **Kaggle** | 2x T4 (32GB total) | Can use batch=2 or seq=4096 |

**Yes, Qwen2.5-Coder-14B fits perfectly in Kaggle 2x T4 with QLoRA.**

## Project Structure

```
verilog-llm/
├── notebooks/
│   └── kaggle_train.ipynb          # Main Kaggle notebook
├── scripts/
│   ├── train.py                     # Headless training script
│   ├── generate_dataset.py          # Synthetic dataset generation
│   └── evaluate.py                  # VerilogEval/RTLLM evaluation
├── configs/
│   └── qwen2.5_coder_14b_qlora.yaml # Training config
├── src/
│   ├── dataset.py                   # Dataset classes
│   ├── model.py                     # Model setup (QLoRA)
│   ├── training.py                  # Training loop
│   └── utils.py                     # Helpers
└── data/                            # Training data (not tracked)
```

## Techniques Used

1. **CodeV**: Chat-FIM-Tag format (`<verilog>` tags, FIM support)
2. **HaVen**: SI-CoT (convert truth tables/state diagrams to text before code)
3. **VeriReason**: Reasoning steps with `<think>` blocks
4. **CraftRTL**: Correct-by-construction synthetic data generation

## Training Config

- **Base**: Qwen/Qwen2.5-Coder-14B-Instruct
- **Method**: QLoRA 4-bit (r=64, α=128, dropout=0.05)
- **LR**: 2e-4 with cosine warmup
- **Epochs**: 3
- **Batch**: 1-2 per device (2x T4 = effective batch 2-4)
- **Max length**: 2048-4096
- **Dataset target**: 100K-300K examples

## Quick Start (Kaggle)

1. Upload `notebooks/kaggle_train.ipynb` to Kaggle
2. Add T4 GPU accelerator (2x T4)
3. Add dataset to Input: `train.jsonl` and `eval.jsonl`
4. HF token is hardcoded in notebook: `HF_TOKEN_PLACEHOLDER`
5. Run all cells

**Tokens configured:**
- HuggingFace: `HF_TOKEN_PLACEHOLDER`
- Kaggle API: `KAGGLE_TOKEN_PLACEHOLDER` (in `scripts/setup_kaggle.py`)

**Notebook includes:**
- Verilog post-processor (fixes begin/end, module/endmodule, parens, semicolons)
- Auto-push to `pabloflores/verilog-qwen-14b-sota` on HF Hub

## Checkpointing

Checkpoints auto-push to HuggingFace Hub every 100 steps. Resume with:
```python
trainer.train(resume_from_checkpoint=True)
```

## Evaluation

After training, run evaluation on:
- **VerilogEval-Machine** (143 problems)
- **VerilogEval-Human** (156 problems)
- **RTLLM v1.1** (29 problems)
- **VerilogEval-FIM** (fill-in-the-middle)

## Expected Results

With 100K+ curated Verilog examples:
- VerilogEval-Machine pass@1: ~65-75%
- VerilogEval-Human pass@1: ~45-55%
- RTLLM functional: ~50-60%

## License

Apache 2.0
