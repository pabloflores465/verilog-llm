#!/usr/bin/env python3
"""
Main training script for Verilog fine-tuning.
Can run headless via Kaggle API or locally.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model_and_tokenizer, setup_lora
from src.dataset import load_and_prepare_dataset
from src.training import create_trainer, merge_and_save


def parse_args():
    parser = argparse.ArgumentParser(description="Train Verilog LLM")
    parser.add_argument("--config", type=str, default="configs/qwen2.5_coder_14b_qlora.yaml")
    parser.add_argument("--local", action="store_true", help="Run locally instead of Kaggle")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA after training")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # HF Token (from env or Kaggle secrets)
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", ""))
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    
    print("=" * 60)
    print("Verilog LLM Fine-tuning")
    print("=" * 60)
    print(f"Base model: {config['model']['base_model']}")
    print(f"LoRA r={config['lora']['r']}, alpha={config['lora']['lora_alpha']}")
    print(f"Batch: {config['training']['per_device_train_batch_size']} per device")
    print(f"Accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print("=" * 60)
    
    # Load model and tokenizer
    print("\n[1/4] Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=config["model"]["base_model"],
        use_4bit=config["quantization"]["load_in_4bit"],
        use_nested_quant=config["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=config["quantization"]["bnb_4bit_compute_dtype"],
    )
    
    # Setup LoRA
    print("\n[2/4] Setting up LoRA adapters...")
    model = setup_lora(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        use_rslora=config["lora"].get("use_rslora", False),
    )
    
    # Load dataset
    print("\n[3/4] Loading and preparing dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        train_path=config["dataset"]["train_file"],
        eval_path=config["dataset"].get("eval_file"),
        tokenizer=tokenizer,
        max_length=config["training"]["max_seq_length"],
        format_type=config["dataset"].get("format", "chat"),
    )
    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval examples: {len(eval_dataset)}")
    
    # Setup trainer
    print("\n[4/4] Setting up trainer...")
    training_args = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
    )
    
    # Resume if requested
    resume_from = None
    if args.resume:
        output_dir = config["training"]["output_dir"]
        if os.path.exists(output_dir):
            resume_from = True  # Auto-detect last checkpoint
            print(f"\nResuming from checkpoint in {output_dir}")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    training_args.train(resume_from_checkpoint=resume_from)
    
    # Save final adapter
    final_path = os.path.join(config["training"]["output_dir"], "adapter_v1")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal adapter saved to {final_path}")
    
    # Merge if requested
    if args.merge:
        merged_path = os.path.join(config["training"]["output_dir"], "merged")
        merge_and_save(model, tokenizer, merged_path)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
