#!/usr/bin/env python3
"""
Build SOTA-quality Verilog training dataset.
Generates examples, validates with iverilog, applies SOTA techniques.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import build_sota_dataset


def main():
    parser = argparse.ArgumentParser(description="Build Verilog SOTA dataset")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--num", type=int, default=5000, help="Number of base examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate", action="store_true", default=True, help="Validate with iverilog")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building SOTA Verilog Dataset")
    print("=" * 60)
    print(f"Target: {args.num} examples")
    print(f"Output: {args.output}")
    print(f"Validation: {'iverilog' if args.validate else 'none'}")
    print("=" * 60)
    
    train, eval_data, fim = build_sota_dataset(
        output_dir=args.output,
        num_examples=args.num,
        seed=args.seed,
    )
    
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Train:   {len(train)} examples")
    print(f"Eval:    {len(eval_data)} examples")
    print(f"FIM:     {len(fim)} examples")
    print(f"Total:   {len(train) + len(eval_data) + len(fim)} examples")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Upload data/ to Kaggle Dataset")
    print("2. Run: python scripts/train.py --config configs/qwen2.5_coder_14b_qlora.yaml")


if __name__ == "__main__":
    main()
