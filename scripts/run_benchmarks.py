#!/usr/bin/env python3
"""
Run benchmark evaluation on trained model.
Supports VerilogEval and RTLLM benchmarks.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark_suite import VerilogEvalBenchmark, RTLLMBenchmark, BenchmarkRunner
from src.model import load_model_and_tokenizer
from src.utils import extract_verilog_code


def generate_fn_factory(model, tokenizer, max_tokens: int = 2048):
    """Create generation function for benchmark."""
    def generate(prompt: str) -> str:
        from transformers import pipeline
        
        messages = [
            {"role": "system", "content": "You are an expert Verilog designer."},
            {"role": "user", "content": prompt},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        return tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    return generate


def main():
    parser = argparse.ArgumentParser(description="Run Verilog benchmarks")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--benchmark_type", choices=["verilogeval", "rtllm"], default="verilogeval")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--n", type=int, default=1, help="Samples per problem")
    parser.add_argument("--functional", action="store_true", help="Run functional tests")
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.base_model,
        use_4bit=True,
    )
    
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Loaded adapter: {args.adapter}")
    
    print(f"\nLoading benchmark: {args.benchmark}")
    if args.benchmark_type == "verilogeval":
        benchmark = VerilogEvalBenchmark(args.benchmark)
    else:
        benchmark = RTLLMBenchmark(args.benchmark)
    
    print(f"Problems: {len(benchmark)}")
    print(f"Samples per problem: {args.n}")
    
    runner = BenchmarkRunner(benchmark, n_samples=args.n)
    generate_fn = generate_fn_factory(model, tokenizer)
    
    print("\nRunning evaluation...")
    metrics = runner.run(generate_fn, output_path=args.output, run_functional=args.functional)
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for key, val in metrics.items():
        if "pct" in key:
            print(f"  {key}: {val:.1f}%")
        else:
            print(f"  {key}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
