#!/usr/bin/env python3
"""
Evaluation script for Verilog models on standard benchmarks.
Supports VerilogEval and RTLLM formats.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model: str, adapter_path: str = None):
    """Load model with optional LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    
    return model, tokenizer


def generate_verilog(model, tokenizer, prompt: str, max_tokens: int = 2048) -> str:
    """Generate Verilog code from prompt."""
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
    
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response


def extract_code(text: str) -> str:
    """Extract and fix Verilog code from model output."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.verilog_postprocess import postprocess_verilog
    return postprocess_verilog(text)


def check_syntax_iverilog(code: str) -> tuple[bool, str]:
    """Check Verilog syntax using Icarus Verilog."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(code)
            tmp_path = f.name
        
        result = subprocess.run(
            ['iverilog', '-g2012', '-o', '/dev/null', tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        os.unlink(tmp_path)
        
        if result.returncode == 0:
            return True, ""
        return False, result.stderr
    except FileNotFoundError:
        return None, "iverilog not installed"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def evaluate_file(model, tokenizer, benchmark_path: str, output_path: str, n: int = 1):
    """Evaluate on a benchmark file."""
    with open(benchmark_path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    
    results = []
    
    for prob in tqdm(problems, desc="Evaluating"):
        prompt = prob.get("prompt", prob.get("instruction", ""))
        canonical = prob.get("canonical_solution", prob.get("code", ""))
        
        generations = []
        for _ in range(n):
            raw = generate_verilog(model, tokenizer, prompt)
            code = extract_code(raw)
            generations.append(code)
        
        # Check syntax
        syntax_pass = []
        for code in generations:
            ok, err = check_syntax_iverilog(code)
            syntax_pass.append(ok)
        
        result = {
            "task_id": prob.get("task_id", ""),
            "prompt": prompt,
            "generations": generations,
            "syntax_correct": syntax_pass,
            "canonical": canonical,
        }
        results.append(result)
    
    # Save results
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Report
    total = len(results)
    any_syntax = sum(1 for r in results if any(r["syntax_correct"]))
    print(f"\nResults: {any_syntax}/{total} ({any_syntax/total*100:.1f}%) pass syntax check")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark JSONL")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--n", type=int, default=1, help="Samples per problem")
    args = parser.parse_args()
    
    print("Loading model...")
    model, tokenizer = load_model(args.base_model, args.adapter)
    
    print(f"\nEvaluating on {args.benchmark}...")
    evaluate_file(model, tokenizer, args.benchmark, args.output, args.n)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
