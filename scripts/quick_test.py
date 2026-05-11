#!/usr/bin/env python3
"""
Quick test suite for Verilog fine-tuned model.
Tests on small set of diverse problems without full benchmark.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.templates import get_all_templates
from src.validator import VerilogValidator
from src.benchmark_suite import VerilogEvalBenchmark, BenchmarkRunner


def create_quick_benchmark(output_path: str = "data/quick_test.jsonl"):
    """Create a quick benchmark from templates (one per category)."""
    import json
    
    templates = get_all_templates()
    categories = {}
    
    # Pick one template per category
    for t in templates:
        cat = t["category"]
        if cat not in categories:
            categories[cat] = t
    
    problems = []
    for cat, t in categories.items():
        problem = {
            "task_id": f"quick_{cat}",
            "prompt": t["spec"],
            "canonical_solution": t["code"],
            "test": "",
            "category": cat,
        }
        problems.append(problem)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p in problems:
            f.write(json.dumps(p) + '\n')
    
    print(f"Quick benchmark saved: {output_path}")
    print(f"Categories: {list(categories.keys())}")
    return output_path


def run_syntax_only_test(benchmark_path: str):
    """Run syntax-only evaluation without model (tests benchmark infrastructure)."""
    import json
    
    print("\nRunning syntax-only validation test...")
    validator = VerilogValidator()
    
    with open(benchmark_path) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    
    results = []
    for p in problems:
        ok, err = validator.check_syntax(p["canonical_solution"])
        results.append((p["task_id"], ok, err))
        status = "✓" if ok else "✗"
        print(f"  {status} {p['task_id']:20s} {p['category']}")
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nSyntax check: {passed}/{total} passed")
    
    return passed == total


def main():
    print("=" * 60)
    print("Verilog LLM Quick Test Suite")
    print("=" * 60)
    
    # 1. Create quick benchmark
    bench_path = create_quick_benchmark()
    
    # 2. Validate all canonical solutions compile
    all_ok = run_syntax_only_test(bench_path)
    
    if all_ok:
        print("\n✓ All templates compile successfully!")
        print("✓ Dataset generator is ready")
        print("✓ Benchmark infrastructure is ready")
    else:
        print("\n✗ Some templates failed compilation")
        print("  Fix templates before generating dataset")
    
    print("\nNext steps:")
    print("  1. python scripts/build_dataset.py --num 5000")
    print("  2. Upload data/ to Kaggle")
    print("  3. Run notebooks/kaggle_train.ipynb")


if __name__ == "__main__":
    main()
