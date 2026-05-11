"""
Benchmark suite for Verilog code generation evaluation.
Supports VerilogEval and RTLLM formats.
"""

import json
import re
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    task_id: str
    prompt: str
    generations: List[str]
    syntax_correct: List[bool]
    functional_correct: List[bool]
    canonical: str
    
    @property
    def pass_at_1(self) -> bool:
        return len(self.syntax_correct) > 0 and self.syntax_correct[0]
    
    @property
    def pass_at_k(self, k: int = 5) -> bool:
        return any(self.syntax_correct[:k])


class VerilogBenchmark:
    """Base class for Verilog benchmarks."""
    
    def __init__(self, benchmark_path: str):
        self.path = Path(benchmark_path)
        self.problems = self._load_problems()
    
    def _load_problems(self) -> List[Dict]:
        """Load benchmark problems from JSONL."""
        problems = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    problems.append(json.loads(line))
        return problems
    
    def get_problem(self, idx: int) -> Dict:
        return self.problems[idx]
    
    def __len__(self):
        return len(self.problems)


class VerilogEvalBenchmark(VerilogBenchmark):
    """
    VerilogEval benchmark loader.
    Format: {"task_id": str, "prompt": str, "canonical_solution": str, "test": str}
    """
    
    def get_prompt(self, problem: Dict) -> str:
        """Format prompt with HaVen SI-CoT for evaluation."""
        prompt = problem.get("prompt", "")
        
        # Add thinking instruction for models trained with reasoning
        enhanced = f"""You are an expert Verilog designer.

Design Task:
{prompt}

Think through the design carefully, then provide complete synthesizable Verilog code.

Format:
<think>
[Your design analysis]
</think>
<answer>
```verilog
[Complete module]
```
</answer>
"""
        return enhanced
    
    def evaluate_generation(
        self,
        generated_code: str,
        problem: Dict,
        run_functional: bool = False,
    ) -> Tuple[bool, bool, str]:
        """
        Evaluate a single generation.
        Returns: (syntax_ok, functional_ok, error_msg)
        """
        code = self._extract_code(generated_code)
        
        # Syntax check
        syntax_ok, err = self._check_syntax(code)
        if not syntax_ok:
            return False, False, err
        
        # Functional check (optional, requires testbench)
        functional_ok = False
        if run_functional and "test" in problem:
            functional_ok, func_err = self._run_testbench(code, problem["test"])
            if not functional_ok:
                err = func_err
        
        return syntax_ok, functional_ok, err
    
    def _extract_code(self, text: str) -> str:
        """Extract Verilog code from generation using post-processor."""
        from .verilog_postprocess import postprocess_verilog
        return postprocess_verilog(text)
    
    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check Verilog syntax with iverilog."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                vfile = Path(tmpdir) / "test.v"
                vfile.write_text(code)
                
                result = subprocess.run(
                    ["iverilog", "-g2012", "-o", str(Path(tmpdir) / "out"), str(vfile)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode == 0:
                    return True, ""
                return False, result.stderr[:500]
        except Exception as e:
            return False, str(e)
    
    def _run_testbench(self, code: str, test: str) -> Tuple[bool, str]:
        """Run testbench against generated code."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write design and testbench
                (Path(tmpdir) / "design.v").write_text(code)
                (Path(tmpdir) / "tb.v").write_text(test)
                
                # Compile
                result = subprocess.run(
                    ["iverilog", "-g2012", "-o", str(Path(tmpdir) / "sim"), 
                     str(Path(tmpdir) / "design.v"), str(Path(tmpdir) / "tb.v")],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode != 0:
                    return False, f"Compile error: {result.stderr[:200]}"
                
                # Run simulation
                run_result = subprocess.run(
                    [str(Path(tmpdir) / "sim")],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                
                # Check for assertion failures or errors
                output = run_result.stdout + run_result.stderr
                if "error" in output.lower() or "fail" in output.lower():
                    return False, output[:500]
                
                return True, ""
        except Exception as e:
            return False, str(e)


class RTLLMBenchmark(VerilogBenchmark):
    """
    RTLLM benchmark loader.
    Format: {"task_id": str, "prompt": str, "canonical_solution": str}
    """
    
    def get_prompt(self, problem: Dict) -> str:
        prompt = problem.get("prompt", "")
        return f"""Design the following RTL module in Verilog:

{prompt}

Provide complete, synthesizable code.
"""


class BenchmarkRunner:
    """Runs evaluation on a benchmark using a model."""
    
    def __init__(self, benchmark: VerilogBenchmark, n_samples: int = 1):
        self.benchmark = benchmark
        self.n = n_samples
    
    def run(
        self,
        generate_fn,
        output_path: Optional[str] = None,
        run_functional: bool = False,
    ) -> Dict:
        """
        Run full benchmark evaluation.
        
        Args:
            generate_fn: Function that takes (prompt) -> str (generated code)
            output_path: Where to save detailed results
            run_functional: Whether to run functional simulation
        
        Returns:
            Dict with aggregate metrics
        """
        results = []
        
        for i, problem in enumerate(self.benchmark.problems):
            print(f"  Evaluating {i+1}/{len(self.benchmark)}: {problem.get('task_id', i)}")
            
            prompt = self.benchmark.get_prompt(problem)
            canonical = problem.get("canonical_solution", "")
            
            generations = []
            syntax_correct = []
            functional_correct = []
            
            for _ in range(self.n):
                raw = generate_fn(prompt)
                code = self.benchmark._extract_code(raw) if hasattr(self.benchmark, '_extract_code') else raw
                generations.append(code)
                
                if isinstance(self.benchmark, VerilogEvalBenchmark):
                    syn_ok, func_ok, err = self.benchmark.evaluate_generation(
                        raw, problem, run_functional
                    )
                else:
                    syn_ok, err = self.benchmark._check_syntax(code) if hasattr(self.benchmark, '_check_syntax') else (True, "")
                    func_ok = False
                
                syntax_correct.append(syn_ok)
                functional_correct.append(func_ok)
            
            result = BenchmarkResult(
                task_id=problem.get("task_id", str(i)),
                prompt=prompt,
                generations=generations,
                syntax_correct=syntax_correct,
                functional_correct=functional_correct,
                canonical=canonical,
            )
            results.append(result)
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        
        # Save detailed results
        if output_path:
            self._save_results(results, output_path)
        
        return metrics
    
    def _compute_metrics(self, results: List[BenchmarkResult]) -> Dict:
        """Compute aggregate metrics."""
        total = len(results)
        
        syntax_pass_1 = sum(1 for r in results if r.syntax_correct[0])
        syntax_pass_k = sum(1 for r in results if any(r.syntax_correct))
        
        func_pass_1 = sum(1 for r in results if r.functional_correct[0])
        func_pass_k = sum(1 for r in results if any(r.functional_correct))
        
        return {
            "total": total,
            "syntax_pass@1": syntax_pass_1,
            "syntax_pass@1_pct": syntax_pass_1 / total * 100 if total else 0,
            "syntax_pass@k": syntax_pass_k,
            "syntax_pass@k_pct": syntax_pass_k / total * 100 if total else 0,
            "func_pass@1": func_pass_1,
            "func_pass@1_pct": func_pass_1 / total * 100 if total else 0,
            "func_pass@k": func_pass_k,
            "func_pass@k_pct": func_pass_k / total * 100 if total else 0,
        }
    
    def _save_results(self, results: List[BenchmarkResult], path: str):
        """Save detailed results to JSONL."""
        with open(path, 'w') as f:
            for r in results:
                f.write(json.dumps({
                    "task_id": r.task_id,
                    "syntax_correct": r.syntax_correct,
                    "functional_correct": r.functional_correct,
                    "generation_0": r.generations[0] if r.generations else "",
                }) + '\n')
        print(f"Detailed results saved to {path}")


def create_verilogeval_format(
    task_id: str,
    prompt: str,
    canonical: str,
    test: str = "",
) -> Dict:
    """Create a problem in VerilogEval format."""
    return {
        "task_id": task_id,
        "prompt": prompt,
        "canonical_solution": canonical,
        "test": test,
    }


def create_rtllm_format(
    task_id: str,
    prompt: str,
    canonical: str,
) -> Dict:
    """Create a problem in RTLLM format."""
    return {
        "task_id": task_id,
        "prompt": prompt,
        "canonical_solution": canonical,
    }
