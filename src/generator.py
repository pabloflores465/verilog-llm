"""
High-quality Verilog dataset generator with compile-time validation.
Integrates techniques from HaVen, CodeV, VeriReason, and CraftRTL.
"""

import random
import json
from typing import List, Dict, Optional
from pathlib import Path

from .templates import get_all_templates, PARAMETER_VARIATIONS
from .validator import VerilogValidator


class VerilogDatasetGenerator:
    """
    Generates curated Verilog training data with:
    - HaVen SI-CoT: Symbolic interpretation before coding
    - CodeV: Language tags + FIM format support
    - VeriReason: Explicit reasoning blocks
    - CraftRTL: Correct-by-construction (validated with iverilog)
    """
    
    def __init__(self, validator: Optional[VerilogValidator] = None, seed: int = 42):
        self.validator = validator or VerilogValidator()
        self.templates = get_all_templates()
        random.seed(seed)
    
    def generate_instruction(self, template: Dict, style: str = "detailed") -> str:
        """
        Generate instruction prompt with HaVen SI-CoT + CodeV tag.
        Multiple styles for diversity.
        """
        spec = template["spec"]
        
        styles = {
            "detailed": f"""<verilog>
You are a professional hardware design engineer specializing in Verilog RTL design.

Design Specification:
{spec}

Before writing any code:
1. Carefully analyze the functional requirements
2. Identify all inputs, outputs, and internal signals needed
3. Determine if the circuit is combinational or sequential
4. Plan the implementation approach (structural, dataflow, or behavioral)
5. Consider edge cases and default values

CRITICAL RULES for correct Verilog:
- Every `begin` MUST have a matching `end`
- Every `module` MUST have a matching `endmodule`
- Every `case` MUST have a matching `endcase`
- Check balance before finishing

Then write complete, synthesizable, industry-standard Verilog-2001 or SystemVerilog code.
Include meaningful signal names and comments.

Respond in this exact format:
<think>
[Your step-by-step analysis and design decisions]
</think>
<answer>
```verilog
[Complete, compilable Verilog module]
```
</answer>""",
            
            "concise": f"""<verilog>
Design a Verilog module for the following specification:
{spec}

Provide complete synthesizable code with reasoning.
Format:
<think>[analysis]</think>
<answer>```verilog
[code]
```</answer>""",
            
            "technical": f"""<verilog>
RTL Design Task:
{spec}

Deliverables:
- Complete module declaration
- All port declarations
- Internal signal declarations
- Combinational/sequential logic
- Proper sensitivity lists
- Synthesis-ready code

Use <think> for design rationale and <answer> for the implementation.
""",
            
            "tutorial": f"""<verilog>
As an expert Verilog instructor, design the following circuit and explain your approach:

{spec}

Walk through your design decisions before presenting the code.
""",
        }
        
        return styles.get(style, styles["detailed"])
    
    def generate_response(self, template: Dict, include_reasoning: bool = True) -> str:
        """Generate response with VeriReason-style reasoning + code."""
        reasoning = template.get("reasoning", "")
        code = template["code"]
        
        if not include_reasoning:
            return f"```verilog\n{code}\n```"
        
        return f"""<think>
{reasoning}
</think>
<answer>
```verilog
{code}
```
</answer>"""
    
    def generate_fim_example(self, template: Dict) -> Optional[Dict]:
        """
        Generate Fill-in-the-Middle example (CodeV FIM technique).
        Returns prefix/suffix/middle for FIM training.
        """
        code = template["code"]
        lines = code.split('\n')
        
        if len(lines) < 5:
            return None
        
        # Pick a split point in the body (not header or endmodule)
        split = random.randint(2, len(lines) - 2)
        prefix = '\n'.join(lines[:split])
        middle = '\n'.join(lines[split:split+2])
        suffix = '\n'.join(lines[split+2:])
        
        return {
            "prefix": prefix,
            "middle": middle,
            "suffix": suffix,
            "source": f"fim_{template['name']}",
        }
    
    def generate_variation(self, template: Dict, idx: int) -> Optional[Dict]:
        """
        Generate a single training example from template.
        Returns None if code fails validation.
        """
        style = random.choice(["detailed", "concise", "technical", "tutorial"])
        
        instruction = self.generate_instruction(template, style)
        response = self.generate_response(template, include_reasoning=True)
        
        example = {
            "spec": template["spec"],
            "instruction": instruction,
            "response": response,
            "code": template["code"],
            "reasoning": template.get("reasoning", ""),
            "category": template["category"],
            "source": template["name"],
            "id": f"{template['name']}_{idx}",
            "style": style,
        }
        
        # Validate syntax
        is_valid, err = self.validator.check_syntax(template["code"])
        if not is_valid:
            example["_valid"] = False
            example["_error"] = err
            return None  # Skip invalid examples
        
        example["_valid"] = True
        return example
    
    def generate_dataset(
        self,
        num_examples: int = 5000,
        include_fim: bool = True,
        validation_ratio: float = 0.1,
        verbose: bool = True,
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Generate full dataset with validation.
        Returns (train_examples, eval_examples).
        """
        all_examples = []
        fim_examples = []
        invalid_count = 0
        
        if verbose:
            print(f"Generating {num_examples} examples from {len(self.templates)} templates...")
        
        for i in range(num_examples):
            template = random.choice(self.templates)
            
            # Generate standard example
            ex = self.generate_variation(template, i)
            if ex:
                all_examples.append(ex)
            else:
                invalid_count += 1
            
            # Generate FIM example
            if include_fim:
                fim = self.generate_fim_example(template)
                if fim:
                    fim_examples.append(fim)
            
            if verbose and (i + 1) % 500 == 0:
                print(f"  Generated {i+1}, valid: {len(all_examples)}, invalid: {invalid_count}")
        
        if verbose:
            print(f"\nGeneration complete:")
            print(f"  Valid examples: {len(all_examples)}")
            print(f"  Invalid examples: {invalid_count}")
            print(f"  FIM examples: {len(fim_examples)}")
        
        # Shuffle and split
        random.shuffle(all_examples)
        split_idx = int(len(all_examples) * (1 - validation_ratio))
        
        train = all_examples[:split_idx]
        eval_data = all_examples[split_idx:]
        
        return train, eval_data, fim_examples
    
    def augment_with_parameter_variations(self, examples: List[Dict]) -> List[Dict]:
        """
        Augment dataset by substituting parameter values.
        e.g., change '8-bit' to '16-bit' in specs.
        """
        augmented = []
        
        for ex in examples:
            augmented.append(ex)
            
            # Check if this template has parameter variations
            for template_name, variations in PARAMETER_VARIATIONS.items():
                if template_name in ex["source"]:
                    for var in variations[1:]:  # Skip first (original)
                        new_ex = ex.copy()
                        new_ex["spec"] = ex["spec"].replace(variations[0], var)
                        new_ex["id"] = f"{ex['id']}_aug_{var.replace('-', '_')}"
                        augmented.append(new_ex)
                    break
        
        return augmented
    
    def save_dataset(
        self,
        train: List[Dict],
        eval_data: List[Dict],
        fim: List[Dict],
        output_dir: str,
    ):
        """Save datasets to JSONL files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        files = {
            "train.jsonl": train,
            "eval.jsonl": eval_data,
            "fim.jsonl": fim,
        }
        
        for fname, data in files.items():
            path = out / fname
            with open(path, 'w', encoding='utf-8') as f:
                for ex in data:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            print(f"Saved {len(data)} examples to {path}")


def build_sota_dataset(
    output_dir: str = "data",
    num_examples: int = 5000,
    seed: int = 42,
):
    """Main entry point to build the SOTA Verilog dataset."""
    generator = VerilogDatasetGenerator(seed=seed)
    
    train, eval_data, fim = generator.generate_dataset(
        num_examples=num_examples,
        include_fim=True,
        validation_ratio=0.1,
        verbose=True,
    )
    
    # Augment with parameter variations
    print("\nAugmenting with parameter variations...")
    train = generator.augment_with_parameter_variations(train)
    print(f"  Train after augmentation: {len(train)}")
    
    generator.save_dataset(train, eval_data, fim, output_dir)
    
    return train, eval_data, fim
