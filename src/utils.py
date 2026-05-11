"""
Utility functions for Verilog LLM project.
"""

import json
import re
from typing import List, Dict, Any, Optional


def extract_verilog_code(text: str) -> str:
    """Extract Verilog code from model output."""
    # Try code blocks first
    patterns = [
        r'```verilog\n(.*?)```',
        r'```\n(.*?)```',
        r'<answer>\n(.*?)\n</answer>',
    ]
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    # Fallback: try to find module declaration
    lines = text.split('\n')
    code_lines = []
    in_module = False
    for line in lines:
        if 'module ' in line or in_module:
            in_module = True
            code_lines.append(line)
        if in_module and 'endmodule' in line:
            break
    return '\n'.join(code_lines) if code_lines else text.strip()


def add_language_tag(prompt: str, tag: str = "<verilog>") -> str:
    """CodeV technique: Add explicit language tag to prompt."""
    if tag not in prompt:
        return f"{tag}\n{prompt}"
    return prompt


def generate_sicot_prompt(spec: str) -> str:
    """
    HaVen SI-CoT: Convert symbolic inputs to text before code generation.
    Handles truth tables, state diagrams, waveforms.
    """
    prompt = f"""You are a professional Verilog designer. 

Design Specification:
{spec}

Before writing code:
1. Analyze the requirements carefully
2. If there are truth tables, state diagrams, or waveforms, interpret them in words
3. Plan the module structure (inputs, outputs, internal signals)
4. Choose the right implementation approach (combinational vs sequential)

Then write complete, synthesizable Verilog code.

Respond in this format:
<think>
[Your analysis and reasoning]
</think>
<answer>
```verilog
[Complete Verilog module]
```
</answer>
"""
    return prompt


def format_chat_example(
    instruction: str,
    response: str,
    system_msg: Optional[str] = None
) -> Dict[str, Any]:
    """Format a single example for chat template."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}


def format_fim_example(prefix: str, suffix: str, middle: str) -> Dict[str, Any]:
    """Format Fill-in-the-Middle example for Qwen2.5-Coder."""
    # Qwen2.5-Coder FIM format
    text = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{middle}"
    return {"text": text}


def validate_verilog_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Basic Verilog syntax validation.
    Returns (is_valid, error_message).
    """
    errors = []
    
    # Check for module/endmodule
    if 'module ' not in code:
        errors.append("Missing 'module' declaration")
    if 'endmodule' not in code:
        errors.append("Missing 'endmodule'")
    
    # Check for unbalanced parentheses
    open_p = code.count('(')
    close_p = code.count(')')
    if open_p != close_p:
        errors.append(f"Unbalanced parentheses: {open_p} open, {close_p} close")
    
    # Check for unbalanced begin/end
    begins = len(re.findall(r'\bbegin\b', code))
    ends = len(re.findall(r'\bend\b', code))
    if begins != ends:
        errors.append(f"Unbalanced begin/end: {begins} begin, {ends} end")
    
    # Check for semicolons (basic)
    lines = code.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not line.startswith('//'):
            # Rough check: assignments and declarations need semicolons
            if any(k in line for k in ['assign ', 'wire ', 'reg ', 'input ', 'output ']):
                if not line.endswith(';'):
                    # Exceptions
                    if not (line.endswith(')') or line.endswith(',')):
                        pass  # Too many false positives, skip
    
    is_valid = len(errors) == 0
    return is_valid, "; ".join(errors) if errors else None


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save to JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    return len(tokenizer.encode(text, add_special_tokens=False))
