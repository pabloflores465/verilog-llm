"""
Dataset classes for Verilog fine-tuning.
"""

import json
from typing import Dict, List, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer

from .utils import (
    add_language_tag,
    generate_sicot_prompt,
    format_chat_example,
    format_fim_example,
)


class VerilogDatasetBuilder:
    """Builds datasets for Verilog fine-tuning with multiple techniques."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        use_language_tags: bool = True,
        use_sicot: bool = True,
        use_reasoning: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_language_tags = use_language_tags
        self.use_sicot = use_sicot
        self.use_reasoning = use_reasoning

    def format_instruction(self, spec: str) -> str:
        """Format instruction with techniques."""
        if self.use_sicot:
            instruction = generate_sicot_prompt(spec)
        else:
            instruction = f"Write a Verilog module for: {spec}"
        
        if self.use_language_tags:
            instruction = add_language_tag(instruction)
        
        return instruction

    def format_response(self, code: str, reasoning: Optional[str] = None) -> str:
        """Format response with reasoning if enabled."""
        if self.use_reasoning and reasoning:
            return f"<think>\n{reasoning}\n</think>\n<answer>\n```verilog\n{code}\n```\n</answer>"
        return f"```verilog\n{code}\n```"

    def build_chat_dataset(self, examples: List[Dict]) -> Dataset:
        """
        Build chat-formatted dataset.
        
        Expected example format:
        {
            "spec": "natural language specification",
            "code": "verilog code",
            "reasoning": "optional reasoning text"
        }
        """
        formatted = []
        for ex in examples:
            instruction = self.format_instruction(ex["spec"])
            response = self.format_response(ex.get("code", ""), ex.get("reasoning"))
            
            messages = [
                {"role": "system", "content": "You are an expert Verilog designer."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)

    def build_fim_dataset(self, examples: List[Dict]) -> Dataset:
        """
        Build Fill-in-the-Middle dataset.
        
        Expected example format:
        {
            "prefix": "module foo (\n  input a,",
            "suffix": "\n);\n  // body\nendmodule",
            "middle": "\n  output b"
        }
        """
        formatted = []
        for ex in examples:
            text = format_fim_example(ex["prefix"], ex["suffix"], ex["middle"])
            formatted.append(text)
        
        return Dataset.from_list(formatted)

    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize examples for training."""
        outputs = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        # For Causal LM, labels = input_ids
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs


def load_and_prepare_dataset(
    train_path: str,
    eval_path: Optional[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    format_type: str = "chat",  # "chat" or "fim"
) -> tuple[Dataset, Optional[Dataset]]:
    """Load and prepare dataset for training."""
    
    # Load raw data
    with open(train_path, 'r') as f:
        train_examples = [json.loads(line) for line in f if line.strip()]
    
    eval_examples = None
    if eval_path:
        with open(eval_path, 'r') as f:
            eval_examples = [json.loads(line) for line in f if line.strip()]
    
    # Build dataset
    builder = VerilogDatasetBuilder(
        tokenizer=tokenizer,
        max_length=max_length,
        use_language_tags=True,
        use_sicot=True,
        use_reasoning=True,
    )
    
    if format_type == "fim":
        train_dataset = builder.build_fim_dataset(train_examples)
    else:
        train_dataset = builder.build_chat_dataset(train_examples)
    
    eval_dataset = None
    if eval_examples:
        if format_type == "fim":
            eval_dataset = builder.build_fim_dataset(eval_examples)
        else:
            eval_dataset = builder.build_chat_dataset(eval_examples)
    
    # Tokenize
    train_dataset = train_dataset.map(
        builder.tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            builder.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
    
    return train_dataset, eval_dataset
