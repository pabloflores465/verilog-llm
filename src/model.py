"""
Model setup with QLoRA for Qwen2.5-Coder-14B.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
    use_4bit: bool = True,
    use_nested_quant: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    device_map: str = "auto",
):
    """
    Load Qwen2.5-Coder-14B with 4-bit quantization.
    
    Memory footprint:
    - Base model 4-bit: ~7.5 GB
    - With LoRA (r=64): +0.3 GB
    - Activations (seq=2048, batch=1): ~2-3 GB
    - Total: ~10-11 GB -> Fits in single T4 (16GB)
    """
    
    # Compute dtype
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    
    # Quantization config
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
    else:
        quantization_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=compute_dtype if not use_4bit else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


def setup_lora(
    model,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: list = None,
    use_rslora: bool = False,
):
    """
    Setup LoRA adapters on the model.
    
    Target modules for Qwen2.5:
    - q_proj, k_proj, v_proj, o_proj (attention)
    - gate_proj, up_proj, down_proj (MLP)
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=use_rslora,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model
