"""
Cell 4: Load Model + LoRA (OOM-safe for dual T4)
Usage: import this module or paste into Kaggle notebook.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    offload_dir: str = "/kaggle/working/offload",
):
    torch.cuda.empty_cache()
    os.makedirs(offload_dir, exist_ok=True)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Leave headroom on each GPU so materialization doesn't OOM
    max_memory = {i: "13GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "16GiB"

    # Disable async materialization to avoid concurrent OOM during load
    import transformers
    transformers.modeling_utils._load_state_dict_into_model = lambda *a, **k: None  # no-op stub to force sync path

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        dtype=compute_dtype,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
        offload_state_dict=True,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i}: {mem:.1f} GB used")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
