#!/usr/bin/env python3
"""
Entrenamiento local con dataset descargado desde Kaggle.
Ejecutar desde laptop:  python scripts/train_local.py

ADVERTENCIA: Qwen 14B 4-bit necesita ~15GB VRAM.
Si no tienes GPU, cambia MODEL_NAME a un modelo 7B o corre en Kaggle.
"""

import os
import sys
import json
import torch
from pathlib import Path

# === CONFIG ===
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
KAGGLE_USERNAME = "pablofloresmollinedo"
DATASET_NAME = "verilog-curated-dataset"

# Rutas locales (dataset descargado con kaggle api)
DATASET_DIR = Path("/tmp/verilog-dataset")  # o donde hayas descargado
TRAIN_PATH = DATASET_DIR / "train.jsonl"
EVAL_PATH  = DATASET_DIR / "eval.jsonl"
OUTPUT_DIR = Path("./checkpoints")

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 7B para laptops sin GPU enorme
# MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"  # 14B si tienes 16GB+ VRAM

HUB_MODEL_ID = "pabloflores/verilog-qwen-local"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 3
LORA_R = 64
LORA_ALPHA = 128
SAVE_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 1. DESCARGAR DATASET SI NO EXISTE ===
if not TRAIN_PATH.exists():
    print("Dataset no encontrado localmente. Descargando con Kaggle API...")
    import subprocess
    subprocess.run([
        sys.executable, "-m", "kaggle", "datasets", "download",
        f"{KAGGLE_USERNAME}/{DATASET_NAME}",
        "-p", str(DATASET_DIR.parent),
        "--unzip"
    ], check=True)
    print("✓ Dataset descargado")
else:
    print(f"✓ Dataset encontrado: {DATASET_DIR}")

# === 2. HF LOGIN ===
from huggingface_hub import login
if HF_TOKEN:
    login(token=HF_TOKEN)
    print("✓ HF Login OK")
else:
    print("⚠️ HF_TOKEN no seteado. Setea: export HF_TOKEN='tu_token'")

# === 3. TOKENIZER ===
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === 4. FORMATO DATASET ===
def format_example(spec, code, reasoning=""):
    instruction = f"""<verilog>
You are a professional Verilog designer.

Design: {spec}

CRITICAL RULES:
- Every `begin` MUST have matching `end`
- Every `module` MUST have matching `endmodule`
- Every `case` MUST have matching `endcase`

Format:
<think>[analysis]</think>
<answer>
```verilog
[code]
```
</answer>"""
    response = f"<think>\n{reasoning}\n</think>\n<answer>\n```verilog\n{code}\n```\n</answer>"
    messages = [
        {"role": "system", "content": "You are an expert Verilog designer."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

from datasets import Dataset

def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                text = format_example(
                    ex.get("spec", ""),
                    ex.get("code", ""),
                    ex.get("reasoning", "")
                )
                examples.append({"text": text})
    return Dataset.from_list(examples)

train_dataset = load_jsonl(TRAIN_PATH)
eval_dataset = load_jsonl(EVAL_PATH) if EVAL_PATH.exists() else None

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")

# === 5. MODELO (con LoRA) ===
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.cuda.empty_cache() if torch.cuda.is_available() else None

compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
) if torch.cuda.is_available() else None

print(f"Loading {MODEL_NAME}...")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

load_kwargs = dict(
    pretrained_model_name_or_path=MODEL_NAME,
    trust_remote_code=True,
)
if bnb_config:
    load_kwargs["quantization_config"] = bnb_config
    load_kwargs["device_map"] = "auto"
    load_kwargs["torch_dtype"] = compute_dtype
    load_kwargs["low_cpu_mem_usage"] = True
else:
    load_kwargs["torch_dtype"] = torch.float32
    print("⚠️ No GPU detectada. Cargando en CPU (MUY LENTO).")
    load_kwargs["device_map"] = {"": "cpu"}

model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

if bnb_config:
    model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === 6. ENTRENAMIENTO ===
from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from trl import SFTTrainer
import inspect

class SaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        print(f"\n💾 Checkpoint guardado: step {state.global_step}")
        return control

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    weight_decay=0.001,
    optim="paged_adamw_8bit" if bnb_config else "adamw_torch",
    group_by_length=True,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    eval_strategy="steps" if eval_dataset else "no",
    eval_steps=200,
    logging_steps=10,
    logging_first_step=True,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=torch.cuda.is_available(),
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Detectar SFTTrainer version
sft_args = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
kwargs = {
    "model": model,
    "train_dataset": train_dataset,
    "eval_dataset": eval_dataset,
    "args": training_args,
    "data_collator": data_collator,
    "callbacks": [SaveCallback()],
}
if "processing_class" in sft_args:
    kwargs["processing_class"] = tokenizer
elif "tokenizer" in sft_args:
    kwargs["tokenizer"] = tokenizer
if "max_seq_length" in sft_args:
    kwargs["max_seq_length"] = MAX_SEQ_LENGTH
if "dataset_text_field" in sft_args:
    kwargs["dataset_text_field"] = "text"

try:
    trainer = SFTTrainer(**kwargs)
except Exception as e:
    print(f"SFTTrainer falló ({e}). Usando Trainer estándar.")
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        data_collator=data_collator, callbacks=[SaveCallback()],
    )

print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60 + "\n")

trainer.train()

# === 7. GUARDAR ===
final_path = OUTPUT_DIR / "adapter_v1"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"\n✓ Modelo guardado en: {final_path}")

# Subir a HF Hub si hay token
if HF_TOKEN:
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path=str(final_path),
            repo_id=HUB_MODEL_ID,
            repo_type="model",
        )
        print(f"✓ Subido a {HUB_MODEL_ID}")
    except Exception as e:
        print(f"✗ Falló upload: {e}")
