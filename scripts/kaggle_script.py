#!/usr/bin/env python3
"""
Kaggle headless training script.
Run via: kaggle kernels push -p /path/to/scripts/
This runs on Kaggle servers, independent of your laptop.
"""

import os
import sys
import json
import torch
from pathlib import Path

# === CONFIG ===
HF_TOKEN = "HF_TOKEN_PLACEHOLDER"
HUB_MODEL_ID = "pabloflores/verilog-qwen-14b-sota"
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
OUTPUT_DIR = "/kaggle/working/checkpoints"

TRAIN_PATH = "/kaggle/input/verilog-curated-dataset/train.jsonl"
EVAL_PATH = "/kaggle/input/verilog-curated-dataset/eval.jsonl"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 3
LORA_R = 64
LORA_ALPHA = 128
SAVE_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Install missing packages for Kaggle environment
import subprocess
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)

print("=" * 60)
print("VERILOG SOTA TRAINING - HEADLESS")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {TRAIN_PATH}")
print(f"Output: {OUTPUT_DIR}")
print(f"Hub: {HUB_MODEL_ID}")
print("=" * 60)

# === 1. HF LOGIN ===
from huggingface_hub import login, HfApi
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
print("✓ HF Login OK")

# === 2. CHECKPOINT MANAGER ===
class CheckpointManager:
    def __init__(self):
        self.hub_model_id = HUB_MODEL_ID
        self.hub_token = HF_TOKEN
        self.output_dir = Path(OUTPUT_DIR)
        self.api = HfApi(token=HF_TOKEN)

    def get_latest_checkpoint(self):
        if not self.output_dir.exists():
            return None
        checkpoints = [d for d in self.output_dir.iterdir()
                       if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        return str(latest)

    def upload_checkpoint(self, checkpoint_dir, step):
        try:
            self.api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.hub_model_id,
                repo_type="model",
                path_in_repo=f"checkpoint-{step}",
                token=self.hub_token,
            )
            print(f"✓ Uploaded checkpoint-{step}")
        except Exception as e:
            print(f"✗ Upload failed: {e}")

ckpt_mgr = CheckpointManager()

# Check for resume
latest_ckpt = ckpt_mgr.get_latest_checkpoint()
if latest_ckpt:
    print(f"✓ Found checkpoint: {latest_ckpt}")
else:
    print("✓ Starting from scratch")

# === 3. POST-PROCESSOR (for inference test) ===
import re

class VerilogPostProcessor:
    def process(self, raw_text, expected_name=None):
        code = self._extract(raw_text)
        code = self._fix_begin_end(code)
        code = self._ensure_module(code, expected_name)
        return code.strip()

    def _extract(self, text):
        for pat in [r'```verilog\s*(.*?)```', r'```\s*(.*?)```', r'<answer>\s*(.*?)\s*</answer>']:
            m = re.search(pat, text, re.DOTALL | re.I)
            if m:
                return m.group(1).strip()
        mod_start = text.find('module ')
        endmod_pos = text.rfind('endmodule')
        if mod_start != -1 and endmod_pos > mod_start:
            return text[mod_start:endmod_pos + len('endmodule')]
        return text.strip()

    def _fix_begin_end(self, code):
        lines = code.split('\n')
        stack = []
        inserts = []
        deletes = set()
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith('//'):
                continue
            begins = s.count('begin')
            ends = s.count('end')
            special = len(re.findall(r'\b(endmodule|endcase|endgenerate|endfunction|endtask)\b', s))
            net = ends - special
            for _ in range(begins):
                stack.append(i)
            for _ in range(net):
                if stack:
                    stack.pop()
                else:
                    deletes.add(i)
        for idx in reversed(stack):
            inserted = False
            for j in range(idx + 1, len(lines)):
                if re.search(r'\b(endmodule|endcase|endgenerate|endfunction|endtask)\b', lines[j]):
                    inserts.append((j, 'end'))
                    inserted = True
                    break
            if not inserted:
                inserts.append((len(lines), 'end'))
        inserts.sort(key=lambda x: x[0], reverse=True)
        for idx, txt in inserts:
            lines.insert(idx, '    ' + txt)
        for idx in sorted(deletes, reverse=True):
            del lines[idx]
        return '\n'.join(lines)

    def _ensure_module(self, code, name):
        has_mod = 'module ' in code and re.search(r'\bmodule\s+\w+', code)
        has_end = 'endmodule' in code
        if has_mod and has_end:
            return code
        mod = name or 'generated_module'
        if not has_mod and not has_end:
            return f"module {mod} ();\n" + code + '\nendmodule\n'
        if has_mod and not has_end:
            return code + '\nendmodule\n'
        return f"module {mod} ();\n" + code

post_processor = VerilogPostProcessor()

# === 4. LOAD MODEL ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("\n[1/5] Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

for i in range(torch.cuda.device_count()):
    mem = torch.cuda.memory_allocated(i) / 1e9
    print(f"  GPU {i}: {mem:.1f} GB used")

# === 5. LOAD DATASET ===
print("\n[2/5] Loading dataset...")

def format_example(spec, code, reasoning=""):
    instruction = f"""<verilog>
You are a professional Verilog designer.

Design: {spec}

CRITICAL RULES:
- Every `begin` MUST have matching `end`
- Every `module` MUST have matching `endmodule`
- Every `case` MUST have matching `endcase`
- Check balance before finishing

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

def load_dataset(train_path, eval_path=None):
    train_ex = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                text = format_example(ex.get("spec", ""), ex.get("code", ""), ex.get("reasoning", ""))
                train_ex.append({"text": text})
    eval_ex = []
    if eval_path and Path(eval_path).exists():
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    text = format_example(ex.get("spec", ""), ex.get("code", ""), ex.get("reasoning", ""))
                    eval_ex.append({"text": text})
    from datasets import Dataset
    return Dataset.from_list(train_ex), Dataset.from_list(eval_ex) if eval_ex else None

train_dataset, eval_dataset = load_dataset(TRAIN_PATH, EVAL_PATH)
print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")

# === 6. TRAINING ===
print("\n[3/5] Setting up training...")

from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

class HFHubCheckpointCallback(TrainerCallback):
    def __init__(self, ckpt_mgr, save_steps):
        self.ckpt_mgr = ckpt_mgr
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            print(f"\n📤 Uploading checkpoint-{state.global_step} to HF Hub...")
            self.ckpt_mgr.upload_checkpoint(str(checkpoint_dir), state.global_step)
        return control

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    weight_decay=0.001,
    optim="paged_adamw_8bit",
    group_by_length=True,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    evaluation_strategy="steps" if eval_dataset else "no",
    eval_steps=200,
    logging_steps=10,
    logging_first_step=True,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="checkpoint",
    hub_token=HF_TOKEN,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=data_collator,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    callbacks=[HFHubCheckpointCallback(ckpt_mgr, SAVE_STEPS)],
)

resume_from = latest_ckpt if latest_ckpt else None

print("\n" + "=" * 60)
print(f"[4/5] STARTING TRAINING")
print(f"Resume: {resume_from or 'scratch'}")
print(f"Steps to save: {SAVE_STEPS}")
print(f"Epochs: {EPOCHS}")
print("=" * 60 + "\n")

try:
    trainer.train(resume_from_checkpoint=resume_from)
except KeyboardInterrupt:
    print("\n⚠️ Interrupted - saving...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted"))
    raise

# === 7. FINAL SAVE ===
print("\n[5/5] Saving final adapter...")
final_path = os.path.join(OUTPUT_DIR, "adapter_v1")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"✓ Saved locally: {final_path}")

print("Uploading adapter_v1 to HF Hub...")
try:
    api.upload_folder(
        folder_path=final_path,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        path_in_repo="adapter_v1",
        token=HF_TOKEN,
    )
    print(f"✓ Uploaded to {HUB_MODEL_ID}/adapter_v1")
except Exception as e:
    print(f"✗ Upload failed: {e}")

# === 8. QUICK TEST ===
print("\n[TEST] Generating sample...")

def generate(prompt, max_tokens=1024):
    messages = [
        {"role": "system", "content": "You are an expert Verilog designer."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.2, top_p=0.95, do_sample=True)
    raw = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return post_processor.process(raw)

test = """<verilog>
Design a 4-bit up-counter with synchronous reset and enable.
When enable is high, count increments on each clock rising edge.
Reset clears count to 0.
"""
print(generate(test))

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"Final adapter: {HUB_MODEL_ID}/adapter_v1")
print("=" * 60)
