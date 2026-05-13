#!/usr/bin/env python3
"""
Kaggle TPU v3-8 training script for Verilog fine-tuning.
Run via: kaggle kernels push -p /path/to/tpu/

TPU v3-8 specs:
- 8 cores, 16GB HBM each = 128GB total
- Native bfloat16 support (optimal for TPU)
- ~420 TFLOPS total (vs ~16 TFLOPS for 2x T4)
- No bitsandbytes (CUDA-only), use bfloat16 instead

SINGLE-CORE MODE (default):
- Uses 7B model which fits in 1 TPU core (14GB in bfloat16)
- Simple, stable, avoids multiprocessing complexity
- ~10-15s per step vs 75s on T4

MULTI-CORE MODE (optional):
- For 14B model, requires FSDP sharding across 2+ cores
- Set USE_14B = True below to enable
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# === CONFIG ===
HUB_MODEL_ID = "Pablo-Flores-Mollinedo/verilog-qwen-14b-sota"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # 7B fits in 1 TPU core
# Set to True for 14B with FSDP (experimental)
USE_14B = False
if USE_14B:
    MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"

OUTPUT_DIR = "/kaggle/working/checkpoints"
TRAIN_PATH = "/kaggle/input/verilog-curated-dataset/train.jsonl"
EVAL_PATH  = "/kaggle/input/verilog-curated-dataset/eval.jsonl"
WORKING_DATASET_DIR = "/kaggle/working/verilog-curated-dataset"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 3
LORA_R = 64
LORA_ALPHA = 128
SAVE_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === INSTALL DEPS ===
# torch_xla comes pre-installed on Kaggle TPU VMs (usually)
# Check if available; if not, attempt install
deps = [
    "transformers>=4.40.0",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "huggingface-hub",
    "sentencepiece",
    "protobuf>=5.29.1,<6.0.0",
]
for pkg in deps:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=False)

# Try to ensure torch_xla is available
try:
    import torch_xla
except ImportError:
    print("⚠️ torch_xla not found, attempting install...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_xla"], check=False)

# === HF TOKEN ===
# IMPORTANT: Do NOT hardcode tokens in this file.
# Set HF_TOKEN via ONE of these methods:
#   1. Kaggle Secrets (recommended): Add 'HF_TOKEN' in Kaggle notebook secrets
#   2. Environment variable: os.environ['HF_TOKEN']
#   3. Kaggle CLI push with env var: HF_TOKEN=xxx kaggle kernels push -p tpu/

HF_TOKEN = None

# 1. Try Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    if HF_TOKEN:
        print("✓ HF_TOKEN loaded from Kaggle Secrets")
except Exception:
    pass

# 2. Fallback to environment variable
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
    if HF_TOKEN:
        print("✓ HF_TOKEN loaded from environment variable")

# 3. Fail if still missing
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Please set it via:\n"
        "  - Kaggle Secrets (recommended): Add secret 'HF_TOKEN' in notebook settings\n"
        "  - Environment variable: export HF_TOKEN=your_token_here\n"
        "  - Or pass when pushing: HF_TOKEN=xxx kaggle kernels push -p tpu/"
    )

# === TPU SETUP ===
import torch
import torch_xla
import torch_xla.core.xla_model as xm

TPU_CORES = xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 8
print(f"✓ TPU detected: {TPU_CORES} cores available")

device = xm.xla_device()
print(f"  Using device: {device}")

# === HF LOGIN ===
from huggingface_hub import login, HfApi
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
print("✓ HF Login OK")

# === CHECKPOINT MANAGER ===
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
latest_ckpt = ckpt_mgr.get_latest_checkpoint()
print(f"✓ {'Found checkpoint: ' + latest_ckpt if latest_ckpt else 'Starting from scratch'}")

# === POST-PROCESSOR ===
class VerilogPostProcessor:
    def process(self, raw_text, expected_name=None):
        code = self._extract(raw_text)
        code = self._fix_begin_end(code)
        code = self._ensure_module(code, expected_name)
        return code.strip()

    def _extract(self, text):
        import re
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
        import re
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
        import re
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

# === LOAD MODEL (TPU: bfloat16, no quantization) ===
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

print("\n[1/5] Loading model...")
print(f"  Model: {MODEL_NAME}")
print(f"  Precision: bfloat16 (TPU native)")

# For TPU v3, bfloat16 is the optimal precision
# No quantization needed - 7B model in bfloat16 = ~14GB, fits in 1 core
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Move to TPU
model = model.to(device)

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === LOAD DATASET ===
print("\n[2/5] Loading dataset...")

from datasets import Dataset

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

def find_dataset():
    import glob
    if Path(TRAIN_PATH).exists():
        return TRAIN_PATH, EVAL_PATH
    for base in glob.glob("/kaggle/input/*verilog*"):
        train = Path(base) / "train.jsonl"
        eval_f = Path(base) / "eval.jsonl"
        if train.exists():
            return str(train), str(eval_f) if eval_f.exists() else None
    working_train = Path(WORKING_DATASET_DIR) / "train.jsonl"
    working_eval = Path(WORKING_DATASET_DIR) / "eval.jsonl"
    if working_train.exists():
        return str(working_train), str(working_eval) if working_eval.exists() else None
    print("  ⚠️ Dataset not mounted. Downloading with kagglehub...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=False)
    import kagglehub
    dataset_path = kagglehub.dataset_download("pablofloresmollinedo/verilog-curated-dataset")
    return str(Path(dataset_path) / "train.jsonl"), str(Path(dataset_path) / "eval.jsonl")

def load_dataset_file(train_path, eval_path=None):
    train_ex = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                text = format_example(ex.get("spec",""), ex.get("code",""), ex.get("reasoning",""))
                train_ex.append({"text": text})
    eval_ex = []
    if eval_path and Path(eval_path).exists():
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    text = format_example(ex.get("spec",""), ex.get("code",""), ex.get("reasoning",""))
                    eval_ex.append({"text": text})
    return Dataset.from_list(train_ex), (Dataset.from_list(eval_ex) if eval_ex else None)

train_path, eval_path = find_dataset()
train_dataset, eval_dataset = load_dataset_file(train_path, eval_path)
print(f"  Train: {len(train_dataset)} | Eval: {len(eval_dataset) if eval_dataset else 0}")

# === TRAINING ===
print("\n[3/5] Setting up training...")

from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from trl import SFTTrainer
import inspect

class HFHubCheckpointCallback(TrainerCallback):
    def __init__(self, ckpt_mgr, save_steps):
        self.ckpt_mgr = ckpt_mgr
        self.save_steps = save_steps
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            print(f"\n📤 Uploading checkpoint-{state.global_step}...")
            self.ckpt_mgr.upload_checkpoint(str(checkpoint_dir), state.global_step)
        return control

sft_args = set(inspect.signature(SFTTrainer.__init__).parameters.keys())

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    weight_decay=0.001,
    optim="adamw_torch",  # Standard AdamW for TPU
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    eval_strategy="no",
    logging_steps=10,
    logging_first_step=True,
    push_to_hub=False,  # Manual upload in callback
    bf16=True,  # TPU native bfloat16
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    # TPU optimizations
    dataloader_drop_last=True,
    dataloader_num_workers=0,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer_kwargs = {
    "model": model,
    "train_dataset": train_dataset,
    "args": training_args,
    "data_collator": data_collator,
    "callbacks": [HFHubCheckpointCallback(ckpt_mgr, SAVE_STEPS)],
}
if "processing_class" in sft_args:
    trainer_kwargs["processing_class"] = tokenizer
elif "tokenizer" in sft_args:
    trainer_kwargs["tokenizer"] = tokenizer
if "max_seq_length" in sft_args:
    trainer_kwargs["max_seq_length"] = MAX_SEQ_LENGTH
if "dataset_text_field" in sft_args:
    trainer_kwargs["dataset_text_field"] = "text"

trainer = SFTTrainer(**trainer_kwargs)

print("\n" + "=" * 60)
print(f"[4/5] STARTING TRAINING ON TPU")
print(f"Model: {MODEL_NAME}")
print(f"Device: {device}")
print(f"Cores available: {TPU_CORES} (using 1 core for simplicity)")
print(f"Save every: {SAVE_STEPS} steps")
print("=" * 60 + "\n")

try:
    trainer.train(resume_from_checkpoint=latest_ckpt)
except KeyboardInterrupt:
    print("\n⚠️ Interrupted - saving...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted"))
    raise

# === FINAL SAVE ===
print("\n[5/5] Saving final adapter...")
final_path = os.path.join(OUTPUT_DIR, "adapter_tpu_v1")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
print(f"✓ Saved locally: {final_path}")

try:
    api.upload_folder(
        folder_path=final_path,
        repo_id=HUB_MODEL_ID,
        repo_type="model",
        path_in_repo="adapter_tpu_v1",
        token=HF_TOKEN,
    )
    print(f"✓ Uploaded to {HUB_MODEL_ID}/adapter_tpu_v1")
except Exception as e:
    print(f"✗ Upload failed: {e}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"Final: {HUB_MODEL_ID}/adapter_tpu_v1")
print("=" * 60)
