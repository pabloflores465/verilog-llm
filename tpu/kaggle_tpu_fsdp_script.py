#!/usr/bin/env python3
"""
Kaggle TPU v3-8 training script with SPMD FSDP for Verilog fine-tuning.
Run via: kaggle kernels push -p /path/to/tpu/

TPU v3-8 specs:
- 8 cores, 16GB HBM each = 128GB total
- Native bfloat16 support
- ~420 TFLOPS total

SPMD FSDP approach:
- Model sharded across 8 TPU cores via SPMD
- 14B model in bfloat16 = ~28GB -> ~3.5GB per core
- 3 epochs, seq 2048 in ~3-4 hours

IMPORTANT: Ejemplos que excedan MAX_SEQ_LENGTH se ELIMINAN (no truncan).
"""

import os
import sys
import json
import subprocess
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# CONFIG
# =============================================================================
HUB_MODEL_ID = "Pablo-Flores-Mollinedo/verilog-qwen-14b-sota"
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
OUTPUT_DIR = "/kaggle/working/checkpoints"

TRAIN_PATH = "/kaggle/input/verilog-curated-dataset/train.jsonl"
EVAL_PATH = "/kaggle/input/verilog-curated-dataset/eval.jsonl"
WORKING_DATASET_DIR = "/kaggle/working/verilog-curated-dataset"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 3
LORA_R = 64
LORA_ALPHA = 128
SAVE_STEPS = 100
WARMUP_RATIO = 0.01
MAX_GRAD_NORM = 0.3
WEIGHT_DECAY = 0.001

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# INSTALL DEPS
# =============================================================================
deps = [
    "transformers>=4.40.0",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "huggingface-hub",
    "sentencepiece",
    "protobuf>=5.29.1,<6.0.0",
    "kagglehub",
]
for pkg in deps:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", pkg], check=False)

# =============================================================================
# HF TOKEN (read from env/secrets only - never hardcode)
# =============================================================================
HF_TOKEN = None

try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    if HF_TOKEN:
        print("✓ HF_TOKEN loaded from Kaggle Secrets")
except Exception:
    pass

if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
    if HF_TOKEN:
        print("✓ HF_TOKEN loaded from environment variable")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Please set it via:\n"
        "  - Kaggle Secrets: Add secret 'HF_TOKEN' in notebook settings\n"
        "  - Environment variable: export HF_TOKEN=your_token"
    )

# =============================================================================
# TPU / SPMD SETUP
# =============================================================================
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
import torch_xla.test.test_utils as test_utils
from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear

# Enable SPMD mode (required for model parallelism)
xr.use_spmd()

# Set environment for TPU
os.environ["PJRT_DEVICE"] = "TPU"
os.environ.setdefault("XLA_USE_BF16", "1")

# Detect TPU devices
num_devices = xr.global_runtime_device_count()
print(f"✓ TPU detected: {num_devices} cores")

# Create mesh: (dp=1, fsdp=data_axis, mp=model_axis)
# For v3-8: (1, 8, 1) -> data parallel across 8 cores, model parallel = 1
model_axis = 1
data_axis = num_devices // model_axis
mesh_shape = (1, data_axis, model_axis)
device_ids = np.array(range(num_devices))
mesh = xs.Mesh(device_ids, mesh_shape, ("dp", "fsdp", "mp"))
print(f"  Mesh shape: {mesh_shape} (dp, fsdp, mp)")

device = xm.xla_device()

# =============================================================================
# HF LOGIN
# =============================================================================
from huggingface_hub import login, HfApi

login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
print("✓ HF Login OK")

# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================
class CheckpointManager:
    def __init__(self, hub_model_id: str, hub_token: str, output_dir: str):
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token
        self.output_dir = Path(output_dir)
        self.api = HfApi(token=hub_token)
        os.makedirs(output_dir, exist_ok=True)

    def upload_checkpoint(self, step: int):
        """Upload checkpoint folder to HF Hub."""
        if not xm.is_master_ordinal():
            return
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        if not checkpoint_dir.exists():
            return
        try:
            self.api.upload_folder(
                folder_path=str(checkpoint_dir),
                repo_id=self.hub_model_id,
                repo_type="model",
                path_in_repo=f"checkpoint-tpu-{step}",
                token=self.hub_token,
            )
            print(f"✓ Uploaded checkpoint-tpu-{step}")
        except Exception as e:
            print(f"✗ Upload failed: {e}")

    def save_adapter(self, model, tokenizer, step: int):
        """Save LoRA adapter locally."""
        if not xm.is_master_ordinal():
            return
        save_path = self.output_dir / f"checkpoint-{step}"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✓ Saved checkpoint to {save_path}")


ckpt_mgr = CheckpointManager(HUB_MODEL_ID, HF_TOKEN, OUTPUT_DIR)

# =============================================================================
# POST-PROCESSOR (for inference test at end)
# =============================================================================
class VerilogPostProcessor:
    def process(self, raw_text: str, expected_name: Optional[str] = None) -> str:
        code = self._extract(raw_text)
        code = self._fix_begin_end(code)
        code = self._ensure_module(code, expected_name)
        return code.strip()

    def _extract(self, text: str) -> str:
        for pat in [
            r"```verilog\s*(.*?)```",
            r"```\s*(.*?)```",
            r"<answer>\s*(.*?)\s*</answer>",
        ]:
            m = re.search(pat, text, re.DOTALL | re.I)
            if m:
                return m.group(1).strip()
        mod_start = text.find("module ")
        endmod_pos = text.rfind("endmodule")
        if mod_start != -1 and endmod_pos > mod_start:
            return text[mod_start:endmod_pos + len("endmodule")]
        return text.strip()

    def _fix_begin_end(self, code: str) -> str:
        lines = code.split("\n")
        stack = []
        inserts = []
        deletes = set()
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            begins = s.count("begin")
            ends = s.count("end")
            special = len(
                re.findall(
                    r"\b(endmodule|endcase|endgenerate|endfunction|endtask)\b", s
                )
            )
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
                if re.search(
                    r"\b(endmodule|endcase|endgenerate|endfunction|endtask)\b",
                    lines[j],
                ):
                    inserts.append((j, "end"))
                    inserted = True
                    break
            if not inserted:
                inserts.append((len(lines), "end"))
        inserts.sort(key=lambda x: x[0], reverse=True)
        for idx, txt in inserts:
            lines.insert(idx, "    " + txt)
        for idx in sorted(deletes, reverse=True):
            del lines[idx]
        return "\n".join(lines)

    def _ensure_module(self, code: str, name: Optional[str]) -> str:
        has_mod = "module " in code and re.search(r"\bmodule\s+\w+", code)
        has_end = "endmodule" in code
        if has_mod and has_end:
            return code
        mod = name or "generated_module"
        if not has_mod and not has_end:
            return f"module {mod} ();\n" + code + "\nendmodule\n"
        if has_mod and not has_end:
            return code + "\nendmodule\n"
        return f"module {mod} ();\n" + code


post_processor = VerilogPostProcessor()

# =============================================================================
# LOAD MODEL + LoRA
# =============================================================================
print("\n[1/6] Loading model...")
print(f"  Model: {MODEL_NAME}")
print(f"  Precision: bfloat16 (TPU native)")
print(f"  Sharding: SPMD across {num_devices} cores")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model in bfloat16 (optimal for TPU v3)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Apply LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Move to TPU
model = model.to(device)

# Patch linear layers for TPU einsum optimization
model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)

# Apply SPMD sharding
from spmd_util import partition_module

partition_module(model, mesh, device=device, verbose=xm.is_master_ordinal())

if xm.is_master_ordinal():
    model.print_trainable_parameters()

xm.rendezvous("model_loaded")

# =============================================================================
# LOAD DATASET
# =============================================================================
print("\n[2/6] Loading dataset...")

def format_example(spec: str, code: str, reasoning: str = "") -> str:
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
    response = (
        f"<think>\n{reasoning}\n</think>\n"
        f"<answer>\n```verilog\n{code}\n```\n</answer>"
    )
    messages = [
        {"role": "system", "content": "You are an expert Verilog designer."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


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
    import kagglehub

    dataset_path = kagglehub.dataset_download(
        "pablofloresmollinedo/verilog-curated-dataset"
    )
    return (
        str(Path(dataset_path) / "train.jsonl"),
        str(Path(dataset_path) / "eval.jsonl"),
    )


def tokenize_and_filter(text: str, max_len: int) -> Optional[Dict]:
    """
    Tokenize text. If length > max_len, return None (ELIMINAR, no truncar).
    """
    tokens = tokenizer(
        text,
        truncation=False,  # NO truncar
        add_special_tokens=True,
    )
    input_ids = tokens["input_ids"]
    if len(input_ids) > max_len:
        return None  # Eliminar ejemplo
    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": input_ids.copy(),
    }


def load_dataset_file(train_path: str, eval_path: Optional[str] = None):
    train_ex = []
    skipped_train = 0
    with open(train_path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            text = format_example(
                ex.get("spec", ""), ex.get("code", ""), ex.get("reasoning", "")
            )
            tok = tokenize_and_filter(text, MAX_SEQ_LENGTH)
            if tok is None:
                skipped_train += 1
                continue
            train_ex.append(tok)

    eval_ex = []
    skipped_eval = 0
    if eval_path and Path(eval_path).exists():
        with open(eval_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                text = format_example(
                    ex.get("spec", ""), ex.get("code", ""), ex.get("reasoning", "")
                )
                tok = tokenize_and_filter(text, MAX_SEQ_LENGTH)
                if tok is None:
                    skipped_eval += 1
                    continue
                eval_ex.append(tok)

    if xm.is_master_ordinal():
        print(f"  Train: {len(train_ex)} (skipped {skipped_train} too long)")
        print(f"  Eval:  {len(eval_ex)} (skipped {skipped_eval} too long)")
    return train_ex, eval_ex


train_path, eval_path = find_dataset()
train_data, eval_data = load_dataset_file(train_path, eval_path)

if len(train_data) == 0:
    raise RuntimeError("No training examples after filtering! Reduce MAX_SEQ_LENGTH.")


# =============================================================================
# PYTORCH DATASET & DATALOADER
# =============================================================================
print("\n[3/6] Building dataloaders...")


class VerilogDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PadCollator:
    """Collate with padding to max length in batch."""

    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = min(
            max(len(ex["input_ids"]) for ex in batch),
            self.max_length,
        )
        input_ids = []
        attention_mask = []
        labels = []
        for ex in batch:
            ids = ex["input_ids"][:max_len]
            mask = ex["attention_mask"][:max_len]
            lbl = ex["labels"][:max_len]
            # Pad
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
                lbl = lbl + [-100] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbl)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


train_dataset = VerilogDataset(train_data)
eval_dataset = VerilogDataset(eval_data) if eval_data else None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    collate_fn=PadCollator(tokenizer.pad_token_id, MAX_SEQ_LENGTH),
)

eval_loader = None
if eval_dataset:
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        collate_fn=PadCollator(tokenizer.pad_token_id, MAX_SEQ_LENGTH),
    )

# Wrap with MpDeviceLoader for TPU
train_loader = pl.MpDeviceLoader(train_loader, device)
if eval_loader:
    eval_loader = pl.MpDeviceLoader(eval_loader, device)

STEPS_PER_EPOCH = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)
TOTAL_STEPS = int((EPOCHS * STEPS_PER_EPOCH))

if xm.is_master_ordinal():
    print(f"  Steps/epoch: {STEPS_PER_EPOCH}")
    print(f"  Total steps: {TOTAL_STEPS} ({EPOCHS} epochs)")

xm.rendezvous("data_ready")

# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================
print("\n[4/6] Setting up optimizer...")

from torch_xla.amp.syncfree import AdamW as SyncFreeAdamW
from transformers import get_cosine_schedule_with_warmup

update_params = list(filter(lambda p: p.requires_grad, model.parameters()))

# TPU-optimized sync-free AdamW
optimizer = SyncFreeAdamW(
    update_params,
    lr=LR,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY,
)

warmup_steps = int(TOTAL_STEPS * WARMUP_RATIO)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=TOTAL_STEPS,
)

if xm.is_master_ordinal():
    print(f"  Optimizer: AdamW (sync-free)")
    print(f"  LR: {LR}, Warmup: {warmup_steps} steps")
    print(f"  Grad accum: {GRAD_ACCUM}")

# =============================================================================
# LOSS FUNCTION
# =============================================================================

def compute_loss(outputs, labels, pad_id: int = tokenizer.pad_token_id):
    """Compute cross-entropy loss with label smoothing."""
    epsilon = 1e-8
    logits = outputs.logits
    # Shift logits and labels for causal LM
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()
    log_probs = -nn.functional.log_softmax(logits, dim=-1)
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)
    padding_mask = labels.eq(pad_id)
    labels = torch.clamp(labels, min=0)
    nll_loss = log_probs.gather(dim=-1, index=labels)
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.bfloat16)
    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)
    num_active = padding_mask.numel() - padding_mask.long().sum()
    nll_loss = nll_loss.sum() / num_active
    smoothed_loss = smoothed_loss.sum() / (num_active * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss


# =============================================================================
# TRAINING LOOP
# =============================================================================
print("\n[5/6] Starting training...")

if xm.is_master_ordinal():
    print("=" * 60)
    print(f"MODEL: {MODEL_NAME}")
    print(f"TPU CORES: {num_devices}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"BATCH: {BATCH_SIZE} per core × {num_devices} cores × {GRAD_ACCUM} accum")
    print(f"SEQ: {MAX_SEQ_LENGTH} (ejemplos largos ELIMINADOS)")
    print(f"STEPS: {TOTAL_STEPS}")
    print(f"SAVE: every {SAVE_STEPS} steps")
    print("=" * 60)

global_step = 0
best_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    xm.master_print(f"\n>>> Epoch {epoch}/{EPOCHS} start {test_utils.now()}")

    epoch_loss = 0.0
    epoch_steps = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Mark sharding on inputs (data parallel across fsdp axis)
        xs.mark_sharding(input_ids, mesh, (0, 1))
        xs.mark_sharding(attention_mask, mesh, (0, 1))
        xs.mark_sharding(labels, mesh, (0, 1))

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(outputs, labels)

        # Scale loss for grad accum
        loss = loss / GRAD_ACCUM
        loss.backward()

        # Accumulate stats
        loss_item = loss.detach().cpu().item() * GRAD_ACCUM
        epoch_loss += loss_item
        epoch_steps += 1

        # Logging
        if (step + 1) % 10 == 0 and xm.is_master_ordinal():
            xm.master_print(
                f"  step {global_step + 1}/{TOTAL_STEPS} | "
                f"loss: {loss_item:.4f} | "
                f"lr: {scheduler.get_last_lr()[0]:.2e} | "
                f"{test_utils.now()}"
            )

        # Gradient accumulation step
        if (step + 1) % GRAD_ACCUM == 0:
            # Clip grads
            torch.nn.utils.clip_grad_norm_(
                update_params, max_norm=MAX_GRAD_NORM * num_devices
            )
            # Step optimizer (includes xm.reduce_gradients + mark_step)
            scheduler.step()
            xm.optimizer_step(optimizer, pin_layout=True, barrier=True)
            optimizer.zero_grad()
            global_step += 1

            # Save checkpoint
            if global_step % SAVE_STEPS == 0:
                ckpt_mgr.save_adapter(model, tokenizer, global_step)
                ckpt_mgr.upload_checkpoint(global_step)

            # Early stop for testing (optional)
            # if global_step >= 10:
            #     break

        del input_ids, attention_mask, labels, outputs, loss

    # Epoch summary
    if xm.is_master_ordinal():
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch} avg loss: {avg_loss:.4f}")

    xm.rendezvous(f"epoch_{epoch}_done")

# =============================================================================
# FINAL SAVE
# =============================================================================
print("\n[6/6] Saving final adapter...")

if xm.is_master_ordinal():
    final_path = Path(OUTPUT_DIR) / "adapter_tpu_v1"
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✓ Saved locally: {final_path}")

    try:
        api.upload_folder(
            folder_path=str(final_path),
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            path_in_repo="adapter_tpu_v1",
            token=HF_TOKEN,
        )
        print(f"✓ Uploaded to {HUB_MODEL_ID}/adapter_tpu_v1")
    except Exception as e:
        print(f"✗ Upload failed: {e}")

xm.rendezvous("training_complete")

# =============================================================================
# TEST INFERENCE
# =============================================================================
if xm.is_master_ordinal():
    print("\n[TEST] Generating sample...")
    model.eval()

    test_prompt = """<verilog>
Design a 4-bit up-counter with synchronous reset and enable.
When enable is high, count increments on each clock rising edge.
Reset clears count to 0.
"""
    messages = [
        {"role": "system", "content": "You are an expert Verilog designer."},
        {"role": "user", "content": test_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
        )
    raw = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True
    )
    result = post_processor.process(raw)
    print(result)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final: {HUB_MODEL_ID}/adapter_tpu_v1")
    print("=" * 60)
