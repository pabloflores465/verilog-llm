#!/usr/bin/env python3
"""
Create Kaggle notebook with proper multi-line cells.
Each line in source must be a separate array element ending with \n.
"""

import json

def cell(cell_type, lines):
    """Create a notebook cell with proper line splitting.
    
    lines: list of strings (each line WITHOUT trailing \n)
    """
    source = [line + '\n' for line in lines]
    # Remove trailing \n from last line (Jupyter convention)
    if source:
        source[-1] = source[-1].rstrip('\n')
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "source": source,
        "outputs": []
    }

cells = []

# Cell 0: Markdown intro
cells.append(cell("markdown", [
    "# Verilog Qwen 14B Training on TPU v3-8 (SPMD FSDP)",
    "",
    "**IMPORTANT:**",
    "- Set accelerator to **TPU VM v3-8** before running",
    "- Add `HF_TOKEN` to Kaggle Secrets (Notebook options → Secrets)",
    "",
    "This notebook fine-tunes Qwen2.5-Coder-14B-Instruct on Verilog dataset using SPMD sharding across 8 TPU cores."
]))

# Cell 1: Install
cells.append(cell("code", [
    "!pip install -q -U transformers>=4.40.0 peft datasets accelerate huggingface-hub sentencepiece",
    "!pip install -q -U \"protobuf>=5.29.1,<6.0.0\" kagglehub",
    "!pip install -q 'torch_xla[tpuvm]'",
    "print('Done installing')"
]))

# Cell 2: Imports and Config
cells.append(cell("code", [
    "import os",
    "import sys",
    "import json",
    "import subprocess",
    "import re",
    "import numpy as np",
    "from pathlib import Path",
    "from typing import Dict, List, Optional",
    "",
    "# === CONFIG ===",
    "HUB_MODEL_ID = 'Pablo-Flores-Mollinedo/verilog-qwen-14b-sota'",
    "MODEL_NAME = 'Qwen/Qwen2.5-Coder-14B-Instruct'",
    "OUTPUT_DIR = '/kaggle/working/checkpoints'",
    "",
    "TRAIN_PATH = '/kaggle/input/verilog-curated-dataset/train.jsonl'",
    "EVAL_PATH = '/kaggle/input/verilog-curated-dataset/eval.jsonl'",
    "WORKING_DATASET_DIR = '/kaggle/working/verilog-curated-dataset'",
    "",
    "MAX_SEQ_LENGTH = 2048",
    "BATCH_SIZE = 1",
    "GRAD_ACCUM = 4",
    "LR = 2e-4",
    "EPOCHS = 3",
    "LORA_R = 64",
    "LORA_ALPHA = 128",
    "SAVE_STEPS = 100",
    "WARMUP_RATIO = 0.01",
    "MAX_GRAD_NORM = 0.3",
    "WEIGHT_DECAY = 0.001",
    "",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)",
    "print('Config loaded')"
]))

# Cell 3: HF Token
cells.append(cell("code", [
    "HF_TOKEN = None",
    "",
    "try:",
    "    from kaggle_secrets import UserSecretsClient",
    "    secrets = UserSecretsClient()",
    "    HF_TOKEN = secrets.get_secret('HF_TOKEN')",
    "    if HF_TOKEN:",
    "        print('HF_TOKEN from Kaggle Secrets')",
    "except Exception:",
    "    pass",
    "",
    "if not HF_TOKEN:",
    "    HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()",
    "    if HF_TOKEN:",
    "        print('HF_TOKEN from env')",
    "",
    "if not HF_TOKEN:",
    "    raise ValueError('HF_TOKEN not found. Add to Kaggle Secrets or set env var.')",
    "",
    "from huggingface_hub import login, HfApi",
    "login(token=HF_TOKEN)",
    "api = HfApi(token=HF_TOKEN)",
    "print('HF Login OK')"
]))

# Cell 4: TPU Setup
cells.append(cell("code", [
    "import torch",
    "import torch.nn as nn",
    "import torch_xla",
    "import torch_xla.core.xla_model as xm",
    "import torch_xla.experimental.xla_sharding as xs",
    "import torch_xla.distributed.parallel_loader as pl",
    "import torch_xla.runtime as xr",
    "import torch_xla.test.test_utils as test_utils",
    "from torch_xla.distributed.fsdp.utils import apply_xla_patch_to_nn_linear",
    "",
    "xr.use_spmd()",
    "os.environ['PJRT_DEVICE'] = 'TPU'",
    "os.environ.setdefault('XLA_USE_BF16', '1')",
    "",
    "num_devices = xr.global_runtime_device_count()",
    "print(f'TPU cores: {num_devices}')",
    "",
    "model_axis = 1",
    "data_axis = num_devices // model_axis",
    "mesh_shape = (1, data_axis, model_axis)",
    "device_ids = np.array(range(num_devices))",
    "mesh = xs.Mesh(device_ids, mesh_shape, ('dp', 'fsdp', 'mp'))",
    "print(f'Mesh: {mesh_shape}')",
    "",
    "device = xm.xla_device()"
]))

# Cell 5: Load Model + LoRA
cells.append(cell("code", [
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "from peft import LoraConfig, get_peft_model",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)",
    "tokenizer.pad_token = tokenizer.eos_token",
    "tokenizer.padding_side = 'right'",
    "",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    MODEL_NAME,",
    "    torch_dtype=torch.bfloat16,",
    "    trust_remote_code=True,",
    ")",
    "",
    "lora_config = LoraConfig(",
    "    r=LORA_R,",
    "    lora_alpha=LORA_ALPHA,",
    "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],",
    "    lora_dropout=0.05,",
    "    bias='none',",
    "    task_type='CAUSAL_LM',",
    ")",
    "model = get_peft_model(model, lora_config)",
    "model = model.to(device)",
    "",
    "model = apply_xla_patch_to_nn_linear(model, xs.xla_patched_nn_linear_forward)",
    "",
    "from spmd_util import partition_module",
    "partition_module(model, mesh, device=device, verbose=xm.is_master_ordinal())",
    "",
    "if xm.is_master_ordinal():",
    "    model.print_trainable_parameters()",
    "",
    "xm.rendezvous('model_loaded')",
    "print('Model ready')"
]))

# Cell 6: Dataset loading
cells.append(cell("code", [
    "def format_example(spec, code, reasoning=''):",
    "    instruction = f'''<verilog>",
    "You are a professional Verilog designer.",
    "",
    "Design: {spec}",
    "",
    "CRITICAL RULES:",
    "- Every `begin` MUST have matching `end`",
    "- Every `module` MUST have matching `endmodule`",
    "- Every `case` MUST have matching `endcase`",
    "",
    "Format:",
    "<think>[analysis]</think>",
    "<answer>",
    "```verilog",
    "[code]",
    "```",
    "</answer>'''",
    "    response = f'<think>\\n{reasoning}\\n</think>\\n<answer>\\n```verilog\\n{code}\\n```\\n</answer>'",
    "    messages = [",
    "        {'role': 'system', 'content': 'You are an expert Verilog designer.'},",
    "        {'role': 'user', 'content': instruction},",
    "        {'role': 'assistant', 'content': response},",
    "    ]",
    "    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)",
    "",
    "def find_dataset():",
    "    import glob",
    "    if Path(TRAIN_PATH).exists():",
    "        return TRAIN_PATH, EVAL_PATH",
    "    for base in glob.glob('/kaggle/input/*verilog*'):",
    "        train = Path(base) / 'train.jsonl'",
    "        eval_f = Path(base) / 'eval.jsonl'",
    "        if train.exists():",
    "            return str(train), str(eval_f) if eval_f.exists() else None",
    "    working_train = Path(WORKING_DATASET_DIR) / 'train.jsonl'",
    "    if working_train.exists():",
    "        working_eval = Path(WORKING_DATASET_DIR) / 'eval.jsonl'",
    "        return str(working_train), str(working_eval) if working_eval.exists() else None",
    "    print('Downloading with kagglehub...')",
    "    import kagglehub",
    "    dataset_path = kagglehub.dataset_download('pablofloresmollinedo/verilog-curated-dataset')",
    "    return str(Path(dataset_path) / 'train.jsonl'), str(Path(dataset_path) / 'eval.jsonl')",
    "",
    "def tokenize_and_filter(text, max_len):",
    "    tokens = tokenizer(text, truncation=False, add_special_tokens=True)",
    "    if len(tokens['input_ids']) > max_len:",
    "        return None",
    "    return {",
    "        'input_ids': tokens['input_ids'],",
    "        'attention_mask': tokens['attention_mask'],",
    "        'labels': tokens['input_ids'].copy(),",
    "    }",
    "",
    "def load_dataset_file(train_path, eval_path=None):",
    "    train_ex = []",
    "    skipped_train = 0",
    "    with open(train_path) as f:",
    "        for line in f:",
    "            if not line.strip(): continue",
    "            ex = json.loads(line)",
    "            text = format_example(ex.get('spec',''), ex.get('code',''), ex.get('reasoning',''))",
    "            tok = tokenize_and_filter(text, MAX_SEQ_LENGTH)",
    "            if tok is None:",
    "                skipped_train += 1",
    "                continue",
    "            train_ex.append(tok)",
    "    eval_ex = []",
    "    skipped_eval = 0",
    "    if eval_path and Path(eval_path).exists():",
    "        with open(eval_path) as f:",
    "            for line in f:",
    "                if not line.strip(): continue",
    "                ex = json.loads(line)",
    "                text = format_example(ex.get('spec',''), ex.get('code',''), ex.get('reasoning',''))",
    "                tok = tokenize_and_filter(text, MAX_SEQ_LENGTH)",
    "                if tok is None:",
    "                    skipped_eval += 1",
    "                    continue",
    "                eval_ex.append(tok)",
    "    if xm.is_master_ordinal():",
    "        print(f'Train: {len(train_ex)} (skipped {skipped_train})')",
    "        print(f'Eval: {len(eval_ex)} (skipped {skipped_eval})')",
    "    return train_ex, eval_ex",
    "",
    "train_path, eval_path = find_dataset()",
    "train_data, eval_data = load_dataset_file(train_path, eval_path)",
    "print(f'Loaded {len(train_data)} training examples')"
]))

# Cell 7: Dataloaders
cells.append(cell("code", [
    "class VerilogDataset(torch.utils.data.Dataset):",
    "    def __init__(self, data):",
    "        self.data = data",
    "    def __len__(self):",
    "        return len(self.data)",
    "    def __getitem__(self, idx):",
    "        return self.data[idx]",
    "",
    "class PadCollator:",
    "    def __init__(self, pad_token_id, max_length):",
    "        self.pad_token_id = pad_token_id",
    "        self.max_length = max_length",
    "    def __call__(self, batch):",
    "        max_len = min(max(len(ex['input_ids']) for ex in batch), self.max_length)",
    "        input_ids, attention_mask, labels = [], [], []",
    "        for ex in batch:",
    "            ids = ex['input_ids'][:max_len]",
    "            mask = ex['attention_mask'][:max_len]",
    "            lbl = ex['labels'][:max_len]",
    "            pad_len = max_len - len(ids)",
    "            if pad_len > 0:",
    "                ids = ids + [self.pad_token_id] * pad_len",
    "                mask = mask + [0] * pad_len",
    "                lbl = lbl + [-100] * pad_len",
    "            input_ids.append(ids)",
    "            attention_mask.append(mask)",
    "            labels.append(lbl)",
    "        return {",
    "            'input_ids': torch.tensor(input_ids, dtype=torch.long),",
    "            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),",
    "            'labels': torch.tensor(labels, dtype=torch.long),",
    "        }",
    "",
    "train_dataset = VerilogDataset(train_data)",
    "train_loader = torch.utils.data.DataLoader(",
    "    train_dataset, batch_size=BATCH_SIZE, shuffle=True,",
    "    drop_last=True, num_workers=0,",
    "    collate_fn=PadCollator(tokenizer.pad_token_id, MAX_SEQ_LENGTH)",
    ")",
    "train_loader = pl.MpDeviceLoader(train_loader, device)",
    "",
    "STEPS_PER_EPOCH = len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM)",
    "TOTAL_STEPS = int(EPOCHS * STEPS_PER_EPOCH)",
    "print(f'Steps/epoch: {STEPS_PER_EPOCH}, Total: {TOTAL_STEPS}')"
]))

# Cell 8: Optimizer
cells.append(cell("code", [
    "from torch_xla.amp.syncfree import AdamW as SyncFreeAdamW",
    "from transformers import get_cosine_schedule_with_warmup",
    "",
    "update_params = list(filter(lambda p: p.requires_grad, model.parameters()))",
    "optimizer = SyncFreeAdamW(update_params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)",
    "warmup_steps = int(TOTAL_STEPS * WARMUP_RATIO)",
    "scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=TOTAL_STEPS)",
    "print(f'Optimizer ready. Warmup: {warmup_steps}, Total: {TOTAL_STEPS}')"
]))

# Cell 9: Training loop
cells.append(cell("code", [
    "class CheckpointManager:",
    "    def __init__(self, hub_model_id, hub_token, output_dir):",
    "        self.hub_model_id = hub_model_id",
    "        self.hub_token = hub_token",
    "        self.output_dir = Path(output_dir)",
    "        self.api = HfApi(token=hub_token)",
    "        os.makedirs(output_dir, exist_ok=True)",
    "    def upload_checkpoint(self, step):",
    "        if not xm.is_master_ordinal(): return",
    "        checkpoint_dir = self.output_dir / f'checkpoint-{step}'",
    "        if not checkpoint_dir.exists(): return",
    "        try:",
    "            self.api.upload_folder(folder_path=str(checkpoint_dir), repo_id=self.hub_model_id,",
    "                                   repo_type='model', path_in_repo=f'checkpoint-tpu-{step}', token=self.hub_token)",
    "            print(f'Uploaded checkpoint-tpu-{step}')",
    "        except Exception as e:",
    "            print(f'Upload failed: {e}')",
    "    def save_adapter(self, model, tokenizer, step):",
    "        if not xm.is_master_ordinal(): return",
    "        save_path = self.output_dir / f'checkpoint-{step}'",
    "        os.makedirs(save_path, exist_ok=True)",
    "        model.save_pretrained(save_path)",
    "        tokenizer.save_pretrained(save_path)",
    "        print(f'Saved checkpoint to {save_path}')",
    "",
    "ckpt_mgr = CheckpointManager(HUB_MODEL_ID, HF_TOKEN, OUTPUT_DIR)",
    "",
    "def compute_loss(outputs, labels, pad_id=tokenizer.pad_token_id):",
    "    epsilon = 1e-8",
    "    logits = outputs.logits[..., :-1, :].contiguous()",
    "    labels = labels[..., 1:].contiguous()",
    "    log_probs = -nn.functional.log_softmax(logits, dim=-1)",
    "    if labels.dim() == log_probs.dim() - 1:",
    "        labels = labels.unsqueeze(-1)",
    "    padding_mask = labels.eq(pad_id)",
    "    labels = torch.clamp(labels, min=0)",
    "    nll_loss = log_probs.gather(dim=-1, index=labels)",
    "    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.bfloat16)",
    "    nll_loss.masked_fill_(padding_mask, 0.0)",
    "    smoothed_loss.masked_fill_(padding_mask, 0.0)",
    "    num_active = padding_mask.numel() - padding_mask.long().sum()",
    "    nll_loss = nll_loss.sum() / num_active",
    "    smoothed_loss = smoothed_loss.sum() / (num_active * log_probs.shape[-1])",
    "    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss",
    "",
    "if xm.is_master_ordinal():",
    "    print('='*60)",
    "    print(f'MODEL: {MODEL_NAME}')",
    "    print(f'TPU CORES: {num_devices}')",
    "    print(f'EPOCHS: {EPOCHS} | STEPS: {TOTAL_STEPS}')",
    "    print('='*60)",
    "",
    "global_step = 0",
    "for epoch in range(1, EPOCHS + 1):",
    "    model.train()",
    "    xm.master_print(f'\\n>>> Epoch {epoch}/{EPOCHS}')",
    "    epoch_loss = 0.0",
    "    epoch_steps = 0",
    "    for step, batch in enumerate(train_loader):",
    "        input_ids = batch['input_ids'].to(device)",
    "        attention_mask = batch['attention_mask'].to(device)",
    "        labels = batch['labels'].to(device)",
    "        xs.mark_sharding(input_ids, mesh, (0, 1))",
    "        xs.mark_sharding(attention_mask, mesh, (0, 1))",
    "        xs.mark_sharding(labels, mesh, (0, 1))",
    "        outputs = model(input_ids=input_ids, attention_mask=attention_mask)",
    "        loss = compute_loss(outputs, labels) / GRAD_ACCUM",
    "        loss.backward()",
    "        loss_item = loss.detach().cpu().item() * GRAD_ACCUM",
    "        epoch_loss += loss_item",
    "        epoch_steps += 1",
    "        if (step + 1) % 10 == 0 and xm.is_master_ordinal():",
    "            xm.master_print(f'  step {global_step+1}/{TOTAL_STEPS} | loss: {loss_item:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}')",
    "        if (step + 1) % GRAD_ACCUM == 0:",
    "            torch.nn.utils.clip_grad_norm_(update_params, max_norm=MAX_GRAD_NORM * num_devices)",
    "            scheduler.step()",
    "            xm.optimizer_step(optimizer, pin_layout=True, barrier=True)",
    "            optimizer.zero_grad()",
    "            global_step += 1",
    "            if global_step % SAVE_STEPS == 0:",
    "                ckpt_mgr.save_adapter(model, tokenizer, global_step)",
    "                ckpt_mgr.upload_checkpoint(global_step)",
    "        del input_ids, attention_mask, labels, outputs, loss",
    "    if xm.is_master_ordinal():",
    "        print(f'  Epoch {epoch} avg loss: {epoch_loss/max(epoch_steps,1):.4f}')",
    "    xm.rendezvous(f'epoch_{epoch}_done')",
    "",
    "print('Training complete!')"
]))

# Cell 10: Save
cells.append(cell("code", [
    "if xm.is_master_ordinal():",
    "    final_path = Path(OUTPUT_DIR) / 'adapter_tpu_v1'",
    "    os.makedirs(final_path, exist_ok=True)",
    "    model.save_pretrained(final_path)",
    "    tokenizer.save_pretrained(final_path)",
    "    print(f'Saved to {final_path}')",
    "    try:",
    "        api.upload_folder(folder_path=str(final_path), repo_id=HUB_MODEL_ID,",
    "                          repo_type='model', path_in_repo='adapter_tpu_v1', token=HF_TOKEN)",
    "        print('Uploaded to HF Hub')",
    "    except Exception as e:",
    "        print(f'Upload failed: {e}')",
    "xm.rendezvous('done')"
]))

# Cell 11: Test
cells.append(cell("code", [
    "if xm.is_master_ordinal():",
    "    model.eval()",
    "    test_prompt = '''<verilog>",
    "Design a 4-bit up-counter with synchronous reset and enable.",
    "When enable is high, count increments on each clock rising edge.",
    "Reset clears count to 0.",
    "'''",
    "    messages = [",
    "        {'role': 'system', 'content': 'You are an expert Verilog designer.'},",
    "        {'role': 'user', 'content': test_prompt},",
    "    ]",
    "    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
    "    inputs = tokenizer(text, return_tensors='pt').to(device)",
    "    with torch.no_grad():",
    "        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2, top_p=0.95, do_sample=True)",
    "    raw = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)",
    "    print('Generated:')",
    "    print(raw)"
]))

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open('kaggle_tpu_fsdp.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f'Created notebook with {len(cells)} cells')
