#!/usr/bin/env python3
"""
Create SIMPLE Kaggle TPU notebook (7B single-core, no SPMD).
"""

import json

def cell(cell_type, lines):
    source = [line + '\n' for line in lines]
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

# Cell 0: Markdown
cells.append(cell("markdown", [
    "# Verilog Qwen 7B Training on TPU v3-8 (Single-Core)",
    "",
    "**Simple & Robust approach:**",
    "- Qwen 7B in bfloat16 on 1 TPU core (~14GB < 16GB HBM)",
    "- Standard transformers Trainer (no SPMD/FSDP complexity)",
    "- ~10-15s/step, 3 epochs in ~4-6 hours",
    "",
    "Set accelerator to **TPU VM v3-8** and add `HF_TOKEN` to Kaggle Secrets."
]))

# Cell 1: Install
cells.append(cell("code", [
    "# Install torch_xla explicitly (Kaggle TPU VMs don't always have it pre-installed)",
    "!pip install -q https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    "!pip install -q transformers>=4.40.0 peft datasets accelerate huggingface-hub sentencepiece",
    "!pip install -q \"protobuf>=5.29.1,<6.0.0\" kagglehub",
    "print('Done installing')"
]))

# Cell 2: Verify TPU
cells.append(cell("code", [
    "import os",
    "import torch",
    "",
    "# TPU detection",
    "try:",
    "    import torch_xla",
    "    import torch_xla.core.xla_model as xm",
    "    device = xm.xla_device()",
    "    print(f'✓ TPU detected: {xm.xrt_world_size()} cores')",
    "    print(f'✓ Device: {device}')",
    "except Exception as e:",
    "    print(f'⚠️ TPU not available: {e}')",
    "    print('Make sure TPU VM v3-8 is selected in notebook settings!')",
    "    raise",
    "",
    "print(f'PyTorch version: {torch.__version__}')",
    "print(f'torch_xla version: {torch_xla.__version__}')"
]))

# Cell 3: Config & imports
cells.append(cell("code", [
    "import os",
    "import sys",
    "import json",
    "import re",
    "from pathlib import Path",
    "",
    "# === CONFIG ===",
    "HUB_MODEL_ID = 'Pablo-Flores-Mollinedo/verilog-qwen-14b-sota'",
    "MODEL_NAME = 'Qwen/Qwen2.5-Coder-7B-Instruct'",
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
    "",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)",
    "print('Config loaded')"
]))

# Cell 4: HF Token
cells.append(cell("code", [
    "HF_TOKEN = None",
    "",
    "try:",
    "    from kaggle_secrets import UserSecretsClient",
    "    secrets = UserSecretsClient()",
    "    HF_TOKEN = secrets.get_secret('HF_TOKEN')",
    "    if HF_TOKEN:",
    "        print('✓ HF_TOKEN from Kaggle Secrets')",
    "except Exception:",
    "    pass",
    "",
    "if not HF_TOKEN:",
    "    HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()",
    "    if HF_TOKEN:",
    "        print('✓ HF_TOKEN from env')",
    "",
    "if not HF_TOKEN:",
    "    raise ValueError('HF_TOKEN not found. Add to Kaggle Secrets.')",
    "",
    "from huggingface_hub import login, HfApi",
    "login(token=HF_TOKEN)",
    "api = HfApi(token=HF_TOKEN)",
    "print('✓ HF Login OK')"
]))

# Cell 5: Load model + LoRA
cells.append(cell("code", [
    "from transformers import AutoModelForCausalLM, AutoTokenizer",
    "from peft import LoraConfig, get_peft_model",
    "",
    "print(f'Loading {MODEL_NAME}...')",
    "",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)",
    "tokenizer.pad_token = tokenizer.eos_token",
    "tokenizer.padding_side = 'right'",
    "",
    "# Load in bfloat16 (native on TPU v3)",
    "model = AutoModelForCausalLM.from_pretrained(",
    "    MODEL_NAME,",
    "    torch_dtype=torch.bfloat16,",
    "    trust_remote_code=True,",
    ")",
    "",
    "# Move to TPU",
    "model = model.to(device)",
    "",
    "lora_config = LoraConfig(",
    "    r=LORA_R, lora_alpha=LORA_ALPHA,",
    "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],",
    "    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM',",
    ")",
    "model = get_peft_model(model, lora_config)",
    "model.print_trainable_parameters()",
    "",
    "# Check memory",
    "mem = xm.get_memory_info(device)['bytes_limit'] / 1e9 if hasattr(xm, 'get_memory_info') else 'N/A'",
    "print(f'TPU memory limit: {mem} GB')"
]))

# Cell 6: Dataset
cells.append(cell("code", [
    "from datasets import Dataset",
    "",
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
    "def load_dataset_file(train_path, eval_path=None):",
    "    train_ex = []",
    "    with open(train_path) as f:",
    "        for line in f:",
    "            if not line.strip(): continue",
    "            ex = json.loads(line)",
    "            text = format_example(ex.get('spec',''), ex.get('code',''), ex.get('reasoning',''))",
    "            train_ex.append({'text': text})",
    "    eval_ex = []",
    "    if eval_path and Path(eval_path).exists():",
    "        with open(eval_path) as f:",
    "            for line in f:",
    "                if not line.strip(): continue",
    "                ex = json.loads(line)",
    "                text = format_example(ex.get('spec',''), ex.get('code',''), ex.get('reasoning',''))",
    "                eval_ex.append({'text': text})",
    "    print(f'Train: {len(train_ex)} | Eval: {len(eval_ex)}')",
    "    return Dataset.from_list(train_ex), (Dataset.from_list(eval_ex) if eval_ex else None)",
    "",
    "train_path, eval_path = find_dataset()",
    "train_dataset, eval_dataset = load_dataset_file(train_path, eval_path)"
]))

# Cell 7: Training
cells.append(cell("code", [
    "from transformers import TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling",
    "from trl import SFTTrainer",
    "import inspect",
    "",
    "class HFHubCheckpointCallback(TrainerCallback):",
    "    def __init__(self, save_steps, hub_model_id, hub_token):",
    "        self.save_steps = save_steps",
    "        self.hub_model_id = hub_model_id",
    "        self.hub_token = hub_token",
    "        self.api = HfApi(token=hub_token)",
    "    def on_save(self, args, state, control, **kwargs):",
    "        checkpoint_dir = Path(args.output_dir) / f'checkpoint-{state.global_step}'",
    "        if checkpoint_dir.exists():",
    "            print(f'\\n📤 Uploading checkpoint-{state.global_step}...')",
    "            try:",
    "                self.api.upload_folder(folder_path=str(checkpoint_dir), repo_id=self.hub_model_id,",
    "                                       repo_type='model', path_in_repo=f'checkpoint-{state.global_step}', token=self.hub_token)",
    "                print(f'✓ Uploaded checkpoint-{state.global_step}')",
    "            except Exception as e:",
    "                print(f'✗ Upload failed: {e}')",
    "        return control",
    "",
    "training_args = TrainingArguments(",
    "    output_dir=OUTPUT_DIR,",
    "    num_train_epochs=EPOCHS,",
    "    per_device_train_batch_size=BATCH_SIZE,",
    "    gradient_accumulation_steps=GRAD_ACCUM,",
    "    learning_rate=LR,",
    "    warmup_steps=50,",
    "    lr_scheduler_type='cosine',",
    "    max_grad_norm=0.3,",
    "    weight_decay=0.001,",
    "    optim='adamw_torch',",
    "    save_strategy='steps',",
    "    save_steps=SAVE_STEPS,",
    "    save_total_limit=3,",
    "    eval_strategy='no',",
    "    logging_steps=10,",
    "    logging_first_step=True,",
    "    push_to_hub=False,",
    "    bf16=True,",
    "    gradient_checkpointing=True,",
    "    gradient_checkpointing_kwargs={'use_reentrant': False},",
    "    report_to='none',",
    "    seed=42,",
    ")",
    "",
    "data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)",
    "",
    "sft_args = set(inspect.signature(SFTTrainer.__init__).parameters.keys())",
    "trainer_kwargs = {",
    "    'model': model,",
    "    'train_dataset': train_dataset,",
    "    'args': training_args,",
    "    'data_collator': data_collator,",
    "    'callbacks': [HFHubCheckpointCallback(SAVE_STEPS, HUB_MODEL_ID, HF_TOKEN)],",
    "}",
    "if 'processing_class' in sft_args:",
    "    trainer_kwargs['processing_class'] = tokenizer",
    "elif 'tokenizer' in sft_args:",
    "    trainer_kwargs['tokenizer'] = tokenizer",
    "if 'max_seq_length' in sft_args:",
    "    trainer_kwargs['max_seq_length'] = MAX_SEQ_LENGTH",
    "if 'dataset_text_field' in sft_args:",
    "    trainer_kwargs['dataset_text_field'] = 'text'",
    "",
    "trainer = SFTTrainer(**trainer_kwargs)",
    "",
    "print('\\n' + '='*60)",
    "print(f'STARTING TRAINING')",
    "print(f'Model: {MODEL_NAME}')",
    "print(f'Device: {device}')",
    "print(f'Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}')",
    "print('='*60)",
    "",
    "trainer.train()"
]))

# Cell 8: Save final
cells.append(cell("code", [
    "final_path = Path(OUTPUT_DIR) / 'adapter_tpu_v1'",
    "os.makedirs(final_path, exist_ok=True)",
    "model.save_pretrained(final_path)",
    "tokenizer.save_pretrained(final_path)",
    "print(f'✓ Saved to {final_path}')",
    "",
    "try:",
    "    api.upload_folder(folder_path=str(final_path), repo_id=HUB_MODEL_ID,",
    "                      repo_type='model', path_in_repo='adapter_tpu_v1', token=HF_TOKEN)",
    "    print('✓ Uploaded to HF Hub')",
    "except Exception as e:",
    "    print(f'✗ Upload failed: {e}')",
    "",
    "print('\\n' + '='*60)",
    "print('TRAINING COMPLETE')",
    "print(f'Final: {HUB_MODEL_ID}/adapter_tpu_v1')",
    "print('='*60)"
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

with open('kaggle_tpu_simple.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f'Created simple notebook with {len(cells)} cells')
