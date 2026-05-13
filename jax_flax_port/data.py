"""
Data pipeline for Verilog training.
Uses tf.data for efficient loading and JAX-compatible output.
"""

import json
import glob
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

from config import TrainConfig


def format_example(
    spec: str,
    code: str,
    reasoning: str = "",
    tokenizer: AutoTokenizer = None,
) -> str:
    """Format a single example into Qwen chat template."""
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
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def find_dataset_paths(cfg: TrainConfig) -> Tuple[str, Optional[str]]:
    """Auto-detect dataset location in Kaggle environment."""
    if Path(cfg.train_path).exists():
        print(f"  ✓ Dataset mounted at: {Path(cfg.train_path).parent}")
        return cfg.train_path, cfg.eval_path

    for base in glob.glob("/kaggle/input/*verilog*"):
        train = Path(base) / "train.jsonl"
        eval_f = Path(base) / "eval.jsonl"
        if train.exists():
            print(f"  ✓ Found dataset at: {base}")
            return str(train), str(eval_f) if eval_f.exists() else None

    working_train = Path(cfg.working_dataset_dir) / "train.jsonl"
    working_eval = Path(cfg.working_dataset_dir) / "eval.jsonl"
    if working_train.exists():
        print(f"  ✓ Dataset found in working dir: {cfg.working_dataset_dir}")
        return str(working_train), str(working_eval) if working_eval.exists() else None

    raise FileNotFoundError(
        f"Dataset not found. Expected at {cfg.train_path} or "
        f"/kaggle/input/*verilog*/train.jsonl"
    )


def load_raw_texts(path: str, tokenizer: AutoTokenizer) -> List[str]:
    """Load JSONL and format all examples."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            text = format_example(
                ex.get("spec", ""),
                ex.get("code", ""),
                ex.get("reasoning", ""),
                tokenizer,
            )
            texts.append(text)
    return texts


def build_tokenized_dataset(
    cfg: TrainConfig,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    skip_overlength: bool = True,
    num_devices: int = 1,
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], int]:
    """Build tf.data.Dataset from JSONL files.
    Returns (train_dataset, eval_dataset, steps_per_epoch).
    If num_devices > 1, global batch size = cfg.batch_size * num_devices.
    """
    train_path, eval_path = find_dataset_paths(cfg)

    print(f"\nLoading raw texts...")
    train_texts = load_raw_texts(train_path, tokenizer)
    print(f"  Train examples: {len(train_texts)}")

    eval_texts = None
    if eval_path and Path(eval_path).exists():
        eval_texts = load_raw_texts(eval_path, tokenizer)
        print(f"  Eval examples: {len(eval_texts)}")

    # Tokenize in Python space (JAX doesn't have a fast tokenizer)
    def tokenize_batch(texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Tokenize a batch and return input_ids + labels."""
        all_ids = []
        all_labels = []
        skipped = 0
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > max_seq_length:
                if skip_overlength:
                    skipped += 1
                    continue
                else:
                    ids = ids[:max_seq_length]
            # Pad to max_seq_length
            if len(ids) < max_seq_length:
                ids = ids + [tokenizer.pad_token_id] * (max_seq_length - len(ids))
            all_ids.append(ids)
            # Labels: same as input_ids for causal LM (mask padding in loss)
            labels = list(ids)
            # Replace padding with -100
            for i in range(len(labels)):
                if labels[i] == tokenizer.pad_token_id:
                    labels[i] = -100
            all_labels.append(labels)
        if skipped:
            print(f"  Skipped {skipped} overlength examples (>{max_seq_length})")
        return np.array(all_ids, dtype=np.int32), np.array(all_labels, dtype=np.int32)

    train_ids, train_labels = tokenize_batch(train_texts)
    print(f"  Tokenized train: {train_ids.shape}")

    eval_ids, eval_labels = None, None
    if eval_texts:
        eval_ids, eval_labels = tokenize_batch(eval_texts)
        print(f"  Tokenized eval: {eval_ids.shape}")

    # Build tf.data.Dataset
    global_batch_size = cfg.batch_size * num_devices
    print(f"  Batch per device: {cfg.batch_size} | Global batch: {global_batch_size}")

    def make_dataset(ids: np.ndarray, labels: np.ndarray) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((ids, labels))
        ds = ds.shuffle(buffer_size=min(10000, len(ids)), reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.batch(global_batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(train_ids, train_labels)
    eval_ds = make_dataset(eval_ids, eval_labels) if eval_ids is not None else None

    # Calculate steps per epoch
    steps_per_epoch = len(train_ids) // global_batch_size
    print(f"  Steps per epoch: {steps_per_epoch}")

    return train_ds, eval_ds, steps_per_epoch


def prepare_batch_for_jax(batch: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tf batch to numpy for JAX."""
    input_ids, labels = batch
    return input_ids.numpy(), labels.numpy()
