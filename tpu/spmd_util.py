"""
SPMD Sharding utilities for TPU training.
Adapted from TPU-Tuner (https://github.com/IsNoobgrammer/TPU-Tuner)

Handles model parallelism sharding across TPU cores using torch_xla SPMD.
"""

import math
import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.experimental.xla_sharding as xs
from transformers import (
    GPTNeoXConfig, T5Config, LlamaConfig, GPT2Config,
    MistralConfig, Qwen2Config, MixtralConfig, PhiConfig, GemmaConfig
)

# Sharding rules: (regex_pattern, (sharding_spec_tuple))
# fsdp = data parallelism axis (shard across cores)
# mp = model parallelism axis (shard within core)
# Order matches mesh axes: ('dp', 'fsdp', 'mp')

QWEN_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
)

LLAMA_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
)

MISTRAL_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.(gate_proj|up_proj)", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("lm_head", ("fsdp", "mp")),
)

GPT2_RULES = (
    ("wte", ("mp", "fsdp")),
    ("wpe", ("mp", "fsdp")),
    ("c_attn", ("fsdp", "mp")),
    ("c_proj", ("mp", "fsdp")),
    ("c_fc", ("fsdp", "mp")),
    ("ln_f", ("fsdp", "mp")),
)

T5_RULES = (
    ("shared$", ("mp", "fsdp")),
    ("embed_tokens$", ("mp", "fsdp")),
    ("q$", ("fsdp", "mp")),
    ("k$", ("fsdp", "mp")),
    ("v$", ("fsdp", "mp")),
    ("o$", ("mp", "fsdp")),
    ("w$", ("fsdp", "mp")),
    ("wi_0$", ("fsdp", "mp")),
    ("wi_1$", ("fsdp", "mp")),
    ("wo$", ("mp", "fsdp")),
    ("lm_head", ("fsdp", "mp")),
)

PHI_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.dense", ("mp", "fsdp")),
    ("mlp\\.fc2", ("mp", "fsdp")),
    ("mlp\\.fc1", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
)

GPTNEOX_RULES = (
    ("gpt_neox\\.embed_in", ("mp", "fsdp")),
    ("attention\\.query_key_value$", ("fsdp", "mp")),
    ("attention\\.dense$", ("mp", "fsdp")),
    ("mlp\\.dense_h_to_4h$", ("fsdp", "mp")),
    ("mlp\\.dense_4h_to_h$", ("mp", "fsdp")),
    ("embed_out", ("fsdp", "mp")),
)

MIXTRAL_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("w1", ("fsdp", "mp")),
    ("w2", ("mp", "fsdp")),
    ("w3", ("fsdp", "mp")),
    ("gate", ("mp", "fsdp")),
    ("lm_head", ("fsdp", "mp")),
)

GEMMA_RULES = (
    ("model\\.embed_tokens", ("mp", "fsdp")),
    ("self_attn\\.(q_proj|k_proj|v_proj)", ("fsdp", "mp")),
    ("self_attn\\.o_proj", ("mp", "fsdp")),
    ("mlp\\.gate_proj", ("fsdp", "mp")),
    ("mlp\\.down_proj", ("mp", "fsdp")),
    ("mlp\\.up_proj", ("fsdp", "mp")),
    ("lm_head", ("fsdp", "mp")),
)

ALL_RULES = [
    (GPTNeoXConfig, GPTNEOX_RULES),
    (T5Config, T5_RULES),
    (LlamaConfig, LLAMA_RULES),
    (GPT2Config, GPT2_RULES),
    (MistralConfig, MISTRAL_RULES),
    (Qwen2Config, QWEN_RULES),
    (MixtralConfig, MIXTRAL_RULES),
    (PhiConfig, PHI_RULES),
    (GemmaConfig, GEMMA_RULES),
]

# Mapping from axis name to mesh axis index
strkey2id = {
    "dp": 0,
    "fsdp": 1,
    "mp": 2,
}


def find_rule(model) -> Tuple:
    """Find the sharding rules for a given model config."""
    for config_cls, rule in ALL_RULES:
        if isinstance(model.config, config_cls):
            return rule
    # Try matching by class name
    model_cls_name = model.config.__class__.__name__
    for config_cls, rule in ALL_RULES:
        if config_cls.__name__ == model_cls_name:
            return rule
    raise ValueError(
        f"Unsupported model config: {model_cls_name}. "
        f"Supported: {[c.__name__ for c, _ in ALL_RULES]}"
    )


def partition_module(model, mesh, device=None, verbose=False):
    """
    Partition model weights across TPU cores using SPMD sharding.

    Args:
        model: The model to partition
        mesh: torch_xla.experimental.xla_sharding.Mesh
        device: Target device (defaults to xm.xla_device())
        verbose: Print sharding decisions
    """
    if device is None:
        device = xm.xla_device()

    partition_specs = find_rule(model)
    rule = [(k, tuple([strkey2id[x] for x in v])) for k, v in partition_specs]

    for name, module in model.named_modules():
        module.to(device)
        if isinstance(module, (nn.Embedding, nn.Linear)):
            for rule_pattern, spec in rule:
                if re.findall(rule_pattern, name.lower()):
                    if verbose:
                        print(f"[shard] {name} -> {spec}")
                    xs.mark_sharding(module.weight, mesh, spec)
                    break
