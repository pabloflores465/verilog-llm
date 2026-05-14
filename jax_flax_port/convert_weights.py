"""
Convert PyTorch Qwen2 weights to Flax parameters.
Run once to produce a JAX checkpoint, then train.
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import jax.numpy as jnp
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.numpy import load_file as load_safetensors

from config import ModelConfig, MODEL_7B, MODEL_14B


def load_pytorch_state_dict(model_name: str) -> Dict[str, np.ndarray]:
    """Download and load PyTorch weights as numpy arrays."""
    print(f"Downloading {model_name} ...")
    local_path = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.safetensors", "config.json"],
        local_dir=None,
    )
    print(f"Downloaded to cache")

    # Find all safetensors files
    files = sorted(Path(local_path).glob("model-*.safetensors"))
    if not files:
        files = [Path(local_path) / "model.safetensors"]

    state_dict = {}
    for f in files:
        if f.exists():
            print(f"  Loading {f.name} ...")
            chunk = load_safetensors(str(f))
            for k, v in chunk.items():
                state_dict[k] = v

    # Also load config to verify
    with open(Path(local_path) / "config.json") as f:
        config = json.load(f)
    print(f"  Config: hidden_size={config['hidden_size']}, layers={config['num_hidden_layers']}")

    return state_dict


def transpose_kernel(w: np.ndarray) -> np.ndarray:
    """Transpose PyTorch Linear weight to Flax Dense kernel.
    PyTorch: [out_features, in_features]
    Flax:    [in_features, out_features]
    """
    return w.T


def convert_to_flax_params(state_dict: Dict[str, np.ndarray], config: ModelConfig) -> Dict[str, Any]:
    """Map PyTorch state dict names to Flax param tree."""
    params = {}

    # Embeddings (same shape)
    params["embed_tokens"] = {
        "embedding": state_dict.pop("model.embed_tokens.weight").astype("float32")
    }

    # Layers
    for i in range(config.num_hidden_layers):
        layer_params = {}
        prefix = f"model.layers.{i}."
        flax_prefix = f"layers_{i}"

        # RMSNorms
        layer_params["input_layernorm"] = {
            "scale": state_dict.pop(f"{prefix}input_layernorm.weight").astype("float32")
        }
        layer_params["post_attention_layernorm"] = {
            "scale": state_dict.pop(f"{prefix}post_attention_layernorm.weight").astype("float32")
        }

        # Attention projections (transpose kernels)
        attn = {}
        attn["q_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}self_attn.q_proj.weight")).astype("float32")}
        }
        attn["k_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}self_attn.k_proj.weight")).astype("float32")}
        }
        attn["v_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}self_attn.v_proj.weight")).astype("float32")}
        }
        attn["o_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}self_attn.o_proj.weight")).astype("float32")}
        }
        layer_params["self_attn"] = attn

        # MLP projections
        mlp = {}
        mlp["gate_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}mlp.gate_proj.weight")).astype("float32")}
        }
        mlp["up_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}mlp.up_proj.weight")).astype("float32")}
        }
        mlp["down_proj"] = {
            "base": {"kernel": transpose_kernel(state_dict.pop(f"{prefix}mlp.down_proj.weight")).astype("float32")}
        }
        layer_params["mlp"] = mlp

        params[flax_prefix] = layer_params

    # Final norm
    params["norm"] = {
        "scale": state_dict.pop("model.norm.weight").astype("float32")
    }

    # LM head
    params["lm_head"] = {
        "kernel": transpose_kernel(state_dict.pop("lm_head.weight")).astype("float32")
    }

    # Report any leftovers (should be empty)
    if state_dict:
        print(f"\n⚠️ Unmapped keys ({len(state_dict)}):")
        for k in sorted(state_dict.keys()):
            print(f"  - {k}")
    else:
        print("\n✓ All PyTorch keys mapped successfully")

    return {"params": params}


def save_jax_checkpoint(flax_params: Dict, output_path: str):
    """Save Flax params with npz (portable)."""
    # Flatten dict
    flat = {}
    def _flatten(d, prefix=""):
        for k, v in d.items():
            new_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, new_key)
            else:
                flat[new_key] = v
    _flatten(flax_params)

    np.savez(output_path, **flat)
    print(f"✓ Saved JAX checkpoint: {output_path}")
    print(f"  Total arrays: {len(flat)}")
    total_mb = sum(v.nbytes for v in flat.values()) / 1e6
    print(f"  Total size: {total_mb:.1f} MB")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="7b", choices=["7b", "14b"])
    parser.add_argument("--output", default="/kaggle/working/qwen_jax_weights.npz")
    args = parser.parse_args()

    cfg = MODEL_7B if args.model == "7b" else MODEL_14B
    print(f"Converting {cfg.model_name} ...")

    state_dict = load_pytorch_state_dict(cfg.model_name)
    flax_params = convert_to_flax_params(state_dict, cfg)
    save_jax_checkpoint(flax_params, args.output)

    # Also save config for reference
    config_path = Path(args.output).with_suffix(".config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": cfg.model_name,
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "vocab_size": cfg.vocab_size,
            "max_position_embeddings": cfg.max_position_embeddings,
        }, f, indent=2)
    print(f"✓ Saved config: {config_path}")


if __name__ == "__main__":
    main()
