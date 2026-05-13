"""
Checkpointing utilities for JAX/Flax training.
Supports local save/load and HF Hub upload.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints as flax_checkpoints
from huggingface_hub import HfApi, create_repo


def flatten_dict(d: Dict, parent_key: str = "") -> Dict[str, np.ndarray]:
    """Flatten nested dict to flat dict with '/' separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}/{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = np.array(v)
    return items


def unflatten_dict(d: Dict[str, np.ndarray]) -> Dict:
    """Unflatten dict with '/' separated keys."""
    result = {}
    for key, value in d.items():
        parts = key.split("/")
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def save_checkpoint(
    state,
    output_dir: str,
    step: int,
    name: Optional[str] = None,
) -> str:
    """Save Flax TrainState to local directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if name:
        ckpt_dir = output_dir / name
    else:
        ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    params_flat = flatten_dict(state.params)
    np.savez(str(ckpt_dir / "params.npz"), **params_flat)

    # Save optimizer state
    opt_state_flat = flatten_dict(state.opt_state)
    np.savez(str(ckpt_dir / "opt_state.npz"), **opt_state_flat)

    # Save metadata
    metadata = {
        "step": int(step),
        "dropout_rng": int(state.dropout_rng[0]) if hasattr(state.dropout_rng, '__len__') else int(state.dropout_rng),
    }
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return str(ckpt_dir)


def load_checkpoint(
    ckpt_dir: str,
    state_template,
) -> Any:
    """Load Flax TrainState from local directory."""
    ckpt_dir = Path(ckpt_dir)

    # Load params
    params_flat = dict(np.load(str(ckpt_dir / "params.npz"), allow_pickle=False))
    params = unflatten_dict(params_flat)
    params = jax.tree_map(jnp.array, params)

    # Load opt_state
    opt_flat = dict(np.load(str(ckpt_dir / "opt_state.npz"), allow_pickle=False))
    opt_state = unflatten_dict(opt_flat)
    opt_state = jax.tree_map(jnp.array, opt_state)

    # Load metadata
    with open(ckpt_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Reconstruct state
    return state_template.replace(
        params=params,
        opt_state=opt_state,
        step=metadata["step"],
    )


def save_jax_checkpoint(flax_params: Dict, output_path: str):
    """Save raw Flax params (e.g. converted weights) as NPZ."""
    flat = flatten_dict(flax_params)
    np.savez(output_path, **flat)
    print(f"✓ Saved JAX checkpoint: {output_path}")
    total_mb = sum(v.nbytes for v in flat.values()) / 1e6
    print(f"  Total size: {total_mb:.1f} MB ({total_mb/1024:.2f} GB)")


def load_jax_checkpoint(path: str) -> Dict:
    """Load raw Flax params from NPZ."""
    flat = dict(np.load(path, allow_pickle=False))
    params = unflatten_dict(flat)
    params = jax.tree_map(jnp.array, params)
    # Unwrap if wrapped in 'params' key
    if "params" in params and len(params) == 1:
        params = params["params"]
    return params


def upload_to_hub(
    local_path: str,
    repo_id: str,
    token: str,
    step: Optional[int] = None,
    name: Optional[str] = None,
):
    """Upload checkpoint folder to HF Hub."""
    api = HfApi(token=token)

    try:
        create_repo(repo_id, repo_type="model", exist_ok=True, token=token)
    except Exception:
        pass

    path_in_repo = name if name else f"checkpoint-{step}"
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
    )
    print(f"✓ Uploaded to {repo_id}/{path_in_repo}")


def get_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find latest checkpoint directory."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and (d.name.startswith("checkpoint-") or d.name == "interrupted")]
    if not checkpoints:
        return None
    # Prefer checkpoint-N over interrupted
    numeric = [d for d in checkpoints if d.name.startswith("checkpoint-")]
    if numeric:
        latest = max(numeric, key=lambda p: int(p.name.split("-")[1]))
    else:
        latest = checkpoints[0]
    return str(latest)
