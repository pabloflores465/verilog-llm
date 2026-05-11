#!/usr/bin/env python3
"""
Setup Kaggle API credentials for headless training.
"""

import os
import json
from pathlib import Path


def setup_kaggle_token(token: str):
    """Write Kaggle API token to ~/.kaggle/kaggle.json"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": "pabloflores", "key": token}))
    kaggle_json.chmod(0o600)
    
    print(f"Kaggle token saved to {kaggle_json}")


def setup_hf_token(token: str):
    """Login to HuggingFace with token."""
    from huggingface_hub import login
    login(token=token)
    print("HuggingFace login successful")


if __name__ == "__main__":
    HF_TOKEN = "HF_TOKEN_PLACEHOLDER"
    KAGGLE_TOKEN = "KAGGLE_TOKEN_PLACEHOLDER"
    
    print("Setting up credentials...")
    setup_kaggle_token(KAGGLE_TOKEN)
    setup_hf_token(HF_TOKEN)
    print("Done!")
