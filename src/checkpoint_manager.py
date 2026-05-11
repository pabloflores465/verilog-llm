"""
Checkpoint manager for Kaggle training.
Handles automatic upload to HuggingFace Hub with resume support.
"""

import os
import json
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, create_repo


class CheckpointManager:
    """Manages checkpoints upload and resume for Kaggle training."""
    
    def __init__(
        self,
        hub_model_id: str = "pabloflores/verilog-qwen-14b-sota",
        hub_token: str = "HF_TOKEN_PLACEHOLDER",
        output_dir: str = "/kaggle/working/checkpoints",
    ):
        self.hub_model_id = hub_model_id
        self.hub_token = hub_token
        self.output_dir = Path(output_dir)
        self.api = HfApi(token=hub_token)
        
        # Create repo if it doesn't exist
        try:
            create_repo(hub_model_id, private=False, token=hub_token, exist_ok=True)
            print(f"HF Hub repo ready: {hub_model_id}")
        except Exception as e:
            print(f"Repo may already exist: {e}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find latest local checkpoint for resume."""
        if not self.output_dir.exists():
            return None
        
        checkpoints = [
            d for d in self.output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        
        if not checkpoints:
            return None
        
        # Sort by step number
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        return str(latest)
    
    def upload_checkpoint(self, checkpoint_dir: str, step: int):
        """Upload a checkpoint folder to HF Hub."""
        try:
            self.api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=self.hub_model_id,
                repo_type="model",
                path_in_repo=f"checkpoint-{step}",
                token=self.hub_token,
            )
            print(f"✓ Uploaded checkpoint-{step} to {self.hub_model_id}")
        except Exception as e:
            print(f"✗ Failed to upload checkpoint-{step}: {e}")
    
    def save_training_state(self, step: int, epoch: float, loss: float):
        """Save training state for resume tracking."""
        state = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "hub_model_id": self.hub_model_id,
        }
        state_file = self.output_dir / "training_state.json"
        state_file.write_text(json.dumps(state, indent=2))
        
        # Also upload to hub root
        try:
            self.api.upload_file(
                path_or_fileobj=str(state_file),
                path_in_repo="training_state.json",
                repo_id=self.hub_model_id,
                token=self.hub_token,
            )
        except Exception:
            pass
    
    def load_training_state(self) -> Optional[dict]:
        """Load training state if exists."""
        state_file = self.output_dir / "training_state.json"
        if state_file.exists():
            return json.loads(state_file.read_text())
        return None
