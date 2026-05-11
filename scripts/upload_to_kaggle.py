#!/usr/bin/env python3
"""
Upload the Verilog dataset to Kaggle.
For Nix environments, creates a tarball for manual upload.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def create_upload_package(data_dir: str = "data/verilog_sota", output: str = "/tmp/verilog-kaggle-upload.zip"):
    """Create a zip file ready for Kaggle dataset upload."""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: {data_dir} not found")
        return False
    
    files = list(data_path.glob("*.jsonl"))
    if not files:
        print(f"Error: No .jsonl files in {data_dir}")
        return False
    
    # Create temp dir with metadata
    tmp = Path("/tmp/kaggle_upload_pkg")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    
    for f in files:
        shutil.copy(f, tmp / f.name)
        print(f"  Added {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    
    # Metadata
    metadata = {
        "title": f"Verilog SOTA Dataset - {datetime.now().strftime('%Y-%m-%d')}",
        "id": "pabloflores/verilog-sota-dataset",
        "licenses": [{"name": "Apache-2.0"}],
        "description": "High-quality Verilog training dataset with 25K examples. Compiled-validated with Icarus Verilog. Includes SI-CoT reasoning, FIM format, and code-specific prompts.",
        "keywords": ["verilog", "hdl", "rtl", "code-generation", "llm"],
    }
    (tmp / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # Zip it
    zip_path = Path(output)
    if zip_path.exists():
        zip_path.unlink()
    
    shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', tmp)
    print(f"\n✓ Package created: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size / 1e6:.1f} MB")
    return True


def try_kaggle_cli_upload(data_dir: str = "data/verilog_sota"):
    """Try to upload using kaggle CLI if available."""
    
    kaggle = shutil.which("kaggle")
    if not kaggle:
        return False
    
    # Setup token
    token = "KAGGLE_TOKEN_PLACEHOLDER"
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(json.dumps({"username": "pabloflores", "key": token}))
    kaggle_json.chmod(0o600)
    
    # Create temp upload folder
    upload_dir = Path("/tmp/kaggle_upload")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)
    
    data_path = Path(data_dir)
    for f in data_path.glob("*.jsonl"):
        shutil.copy(f, upload_dir / f.name)
    
    metadata = {
        "title": f"Verilog SOTA Dataset - {datetime.now().strftime('%Y-%m-%d')}",
        "id": "pabloflores/verilog-sota-dataset",
        "licenses": [{"name": "Apache-2.0"}],
        "description": "High-quality Verilog training dataset with 25K examples.",
        "keywords": ["verilog", "hdl", "rtl", "code-generation", "llm"],
    }
    (upload_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    
    print("Uploading via kaggle CLI...")
    result = subprocess.run(
        ["kaggle", "datasets", "create", "-p", str(upload_dir), "-r", "zip", "--quiet"],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("✓ Uploaded!")
        return True
    else:
        # Try update
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(upload_dir), "-r", "zip", 
             "-m", f"Update {datetime.now().isoformat()}", "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✓ Updated!")
            return True
    
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("Uploading Verilog Dataset to Kaggle")
    print("=" * 60)
    
    # Try CLI first
    if try_kaggle_cli_upload():
        print("\nDone! Dataset uploaded via CLI.")
        print("Next: Create Kaggle notebook, add 'verilog-sota-dataset' as input")
    else:
        print("Kaggle CLI not available (Nix env). Creating manual upload package...\n")
        if create_upload_package():
            print("\n" + "=" * 60)
            print("MANUAL UPLOAD REQUIRED")
            print("=" * 60)
            print("\n1. Go to https://www.kaggle.com/datasets")
            print("2. Click 'New Dataset'")
            print("3. Upload: /tmp/verilog-kaggle-upload.zip")
            print("4. Set title: 'verilog-sota-dataset'")
            print("5. Click 'Create'")
            print("\nThen in your Kaggle notebook:")
            print("  Add Input → Datasets → search 'verilog-sota'")
            print("=" * 60)
