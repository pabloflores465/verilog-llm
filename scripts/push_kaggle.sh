#!/bin/bash
# Push headless training script to Kaggle
# This runs on Kaggle servers, not your laptop!

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Pushing Verilog SOTA to Kaggle (headless)"
echo "=========================================="

# Check if kaggle CLI exists
if ! command -v kaggle &> /dev/null; then
    echo "❌ kaggle CLI not found"
    echo "Install it first:"
    echo "  pip install kaggle"
    echo "Or upload manually via Kaggle web UI"
    exit 1
fi

# Push the script
echo ""
echo "Pushing kernel to Kaggle..."
kaggle kernels push -p "$SCRIPT_DIR"

echo ""
echo "✅ Done!"
echo ""
echo "Your training is now running on Kaggle's servers."
echo "You can close your laptop - it will keep running."
echo ""
echo "Monitor progress:"
echo "  https://www.kaggle.com/pabloflores/verilog-sota-training"
echo ""
echo "Checkpoints auto-upload to:"
echo "  https://huggingface.co/pabloflores/verilog-qwen-14b-sota"
echo ""
echo "Check logs with:"
echo "  kaggle kernels output pabloflores/verilog-sota-training --log"
