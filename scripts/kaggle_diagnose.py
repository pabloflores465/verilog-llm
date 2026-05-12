"""
Diagnóstico de paths en Kaggle. Pega esto en una celda de tu notebook.
"""

import os
import glob

print("=== /kaggle/input ===")
for p in glob.glob("/kaggle/input/*"):
    print(f"  {p}")
    if os.path.isdir(p):
        for f in sorted(os.listdir(p)):
            fp = os.path.join(p, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                print(f"    {f} ({size:,} bytes)")

print("\n=== Verificar paths específicos ===")
paths = [
    "/kaggle/input/verilog-curated-dataset/train.jsonl",
    "/kaggle/input/verilog-curated-dataset/eval.jsonl",
    "/kaggle/input/verilog-curated-dataset/fm.jsonl",
]
for p in paths:
    exists = "✓" if os.path.exists(p) else "✗"
    print(f"  {exists} {p}")
