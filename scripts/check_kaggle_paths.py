"""Diagnóstico: lista archivos en /kaggle/input/ para ver qué datasets están disponibles."""

import os

def ls(path, indent=0):
    prefix = "  " * indent
    try:
        items = sorted(os.listdir(path))
    except FileNotFoundError:
        print(f"{prefix}[NOT FOUND] {path}")
        return
    except PermissionError:
        print(f"{prefix}[NO PERM] {path}")
        return

    print(f"{prefix}{path}/")
    for item in items:
        full = os.path.join(path, item)
        if os.path.isdir(full):
            ls(full, indent + 1)
        else:
            size = os.path.getsize(full)
            print(f"{prefix}  {item} ({size:,} bytes)")


print("=== /kaggle/input ===")
ls("/kaggle/input")

print("\n=== /kaggle/input/verilog-curated-dataset (directo) ===")
ls("/kaggle/input/verilog-curated-dataset")

print("\n=== Posibles variantes de nombre ===")
import glob
for p in glob.glob("/kaggle/input/*verilog*"):
    print(f"  {p}")
