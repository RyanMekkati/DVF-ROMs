#!/usr/bin/env python3
import os, shutil

# ↳ change this to your folder of PNGs
src = "/Users/ryanmekkati/.../CineMRI_pgm"

# these will be created under `src/series_even` and `src/series_odd`
even_dir = os.path.join(src, "series_even")
odd_dir  = os.path.join(src, "series_odd")
os.makedirs(even_dir, exist_ok=True)
os.makedirs(odd_dir,  exist_ok=True)

# grab and sort all your slice_*.png files
files = sorted(f for f in os.listdir(src) if f.startswith("slice_") and f.endswith(".png"))

for idx, fname in enumerate(files):
    src_path = os.path.join(src, fname)
    if idx % 2 == 0:
        shutil.move(src_path, os.path.join(even_dir, fname))
    else:
        shutil.move(src_path, os.path.join(odd_dir, fname))

print(f"Moved {len(files)//2 + len(files)%2} → {even_dir}")
print(f"Moved {len(files)//2} → {odd_dir}")