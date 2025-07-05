#!/usr/bin/env python3
"""
split_even_odd.py

Given a folder of PNGs, move every other file into two subfolders:
  - series_even (files with even indices 0,2,4…)
  - series_odd  (files with odd  indices 1,3,5…)
"""

import os
import shutil
import argparse

def split_even_odd(src_dir, even_name="series_even", odd_name="series_odd"):
    # create target subfolders
    even_dir = os.path.join(src_dir, even_name)
    odd_dir  = os.path.join(src_dir, odd_name)
    os.makedirs(even_dir, exist_ok=True)
    os.makedirs(odd_dir,  exist_ok=True)

    # grab and sort all .png files
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    if not files:
        print("No PNGs found in", src_dir)
        return

    # move them
    for idx, fname in enumerate(files):
        src_path = os.path.join(src_dir, fname)
        dst_dir  = even_dir if idx % 2 == 0 else odd_dir
        shutil.move(src_path, os.path.join(dst_dir, fname))

    print(f"Moved {len(files)//2 + len(files)%2} files to '{even_name}'")
    print(f"Moved {len(files)//2} files to '{odd_name}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split PNGs in a folder into two series_even/series_odd subfolders"
    )
    p.add_argument("src_dir", help="Folder containing your .png files")
    p.add_argument(
        "--even-name", default="series_even",
        help="Name of the subfolder for the even-indexed files"
    )
    p.add_argument(
        "--odd-name",  default="series_odd",
        help="Name of the subfolder for the odd-indexed files"
    )
    args = p.parse_args()
    split_even_odd(args.src_dir, args.even_name, args.odd_name)