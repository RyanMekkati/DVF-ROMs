#!/usr/bin/env python3
"""
Convert all .ima (DICOM) files in a folder into 8-bit grayscale PNGs.
"""

import os
import argparse

import pydicom
import numpy as np
from PIL import Image

def dicom_to_png(dcm_path, png_path, clip_percentile=(1, 99)):
    # read DICOM
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)

    # optional: window/level via percentiles to boost contrast
    lo, hi = np.percentile(arr, clip_percentile)
    arr = np.clip(arr, lo, hi)

    # normalize to [0,255]
    arr = (arr - lo) / (hi - lo) * 255.0
    arr = np.round(arr).astype(np.uint8)

    # save PNG
    Image.fromarray(arr).save(png_path)

def main():
    p = argparse.ArgumentParser(
        description="Batch-convert .ima (DICOM) → 8-bit PNG"
    )
    p.add_argument("in_dir", help="Folder containing .ima files")
    p.add_argument("out_dir", help="Folder to write PNGs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # gather and sort all .ima files
    files = sorted(f for f in os.listdir(args.in_dir)
                   if f.lower().endswith(".ima"))

    for idx, fname in enumerate(files, 1):
        dcm_path = os.path.join(args.in_dir, fname)
        png_name = f"slice_{idx:03d}.png"
        png_path = os.path.join(args.out_dir, png_name)
        print(f"Converting {fname} → {png_name}...")
        dicom_to_png(dcm_path, png_path)

if __name__ == "__main__":
    main()