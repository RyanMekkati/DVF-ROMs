#!/usr/bin/env python
"""
inspect_mha.py

A combined tool to:
 1) Inspect .mha/.mhd files for metadata and basic stats.
 2) Optionally visualize an axial slice and overlay a physical-space ROI box.

Usage:
  # Just inspect metadata:
  python inspect_mha.py /path/to/volume.mha

  # Inspect and show an ROI on slice 42:
  python inspect_mha.py /path/to/volume.mha --cx -100 --cy -100 --cz -120 \
      --dx 64 --dy 64 --dz 32 --slice 42
"""
import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def inspect_mha(file_path):
    """Print metadata and basic intensity stats for the given MHA/MHD file."""
    img = sitk.ReadImage(file_path)
    size = img.GetSize()            # (x, y, z)
    spacing = img.GetSpacing()      # (sx, sy, sz)
    origin = img.GetOrigin()        # (ox, oy, oz)
    direction = img.GetDirection()  # flattened 3x3
    arr = sitk.GetArrayFromImage(img)
    print(f"File: {os.path.basename(file_path)}")
    print(f"  Size (x, y, z):        {size}")
    print(f"  Spacing (x, y, z):     {spacing}")
    print(f"  Origin (x, y, z):      {origin}")
    print(f"  Direction cosines:     {direction}")
    print(f"  Array shape (z, y, x): {arr.shape}")
    print(f"  Intensity min/max:     {arr.min()} / {arr.max()}")
    print("-" * 40)

def plot_roi_on_slice(file_path, phys_center, phys_size, slice_idx=None):
    """
    Display an axial slice with a red rectangle marking the given physical-space ROI.
    phys_center: (cx, cy, cz) in mm
    phys_size:   (dx, dy, dz) in mm
    slice_idx:   slice index (0-based) or None for middle
    """
    img = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(img)  # shape: (z, y, x)

    # Determine slice index
    if slice_idx is None:
        slice_idx = arr.shape[0] // 2
    slice_img = arr[slice_idx]

    # Compute ROI in voxel indices
    origin  = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    phys_center = np.array(phys_center)
    phys_size   = np.array(phys_size)
    phys_corner = phys_center - phys_size/2
    start_idx = np.floor((phys_corner - origin) / spacing).astype(int)
    size_vox  = np.ceil(phys_size / spacing).astype(int)

    # Plot slice and rectangle
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.imshow(slice_img, cmap='gray', origin='lower')
    rect = patches.Rectangle(
        (start_idx[0], start_idx[1]),
        size_vox[0], size_vox[1],
        linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Slice {slice_idx}: ROI idx {start_idx.tolist()} size {size_vox.tolist()}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and optionally plot ROI on .mha/.mhd volumes.")
    parser.add_argument('file', help='Path to .mha or .mhd file')
    parser.add_argument('--cx', type=float, help='ROI center x (mm)')
    parser.add_argument('--cy', type=float, help='ROI center y (mm)')
    parser.add_argument('--cz', type=float, help='ROI center z (mm)')
    parser.add_argument('--dx', type=float, help='ROI size dx (mm)')
    parser.add_argument('--dy', type=float, help='ROI size dy (mm)')
    parser.add_argument('--dz', type=float, help='ROI size dz (mm)')
    parser.add_argument('--slice', type=int, help='Axial slice index to display')
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"File not found: {args.file}")
        return

    # Always print metadata
    inspect_mha(args.file)

    # If all ROI args provided, plot ROI
    roi_args = [args.cx, args.cy, args.cz, args.dx, args.dy, args.dz]
    if all(v is not None for v in roi_args):
        plot_roi_on_slice(
            args.file,
            phys_center=(args.cx, args.cy, args.cz),
            phys_size=(args.dx,  args.dy,  args.dz),
            slice_idx=args.slice)

if __name__ == '__main__':
    main()

