#!/usr/bin/env python3
import os
import glob
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def compute_pod_modes(dvf_paths, r):
    # Load all DVFs into a 2D data matrix X: (3*D*H*W, T)
    flattened = []
    for fp in dvf_paths:
        arr = nib.load(fp).get_fdata(dtype=np.float32)
        arr = np.squeeze(arr)              # (D, H, W, 3)
        arr = np.moveaxis(arr, -1, 0)      # (3, D, H, W)
        C, D, H, W = arr.shape
        flattened.append(arr.reshape(C*D*H*W))
    X = np.stack(flattened, axis=1)        # (3*D*H*W, T)

    # Center
    X_mean = X.mean(axis=1, keepdims=True) # (3*D*H*W, 1)
    Xc     = X - X_mean

    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    modes  = U[:, :r]                      # (3*D*H*W, r)
    coeffs = modes.T @ Xc                  # (r, T)

    return X_mean, modes, coeffs

def visualize_reconstructions(dvf_paths, X_mean, modes, coeffs, grid):
    C, D, H, W = 3, *grid
    for i, fp in enumerate(dvf_paths):
        # Original
        arr = nib.load(fp).get_fdata(dtype=np.float32)
        arr = np.squeeze(arr)              # (D, H, W, 3)
        arr = np.moveaxis(arr, -1, 0)      # (3, D, H, W)

        # Reconstruct
        recon_flat = X_mean.flatten() + modes @ coeffs[:, i]
        recon = recon_flat.reshape(3, D, H, W)

        # Amplitudes
        orig_amp  = np.linalg.norm(arr,   axis=0)  # (D, H, W)
        recon_amp = np.linalg.norm(recon, axis=0)

        # Plot middle axial slice
        z = D // 2
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(orig_amp[z], cmap='jet')
        plt.title(f'Orig amp slice {z}')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(recon_amp[z], cmap='jet')
        plt.title(f'Recon amp slice {z}')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.suptitle(os.path.basename(fp))
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder of DVF .nii.gz files")
    parser.add_argument("--r",       type=int, default=4, help="Number of POD modes")
    args = parser.parse_args()

    dvf_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.nii.gz")))
    if not dvf_paths:
        raise FileNotFoundError(f"No .nii.gz files in {args.data_dir}")

    # get grid from first file
    arr0 = nib.load(dvf_paths[0]).get_fdata(dtype=np.float32)
    D, H, W = arr0.shape[:3]

    X_mean, modes, coeffs = compute_pod_modes(dvf_paths, args.r)
    visualize_reconstructions(dvf_paths, X_mean, modes, coeffs, (D, H, W))

if __name__ == "__main__":
    main()

