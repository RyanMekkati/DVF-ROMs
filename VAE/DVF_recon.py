#!/usr/bin/env python3
import os
import glob
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from VAE_model import BetaVAE3D, compute_dataset_stats

def main():
    parser = argparse.ArgumentParser("VAE DVF Amplitude Vis")
    parser.add_argument("--data_dir",   required=True, help="Folder of DVF .nii or .nii.gz")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE .pth checkpoint")
    parser.add_argument("--latent_dim", type=int, default=4, help="Size of latent z")
    parser.add_argument("--device",     default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    # Device
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Gather files
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii*")))
    if not files:
        raise FileNotFoundError(f"No DVFs found in {args.data_dir}")

    # Compute normalization stats on train split
    from sklearn.model_selection import train_test_split
    train_fs, _ = train_test_split(files, test_size=0.2, random_state=42)
    mean, std = compute_dataset_stats(train_fs)
    if torch.is_tensor(mean): mean = mean.cpu().numpy()
    if torch.is_tensor(std):  std  = std.cpu().numpy()
    mean_t = torch.from_numpy(mean).view(1,-1,1,1,1).to(dev)
    std_t  = torch.from_numpy(std).view(1,-1,1,1,1).to(dev)

    # Load model
    # infer C,D,H,W
    arr0 = nib.load(files[0]).get_fdata(dtype=np.float32)
    if arr0.ndim==5 and arr0.shape[3]==1: arr0 = np.squeeze(arr0,3)
    if arr0.ndim==4 and arr0.shape[-1] in (1,3):
        D,H,W,C = arr0.shape
    else:
        D,H,W,C = arr0.shape[1], arr0.shape[2], arr0.shape[3], arr0.shape[0]
    model = BetaVAE3D(in_ch=C, latent_dim=args.latent_dim, grid=(D,H,W)).to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev)
    model.load_state_dict(ckpt)
    # deterministic
    model.reparameterize = lambda mu, logvar: mu
    model.eval()

    # Loop & visualize
    for fp in files:
        name = os.path.basename(fp)
        arr = nib.load(fp).get_fdata(dtype=np.float32)
        # squeeze singleton
        if arr.ndim==5 and arr.shape[3]==1: arr = np.squeeze(arr,3)
        # to (C,D,H,W)
        if arr.ndim==4 and arr.shape[-1] in (1,3):
            arr = np.moveaxis(arr, -1, 0)
        arr_raw = arr.copy()

        # normalize + forward
        x = torch.from_numpy(arr).unsqueeze(0).to(dev)
        dvf_norm = (x - mean_t) / std_t
        with torch.no_grad():
            recon_norm, mu, logvar = model(dvf_norm)
        recon_raw = (recon_norm * std_t + mean_t).squeeze(0).cpu().numpy()

        # compute amplitude maps
        orig_amp  = np.linalg.norm(arr_raw,  axis=0)  # (D,H,W)
        recon_amp = np.linalg.norm(recon_raw, axis=0)

        # mid‐slice
        z = D//2
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(orig_amp[z], cmap='jet')
        plt.title(f"{name}\nOriginal amp (slice {z})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.imshow(recon_amp[z], cmap='jet')
        plt.title(f"Reconstructed amp (slice {z})")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

