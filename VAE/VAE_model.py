#!/usr/bin/env python3
# DVF_betaVAE.py

import os
import glob
import argparse
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DVFDataset(Dataset):
    def __init__(self, paths, mean=None, std=None):
        self.paths = paths
        self.mean  = mean
        self.std   = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = nib.load(p)
        arr = img.get_fdata(dtype=np.float32)    # e.g. (X,Y,Z,3) or maybe (X,Y,Z,1,3)

        # remove any singleton dims so we end up with exactly 4 dims and 3 channels last
        arr = np.squeeze(arr)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise RuntimeError(f"Unexpected DVF array shape {arr.shape} loading {p}")

        # now arr.shape == (X, Y, Z, 3)
        # reorder to (C=3, D=Z, H=Y, W=X)
        dvf = torch.from_numpy(arr).permute(3, 2, 1, 0).contiguous()

        # normalize if requested
        if self.mean is not None:
            dvf = (dvf - self.mean[:,None,None,None]) / self.std[:,None,None,None]

        return dvf


class BetaVAE3D(nn.Module):
    def __init__(self, in_ch=3, latent_dim=4, grid=(64,64,64)):
        super().__init__()
        D,H,W = grid
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv3d(32, 64, 4, 2, 1),    nn.ReLU(True),
            nn.Conv3d(64,128, 4, 2, 1),    nn.ReLU(True),
        )
        d2,h2,w2 = D//8, H//8, W//8
        flat = 128 * d2 * h2 * w2
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, flat)
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose3d(64,  32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose3d(32, in_ch,4, 2, 1),          # back to (D,H,W)
        )

    def encode(self, x):
        x = self.enc(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, out_shape):
        x = F.relu(self.fc_dec(z))
        D,H,W = out_shape
        d2,h2,w2 = D//8, H//8, W//8
        x = x.view(-1, 128, d2, h2, w2)
        x = self.dec(x)
        # in case sizes drift, upsample
        if x.shape[-3:] != tuple(out_shape):
            x = F.interpolate(x, size=out_shape, mode='trilinear', align_corners=False)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.shape[-3:])
        return recon, mu, logvar

def compute_dataset_stats(paths):
    sum_ = torch.zeros(3)
    sumsq = torch.zeros(3)
    nvox = 0
    for p in paths:
        arr = nib.load(p).get_fdata(dtype=np.float32)   # (X,Y,Z,3)
        vox = torch.from_numpy(arr).reshape(-1, 3)      # use reshape!
        sum_   += vox.sum(dim=0)
        sumsq  += (vox**2).sum(dim=0)
        nvox   += vox.shape[0]
    mean = sum_ / nvox
    var  = sumsq / nvox - mean**2
    std  = torch.sqrt(var.clamp_min(1e-8))
    return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True, help="root of DVF .nii.gz files")
    p.add_argument("--latent_dim", type=int, default=4)
    p.add_argument("--beta",       type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--device",     default="cuda")
    opt = p.parse_args()

    # scan for DVFs
    all_paths = glob.glob(os.path.join(opt.data_dir, "**", "*.nii.gz"), recursive=True)
    train_p, val_p = train_test_split(all_paths, test_size=0.2, random_state=42)
    mean, std = compute_dataset_stats(train_p)
    print("Loaded", len(train_p), "train and", len(val_p), "val DVFs")
    print("Dataset mean:", mean, "std:", std)

    train_ds = DVFDataset(train_p, mean, std)
    val_ds   = DVFDataset(val_p,   mean, std)
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=opt.batch_size, shuffle=False, num_workers=2)

    # get grid shape from first sample
    sample = train_ds[0]
    grid   = sample.shape[-3:]  # (D,H,W)
    print("Grid shape:", grid)

    model = BetaVAE3D(in_ch=3, latent_dim=opt.latent_dim, grid=grid).to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(1, opt.epochs+1):
        model.train()
        train_loss = 0
        for dvf in train_dl:
            dvf = dvf.to(opt.device)
            recon, mu, logvar = model(dvf)
            rloss = F.mse_loss(recon, dvf, reduction="sum")
            kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss  = rloss + opt.beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for dvf in val_dl:
                dvf = dvf.to(opt.device)
                recon, mu, logvar = model(dvf)
                rloss = F.mse_loss(recon, dvf, reduction="sum")
                kl    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (rloss + opt.beta * kl).item()
        val_loss /= len(val_ds)

        print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    # save final model
    torch.save(model.state_dict(), "dvf_betaVAE.pth")
    print("Model saved to dvf_betaVAE.pth")

if __name__ == "__main__":
    main()
