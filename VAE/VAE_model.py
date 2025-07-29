#!/usr/bin/env python3
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
        arr = nib.load(self.paths[idx]).get_fdata(dtype=np.float32)
        arr = np.squeeze(arr)  # (D,H,W,3)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise RuntimeError(f"Unexpected DVF shape {arr.shape}")
        dvf = torch.from_numpy(arr).permute(3,2,1,0).contiguous()  # (C,D,H,W)
        if self.mean is not None:
            dvf = (dvf - self.mean[:,None,None,None]) / self.std[:,None,None,None]
        return dvf

class BetaVAE3D(nn.Module):
    def __init__(self, in_ch=3, latent_dim=4, grid=(32,64,64)):
        super().__init__()
        D,H,W = grid

        # --- Encoder: conv1 → conv2 → conv3 → conv4
        self.conv1 = nn.Conv3d(in_ch,   32, 4, 2, 1); self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32,     64, 4, 2, 1); self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64,    128, 4, 2, 1); self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128,   256, 4, 2, 1); self.bn4 = nn.BatchNorm3d(256)

        # compute flattened size at 1/16 resolution
        d4,h4,w4 = D//16, H//16, W//16
        flat     = 256 * d4 * h4 * w4
        self.fc_mu     = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, flat)

        # --- Decoder w/ 4‑level U‑Net skips
        self.deconv4 = nn.ConvTranspose3d(256*2, 128, 4, 2, 1); self.bn5 = nn.BatchNorm3d(128)
        self.deconv3 = nn.ConvTranspose3d(128*2,  64, 4, 2, 1); self.bn6 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64*2,   32, 4, 2, 1); self.bn7 = nn.BatchNorm3d(32)
        self.deconv1 = nn.ConvTranspose3d(32*2, in_ch,4, 2, 1)

    def encode(self, x):
        e1 = F.leaky_relu(self.bn1(self.conv1(x)),  0.1)
        e2 = F.leaky_relu(self.bn2(self.conv2(e1)), 0.1)
        e3 = F.leaky_relu(self.bn3(self.conv3(e2)), 0.1)
        e4 = F.leaky_relu(self.bn4(self.conv4(e3)), 0.1)
        flat = e4.view(e4.size(0), -1)
        mu, logvar = self.fc_mu(flat), self.fc_logvar(flat)
        return mu, logvar, (e1,e2,e3,e4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, skips):
        e1,e2,e3,e4 = skips
        batch = z.size(0)
        # FC → reshape to (B,256,d4,h4,w4)
        x = F.leaky_relu(self.fc_dec(z), 0.1).view(batch, 256, *e4.shape[-3:])
        # up 1: 1/16→1/8
        d4 = torch.cat([x,e4], dim=1)
        d4 = F.leaky_relu(self.bn5(self.deconv4(d4)), 0.1)
        # up 2: 1/8→1/4
        d3 = torch.cat([d4,e3], dim=1)
        d3 = F.leaky_relu(self.bn6(self.deconv3(d3)), 0.1)
        # up 3: 1/4→1/2
        d2 = torch.cat([d3,e2], dim=1)
        d2 = F.leaky_relu(self.bn7(self.deconv2(d2)), 0.1)
        # up 4: 1/2→full
        d1 = torch.cat([d2,e1], dim=1)
        return self.deconv1(d1)

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar

def compute_dataset_stats(paths):
    sum_, sumsq = torch.zeros(3), torch.zeros(3)
    nvox = 0
    for p in paths:
        arr = nib.load(p).get_fdata(dtype=np.float32)
        arr = np.squeeze(arr)              # (D,H,W,3)
        vox = torch.from_numpy(arr).reshape(-1,3)
        sum_  += vox.sum(dim=0)
        sumsq += (vox**2).sum(dim=0)
        nvox  += vox.shape[0]
    mean = sum_ / nvox
    var  = sumsq / nvox - mean**2
    std  = torch.sqrt(var.clamp_min(1e-8))
    return mean, std

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True, help="path to folder of .nii.gz DVFs")
    p.add_argument("--latent_dim", type=int, default=4)
    p.add_argument("--beta",       type=float, default=1.0)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--device",     default="cuda")
    opt = p.parse_args()

    # load DVF paths
    all_paths = sorted(glob.glob(os.path.join(opt.data_dir, "*.nii.gz")))
    if not all_paths:
        raise FileNotFoundError(f"No .nii.gz files found in {opt.data_dir}")
    train_p, val_p = train_test_split(all_paths, test_size=0.2, random_state=42)
    mean, std = compute_dataset_stats(train_p)
    print(f"Loaded {len(train_p)} train / {len(val_p)} val DVFs")
    print("Dataset mean:", mean, "std:", std)

    # data loaders
    train_ds = DVFDataset(train_p, mean, std)
    val_ds   = DVFDataset(val_p,   mean, std)
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=opt.batch_size, shuffle=False, num_workers=2)

    # build & train
    grid  = train_ds[0].shape[-3:]
    model = BetaVAE3D(in_ch=3, latent_dim=opt.latent_dim, grid=grid).to(opt.device)
    optm  = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for ep in range(1, opt.epochs+1):
        model.train()
        tloss = 0.0
        for dvf in train_dl:
            dvf = dvf.to(opt.device)
            recon, mu, logvar = model(dvf)
            rloss = F.mse_loss(recon, dvf, reduction="mean")
            if opt.beta > 0.0:
                kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rloss + opt.beta * kl
            else:
                loss = rloss
            optm.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optm.step()
            tloss += loss.item() * dvf.size(0)
        tloss /= len(train_p)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for dvf in val_dl:
                dvf = dvf.to(opt.device)
                recon, mu, logvar = model(dvf)
                rloss = F.mse_loss(recon, dvf, reduction="mean")
                if opt.beta > 0.0:
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    vloss += (rloss + opt.beta*kl).item() * dvf.size(0)
                else:
                    vloss += rloss.item() * dvf.size(0)
        vloss /= len(val_p)

        print(f"Epoch {ep:3d} | Train {tloss:.6f} | Val {vloss:.6f}")

    torch.save(model.state_dict(), "dvf_unet_betaVAE.pth")
    print("Saved dvf_unet_betaVAE.pth")

if __name__ == "__main__":
    main()

