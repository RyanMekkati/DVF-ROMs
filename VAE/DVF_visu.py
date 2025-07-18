#!/usr/bin/env python3
import glob, os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ————— CONFIG ——————
# Folder where your real DVFs live:
real_dir   = "/home/ryan/Documents/GitHub/DVF-Algorithms-2D-3D/DVF-data/4DCT-Dicom_P1"
# Pick the first real DVF there:
real_paths = glob.glob(os.path.join(real_dir, "*.nii.gz"))
if not real_paths:
    raise FileNotFoundError(f"No .nii.gz found in {real_dir}")
real_path  = real_paths[0]

# Your VAE‑generated DVF:
sample_path = "sample_dvf_00.nii.gz"
if not os.path.exists(sample_path):
    raise FileNotFoundError(f"Sample file not found: {sample_path}")

# ————— LOADING ——————
def load_dvf(path):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)  # (X,Y,Z,3)
    arr = np.squeeze(arr)
    assert arr.ndim == 4 and arr.shape[-1] == 3, f"Bad shape {arr.shape}"
    return arr

real_arr = load_dvf(real_path)
samp_arr = load_dvf(sample_path)

# ————— SELECT SLICE ——————
# take the middle Z slice
Z = real_arr.shape[2]
slice_idx = Z // 2
real_sl = real_arr[:, :, slice_idx, :]   # (X,Y,3)
samp_sl = samp_arr[:, :, slice_idx, :]   # (X,Y,3)

# ————— BUILD GRID & SUBSAMPLE ——————
X, Y = real_sl.shape[:2]
step = max(X // 64, 4)  # ~64 arrows per row
x = np.arange(0, X, step)
y = np.arange(0, Y, step)
Xg, Yg = np.meshgrid(x, y, indexing="ij")

Ur = real_sl[x][:, y, 0]
Vr = real_sl[x][:, y, 1]
Us = samp_sl[x][:, y, 0]
Vs = samp_sl[x][:, y, 1]

# ————— PLOT ——————
fig, ax = plt.subplots(figsize=(6,6))
# background = magnitude of real DVF
mag = np.linalg.norm(real_sl, axis=-1).T
ax.imshow(mag, cmap="gray", origin="lower", alpha=0.6)
# real in blue, sample in red
ax.quiver(Xg, Yg, Ur.T, Vr.T, color="blue", scale=10, label="Real")
ax.quiver(Xg, Yg, Us.T, Vs.T, color="red",  scale=10, label="Sample")
ax.set_title(f"Real vs Sample DVF (slice {slice_idx})")
ax.axis("off")
ax.legend(loc="upper right")
plt.show()
