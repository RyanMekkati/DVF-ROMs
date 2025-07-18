#!/usr/bin/env python3
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ———— CONFIG ————
# Path to one of your real DVFs:
real_path   = "/home/ryan/Documents/GitHub/DVF-Algorithms-2D-3D/DVF-data/4DCT-Dicom_P1/4DCT-Dicom_P1_P00toP10_dvf_DEMONS.nii.gz"
# Path to your reconstructed DVF:
recon_path  = "recon_dvf.nii.gz"

# ———— LOAD DATA ————
# Real
img_r   = nib.load(real_path)
arr_r   = img_r.get_fdata(dtype=np.float32)  # (X,Y,Z,3)
arr_r   = np.squeeze(arr_r)
# Recon
img_c   = nib.load(recon_path)
arr_c   = img_c.get_fdata(dtype=np.float32)
arr_c   = np.squeeze(arr_c)

# ———— SELECT A SLICE ————
Z = arr_r.shape[2]
slice_idx = Z // 2
slice_r = arr_r[:, :, slice_idx, :]  # (X,Y,3)
slice_c = arr_c[:, :, slice_idx, :]  # (X,Y,3)

# ———— MAKE A GRID ————
X, Y = slice_r.shape[:2]
step = max(X // 64, 4)            # adjust density
x = np.arange(0, X, step)
y = np.arange(0, Y, step)
Xg, Yg = np.meshgrid(x, y, indexing="ij")

# Vector components (just XY plane vectors)
U_r = slice_r[x][:, y, 0]
V_r = slice_r[x][:, y, 1]
U_c = slice_c[x][:, y, 0]
V_c = slice_c[x][:, y, 1]

# scale amplitudes by 3 for better visibility
U_r *= 3
V_r *= 3
U_c *= 10
V_c *= 10

# ———— PLOT ————
fig, ax = plt.subplots(figsize=(6,6))
# background = real DVF magnitude
mag = np.linalg.norm(slice_r, axis=-1).T  # transpose so origin='lower' is correct
ax.imshow(mag, cmap="gray", origin="lower", alpha=0.6)

# original in blue
ax.quiver(
    Xg, Yg,
    U_r.T, V_r.T,
    color="blue", angles="xy", scale_units="xy", scale=1,
    width=0.002, label="Original"
)
# reconstructed in red
ax.quiver(
    Xg, Yg,
    U_c.T, V_c.T,
    color="red",  angles="xy", scale_units="xy", scale=1,
    width=0.002, label="Reconstructed"
)

ax.set_title(f"DVF Vectors Comparison (slice {slice_idx})")
ax.axis("off")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
