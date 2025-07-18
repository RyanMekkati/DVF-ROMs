import torch
import nibabel as nib
import numpy as np
from DVF_betaVAE import BetaVAE3D

# 1) Config
device     = "cuda"
latent_dim = 8
grid       = (64,64,32)                   # D, H, W
checkpoint = "dvf_betaVAE.pth"

# 2) Stats from training (exact values you saw)
mean = torch.tensor([ 0.0433,  0.0082, -0.0082], device=device).view(1,3,1,1,1)
std  = torch.tensor([ 0.4322,  0.3908,  0.4063], device=device).view(1,3,1,1,1)

# 3) Load VAE
model = BetaVAE3D(in_ch=3, latent_dim=latent_dim, grid=grid).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

# 4) Sample, denormalize, save
for i in range(5):
    z = torch.randn(1, latent_dim, device=device)
    # decode → normalized field (shape [1,3,D,H,W])
    dvf_norm = model.decode(z, out_shape=grid)
    # undo z‑score normalization → physical units (mm)
    dvf = dvf_norm * std + mean
    arr = dvf.squeeze(0).cpu().permute(1,2,3,0).numpy()  # (D,H,W,3)
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    out = f"sample_dvf_denorm_{i:02d}.nii.gz"
    nib.save(img, out)
    print("Saved", out)
