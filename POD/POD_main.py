import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt


# 1) CONFIGURATION
data_dir = '/home/ryan/Documents/GitHub/DVF-ROMs/DVF/DVF-data/4DCT-Dicom_P1'  # adjust if needed
pattern  = '4DCT-Dicom_P1_*_dvf_DEMONS.nii.gz'
out_dir  = '/home/ryan/Documents/GitHub/DVF-ROMs/DVF/DVF-data/POD'  # where to write mode_01.nii.gz, etc.

# 2) COLLECT FILES
import glob
dvf_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
T = len(dvf_paths)
if T < 2:
    raise RuntimeError(f"Found only {T} DVFs – need at least 2")

# 3) READ FIRST TO GET SPATIAL METADATA & SHAPE
first_img = sitk.ReadImage(dvf_paths[0])
affine_spc = first_img.GetSpacing(), first_img.GetOrigin(), first_img.GetDirection()
arr0       = sitk.GetArrayFromImage(first_img)          # shape: (z, y, x, 3)
# reorder to (x, y, z, 3) for flattening
dvf0       = np.transpose(arr0, (2, 1, 0, 3))
Nx, Ny, Nz, Nc = dvf0.shape
assert Nc == 3, "Expect 3‑component vector field"

# 4) LOAD ALL DVFs INTO A NUMPY ARRAY
DVFs = np.zeros((T, Nx, Ny, Nz, 3), dtype=np.float32)
for i, fp in enumerate(dvf_paths):
    img = sitk.ReadImage(fp)
    arr = sitk.GetArrayFromImage(img)
    DVFs[i] = np.transpose(arr, (2, 1, 0, 3))

# 5) FLATTEN & CENTER
X      = DVFs.reshape(T, -1)      # shape: (T, 3*Nx*Ny*Nz)
X_mean = X.mean(axis=0)
Xc     = X - X_mean

# 6) SVD → POD
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
explained_var = (S**2) / (S**2).sum()
cum_var       = np.cumsum(explained_var)

# pick r to explain ≥99% variance
r = int(np.searchsorted(cum_var, 0.99) + 1)
print(f"Keeping r = {r} modes (cumulative variance = {cum_var[r-1]:.4f})")

modes  = Vt[:r, :]                 # (r, 3*Nx*Ny*Nz)
coeffs = U[:, :r] * S[:r]          # (T, r)

# Updated Metrics Block with Percentage Error

# 7) METRICS: per‑DVF reconstruction error (including percent)
records = []
for i, fp in enumerate(dvf_paths):
    orig      = X[i]
    recon     = X_mean + coeffs[i].dot(modes)
    diff      = orig - recon
    mse       = np.mean(diff**2)
    rmse      = np.sqrt(mse)
    rel       = rmse / (np.sqrt(np.mean(orig**2)) + 1e-12)
    error_pct = rel * 100.0
    rec = {
        'DVF': os.path.basename(fp),
        'rmse': rmse,
        'error_pct': error_pct
    }
    for k in range(r):
        rec[f'coef_mode{k+1}'] = coeffs[i, k]
    records.append(rec)

df = pd.DataFrame(records)

# Print just DVF name, RMSE (in original units), and percent error
print(df[['DVF','rmse','error_pct']])


# 8) SAVE EACH MODE AS A 3D VECTOR NIfTI
spacing, origin, direction = affine_spc
for k in range(r):
    mode_flat = modes[k]
    mode_arr  = mode_flat.reshape((Nx, Ny, Nz, 3))
    # back to (z,y,x,3) for SimpleITK
    sitk_arr  = np.transpose(mode_arr, (2, 1, 0, 3))
    img_mode  = sitk.GetImageFromArray(sitk_arr, isVector=True)
    img_mode.SetSpacing(spacing)
    img_mode.SetOrigin(origin)
    img_mode.SetDirection(direction)
    out_fp = os.path.join(out_dir, f'mode_{k+1:02d}.nii.gz')
    sitk.WriteImage(img_mode, out_fp)
    print(f"Wrote mode {k+1} → {out_fp}")
