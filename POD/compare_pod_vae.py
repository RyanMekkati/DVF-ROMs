#!/usr/bin/env python3
import os, glob, argparse
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAE_DIR    = os.path.join(SCRIPT_DIR, os.pardir, "VAE")
sys.path.insert(0, VAE_DIR)
from VAE_model import BetaVAE3D, compute_dataset_stats

def compute_pod(dvf_paths, r):
    # flatten and stack
    mats = []
    for fp in dvf_paths:
        arr = nib.load(fp).get_fdata(dtype=np.float32)
        arr = np.squeeze(arr)              # (D,H,W,3)
        arr = np.moveaxis(arr, -1, 0)      # (3,D,H,W)
        mats.append(arr.reshape(-1))
    X = np.stack(mats, axis=1)             # (3*D*H*W, T)
    mean = X.mean(axis=1, keepdims=True)
    Xc   = X - mean
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    modes  = U[:, :r]                      # (3*D*H*W, r)
    coeffs = modes.T @ Xc                  # (r, T)
    return mean, modes, coeffs

def pod_reconstruct(mean, modes, coeffs, idx, shape):
    flat = (mean.flatten() + modes @ coeffs[:, idx])
    return flat.reshape(shape)

def vae_reconstruct(fp, model, mean_t, std_t, dev):
    arr = nib.load(fp).get_fdata(dtype=np.float32)
    if arr.ndim==5 and arr.shape[3]==1: arr = np.squeeze(arr,3)
    if arr.ndim==4 and arr.shape[-1] in (1,3):
        arr = np.moveaxis(arr, -1, 0)
    arr_ch = arr.copy()  # (C,D,H,W)
    x = torch.from_numpy(arr).unsqueeze(0).to(dev)
    dvf_norm = (x - mean_t)/std_t
    with torch.no_grad():
        recon_norm, mu, _ = model(dvf_norm)
    recon = (recon_norm*std_t + mean_t).squeeze(0).cpu().numpy()
    return arr_ch, recon, mu.cpu().numpy()

def amp(vol):
    return np.linalg.norm(vol, axis=0)

def rmse_pct(orig, recon):
    diff = recon-orig
    rmse = np.sqrt(np.mean(diff**2))
    rms0 = np.sqrt(np.mean(orig**2))
    return rmse, rmse/(rms0+1e-12)*100

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pod_modes", type=int, default=4)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--latent_dim", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # gather DVFs
    fps = sorted(glob.glob(os.path.join(args.data_dir, "*.nii*")))
    if not fps: raise FileNotFoundError(f"No DVF in {args.data_dir}")
    T = len(fps)

    # POD on the *entire* sequence (we’ll reconstruct all of them)
    train = fps
    mean_pod, modes, coeffs_pod = compute_pod(train, args.pod_modes)

    # VAE setup
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mean, std = compute_dataset_stats(train)
    if torch.is_tensor(mean): mean=mean.cpu().numpy()
    if torch.is_tensor(std):  std=std.cpu().numpy()
    mean_t = torch.from_numpy(mean).view(1,-1,1,1,1).to(dev)
    std_t  = torch.from_numpy(std).view(1,-1,1,1,1).to(dev)

    # load model
    # infer grid from first DVF
    arr0 = nib.load(fps[0]).get_fdata(dtype=np.float32); 
    if arr0.ndim==5 and arr0.shape[3]==1: arr0=np.squeeze(arr0,3)
    D,H,W = arr0.shape[:3]; C = arr0.shape[3] if arr0.ndim==4 else arr0.shape[0]
    model = BetaVAE3D(in_ch=C, latent_dim=args.latent_dim, grid=(D,H,W)).to(dev)
    ckpt = torch.load(args.vae_ckpt, map_location=dev)
    model.load_state_dict(ckpt)
    model.reparameterize = lambda mu, logvar: mu
    model.eval()

    # containers
    rmses_pod, pct_pod = [], []
    rmses_vae, pct_vae = [], []
    coeffs_vae = []

    # compare each
    shape = (3,D,H,W)
    for i, fp in enumerate(fps):
        # POD
        recon_pod = pod_reconstruct(mean_pod, modes, coeffs_pod, i, (3*D*H*W,))
        recon_pod = recon_pod.reshape(shape)
        orig, recon_vae, mu = vae_reconstruct(fp, model, mean_t, std_t, dev)

        # amplitudes
        amp_o  = amp(orig)
        amp_p  = amp(recon_pod)
        amp_v  = amp(recon_vae)

        # metrics
        r_p, e_p = rmse_pct(amp_o, amp_p)
        r_v, e_v = rmse_pct(amp_o, amp_v)
        rmses_pod.append(r_p); pct_pod.append(e_p)
        rmses_vae.append(r_v); pct_vae.append(e_v)
        coeffs_vae.append(mu)

        # plot midslice
        z = D//2
        plt.figure(figsize=(12,4))
        for j,(im,title) in enumerate(zip(
            [amp_o[z], amp_p[z], amp_v[z]],
            ["Orig amp", f"POD amp (r={args.pod_modes})", "VAE amp"]
        )):
            ax=plt.subplot(1,3,j+1)
            plt.imshow(im, cmap='jet'); plt.axis('off'); plt.title(title)
        plt.suptitle(os.path.basename(fp))
        plt.tight_layout(); plt.show()

    # numeric summary
    print("=== RMSE & %Error (amp) ===")
    print("Frame | POD RMSE | POD %  | VAE RMSE | VAE %")
    for i in range(T):
        print(f"{i:3d}  | {rmses_pod[i]:7.4f} | {pct_pod[i]:6.2f}% | {rmses_vae[i]:7.4f} | {pct_vae[i]:6.2f}%")

    # plot coefficient trajectories
    coeffs_vae = np.vstack(coeffs_vae).T  # (latent_dim, T)
    plt.figure()
    for j in range(args.pod_modes):
        plt.plot(coeffs_pod[j],  '-o', label=f"POD mode {j+1}")
    for j in range(args.latent_dim):
        plt.plot(coeffs_vae[j], '--x', label=f"VAE z{j+1}")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.xlabel("Frame index"); plt.ylabel("Coeff value")
    plt.title("POD vs VAE coefficient trajectories")
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
