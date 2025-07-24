import os
import glob
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from VAE_model import BetaVAE3D  # adjust import if your class/module name differs

def main():
    parser = argparse.ArgumentParser(
        description="Latent histograms & reconstruction accuracy"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory of DVF .nii.gz files"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained VAE checkpoint (.pth)"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=4,
        help="Dimensionality of the latent space"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="cpu or cuda"
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Gather DVF files
    file_list = sorted(glob.glob(os.path.join(args.data_dir, "*.nii.gz")))
    if not file_list:
        raise RuntimeError(f"No .nii.gz files found under {args.data_dir}")

    # Infer input channels and spatial grid from first DVF
    first_img = nib.load(file_list[0])
    first = first_img.get_fdata(dtype=np.float32)
    first = np.squeeze(first)

    # Dynamic shape handling
    if first.ndim == 4:
        # possible shapes: (C, D, H, W) or (D, H, W, C)
        if first.shape[0] in (1, 3):
            in_ch, D, H, W = first.shape
        elif first.shape[-1] in (1, 3):
            D, H, W, in_ch = first.shape
        else:
            raise ValueError(f"Unexpected 4D DVF shape: {first.shape}")
    elif first.ndim == 3:
        # assume channels last with implicit single DVF channel
        in_ch = 1
        D, H, W = first.shape
    else:
        raise ValueError(f"Unexpected DVF array shape: {first.shape}")

    # Build and load the model
    model = BetaVAE3D(
        in_ch=in_ch,
        latent_dim=args.latent_dim,
        grid=(D, H, W)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # Containers for results
    latent_codes    = []
    mses            = []
    epe_means       = []
    energy_retained = []

    # Process each DVF file
    for fpath in file_list:
        img = nib.load(fpath)
        arr = img.get_fdata(dtype=np.float32)
        arr = np.squeeze(arr)
        # Reorder channels to first if needed
        if arr.ndim == 4 and arr.shape[-1] in (1, 3):
            arr = np.moveaxis(arr, -1, 0)

        tensor = torch.from_numpy(arr).unsqueeze(0).to(device)  # shape (1,C,D,H,W)
        with torch.no_grad():
            recon, mu, logvar = model(tensor)

        recon = recon.cpu().numpy().squeeze(0)
        mu    = mu.cpu().numpy().squeeze(0)

        # Compute reconstruction metrics
        err_vec = recon - arr
        mse = np.mean(err_vec ** 2)
        epe_map = np.linalg.norm(err_vec, axis=0)
        mean_epe = np.mean(epe_map)

        # Compute percent energy retained
        # move channel to last for norm calculation
        if recon.ndim == 4:
            recon_ch_last = np.moveaxis(recon, 0, -1)
            arr_ch_last   = np.moveaxis(arr,   0, -1)
        else:
            recon_ch_last = recon
            arr_ch_last   = arr
        E_orig = np.sum(np.linalg.norm(arr_ch_last,   axis=-1) ** 2)
        E_err  = np.sum(np.linalg.norm(recon_ch_last - arr_ch_last, axis=-1) ** 2)
        pct_ret = (1.0 - E_err / E_orig) * 100.0

        latent_codes.append(mu)
        mses.append(mse)
        epe_means.append(mean_epe)
        energy_retained.append(pct_ret)

    latent_codes    = np.stack(latent_codes, axis=0)
    mses            = np.array(mses)
    epe_means       = np.array(epe_means)
    energy_retained = np.array(energy_retained)

    # Plot latent histograms
    dims = latent_codes.shape[1]
    fig, axes = plt.subplots(max(1, dims), 1, figsize=(6, 3 * dims))
    if dims == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.hist(latent_codes[:, i], bins=10, edgecolor='black')
        ax.set_title(f"Latent dim {i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Plot reconstruction accuracy histograms
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    ax[0].hist(mses, bins=10, edgecolor='black')
    ax[0].set_title("Reconstruction MSE across DVFs")
    ax[0].set_xlabel("MSE")
    ax[0].set_ylabel("Frequency")

    ax[1].hist(epe_means, bins=10, edgecolor='black')
    ax[1].set_title("Mean EPE across DVFs")
    ax[1].set_xlabel("Mean EPE")
    ax[1].set_ylabel("Frequency")

    ax[2].hist(energy_retained, bins=10, edgecolor='black')
    ax[2].set_title("Percent Energy Retained")
    ax[2].set_xlabel("% Energy Retained")
    ax[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    # Print numeric summary
    print("Reconstruction Accuracy & Energy Retention Summary:")
    for idx, fname in enumerate(file_list):
        print(f"{os.path.basename(fname)}: "
              f"MSE={mses[idx]:.4e}, "
              f"Mean EPE={epe_means[idx]:.4e}, "
              f"Energy Retained={energy_retained[idx]:.2f}%")

if __name__ == "__main__":
    main()
