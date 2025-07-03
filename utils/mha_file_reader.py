import SimpleITK as sitk
import numpy as np
import glob, os
import matplotlib.pyplot as plt

# 1) Read in all your .mha files:
mha_folder = "/Users/ryanmekkati/Library/Mobile Documents/com~apple~CloudDocs/Projet PhD/4DCT/P113"
files = sorted(glob.glob(os.path.join(mha_folder, "*.mha")))
volumes = [sitk.GetArrayFromImage(sitk.ReadImage(f)) for f in files]
data = np.stack(volumes, axis=0)
print("Loaded data shape:", data.shape)  # (T, Z, Y, X)

# 2) Prepare output folder for PNGs
os.makedirs("viz", exist_ok=True)

# 3) Loop over time points
T, Z, H, W = data.shape
mid_slice = Z // 2

for t in range(T):
    frame = data[t, mid_slice, :, :]    # pick central slice
    # show
    plt.imshow(frame, cmap="gray")
    plt.title(f"Phase {t} (slice {mid_slice})")
    plt.axis("off")
    plt.pause(0.1)                      # pause so you can see it
    # save
    outname = f"viz/phase_{t:03d}.png"
    plt.imsave(outname, frame, cmap="gray")

plt.show()