import nibabel as nib
import numpy as np
import os
import sys
from nilearn.masking import apply_mask
from sklearn.preprocessing import StandardScaler             # quick z-scoring
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from nilearn.masking import apply_mask
from pathlib import Path


pre_splora_path = '/data/csasi/MindEyeV2/src/data/betas/movie_betas/lostintranslation_preprocessed_NSDgeneral.nii.gz'
post_splora_path = '/home/csasi/decoding/dmt_002/cat_betas_lost_in_translation.nii.gz'

mask_path = '/data/csasi/MindEyeV2/data/processed_outputs/binary_mask.nii.gz'


def load_and_mask(nifti_path, mask_img):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    if data.ndim == 3:                         # ← only one frame
        raise RuntimeError(
            f"{Path(nifti_path).name} is 3-D; expected 4-D. "
            "Stack your β windows with fslmerge/3dTcat first."
        )
    return apply_mask(img, mask_img)           # (T, Nvox)

mask_img = nib.load(mask_path)
raw_ts  = load_and_mask(pre_splora_path , mask_img)
beta_ts = load_and_mask(post_splora_path, mask_img)   # will raise if 3-D


# -----------------------------------------------------------
# 0.  Paths
# -----------------------------------------------------------
mask_file  = mask_path
raw_file   = pre_splora_path
beta_file  = post_splora_path


mask_img  = nib.load(mask_file)           # binary mask
raw_img   = nib.load(raw_file)            # 4-D BOLD  (X,Y,Z,Traw)
beta_img  = nib.load(beta_file)           # 4-D β      (X,Y,Z,Tbeta)

raw_ts   = apply_mask(raw_img , mask_img)   # →  (Traw , Nvox)
beta_ts  = apply_mask(beta_img, mask_img)   # →  (Tbeta, Nvox)

# (optional) z-score each time-point across voxels
scaler   = StandardScaler(with_mean=True, with_std=True)
raw_ts   = scaler.fit_transform(raw_ts)     # shape preserved
beta_ts  = scaler.fit_transform(beta_ts)

# -----------------------------------------------------------
# 2.  Equalise number of frames so that RSMs are comparable
# -----------------------------------------------------------
n_frames = min(raw_ts.shape[0], beta_ts.shape[0])
raw_ts  = raw_ts [:n_frames]
beta_ts = beta_ts[:n_frames]

# -----------------------------------------------------------
# 3.  Representational-similarity matrices (frame × frame)
# -----------------------------------------------------------
rsm_raw  = np.corrcoef(raw_ts)             # (n_frames, n_frames)
rsm_beta = np.corrcoef(beta_ts)

# -----------------------------------------------------------
# 4.  Quick visual side-by-side
# -----------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
vmax = max(abs(rsm_raw).max(), abs(rsm_beta).max())

im0 = ax[0].imshow(rsm_raw , vmin=-vmax, vmax=vmax, cmap='viridis')
ax[0].set_title("Raw BOLD  RSM");  ax[0].set_xlabel("frame"); ax[0].set_ylabel("frame")

im1 = ax[1].imshow(rsm_beta, vmin=-vmax, vmax=vmax, cmap='viridis')
ax[1].set_title("SPLORA β  RSM"); ax[1].set_xlabel("frame"); ax[1].set_ylabel("frame")

fig.colorbar(im1, ax=ax.ravel().tolist(), shrink=.7); plt.show()

# -----------------------------------------------------------
# 5.  “Viable signal” check (window‐wise standard deviation)
# -----------------------------------------------------------
beta_std = beta_ts.std(axis=1)             # σ for each frame/window
from statsmodels.robust import mad
cut = np.median(beta_std) + 2 * mad(beta_std)

pct = (beta_std > cut).mean()*100
print(f"{pct:.1f}% of β time-points exceed (median+2·MAD) signal threshold")

plt.figure(figsize=(5,3))
plt.hist(beta_std, bins=60); plt.axvline(cut, color='red'); 
plt.xlabel("β frame σ"); plt.ylabel("#frames"); plt.title("SPLORA β variance distribution")
plt.show()


