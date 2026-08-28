# Script to compare the calculation of a RECIST longest diameter from a 3D tumour segmentation mask.
# Compared method from MedSAM2 implementation and BHKLab implementation by Kaitlyn Kobayashi. 

import cv2
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from damply import dirs
from readii.image_processing import displayCTSegOverlay

from ab_testing.utils.annotations import get_line_from_recist, get_recist_pts
from ab_testing.utils.conver_nifti_to_recist_npz import (
                 compute_recist_line,
                 generate_recist,
                 read_case,
)

cool_cmap = mpl.colormaps['cool']
binary_magenta = mcolors.ListedColormap([(0,0,0,0.1), cool_cmap(400)])

sample_scan = "LUNG1-012_0011/CT_55636694"
sample_mask = "LUNG1-012_0011/RTSTRUCT_38424583"

sample_id = sample_scan.split("/", maxsplit=1)[0]

scan_path = dirs.RAWDATA / "TCIA_NSCLC-Radiomics/images/mit_NSCLC-Radiomics" / f"{sample_scan}" / "CT.nii.gz"
mask_path = dirs.RAWDATA / "TCIA_NSCLC-Radiomics/images/mit_NSCLC-Radiomics" / f"{sample_mask}" / "GTV.nii.gz"

scan, mask, spacing, direction, origin, reader = read_case(scan_path, mask_path)

key_slice = int(np.argmax(np.sum(mask, axis=(1, 2))))

r_fig, r_ax = plt.subplots(1,3, figsize=(21,8.5), layout='tight')
plt.suptitle(f'RECIST line comparison for NSCLC-Radiomics {sample_id}')

# MedSAM2 method
p1, p2 = compute_recist_line(mask[key_slice])
m_recist_line, short_ids = generate_recist(mask, spacing)

displayCTSegOverlay(scan, mask, sliceIdx=key_slice, dispMin=-1350, dispMax=150, ax=r_ax[0], alpha=0.3)
r_ax[0].imshow(m_recist_line[key_slice], cmap=binary_magenta, alpha=0.7)
r_ax[0].set_title(f"MedSAM2 \n Coordinates: p1={p1}, p2={p2}")

# Kaitlyn's method
recist_pts = get_recist_pts(mask[key_slice])
k_recist_line = get_line_from_recist(recist_pts, key_slice, mask.shape)
f_k_pts = f"Coordinates: p1=[{recist_pts[2]:.2f} {recist_pts[3]:.2f}], p2=[{recist_pts[0]:.2f} {recist_pts[1]:.2f}]"

displayCTSegOverlay(scan, mask, sliceIdx=key_slice, dispMin=-1350, dispMax=150, ax=r_ax[1], alpha=0.3)
r_ax[1].imshow(k_recist_line[key_slice], cmap=binary_magenta, alpha=0.7)
r_ax[1].set_title(f"BHKLab \n {f_k_pts}")

# Draw line like MedSAM method with Kaitlyn's coordinates
k_thick_line = np.zeros_like(mask, dtype=np.uint16)
cv2.line(k_thick_line[key_slice], (int(recist_pts[0]), int(recist_pts[1])), (int(recist_pts[2]), int(recist_pts[3])),
                 color=1, thickness=2)
f_kt_pts = f"Coordinates: p1=[{int(recist_pts[2])} {int(recist_pts[3])}], p2=[{int(recist_pts[0])} {int(recist_pts[1])}]"

displayCTSegOverlay(scan, mask, sliceIdx=key_slice, dispMin=-1350, dispMax=150, ax=r_ax[2], alpha=0.3)
r_ax[2].imshow(k_thick_line[key_slice], cmap=binary_magenta, alpha=0.7)
r_ax[2].set_title(f"BHKLab drawn like MedSAM2 \n {f_kt_pts}")

r_fig.savefig(dirs.RESULTS / f"recist_comparison_{sample_id}.png")