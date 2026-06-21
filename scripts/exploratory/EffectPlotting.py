#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation + colored overlays with adjustable dark edges
- Anatomical background visible in grayscale
- Black background
- Each overlay color keeps its hue
- Edges of each overlay are slightly darker
- Edge thickness adjustable
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
data_path = str(Path(__file__).resolve().parent.parent.parent / 'data') + '/'

anat_path = data_path + 'NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz'
cortical_path = data_path + 'cortical_colored_effect.nii.gz'
subcortical_path = data_path + 'subcortical_colored_effect.nii.gz'

# -----------------------------
# Load images
# -----------------------------
anat_img = nib.load(anat_path)
cortical_img = nib.load(cortical_path)
subcortical_img = nib.load(subcortical_path)

anat_data = anat_img.get_fdata()
cortical_data = cortical_img.get_fdata()
subcortical_data = subcortical_img.get_fdata()

# -----------------------------
# Slice settings
# -----------------------------
slice_indices = range(10, 295, 16)
slice_reverse = slice_indices[::-1]
n_rows = 3
n_cols = 6

# -----------------------------
# Edge settings
# -----------------------------
edge_dark_factor = 0.8  # how dark the edges are (0 = black, 1 = same as color)
edge_thickness = 2       # thickness of the edge in pixels

# -----------------------------
# Functions
# -----------------------------
def normalize_rgba(img):
    img = img.copy()
    if img.ndim == 3 and img.shape[2] in [3, 4]:
        img_max = img[..., :3].max()
        if img_max > 0:
            img[..., :3] = img[..., :3] / img_max
    else:
        img = np.repeat(img[..., np.newaxis] / img.max(), 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 4:
        img[..., 3] = np.clip(img[..., 3], 0, 1)
    else:
        img = np.dstack((img, np.ones(img.shape[:2])))
    return img

def blend_rgba(fg1, fg2):
    """Blend two RGBA images (standard)."""
    def to_rgba(fg):
        if fg.shape[2] == 3:
            alpha = np.ones(fg.shape[:2])
            return np.dstack((fg, alpha))
        return fg
    fg1 = to_rgba(fg1)
    fg2 = to_rgba(fg2)
    alpha1 = fg1[..., 3][..., np.newaxis]
    alpha2 = fg2[..., 3][..., np.newaxis]
    out_rgb = fg1[..., :3]*alpha1*(1-alpha2) + fg2[..., :3]*alpha2 + fg1[..., :3]*alpha1*alpha2
    out_alpha = np.clip(alpha1 + alpha2*(1-alpha1), 0, 1)
    return np.dstack((out_rgb, out_alpha))

# -----------------------------
# Plot slices
# -----------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), facecolor='black')
fig.patch.set_facecolor('black')

for i, slice_idx in enumerate(slice_reverse):
    ax = axes.flat[i]
    ax.set_facecolor('black')

    # -----------------------------
    # Background grayscale
    # -----------------------------
    bg_slice = np.rot90(anat_data[:, slice_idx, :])
    if bg_slice.max() > 0:
        bg_gray = bg_slice / bg_slice.max()
    else:
        bg_gray = bg_slice
    bg_rgb = np.stack([bg_gray]*3, axis=-1)

    # -----------------------------
    # Foreground overlays
    # -----------------------------
    fg_cortical = (np.rot90(cortical_data[:, slice_idx, :, :])
                   if cortical_data.ndim == 4
                   else np.expand_dims(np.rot90(cortical_data[:, slice_idx, :]), axis=-1))
    fg_subcortical = (np.rot90(subcortical_data[:, slice_idx, :, :])
                      if subcortical_data.ndim == 4
                      else np.expand_dims(np.rot90(subcortical_data[:, slice_idx, :]), axis=-1))
    fg_cortical = normalize_rgba(fg_cortical)
    fg_subcortical = normalize_rgba(fg_subcortical)
    blended_fg = blend_rgba(fg_cortical, fg_subcortical)

    combined_rgb = bg_rgb.copy()

    # -----------------------------
    # Find unique colors
    # -----------------------------
    overlay_mask = np.any(blended_fg[..., :3] > 0, axis=-1)
    color_pixels = blended_fg[..., :3][overlay_mask]
    color_pixels_rounded = (color_pixels*255).astype(int)
    unique_colors = np.unique(color_pixels_rounded, axis=0)/255.0

    # -----------------------------
    # Fill colors and edges
    # -----------------------------
    for color in unique_colors:
        # Mask for this color
        mask = np.all(np.isclose(blended_fg[..., :3], color, atol=0.01), axis=-1)
        if not mask.any():
            continue
        # Edge detection via dilation
        edge_mask = binary_dilation(mask, iterations=edge_thickness) & ~mask

        # Fill interior
        combined_rgb[mask] = color
        # Darken edges
        combined_rgb[edge_mask] = color * edge_dark_factor

    ax.imshow(combined_rgb, interpolation='nearest')
    ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0.04)
plt.show()
