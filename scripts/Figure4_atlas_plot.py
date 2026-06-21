#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:56:25 2024

@author: sjoerdmurris
"""

import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
data_path = str(Path(__file__).resolve().parent.parent / 'data') + '/'
file_path = data_path + 'Pubmed_Overview_Final_Tasks.xlsx'

# Load Excel data
df = pd.read_excel(file_path, sheet_name='Study_Selection')

# NIfTI task files (4th dimension = colour information)
cortical_path = data_path + 'cortical_colored_tasks.nii.gz'
subcortical_path = data_path + 'subcortical_colored_tasks.nii.gz'

# Load NIfTI images
cortical_img = nib.load(cortical_path)
subcortical_img = nib.load(subcortical_path)

# Extract data arrays
cortical_data = cortical_img.get_fdata()
subcortical_data = subcortical_img.get_fdata()

print("Cortical data shape:", cortical_data.shape)
print("Subcortical data shape:", subcortical_data.shape)

# -----------------------------
# Plot settings
# -----------------------------
slice_indices = range(10, 295, 16)
slice_reverse = slice_indices[::-1]

n_rows = 3
n_cols = 6

# -----------------------------
# Create figure with subplots
# -----------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))

# Loop over slices
for i, slice_idx in enumerate(slice_reverse):
    ax = axes[i] if n_rows == 1 else axes.flat[i]
    
    # Extract 2D slices from cortical and subcortical images
    # If 4th dimension exists, take the first channel (R) or combine channels as needed
    if cortical_data.ndim == 4:
        foreground_slice = np.rot90(cortical_data[:, slice_idx, :, 0])  # First color channel
    else:
        foreground_slice = np.rot90(cortical_data[:, slice_idx, :])
    
    if subcortical_data.ndim == 4:
        background_slice = np.rot90(subcortical_data[:, slice_idx, :, 0])
    else:
        background_slice = np.rot90(subcortical_data[:, slice_idx, :])
    
    # Alpha mask: zero values fully transparent
    alpha_mask = np.ones_like(foreground_slice) * 0.6
    alpha_mask[foreground_slice == 0] = 0
    
    # Custom colormap for foreground
    custom_colors = ['gray', 'gray', '#befdb7', '#65ab7c', '#05472a', 'black']
    cmap = ListedColormap(custom_colors)
    boundaries = [-0.1, 0.9, 1.9, 5.9, 9.9, 39.9, 200]
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    
    # Plot background first
    ax.imshow(background_slice, cmap='gray')
    
    # Overlay foreground with transparency
    ax.imshow(foreground_slice, cmap=cmap, norm=norm, alpha=alpha_mask)
    
    ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0.04)
plt.show()
