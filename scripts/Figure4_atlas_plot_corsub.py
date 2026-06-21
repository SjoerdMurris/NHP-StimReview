#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:40:59 2024

@author: sjoerdmurris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:56:25 2024

@author: sjoerdmurris
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm


## Import previous nifti file and visualize slices
# Read in the results and the NMT images (nifti files)
from pathlib import Path
data_path     = str(Path(__file__).resolve().parent.parent / 'data') + '/'
template_file = data_path + 'NMT_v2.0_sym/NMT_v2.0_sym_SS.nii.gz'

NMT_file = nib.load(template_file)
NMT_data = NMT_file.get_fdata()

COR_file = nib.load(data_path + 'cor_output_opt.nii.gz')
SUB_file = nib.load(data_path + 'sub_output_opt.nii.gz')

COR_data = COR_file.get_fdata()
SUB_data = SUB_file.get_fdata()

RES_file = nib.load(data_path + 'output_opt.nii.gz')
RES_data = RES_file.get_fdata()

# Round the results
rounded_cor  = np.round(COR_data)
rounded_sub  = np.round(SUB_data)
rounded_data = np.round(RES_data)

# Plot the 2D slices
slice_indices = range(10,295,16)
slice_reverse  = slice_indices[::-1]


# Number of rows and columns in the grid of subplots
n_rows = 3  # Change this based on the number of slices
n_cols = 6 #len(slice_indices)

# Create a figure with multiple subplots
fig, axes = plt.subplots(n_rows, n_cols, dpi=600)

# Loop over each slice index and plot it in a subplot
for i, slice_idx in enumerate(slice_reverse):
    ax = axes[i] if n_rows == 1 else axes.flat[i]  # For single or multiple rows
    
    # Extract and rotate the 2D slices from both the foreground and background
    cortical_slice    = np.rot90(rounded_cor[:, slice_idx, :])
    subcortical_slice = np.rot90(rounded_sub[:, slice_idx, :])
    background_slice  = np.rot90(NMT_data[:, slice_idx, :])
    foreground_slice  = np.rot90(rounded_data[:, slice_idx, :])
    
    # Create a mask where values in the foreground equal to 0 are fully transparent
    alpha_mask_cor = np.ones_like(cortical_slice) * 1
    alpha_mask_cor[cortical_slice == 0] = 0  # Set alpha to 0 for zero values
    
    alpha_mask_sub = np.ones_like(subcortical_slice) * 1
    alpha_mask_sub[subcortical_slice == 0] = 0  # Set alpha to 0 for zero values
    
    alpha_def = np.ones_like(foreground_slice) * 1
    alpha_def[foreground_slice == 0] = 0
    
    # Define the threshold for values that should be white
    threshold = 10.3

    # Create the custom colormap
    # Use the 'gray' colormap as a base, but modify the first part to be white
    cmap = plt.cm.gray  # Starting colormap (grayscale)
    colors = cmap(np.linspace(0, 1, 256))  # Extract the grayscale colors

    # Modify the low values (close to zero) to white
    colors[:int(threshold * 256)] = [1, 1, 1, 1]  # Set the first portion of the colormap to white

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_white', colors)
    
    # Define custom colors for specific values
    cor_colors = ['none', '#b7c9e2', '#75bbfd', '#01386a', 'black'] # #d1ffbd,#154406 '#50a747', '#062e03'
    sub_colors = ['none', '#c8aca9', '#c85a53', '#840000', 'black']
    
    cmap_cor = ListedColormap(cor_colors)
    cmap_sub = ListedColormap(sub_colors)

    # Define boundaries for mapping values to colors
    #boundaries = [-0.1, 0.9, 4.9, 19.9, 39.9, 200]  # Upper bounds for each color
    #boundaries = [-0.1, 0.9, 1.9, 5.9, 9.9, 29.9, 200]  # Upper bounds for each color
    boundaries = [-0.1, 0.9, 1.9, 4.9, 9.9, 200]
    norm = BoundaryNorm(boundaries, cmap_cor.N, clip=True)
   
    # Overlay the foreground image with transparency
    im_orig = ax.imshow(alpha_def, cmap='gray_r', vmin=0, vmax=5)
    im      = ax.imshow(cortical_slice, cmap=cmap_cor, norm=norm, alpha=alpha_mask_cor)  # Overlay with transparency
    im2     = ax.imshow(subcortical_slice, cmap=cmap_sub, norm=norm, alpha=alpha_mask_sub)
    
    #ax.set_title(f'Slice {slice_idx}')
    ax.axis('off')  # Turn off the axis labels for cleaner plots

# Manually reduce the space between subplots
# Adjust spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0.04)

plt.show()