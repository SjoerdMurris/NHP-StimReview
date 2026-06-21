#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:40:04 2025

@author: sjoerdmurris
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.colors import to_rgb, to_hex
from pathlib import Path

# Read data
data_path = str(Path(__file__).resolve().parent.parent.parent / 'data') + '/'
file_path = data_path + 'Pubmed_Overview_Final_StudySelection.csv'
df = pd.read_csv(file_path)

# Task colors
task_colors = {
    'no task & resting-state': '#e0e0e0',
    'eye movement & visual': '#4a90e2',
    'visual perception & discrimination': '#7fb3d5',
    'motor, reaching & grasping': '#6bb06b',
    'conditioning, reward & motivation': '#e2c56b',
    'cognitive, memory & decision-making': '#a78fb3',
    'somatosensory & tactile': '#e28a7f'
}
task_categories = list(task_colors.keys())

# Helper function to prepare stacked data including empty ROIs
def prepare_stacked_data(df_subset):
    # Get all Brain_Parc_Numb values in order
    all_numbers = df[['Brain_Parc_Numb']].drop_duplicates().sort_values('Brain_Parc_Numb')
    stacked_data = []

    roi_labels = []
    atlas_numbers = []
    
    for num in all_numbers['Brain_Parc_Numb']:
        df_roi = df_subset[df_subset['Brain_Parc_Numb'] == num]
        counts = df_roi['Task_category'].value_counts()
        # Keep the order of task_categories
        stacked_data.append([counts.get(task, 0) for task in task_categories])
        # If ROI exists in df_subset, get label, else empty string
        roi_label = df_roi['Brain_Parc'].iloc[0] if not df_roi.empty else ''
        roi_labels.append(roi_label)
        atlas_number = df_roi['Brain_ParcLabel'].min()
        atlas_numbers.append(atlas_number)
        
    return roi_labels, np.array(stacked_data), atlas_numbers

# Split into cortical and subcortical
cortical_df = df[df['Cortical_Subcortical'] == 'cortical']
subcortical_df = df[df['Cortical_Subcortical'] == 'subcortical']

# Prepare data
cortical_labels, cortical_array, cortical_atlas = prepare_stacked_data(cortical_df)
subcortical_labels, subcortical_array, subcortical_atlas = prepare_stacked_data(subcortical_df)

# Plotting function
def plot_stacked_bars(rois, stacked_array, title):
    fig, ax = plt.subplots(figsize=(len(rois)*0.5, 5))
    x_pos = np.arange(len(rois))
    bottoms = np.zeros(len(rois))

    for i, task in enumerate(task_categories):
        values = stacked_array[:, i]
        ax.bar(x_pos, values, bottom=bottoms, color=task_colors[task],
               edgecolor='k', width=0.6, label=task)
        bottoms += values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(rois, rotation=45, ha='right')
    ax.set_ylabel('Number of Studies')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Plot cortical and subcortical
plot_stacked_bars(cortical_labels, cortical_array, 'Cortical ROIs - Task Categories')
plot_stacked_bars(subcortical_labels, subcortical_array, 'Subcortical ROIs - Task Categories')

## Get blended colours for the tasks
# Convert to RGB array (shape = 7x3)
color_list = np.array([to_rgb(c) for c in task_colors.values()])

# --- Define a reusable blending function ---
def blend_colors(count_array, color_list, empty_color="#000000"):
    """
    Compute a blended hex color for each row in count_array
    based on weights and corresponding RGB colors.
    Rows with all zeros are assigned a slightly off-white color.
    
    Parameters:
    - count_array: 2D numpy array of counts (rows = regions, columns = tasks)
    - color_list: list or array of colors in RGB or hex (length = number of columns)
    - empty_color: hex color to use if the row is all zeros (default: very light gray)
    
    Returns:
    - numpy array of hex colors (one per row)
    """
    # Convert color_list to RGB floats if in hex
    rgb_list = np.array([to_rgb(c) if isinstance(c, str) else c for c in color_list])

    blended = []
    for row in count_array:
        total = np.sum(row)
        if total == 0:
            blended.append(empty_color)  # slightly off-white for empty rows
        else:
            weighted_rgb = np.average(rgb_list, axis=0, weights=row)
            blended.append(to_hex(weighted_rgb))
    return np.array(blended)

# --- Compute blended colors for both arrays ---
cortical_blended = blend_colors(cortical_array, color_list)
subcortical_blended = blend_colors(subcortical_array, color_list)

CHARM_path  = data_path + 'NMT_v2.0_sym/supplemental_CHARM/'
SARM_path   = data_path + 'NMT_v2.0_sym/supplemental_SARM/'

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def assign_colors_by_presence(atlas_files, roi_values, roi_colors):
    """
    Loop through atlas .nii files and assign colors to ROIs dynamically based on presence.
    
    Parameters
    ----------
    atlas_files : list of str
        Paths to atlas NIfTI files, order doesn't matter but usually coarse->fine
    roi_values : array-like
        ROI numbers to check and color
    roi_colors : array-like
        Color for each ROI (hex or RGB)
    
    Returns
    -------
    color_volume : np.ndarray
        4D RGB volume with assigned colors
    affine : np.ndarray
        NIfTI affine
    header : nib.Nifti1Header
        NIfTI header
    """
    
    # Load first atlas to get volume shape
    ref_img = nib.load(atlas_files[0])
    shape = ref_img.shape
    color_volume = np.zeros(shape + (3,), dtype=float)
    affine = ref_img.affine
    header = ref_img.header
    
    # Convert colors to RGB
    colors_rgb = np.array([hex_to_rgb(c) if isinstance(c, str) else c for c in roi_colors])
    
    # Track which ROIs have been assigned
    assigned_rois = set()
    
    for atlas_file in atlas_files:
        atlas_data = nib.load(atlas_file).get_fdata().astype(int)
        
        for idx, roi in enumerate(roi_values):
            if roi in assigned_rois:
                continue  # Already assigned
            mask = atlas_data == roi
            if np.any(mask):
                color_volume[mask] = colors_rgb[idx]
                assigned_rois.add(roi)
    
    missing_rois = set(roi_values) - assigned_rois
    if missing_rois:
        print("Warning: these ROIs were not found in any atlas file:", missing_rois)
    
    return color_volume, affine, header

cortical_atlas_files    = [CHARM_path + f'CHARM_{i}_in_NMT_v2.0_sym.nii.gz' for i in range(1, 7)]
subcortical_atlas_files = [SARM_path + f'SARM_{i}_in_NMT_v2.0_sym.nii.gz' for i in range(1, 7)]

cor_color_vol, cor_affine, cor_header = assign_colors_by_presence(cortical_atlas_files, cortical_atlas, cortical_blended)
sub_color_vol, sub_affine, sub_header = assign_colors_by_presence(subcortical_atlas_files, subcortical_atlas, subcortical_blended)

# Optional: create a NIfTI image if you want to save or visualize
cor_color_nifti = nib.Nifti1Image(cor_color_vol, cor_affine, cor_header)
sub_color_nifti = nib.Nifti1Image(sub_color_vol, sub_affine, sub_header)

nib.save(nib.Nifti1Image(cor_color_vol, cor_affine, cor_header), data_path + 'cortical_colored_tasks.nii.gz')
nib.save(nib.Nifti1Image(sub_color_vol, sub_affine, sub_header), data_path + 'subcortical_colored_tasks.nii.gz')


