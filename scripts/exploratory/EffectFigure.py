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
data_path   = str(Path(__file__).resolve().parent.parent.parent / 'data') + '/'
output_path = str(Path(__file__).resolve().parent.parent.parent / 'outputs') + '/'
file_path = data_path + 'Pubmed_Overview_Final_StudySelection.csv'
df = pd.read_csv(file_path)

# Effector-based task colors (refined)
task_colors = {
    'direct': '#e41a1c',
    'indirect': '#fdae61',
    'mixed': 'indianred',
    'no': 'lightyellow',
}

task_categories = list(task_colors.keys())


def prepare_stacked_data(df_subset):
    all_numbers = df[['Brain_Parc_Numb']].drop_duplicates().sort_values('Brain_Parc_Numb')
    stacked_data = []
    roi_labels = []
    atlas_numbers = []

    for num in all_numbers['Brain_Parc_Numb']:
        df_roi = df_subset[df_subset['Brain_Parc_Numb'] == num]
        counts = df_roi['Effect'].value_counts()
        stacked_data.append([counts.get(task, 0) for task in task_categories])
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


def plot_stacked_bars(rois, stacked_array, out_prefix=None):
    fig, ax = plt.subplots(figsize=(len(rois) * 0.5, 5))
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
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if out_prefix is not None:
        plt.savefig(out_prefix + '.svg', format='svg', bbox_inches='tight')
        plt.savefig(out_prefix + '.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: {out_prefix}.svg / .png")

    plt.show()


# Plot cortical and subcortical — no title argument
plot_stacked_bars(cortical_labels, cortical_array,
                  out_prefix=output_path + 'cortical_stacked_bars')
plot_stacked_bars(subcortical_labels, subcortical_array,
                  out_prefix=output_path + 'subcortical_stacked_bars')


## Get blended colours for the tasks
color_list = np.array([to_rgb(c) for c in task_colors.values()])


def blend_colors(count_array, color_list, empty_color="#000000"):
    rgb_list = np.array([to_rgb(c) if isinstance(c, str) else c for c in color_list])
    blended = []
    for row in count_array:
        total = np.sum(row)
        if total == 0:
            blended.append(empty_color)
        else:
            weighted_rgb = np.average(rgb_list, axis=0, weights=row)
            blended.append(to_hex(weighted_rgb))
    return np.array(blended)


cortical_blended = blend_colors(cortical_array, color_list)
subcortical_blended = blend_colors(subcortical_array, color_list)

CHARM_path = data_path + 'NMT_v2.0_sym/supplemental_CHARM/'
SARM_path  = data_path + 'NMT_v2.0_sym/supplemental_SARM/'


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def assign_colors_by_presence(atlas_files, roi_values, roi_colors):
    ref_img = nib.load(atlas_files[0])
    shape = ref_img.shape
    color_volume = np.zeros(shape + (3,), dtype=float)
    affine = ref_img.affine
    header = ref_img.header

    colors_rgb = np.array([hex_to_rgb(c) if isinstance(c, str) else c
                           for c in roi_colors])
    assigned_rois = set()

    for atlas_file in atlas_files:
        atlas_data = nib.load(atlas_file).get_fdata().astype(int)
        for idx, roi in enumerate(roi_values):
            if roi in assigned_rois:
                continue
            mask = atlas_data == roi
            if np.any(mask):
                color_volume[mask] = colors_rgb[idx]
                assigned_rois.add(roi)

    missing_rois = set(roi_values) - assigned_rois
    if missing_rois:
        print("Warning: these ROIs were not found in any atlas file:", missing_rois)

    return color_volume, affine, header


cortical_atlas_files    = [CHARM_path + f'CHARM_{i}_in_NMT_v2.0_sym.nii.gz'
                           for i in range(1, 7)]
subcortical_atlas_files = [SARM_path  + f'SARM_{i}_in_NMT_v2.0_sym.nii.gz'
                           for i in range(1, 7)]

cor_color_vol, cor_affine, cor_header = assign_colors_by_presence(
    cortical_atlas_files, cortical_atlas, cortical_blended)
sub_color_vol, sub_affine, sub_header = assign_colors_by_presence(
    subcortical_atlas_files, subcortical_atlas, subcortical_blended)

nib.save(nib.Nifti1Image(cor_color_vol, cor_affine, cor_header),
         data_path + 'cortical_colored_effect.nii.gz')
nib.save(nib.Nifti1Image(sub_color_vol, sub_affine, sub_header),
         data_path + 'subcortical_colored_effect.nii.gz')


## Run statistics

import statsmodels.formula.api as smf
import re


def parse_frequency(val):
    if pd.isna(val):
        return np.nan
    val = str(val).lower().replace('hz', '').strip()
    if 'single' in val or 'pulse' in val:
        return 0.5
    numbers = re.findall(r'[\d.]+', val)
    if len(numbers) == 0:
        return np.nan
    return np.mean([float(n) for n in numbers])


df['Frequency_Hz'] = df['Frequency'].apply(parse_frequency)
print(df['Frequency_Hz'].describe())
print(f"NaNs: {df['Frequency_Hz'].isna().sum()}")

df['Frequency_log'] = np.log1p(df['Frequency_Hz'])


def parse_amplitude(row):
    try:
        lower  = float(row['Amplitude_lower'])
        higher = float(row['Amplitude_higher'])
    except (ValueError, TypeError):
        return np.nan

    mean_amp = np.mean([lower, higher])
    unit = str(row['Amplitude_unit']).strip().lower()

    if unit == 'ma':
        mean_amp = mean_amp * 1000
    elif unit in ['microa', 'µa', 'ua', 'μa']:
        pass
    else:
        return np.nan

    return mean_amp


df['Amplitude_uA'] = df.apply(parse_amplitude, axis=1)
print(df['Amplitude_uA'].describe())
print(f"NaNs: {df['Amplitude_uA'].isna().sum()}")

df['Amplitude_log'] = np.log1p(df['Amplitude_uA'])

df_model = df[
    (df['Type_stimulation'] == 'electrical') &
    (df['Cortical_Subcortical'] != 'neither')
][['Effect', 'Electrode_type', 'Waveform', 'Year', 'Frequency_log',
   'Brain_Parc']].copy()

df_model['Waveform'] = df_model['Waveform'].str.strip()
df_model.loc[df_model['Waveform'].str.contains(',|and|/', na=False), 'Waveform'] = 'both'
df_model = df_model[df_model['Waveform'].isin(['monophasic', 'biphasic'])]
df_model = df_model.dropna().copy()
df_model['outcome_binary'] = (df_model['Effect'] == 'direct').astype(int)

print(f"N after filtering: {len(df_model)}")
print(f"Unique ROIs: {df_model['Brain_Parc'].nunique()}")
print(f"Direct: {df_model['outcome_binary'].sum()}, "
      f"Other: {(df_model['outcome_binary']==0).sum()}")

model = smf.mixedlm(
    'outcome_binary ~ C(Electrode_type) + C(Waveform) + Year + Frequency_log',
    data=df_model,
    groups=df_model['Brain_Parc']
).fit()
print(model.summary())

## Run three models (mixed, no, indirect, direct)
for other in ['indirect', 'no', 'mixed']:
    df_sub = df_model[df_model['Effect'].isin(['direct', other])].copy()
    df_sub = df_sub.dropna(subset=['Amplitude_log']).copy()
    df_sub['outcome_binary'] = (df_sub['Effect'] == 'direct').astype(int)

    print(f"\n{'='*60}")
    print(f"direct vs {other} (N={len(df_sub)}, "
          f"direct={df_sub['outcome_binary'].sum()}, "
          f"{other}={(df_sub['outcome_binary']==0).sum()})")
    print('='*60)

    model = smf.mixedlm(
        'outcome_binary ~ C(Electrode_type) + C(Waveform) + Amplitude_log + Frequency_log',
        data=df_sub,
        groups=df_sub['Brain_Parc']
    ).fit()
    print(model.summary())