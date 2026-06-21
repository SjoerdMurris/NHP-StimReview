#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:41:29 2024

@author: sjoerdmurris
"""
# Summarize data for review paper
# Import tabular data

import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path

"(1) read in the data from the excel file and atlas"
data_path       = str(Path(__file__).resolve().parent.parent / 'data') + '/'
overview_path   = data_path + 'Pubmed_Overview_Final.xlsx'

df              = pd.read_excel(overview_path, sheet_name='Study_Selection')

charm_path      = data_path + '/tables_CHARM/CHARM_key_all.txt'
sarm_path       = data_path + '/tables_SARM/SARM_key_all.txt'

charm_df        = pd.read_csv(charm_path, sep='\s+', header=0)
sarm_df         = pd.read_csv(sarm_path, sep='\s+', header=0)


"(2) clean data: get rid of no access articles"
df_cleaned = df[df['Journal'] != 'NO ACCESS']

# Get rid of instances that have no atlas index (i.e. white matter tracts)
df_filtered = df_cleaned[~df_cleaned['Brain_Atlas'].apply(lambda x: isinstance(x, str))]

# Option to generate a map for esfMRI studies
df_select   = df_filtered[df_filtered['Concurrent_fMRI'] == 'YES']
#df_select   = df_filtered[df_filtered['Type_stimulation'] == 'optogenetic']
df_filtered = df_select

"(3) separate cortical from subcortical entries"
df_sub = df_filtered[df_filtered['Cortical_Subcortical'] == 'subcortical']
df_cor = df_filtered[df_filtered['Cortical_Subcortical'] == 'cortical']

# Add a column based on the first_level entry
counts_sub = df_sub['Brain_Atlas'].value_counts()
counts_cor = df_cor['Brain_Atlas'].value_counts()

# Obtain the indices and first_level values from the atlas txt files
df_counts_sub = pd.DataFrame({'Index': counts_sub.keys(), 'Count': counts_sub.values})
df_counts_sub['Match_Index'] = df_counts_sub['Index'].apply(lambda x: sarm_df.index[sarm_df['Index'] == x].tolist())
df_counts_sub['First_Level'] = df_counts_sub['Match_Index'].apply(lambda indices: [sarm_df['First_Level'].iloc[i] for i in indices] if indices else [None])

df_counts_cor = pd.DataFrame({'Index': counts_cor.keys(), 'Count': counts_cor.values})
df_counts_cor['Match_Index'] = df_counts_cor['Index'].apply(lambda x: charm_df.index[charm_df['Index'] == x].tolist())
df_counts_cor['First_Level'] = df_counts_cor['Match_Index'].apply(lambda indices: [charm_df['First_Level'].iloc[i] for i in indices] if indices else [None])


"(4) Read in the nifti files"
# First specify the paths for CHARM/SARM atlas
CHARM_path  = data_path + 'NMT_v2.0_sym/supplemental_CHARM/'
SARM_path   = data_path + 'NMT_v2.0_sym/supplemental_SARM/'

first_level_values = np.arange(2,7,1)

# Read in CHARM, SARM first level maps
nifti_file_cor = nib.load(CHARM_path + 'CHARM_1_in_NMT_v2.0_sym.nii.gz')
nifti_file_sub = nib.load(SARM_path + 'SARM_1_in_NMT_v2.0_sym.nii.gz')
nifti_data_cor = nifti_file_cor.get_fdata()
nifti_data_sub = nifti_file_sub.get_fdata()

# Generate an empty matrix of the same size
empty_matrix = nifti_data_cor.copy()
empty_matrix.fill(0)

# Create a mask for both CHARM and SARM combined
mask_matrix = nifti_data_cor.copy()
mask_matrix[mask_matrix != 0] = 1

mask_sub_matrix = nifti_data_sub.copy()
mask_sub_matrix[mask_sub_matrix != 0] = 1

# There is some overlap between CHARM and SARM, but ignoring it in here
mask_combined = mask_matrix + mask_sub_matrix
mask_combined[mask_combined != 0] = 1

size_matrix = nifti_data_cor
size_matrix.fill(0)

# Add separate matrices for cortical and subcortical outputs
cor_matrix = empty_matrix.copy()
sub_matrix = empty_matrix.copy()

for flv in first_level_values:
    nifti_file = nib.load(CHARM_path + 'CHARM_' + str(flv) + '_in_NMT_v2.0_sym.nii.gz')
    nifti_data = nifti_file.get_fdata()
    indices = df_counts_cor.index[df_counts_cor['First_Level'].apply(lambda x: flv in x)].tolist()
    
    for index in indices:
        ROI_value = df_counts_cor['Index'][index]
        ROI_indices = np.where(nifti_data == ROI_value)
        empty_matrix[ROI_indices] = empty_matrix[ROI_indices] + df_counts_cor['Count'][index]
        size_matrix[ROI_indices]  = size_matrix[ROI_indices] + (df_counts_cor['Count'][index] / len(ROI_indices[1]))
        cor_matrix[ROI_indices]   = cor_matrix[ROI_indices] + df_counts_cor['Count'][index]
        
for flv in first_level_values:
    nifti_file = nib.load(SARM_path + 'SARM_' + str(flv) + '_in_NMT_v2.0_sym.nii.gz')
    nifti_data = nifti_file.get_fdata()
    indices = df_counts_sub.index[df_counts_sub['First_Level'].apply(lambda x: flv in x)].tolist()
    
    for index in indices:
        ROI_value = df_counts_sub['Index'][index]
        ROI_indices = np.where(nifti_data == ROI_value)
        empty_matrix[ROI_indices] = empty_matrix[ROI_indices] + df_counts_sub['Count'][index]
        size_matrix[ROI_indices]  = size_matrix[ROI_indices] + (df_counts_sub['Count'][index] / len(ROI_indices[1]))
        sub_matrix[ROI_indices]   = sub_matrix[ROI_indices] + df_counts_sub['Count'][index]
        
masked_matrix = empty_matrix + mask_combined
output_nifti = nib.Nifti1Image(masked_matrix, nifti_file.affine, nifti_file.header)
nib.save(output_nifti, data_path + 'output_opt.nii.gz')

output_cor   = nib.Nifti1Image(cor_matrix, nifti_file.affine, nifti_file.header)
nib.save(output_cor, data_path + 'cor_output_opt.nii.gz')

output_sub   = nib.Nifti1Image(sub_matrix, nifti_file.affine, nifti_file.header)
nib.save(output_sub, data_path + 'sub_output_opt.nii.gz')

output_nifti2 = nib.Nifti1Image(size_matrix, nifti_file.affine, nifti_file.header)
nib.save(output_nifti2, data_path + 'size_output_opt.nii.gz')