# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 18:04:32 2024

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import numpy as np
from utils import create_binary_array
from matplotlib.transforms import ScaledTranslation

"(0) Specify general parameters for the figures"
data_label = 'subcortical' # Specify cortical or subcortical to plot

"(1) read in the data from the excel file"
file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Study_Selection')

"(2) parse the amplitude stimulation data"
# Convert to cortical or subcortical
cort_df      = df[df['Cortical_Subcortical'] == data_label]

# First sort by electrical/optogenetic stimulation
elec_df      = cort_df[cort_df['Type_stimulation'] == 'electrical']
opt_df       = cort_df[cort_df['Type_stimulation'] == 'optogenetic']

# Convert milli-ampere values to micro-ampere
elec_df.loc[elec_df['Amplitude_unit'] == 'mA', 'Amplitude_lower'] *= 1000
elec_df.loc[elec_df['Amplitude_unit'] == 'mA', 'Amplitude_higher'] *= 1000

# Split up by macro and micro electrode
micro_df            = elec_df[elec_df['Electrode_type'].str.contains('Microelectrode', case=False, na=False)]
macro_df            = elec_df[elec_df['Electrode_type'].str.contains('Macroelectrode', case=False, na=False)]

# Separate A, V and unclear
df_amplitude        = elec_df[elec_df['Amplitude_unit'].astype(str).str.contains(r'[A]', regex=True)]
df_voltage          = elec_df[elec_df['Amplitude_unit'].astype(str).str.contains(r'[V]', regex=True)]
df_unclear          = elec_df[elec_df['Amplitude_unit'].astype(str).str.contains(r'[unclear]', regex=True)]

" (3) Create the main figure with specifications"
# Create a main figure
# Create a figure
fig = plt.figure(figsize=(10.5, 8), dpi = 600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(2, 2, height_ratios=[6, 16], width_ratios=[1,3])  # Top rows shorter than bottom

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Top left
ax2 = fig.add_subplot(gs[0, 1])  # Top right
ax3 = fig.add_subplot(gs[1, 0])  # Bottom left
ax4 = fig.add_subplot(gs[1, 1])  # Bottom right

" (4) Create a pie chart that shows the distribution of stimulation types - micro and macro-electrode"
# Add labels and title
sizes    = [len(micro_df), len(macro_df), len(df_voltage) + len(df_unclear)]
colors   = ['indianred', 'darkorange', 'silver']

ax1.pie(sizes, colors=colors, wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}, startangle=90)
ax1.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)

"(5) Bar graph with counts -- per amplitude"
df_amplitude = df_amplitude.copy()  # Add this line

# Create a new column with arrays from Lower_Value to Higher_Value
df_amplitude.loc[:, 'Range'] = df_amplitude.apply(
    lambda row: np.arange(int(row['Amplitude_lower']), int(row['Amplitude_higher']) + 1), axis=1
)

#df_amplitude['Range'] = df_amplitude.apply(lambda row: np.arange(int(row['Amplitude_lower']), int(row['Amplitude_higher']) + 1), axis=1)

# Create binary arrays
df_amplitude.loc[:, 'BinaryArrays'] = df_amplitude['Range'].apply(
    create_binary_array, start_value=0, end_value=10000
)

#df_amplitude['BinaryArrays']   = df_amplitude['Range'].apply(create_binary_array, start_value=0, end_value=10000)

# Split up based on Micro or Macro electrode
Micro_array = df_amplitude[df_amplitude['Electrode_type'].str.contains('Microelectrode', case=False, na=False)]
Macro_array = df_amplitude[df_amplitude['Electrode_type'].str.contains('Macroelectrode', case=False, na=False)]

# Collapse binary arrays
collapsed_micro = np.sum(np.array(Micro_array['BinaryArrays'].tolist()), axis=0)
collapsed_macro = np.sum(np.array(Macro_array['BinaryArrays'].tolist()), axis=0)

# amplitudes (skip 0)
amps = np.arange(1, len(collapsed_micro))
micro_vals = collapsed_micro[1:]
macro_vals = collapsed_macro[1:]

# Define log bins: 10 bins per decade from 1 to 10000
bins_per_decade = 10
bin_edges = np.logspace(0, np.log10(10000), num=4*bins_per_decade + 1)  # 4 decades

# Compute binned averages, filling empty bins with the closest value
def compute_binned_values(vals, amps, bin_edges):
    binned = []
    for i in range(len(bin_edges)-1):
        mask = (amps >= bin_edges[i]) & (amps < bin_edges[i+1])
        if np.any(mask):
            avg = vals[mask].mean()
        else:
            # Use closest amplitude to bin center
            bin_center = np.sqrt(bin_edges[i] * bin_edges[i+1])
            closest_idx = np.argmin(np.abs(amps - bin_center))
            avg = vals[closest_idx]
        binned.append(avg)
    return np.array(binned)

micro_binned = compute_binned_values(micro_vals, amps, bin_edges)
macro_binned = compute_binned_values(macro_vals, amps, bin_edges)

# Compute bar centers and widths
centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean
widths = bin_edges[1:] - bin_edges[:-1]

# Plot bars
ax2.bar(centers, micro_binned, width=widths, align='center', color='indianred', edgecolor='none', zorder=2)
ax2.bar(centers, macro_binned, width=widths, align='center', color='darkorange', edgecolor='none', alpha=0.8, zorder=2)

# Keep x-axis logarithmic
ax2.set_xscale('log')
ax2.set_xlim(1, 10000)
ax2.set_xlabel('Amplitude (µA)')

# Move y-axis to the right
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_ylabel('Reports (#)')
ax2.set_ylim(0, 400)

# Clean style
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Force tick labels at nice positions
ax2.set_xticks([1, 10, 100, 1000, 10000])
ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax2.ticklabel_format(style='plain', axis='x')

ax2.xaxis.set_minor_locator(ticker.NullLocator())

## Calculating and plotting midpoints and widths
Micro_array['Amplitude_midpoint'] = (Micro_array['Amplitude_lower'] + Micro_array['Amplitude_higher']) / 2
Micro_array['Amplitude_width']    = Micro_array['Amplitude_higher'] - Micro_array['Amplitude_lower']

Macro_array['Amplitude_midpoint'] = (Macro_array['Amplitude_lower'] + Macro_array['Amplitude_higher']) / 2
Macro_array['Amplitude_width']    = Macro_array['Amplitude_higher'] - Macro_array['Amplitude_lower']

# Bootstrapping function
def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    data = np.array(data)
    boot_medians = []
    n = len(data)
    for _ in range (n_bootstrap):
        sample = np.random.choice(data, size=n, replace = True)
        boot_medians.append(np.median(sample))
    lower = np.percentile(boot_medians, (100-ci)/2)
    upper = np.percentile(boot_medians, 100-(100-ci)/2)
    return np.median(data), (lower, upper)

micro_mid_median, micro_mid_ci     = bootstrap_ci(Micro_array['Amplitude_midpoint'])
micro_width_median, micro_width_ci = bootstrap_ci(Micro_array['Amplitude_width'])

macro_mid_median, macro_mid_ci     = bootstrap_ci(Macro_array['Amplitude_midpoint'])
macro_width_median, macro_width_ci = bootstrap_ci(Macro_array['Amplitude_width'])

# Plot the values
#ax2.hlines(y=350, xmin=micro_mid_median - micro_width_median/2, xmax=micro_mid_median + micro_width_median/2, color='indianred', linewidth=5)
#ax2.hlines(y=350, xmin=macro_mid_median - macro_width_median/2, xmax=macro_mid_median + macro_width_median/2, color='darkorange', linewidth=5)

ax2.scatter(micro_mid_median, 350, s=100, facecolors='white', edgecolors='indianred', linewidths=5, zorder=5)
ax2.scatter(macro_mid_median, 350, s=100, facecolors='white', edgecolors='darkorange', linewidths=5, zorder=5)
ax2.axvline(x=micro_mid_median, ymin=0, ymax=350, color='#d8dcd6', linestyle='--', linewidth=1)
ax2.axvline(x=macro_mid_median, ymin=0, ymax=350, color='#d8dcd6', linestyle='--', linewidth=1)

# Plot a figure for each ROI - similar to frequency figure
sub_combined = pd.concat([Micro_array, Macro_array], ignore_index=True)

# Get rid of an instance of ROIs outside of SARM
sub_combined = sub_combined[sub_combined['Brain_Parc_Numb'] != 299]

brain_numbers = sorted(sub_combined['Brain_Parc_Numb'].unique())

y_counter = 1
sep_counter = 0

# Store brain area labels for y-ticks
brain_area_labels = []
graph_data        = []

# Add padding to the brain areas
arr = np.arange(1, brain_numbers[-1]+1)  # Create array from 1 to n
new_brain_numbers = np.concatenate(([0], arr, [0]))  # Ensure 0 is in a list

## Get brain numbers that are empty
diff1 = np.setdiff1d(new_brain_numbers, brain_numbers)
cort_miss_labels = ['LPons','LVPal']

# Reverse the order of the ROIs
reversed_brain_numbers = new_brain_numbers[::-1]

# Iterate over each unique value in the labels (rows on the y-axis)
for label in reversed_brain_numbers:
    
    # Isolate rows with this brain area
    if label == 0:        
        # Draw an empty line
        ax4.axhline(y=y_counter, color='white', linestyle='dotted', linewidth=0.5)
        
        # Add empty label
        brain_area_label = ''
        brain_area_labels.append(brain_area_label)
        
    elif label in diff1:
        # Draw an empty line
        ax4.axhline(y=y_counter, color='gray', linestyle='dotted', linewidth=0.5)
        
        brain_area_label = cort_miss_labels[sep_counter]
        brain_area_labels.append(brain_area_label)
        sep_counter = sep_counter + 1
        
    else:
        filtered_df = sub_combined[sub_combined['Brain_Parc_Numb'] == label]
       
        # Add label from df_sub['Brain_area'] for the y-axis
        brain_area_label = sub_combined.loc[sub_combined['Brain_Parc_Numb'] == label, 'Brain_Parc'].iloc[0]  # Get the corresponding Brain area label
        brain_area_labels.append(brain_area_label)  # Store the brain area label

        # Plot a horizontal dotted line for each row
        ax4.axhline(y=y_counter, color='gray', linestyle='dotted', linewidth=0.5)
    
        # Include amplitude range measure if not empty (NaN)
        if filtered_df['Electrode_type'].str.contains('Microelectrode', case=False, na=False).any():
            indices = filtered_df[filtered_df['Electrode_type'].str.contains('Microelectrode', case=False, na=False)]
            values = np.concatenate(indices['Range'].values)
            unique_values, counts = np.unique(values, return_counts=True)
        
            # Data to add for bar graphs (per area)
            row_graph = {'Index': y_counter, 'Category': 'Micro_Electrode', 'Brain_Parc': label, 'Amplitude': 'unkown', 'Count': len(indices['Range'])}
            graph_data.append(row_graph)
            
            #x = np.linspace(1, 10001, 10001)  # Your x-values (10001 points)
            #collapsed_array = np.sum(np.array(indices['BinaryArrays'].tolist()), axis=0)
            
            # Determine the size of each dot based on x (you can scale this as needed)
            #dot_sizes = collapsed_array * 2  # Multiply by a scaling factor for visibility (adjust as needed)
            
            # Plot each x as a dot on the line at y_target
            #ax4.scatter(x, np.full_like(collapsed_array, y_counter), s=dot_sizes, color='indianred', alpha=0.05)
            
            mid_med = indices['Amplitude_midpoint'].median()
            wid_med = indices['Amplitude_width'].median()
            min_val = indices['Amplitude_lower'].min()
            max_val = indices['Amplitude_higher'].max()
            
            #ax4.hlines(y_counter, xmin=mid_med - wid_med/2, xmax=mid_med + wid_med/2, color='indianred', linewidth=3)
            ax4.scatter(mid_med, y_counter, s=50, facecolors='indianred', zorder=5)
            ax4.hlines(y_counter, xmin=min_val, xmax=max_val, color='indianred', alpha=0.5, linewidth=3, zorder=1)
            
            
        if filtered_df['Electrode_type'].str.contains('Macroelectrode', case=False, na=False).any():
            indices = filtered_df[filtered_df['Electrode_type'].str.contains('Macroelectrode', case=False, na=False)]
            values = np.concatenate(indices['Range'].values)
        
            # Data to add for bar graphs (per area)
            row_graph = {'Index': y_counter, 'Category': 'Macro_Electrode', 'Brain_Parc': label, 'Amplitude': 'unkown', 'Count': len(indices['Range'])}
            graph_data.append(row_graph)
            
            #x = np.linspace(1, 10001, 10001)  # Your x-values (10001 points)
            #collapsed_array = np.sum(np.array(indices['BinaryArrays'].tolist()), axis=0)
            
            # Determine the size of each dot based on x (you can scale this as needed)
            #dot_sizes = collapsed_array * 2  # Multiply by a scaling factor for visibility (adjust as needed)
            
            # Plot each x as a dot on the line at y_target
            #ax4.scatter(x, np.full_like(collapsed_array, y_counter), s=dot_sizes, color='darkorange', alpha=0.05)
            
            mac_mid_med = indices['Amplitude_midpoint'].median()
            mac_wid_med = indices['Amplitude_width'].median()
            mac_min_val = indices['Amplitude_lower'].min()
            mac_max_val = indices['Amplitude_higher'].max()
            
            #ax4.hlines(y_counter, xmin=mac_mid_med - mac_wid_med/2, xmax=mac_mid_med + mac_wid_med/2, color='darkorange', linewidth=3)
            ax4.scatter(mac_mid_med, y_counter, s=50, facecolors='darkorange', zorder=5)
            ax4.hlines(y_counter, xmin=mac_min_val, xmax=mac_max_val, color='darkorange', alpha=0.5, linewidth=3, zorder=1)
            
            print('Done')
            
    y_counter += 2

# Set labels
ax4.set_xlabel('Amplitude (µA)')
ax4.set_xscale('log')
ax4.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax4.ticklabel_format(style='plain', axis='x')
ax4.set_xlim(1, 10000)
ax4.set_xticks([1, 10, 100, 1000, 10000])

# Configuration of the axes, lay-out
ax4.set_yticks(list(range(1, len(new_brain_numbers)*2 + 1, 2)))  # Adjust this to your actual y values
ax4.set_yticklabels(brain_area_labels, fontsize=8)  # Your string labels
ax4.set_ylim(0, len(new_brain_numbers)*2 - 1)

left_ticks  = ax4.get_yticks()
left_labels = ax4.get_yticklabels() 

# Set right y-axis ticks and labels
ax_inv = ax4.twinx()  # Create twin axes sharing the x-axis
ax_inv.set_yticks(left_ticks)  # Copy ticks
ax_inv.set_yticklabels(left_labels)  # Copy labels

#tick_lengths = tick_lengths_or[::-1]
middle_values = np.full(55,5)
tick_lengths = np.concatenate(([0], middle_values, [0]))

# Adjust x-axis tick lengths
for tick, length in zip(ax_inv.yaxis.get_major_ticks(), tick_lengths):
    tick.tick1line.set_markersize(length)  # Length of ticks pointing inward
    tick.tick2line.set_markersize(length)  # Length of ticks pointing outward

# Adjust the positions of the labels
for label, length in zip(ax_inv.yaxis.get_majorticklabels(), tick_lengths):
    offset = ScaledTranslation(length / 75, 0, fig.dpi_scale_trans)  # Adjust offset for inverted axis
    label.set_transform(label.get_transform() + offset)

ax_inv.spines['top'].set_visible(False)
ax_inv.spines['left'].set_visible(False)

ax4.spines['top'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.yaxis.set_visible(False) 

"(5) Bar graph with counts -- per brain area"
# Dataframe based on data from previous graph
bar_df = pd.DataFrame(graph_data)

# Parcellation -- Microelectrode
df_micro                = bar_df[bar_df['Category'] == 'Micro_Electrode']
micro_brain             = df_micro.groupby('Index')['Count'].sum()
micro_frequency         = df_micro.groupby('Amplitude')['Count'].sum()

# Parcellation -- Macroelectrode
df_macro                = bar_df[bar_df['Category'] == 'Macro_Electrode']
macro_brain             = df_macro.groupby('Index')['Count'].sum()
macro_frequency         = df_macro.groupby('Amplitude')['Count'].sum()

## Create bar graph per brain area
# Set default bar_width size
bar_width = 1

## Make sure datasets are the same size
# Add zeros if keys are empty
key_range   = range(1, len(brain_area_labels) * 2 + 1)  # From 1 to 4 (inclusive)

micro_brain_count        = {key: micro_brain.get(key, 0) for key in key_range}
macro_brain_count        = {key: macro_brain.get(key, 0) for key in key_range}

ax3.barh(list(micro_brain_count.keys()), list(micro_brain_count.values()), color='indianred', edgecolor='white', height=bar_width)
ax3.barh(list(macro_brain_count.keys()), list(macro_brain_count.values()), left=list(micro_brain_count.values()), color='darkorange', edgecolor='white', height=bar_width)

# Invert the x-axis
ax3.set_xlim(0, 100)
ax3.invert_xaxis()

ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.yaxis.set_visible(False)
ax3.set_xlabel('Reports (#)')

ax3.set_ylim(0,len(new_brain_numbers)*2 - 1)

# Save output as a .png and .svg file
plt.tight_layout()

output_path = '../outputs/'
plt.savefig(output_path + 'Figure5_Subcortical.svg', format='svg', dpi=600)
plt.savefig(output_path + 'Figure5_Subcortical.png', format='png', dpi=600)

plt.show()

## Final analysis
# Convert columns to numeric arrays
x_micro = np.array(Micro_array['Amplitude_midpoint'], dtype=float)
y_micro = np.array(Micro_array['Amplitude_width'], dtype=float)
x_macro = np.array(Macro_array['Amplitude_midpoint'], dtype=float)
y_macro = np.array(Macro_array['Amplitude_width'], dtype=float)

# Define log bins (1–10000 µA, 10 bins per decade)
bins_per_decade = 10
bin_edges = np.logspace(0, np.log10(10000), num=4*bins_per_decade + 1)

# Create figure and GridSpec
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[6, 16], width_ratios=[1, 3], hspace=0.3, wspace=0.3)

# --- 1. Top-left (empty)
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')

# --- 2. Top-right: Histogram of Amplitude_midpoint
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(x_micro, bins=bin_edges, color='indianred', alpha=0.7)
ax2.hist(x_macro, bins=bin_edges, color='darkorange', alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlim(1, 10000)
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Amplitude Midpoint')
ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()


# --- 3. Bottom-left: Histogram of Amplitude_width (horizontal, mirrored)
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(y_micro, bins=bin_edges, color='indianred', alpha=0.7, orientation='horizontal')
ax3.hist(y_macro, bins=bin_edges, color='darkorange', alpha=0.7, orientation='horizontal')
ax3.set_yscale('log')
ax3.set_ylim(1, 10000)
ax3.set_xlabel('Count')
ax3.set_title('Distribution of Amplitude Width')

# Mirror histogram by inverting x-axis
ax3.invert_xaxis()

# Remove y-axis ticks and labels
ax3.set_yticks([])

# Show x-axis ticks
ax3.xaxis.set_ticks_position('bottom')

# Remove gridlines
ax3.grid(False)

# --- 4. Bottom-right: Scatter + regression (log–log)
ax4 = fig.add_subplot(gs[1, 1])  # Only bottom-right cell, does NOT overlap ax2
ax4.scatter(x_micro, y_micro, color='indianred', s=15, alpha=0.7)
ax4.scatter(x_macro, y_macro, color='darkorange', s=15, alpha=0.7)

# Filter out zeros or negatives before log-transform
mask_micro = (x_micro > 0) & (y_micro > 0)
mask_macro = (x_macro > 0) & (y_macro > 0)
log_x_micro, log_y_micro = np.log10(x_micro[mask_micro]), np.log10(y_micro[mask_micro])
log_x_macro, log_y_macro = np.log10(x_macro[mask_macro]), np.log10(y_macro[mask_macro])

# Regression lines
coeff_micro = np.polyfit(log_x_micro, log_y_micro, 1)
coeff_macro = np.polyfit(log_x_macro, log_y_macro, 1)
x_vals = np.logspace(0, 4, 100)
ax4.plot(x_vals, 10**np.poly1d(coeff_micro)(np.log10(x_vals)), color='indianred', linestyle='--', linewidth=1.5, alpha=0.8)
ax4.plot(x_vals, 10**np.poly1d(coeff_macro)(np.log10(x_vals)), color='darkorange', linestyle='--', linewidth=1.5, alpha=0.8)

# Regression coefficients text
textstr = (f"Micro: slope={coeff_micro[0]:.3f}, intercept={coeff_micro[1]:.3f}\n"
           f"Macro: slope={coeff_macro[0]:.3f}, intercept={coeff_macro[1]:.3f}")
ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# Axis formatting
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlim(1, 10000)
ax4.set_ylim(1, 10000)
ax4.set_xlabel('Amplitude Midpoint (µA)')
ax4.set_ylabel('Amplitude Width (µA)')
ax4.set_title('Amplitude Midpoint vs Width')
ax4.set_xticks([1, 10, 100, 1000, 10000])
ax4.set_yticks([1, 10, 100, 1000, 10000])
ax4.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax4.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
ax4.ticklabel_format(style='plain', axis='both')
ax4.xaxis.set_minor_locator(ticker.NullLocator())
ax4.yaxis.set_minor_locator(ticker.NullLocator())
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

plt.tight_layout()
plt.show()

