# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:26:27 2025

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from utils import create_binary_array
from matplotlib.ticker import LogLocator
from matplotlib.ticker import FixedLocator
import matplotlib.ticker as ticker

file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Optogenetic_Selection')

print('Number of studies: ' + str(len(df['Title'].unique())))

"(1) Create the main figure with specifications"
# Create a main figure
fig = plt.figure(figsize=(8.5, 4), dpi=600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(2, 4)

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Top left
ax2 = fig.add_subplot(gs[0, 1])  # Top middle
ax3 = fig.add_subplot(gs[0, 2])  # Top right
ax4 = fig.add_subplot(gs[1, 0])  # Bottom left
ax5 = fig.add_subplot(gs[1, 1])  # Bottom middle
ax6 = fig.add_subplot(gs[1, 2])  # Bottom right

"(2) Serotype Subfigure"
# Get a pie chart of the serotypes
# Split the string on commas
df['Viral vector'] = df['Viral vector'].str.split(',')
df_exploded  = df.explode('Viral vector').reset_index(drop=True)
vector_sizes = df_exploded['Viral vector'].value_counts()

labels = ['Lenti', 'AAV5', 'AAV2/5', 'AAV8', 'AAV1', 'AAV9','Other']
sizes  = [vector_sizes.values[1], vector_sizes.values[0], vector_sizes.values[2], vector_sizes.values[3], vector_sizes.values[4], vector_sizes.values[5], vector_sizes.values[6::].sum()]
colors = ['slateblue', 'saddlebrown', 'chocolate', 'sandybrown', 'peachpuff','linen','silver']

ax1.pie(sizes, colors=colors, startangle=90,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax1.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)

"(3) Opsin Subfigure"
# Get a pie chart of the opsins
# Split the string on commas
df['Opsin']  = df['Opsin'].str.split(',')
df_opsin     = df.explode('Opsin').reset_index(drop=True)
opsin_sizes = df_opsin['Opsin'].value_counts()

labels = ['ChR2', 'C1V1', 'ArchT', 'Jaws', 'Other']
sizes  = [opsin_sizes.values[0], opsin_sizes.values[1], opsin_sizes.values[2], opsin_sizes.values[3], opsin_sizes.values[4::].sum()]
colors = ['cyan', 'yellowgreen', 'gold', 'crimson', 'silver']

ax2.pie(sizes, colors=colors, startangle=90,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax2.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax2.add_artist(centre_circle)

"(4) Amplitude Subfigure"
# Amplitude figure for optogenetics
df_mWmm2   = df[df['Amplitude_unit'].str.contains("mW/mm2", na=False)]
df_mW      = df[df['Amplitude_unit'].str.strip() == "mW"]
df_unclear = df[df['Amplitude_unit'].str.contains("unclear", na=False)]

# Create a new column with arrays from Lower_Value to Higher_Value
df_mWmm2.loc[:, 'Range'] = df_mWmm2.apply(
    lambda row: np.arange(int(row['Amplitude_lower']), int(row['Amplitude_higher']) + 1), axis=1)
df_mW.loc[:, 'Range'] = df_mW.apply(
    lambda row: np.arange(int(row['Amplitude_lower']), int(row['Amplitude_higher']) + 1), axis=1)

# Create binary arrays
df_mWmm2.loc[:, 'BinaryArrays'] = df_mWmm2['Range'].apply(
    create_binary_array, start_value=0, end_value=2000)
df_mW.loc[:, 'BinaryArrays'] = df_mW['Range'].apply(
    create_binary_array, start_value=0, end_value=2000)

df_mWmm2['Amplitude_midpoint'] = (df_mWmm2['Amplitude_lower'] + df_mWmm2['Amplitude_higher']) / 2
df_mWmm2['Amplitude_width']    = df_mWmm2['Amplitude_higher'] - df_mWmm2['Amplitude_lower']

mWmm2_med_midpoint  = df_mWmm2['Amplitude_midpoint'].median()
mWmm2_med_width     = df_mWmm2['Amplitude_width'].median()

df_mW['Amplitude_midpoint'] = (df_mW['Amplitude_lower'] + df_mW['Amplitude_higher']) / 2
df_mW['Amplitude_width']    = df_mW['Amplitude_higher'] - df_mW['Amplitude_lower']

mW_med_midpoint = df_mW['Amplitude_midpoint'].median()
mW_med_width    = df_mW['Amplitude_width'].median()

collapsed_mWmm2 = np.sum(np.array(df_mWmm2['BinaryArrays'].tolist()), axis=0)
collapsed_mW = np.sum(np.array(df_mW['BinaryArrays'].tolist()), axis=0)

# --- Collapse binary arrays
collapsed_mWmm2 = np.sum(np.array(df_mWmm2['BinaryArrays'].tolist()), axis=0)
collapsed_mW    = np.sum(np.array(df_mW['BinaryArrays'].tolist()), axis=0)

# --- Define amplitude values (skip 0)
amps = np.arange(1, len(collapsed_mWmm2))
mWmm2_vals = collapsed_mWmm2[1:]
mW_vals    = collapsed_mW[1:]

# --- Define logarithmic bins (1–2000, 10 bins per decade)
bins_per_decade = 10
bin_edges = np.logspace(0, np.log10(2000), num=3*bins_per_decade + 1)  # up to 2000 → ~3 decades

# --- Compute binned averages, filling empty bins with the closest value
def compute_binned_values(vals, amps, bin_edges):
    binned = []
    for i in range(len(bin_edges) - 1):
        mask = (amps >= bin_edges[i]) & (amps < bin_edges[i + 1])
        if np.any(mask):
            avg = vals[mask].mean()
        else:
            # use closest amplitude to bin center
            bin_center = np.sqrt(bin_edges[i] * bin_edges[i + 1])
            closest_idx = np.argmin(np.abs(amps - bin_center))
            avg = vals[closest_idx]
        binned.append(avg)
    return np.array(binned)

mWmm2_binned = compute_binned_values(mWmm2_vals, amps, bin_edges)
mW_binned    = compute_binned_values(mW_vals, amps, bin_edges)

# --- Compute bar centers and widths
centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean
widths  = bin_edges[1:] - bin_edges[:-1]

# --- Plot bars
ax3.bar(centers, mWmm2_binned, width=widths, align='center', color='indianred', edgecolor='none', zorder=2)
ax3.bar(centers, mW_binned,    width=widths, align='center', color='darkorange', edgecolor='none', alpha=0.8, zorder=3)

# --- Overlay median markers
ax3.scatter(mWmm2_med_midpoint, 17, s=50, facecolors='white', edgecolors='indianred', linewidths=2, zorder=5)
ax3.scatter(mW_med_midpoint,    19, s=50, facecolors='white', edgecolors='darkorange', linewidths=2, zorder=5)
ax3.axvline(x=mWmm2_med_midpoint, ymin=0, ymax=18, color='#d8dcd6', linestyle='--', linewidth=1)
ax3.axvline(x=mW_med_midpoint,   ymin=0, ymax=18, color='#d8dcd6', linestyle='--', linewidth=1)

# --- Style and labels
ax3.set_xscale('log')
ax3.set_xlim(1, 2000)
ax3.set_xlabel('Amplitude (mW/mm²)')
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()
ax3.set_ylabel('Reports (#)')
ax3.set_ylim(0, 20)
ax3.spines['left'].set_visible(False)
ax3.spines['top'].set_visible(False)

# --- Ticks and formatting
ax3.set_xticks([1, 10, 100, 1000])
ax3.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax3.ticklabel_format(style='plain', axis='x')
ax3.xaxis.set_minor_locator(ticker.NullLocator())

# --- Add inset pie chart (unchanged)
pie_sizes  = [len(df_mW), len(df_mWmm2), len(df_unclear)]
pie_colors = ['darkorange', 'indianred', 'silver']

inset_ax = ax3.inset_axes([-0.1, 0.7, 0.3, 0.3])
inset_ax.pie(
    pie_sizes,
    colors=pie_colors,
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},)

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
inset_ax.add_artist(centre_circle)

# Plot mean values for higher and lower - mW/mm2
mWmm2_mean_low    = df_mWmm2['Amplitude_lower'].mean()
mWmm2_median_low  = df_mWmm2['Amplitude_lower'].median()

mWmm2_mean_high   = df_mWmm2['Amplitude_higher'].mean()
mWmm2_median_high = df_mWmm2['Amplitude_higher'].median()

# Plot mean values for higher and lower - mW/mm2
mW_mean_low    = df_mW['Amplitude_lower'].mean()
mW_median_low  = df_mW['Amplitude_lower'].median()

mW_mean_high   = df_mW['Amplitude_higher'].mean()
mW_median_high = df_mW['Amplitude_higher'].median()

"(5) Frequency Subfigure"
# Fourth figure -> Frequencies (include continuous)
# Separate continuous from everything else
df_continuous = df[df['Frequency'].str.contains("continuous", na=False)]
df_filtered   = df[~df['Frequency'].str.contains("continuous", na=False)]

# Separate single values from ranges and multiple values
df_singular    = df_filtered[~df_filtered['Frequency'].astype(str).str.contains(r'[,-]', regex=True)]
singular_array = df_singular['Frequency'] 
singular_frequency_count = singular_array.value_counts()
sing_values = singular_frequency_count.keys().to_numpy()
sing_counts = singular_frequency_count.values

# Separate range and multiple values
df_multiple = df_filtered[df_filtered['Frequency'].astype(str).str.contains(r'[,]', regex=True)]
df_range    = df_filtered[df_filtered['Frequency'].astype(str).str.contains(r'[-]', regex=True)]
df_range    = df_range[~df_range['Frequency'].astype(str).str.contains(r'[,]', regex=True)]

# Filter multiple frequency data
multiple_values = df_multiple['Frequency'].dropna().str.split(',').explode().str.strip().tolist()
string_array    = np.array(multiple_values)
multiple_array  = pd.to_numeric(string_array, errors='coerce')
mul_values, mul_counts = np.unique(multiple_array, return_counts=True)

# Filter range frequency data
# Split the column at '-' into two new columns
split_df = df_range['Frequency'].str.split('-', expand=True)

# Convert to numeric
low_values = pd.to_numeric(split_df[0])
high_values = pd.to_numeric(split_df[1])

# Convert to NumPy arrays if needed
low_range  = low_values.to_numpy()
high_range = high_values.to_numpy()

# Create a new column with arrays from Lower_Value to Higher_Value
# Create DataFrame with one column
df_range = pd.DataFrame({'Frequency_Lower' : low_range, 'Frequency_Higher' : high_range})
df_range.loc[:, 'Range'] = df_range.apply(
    lambda row: np.arange(int(row['Frequency_Lower']), int(row['Frequency_Higher']) + 1), axis=1)

df_range.loc[:, 'BinaryArrays'] = df_range['Range'].apply(
    create_binary_array, start_value=0, end_value=350)

collapsed_range = np.sum(np.array(df_range['BinaryArrays'].tolist()), axis=0)

# Create a common x-axis (all unique x-values from both datasets)
all_x_vals = np.unique(np.concatenate((sing_values, mul_values)))

# Align the data with zero-padding for missing x-values
sing_counts = np.array([sing_counts[np.where(sing_values == x)[0][0]] if x in sing_values else 0 for x in all_x_vals])
mul_counts  = np.array([mul_counts[np.where(mul_values == x)[0][0]] if x in mul_values else 0 for x in all_x_vals])

## Create the bar graph with frequency data
ax5.bar(all_x_vals, sing_counts, color='royalblue', edgecolor = 'white', linewidth=0.5, width=15, zorder=4)
ax5.bar(all_x_vals, mul_counts, bottom=sing_counts, color='darkviolet', edgecolor = 'white', linewidth=0.5, width=15, zorder=2)
ax5.bar(np.arange(0,350+1,step=1), collapsed_range, color='indianred', edgecolor='none', width=10, alpha=1, zorder=2)

# Get the median values for the Frequency
combined_values  = np.concatenate([singular_array.to_numpy(), multiple_array])
median_frequency = np.median(combined_values)

# Remove the left and top spines
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.set_yticklabels([])
ax5.set_yticks([])
ax5.set_xticks([0, 100, 200, 300])
ax5.set_xlim(-10, 350)
ax5.set_ylim(0, 12)
ax5.set_xlabel('Frequency (Hz)')

ax5.scatter(median_frequency, 11, s=50, facecolors='white', edgecolors='royalblue', linewidths=2, zorder=5)
ax5.axvline(x=median_frequency, ymin=0, ymax=10, color='#d8dcd6', linestyle='--', linewidth=1)

# Get a subplot with pulse-width information
inset_ax2 = ax5.inset_axes([0.4, 0.5, 0.5, 0.5])  # [x, y, width, height] relative to the subplot

# Get the mean value for columns
def get_mean_or_value(x):
    if '-' in x:
        parts = x.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    else:
        return float(x)

df_filtered['Pulse_width'] = df_filtered['Pulse_width'].str.replace(' ms', '', regex=False)
df_filtered['Pulse_width'] = df_filtered['Pulse_width'].apply(get_mean_or_value)
median_pulse = df_filtered['Pulse_width'].median()
pulse_counts = df_filtered['Pulse_width'].value_counts()

inset_ax2.bar(pulse_counts.keys(), pulse_counts.values, color = '#82cafc', width = 3)
inset_ax2.spines['top'].set_visible(False)
inset_ax2.spines['right'].set_visible(False)
inset_ax2.set_xlim(-10, 50)
inset_ax2.set_ylim(0, 10)

inset_ax2.scatter(median_pulse, 8, s=50, facecolors='white', edgecolors='#82cafc', linewidths=2, zorder=5)
inset_ax2.axvline(x=median_pulse, ymin=0, ymax=8, color='#d8dcd6', linestyle='--', linewidth=1)

## Pie chart with types of stimulation
# Figure -> Frequency or continuous
labels = ['Continuous', 'Single', 'Multiple', 'Range']
sizes  = [len(df_continuous), len(df_singular), len(df_multiple), len(df_range)]
colors = ['chocolate', 'royalblue', 'darkviolet','indianred']

ax4.pie(sizes, colors=colors, startangle=90,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax4.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax4.add_artist(centre_circle)

## Final bar chart with time intervals
df['Timespan'] = df['Timespan'].str.replace(' ms', '', regex=False)

df_continuous = df[df['Frequency'].str.contains("continuous", na=False)]
df_filtered   = df[~df['Frequency'].str.contains("continuous", na=False)]

# preparing data (taking the mean of range values)
df_continuous['Timespan'] = df_continuous['Timespan'].apply(get_mean_or_value)
df_filtered['Timespan']   = df_filtered['Timespan'].apply(get_mean_or_value)

con_values  = df_continuous['Timespan'].value_counts()
freq_values = df_filtered['Timespan'].value_counts()

# Create the actual figure
ax6.bar(con_values.keys(), con_values.values, color='chocolate', edgecolor = 'white', linewidth=0.25, width=150, zorder=4)
ax6.bar(freq_values.keys(), freq_values.values, color='royalblue', edgecolor = 'white', linewidth=0.25, width=150, zorder=4)
ax6.set_xlim(0, 2000)
ax6.set_xticks([0, 1000, 2000])
ax6.yaxis.tick_right()
ax6.spines['top'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.set_ylim(0,12)

con_median = df_continuous['Timespan'].median()
con_freq   = df_filtered['Timespan'].median()

ax6.scatter(con_median, 11, s=50, facecolors='white', edgecolors='chocolate', linewidths=2, zorder=5)
ax6.axvline(x=con_median, ymin=0, ymax=11, color='#d8dcd6', linestyle='--', linewidth=1)

ax6.scatter(con_freq, 11, s=50, facecolors='white', edgecolors='royalblue', linewidths=2, zorder=5)
ax6.axvline(x=con_freq, ymin=0, ymax=11, color='#d8dcd6', linestyle='--', linewidth=1)


## Save the figure in .svg and .png
# Save output as a .png and .svg file
output_path = '../outputs/'
plt.savefig(output_path + 'Figure8.svg', format='svg', dpi=600)
plt.savefig(output_path + 'Figure8.png', format='png', dpi=600)

plt.show()