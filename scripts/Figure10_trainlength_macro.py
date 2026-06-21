# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:41:02 2024

@author: Sjoerd Murris
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as ticker

from utils import create_binary_array

"(0) specify some of the parameters"
# Specify the colours used for the plots
single_frequency_colour   = 'royalblue'
multiple_frequency_colour = 'darkviolet'
range_frequency_colour    = 'indianred'
pulse_frequency_colour    = 'gold'
seconds_frequency_colour  = 'darkorange'
long_frequency_colour     = 'chocolate'
unclear_frequency_colour  = 'black'

"(1) read in the data from the excel file"
# Turn off errors related to copy warnings in python
pd.options.mode.chained_assignment = None

file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Study_Selection')

"(2) clean data: get rid of no access articles"
df_cleaned = df[df['Journal'] != 'NO ACCESS']

"(3) Parse the timespan data"
df_singlepulses     = df_cleaned[df_cleaned['Timespan'].str.contains('single pulse', case=False, na=False)]
df_long             = df_cleaned[df_cleaned['Timespan'].str.contains('continuous', case=False, na=False) |
                             df_cleaned['Timespan'].str.contains('min', case=False, na=False) |
                             df_cleaned['Timespan'].str.contains('day', case=False, na=False)]
df_seconds          = df_cleaned[df_cleaned['Timespan'].str.contains('variable', case=False, na=False) |
                                 df_cleaned['Timespan'].str.contains('second', case=False, na=False) |
                                 df_cleaned['Timespan'].str.contains('depends', case=False, na=False)]
df_miliseconds      = df_cleaned[df_cleaned['Timespan'].str.contains('ms', case=False, na=False)]
df_unclear          = df_cleaned[df_cleaned['Timespan'].str.contains('unclear', case=False, na=False)]

"(4) Split up the miliseconds dataframe in < 1s and > 1s"
# Remove the string ' ms' from the column
df_miliseconds['Timespan'] = df_miliseconds['Timespan'].str.replace(' ms', '', regex=False)

# Get the range values as a separate dataframe
df_range                           = df_miliseconds[df_miliseconds['Timespan'].str.contains('-', case=False, na=False)]
df_range[['Lower', 'Higher']]      = df_range['Timespan'].str.extract(r'(\d+)-(\d+)')

# Ensure columns are numeric
df_range['Lower']                  = pd.to_numeric(df_range['Lower'], errors='coerce')
df_range['Higher']                 = pd.to_numeric(df_range['Higher'], errors='coerce')
df_range['Timespan_Range']         = df_range.apply(lambda row: list(range(row['Lower'], row['Higher'] + 1)), axis=1)

df_multiple                        = df_miliseconds[df_miliseconds['Timespan'].str.contains(',', case=False, na=False)] 

# Use pd.to_numeric to check for numeric values
df_single = df_miliseconds[pd.to_numeric(df_miliseconds['Timespan'], errors='coerce').notna()]

"(5) Create a pie chart showing the distributions of different timespans"
# Create a pie chart showing distribution of reporting frequencies
# Add labels and title
sizes    = [len(df_single), len(df_multiple), len(df_range), len(df_singlepulses), len(df_seconds), len(df_long), len(df_unclear)]
labels   = ['Single Interval', 'Multiple Intervals', 'Interval Range', 'Single Pulses', 'Seconds', 'Minutes', 'Unclear']

colors = [single_frequency_colour, multiple_frequency_colour, range_frequency_colour, pulse_frequency_colour, seconds_frequency_colour, long_frequency_colour, unclear_frequency_colour]

# Create a figure
fig = plt.figure(figsize=(8.5, 3), dpi = 600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(1, 2, width_ratios=[1,3])  # Top rows shorter than bottom

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Left
ax2 = fig.add_subplot(gs[0, 1])  # Middle

# Add subplot pie charts: for macro-electrode data
# Specify macro-electrode statistics
macro_single   = df_single[df_single['Electrode_type'] == 'Macroelectrode']
macro_multiple = df_multiple[df_multiple['Electrode_type'] == 'Macroelectrode']
macro_range    = df_range[df_range['Electrode_type'] == 'Macroelectrode']
macro_sp       = (df_singlepulses['Electrode_type'] == 'Macroelectrode').sum()
macro_sec      = (df_seconds['Electrode_type'] == 'Macroelectrode').sum()
macro_long     = (df_long['Electrode_type'] == 'Macroelectrode').sum()
macro_unclear  = (df_unclear['Electrode_type'] == 'Macroelectrode').sum()

macro_single_size   = len(macro_single)
macro_multiple_size = len(macro_multiple)
macro_range_size    = len(macro_range)

macro_sizes    = [macro_single_size, macro_multiple_size, macro_range_size, macro_sp, macro_sec, macro_long, macro_unclear]

ax1.pie(macro_sizes, colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, startangle=90)
ax1.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)

"(6) Bar graph with counts -- per interval"
# Create binary arrays
single_values = macro_single['Timespan'].explode()
single_values = single_values.astype(int)
value_counts  = single_values.value_counts()

# Extract all of the multiple timespan values
# Split the column values by ',' and flatten them into a single array
multiple_values = macro_multiple['Timespan'].str.split(',').explode()
multiple_values = multiple_values.astype(int)
multiple_counts = multiple_values.value_counts()

# Full vector of x-values
x = np.arange(0, 60001, step=1)

# Reindex so missing values become 0
single_aligned_counts   = value_counts.reindex(x, fill_value=0)
multiple_aligned_counts = multiple_counts.reindex(x, fill_value=0)

# Plot the range values
macro_range['Binary_ms'] = macro_range['Timespan_Range'].apply(create_binary_array, start_value=0, end_value=60000)
collapsed_array       = np.sum(np.array(macro_range['Binary_ms'].tolist()), axis=0)

# --- Define x-values (1–60000 ms)
x = np.arange(1, 60001)

# --- Align single and multiple counts
single_aligned_counts   = value_counts.reindex(x, fill_value=0)
multiple_aligned_counts = multiple_counts.reindex(x, fill_value=0)

# --- Define log bins (10 bins per decade)
bins_per_decade = 10
bin_edges = np.logspace(0, np.log10(60000), num=5*bins_per_decade + 1)  # 1–10–100–1k–10k–60k
centers   = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean
widths    = bin_edges[1:] - bin_edges[:-1]

# --- Helper function: sum counts in each bin (for single/multiple)
def compute_binned_sums(vals, x, bin_edges):
    vals = np.asarray(vals)
    x = np.asarray(x)
    n = min(len(vals), len(x))
    vals, x = vals[:n], x[:n]
    binned = []
    for i in range(len(bin_edges) - 1):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.any(mask):
            s = vals[mask].sum()
        else:
            # fallback: use closest available value
            bin_center = np.sqrt(bin_edges[i] * bin_edges[i+1])
            closest_idx = np.argmin(np.abs(x - bin_center))
            s = vals[closest_idx]
        binned.append(s)
    return np.array(binned)

# --- Helper function: mean per bin (for ranged data)
def compute_binned_means(vals, x, bin_edges):
    vals = np.asarray(vals)
    x = np.asarray(x)
    n = min(len(vals), len(x))
    vals, x = vals[:n], x[:n]
    binned = []
    for i in range(len(bin_edges) - 1):
        mask = (x >= bin_edges[i]) & (x < bin_edges[i+1])
        if np.any(mask):
            avg = vals[mask].mean()
        else:
            # fallback: use nearest value
            bin_center = np.sqrt(bin_edges[i] * bin_edges[i+1])
            closest_idx = np.argmin(np.abs(x - bin_center))
            avg = vals[closest_idx]
        binned.append(avg)
    return np.array(binned)

# --- Compute binned values
single_binned    = compute_binned_sums(single_aligned_counts.values, x, bin_edges)
multiple_binned  = compute_binned_sums(multiple_aligned_counts.values, x, bin_edges)
collapsed_binned = compute_binned_means(collapsed_array, x, bin_edges)

# --- Plot stacked bars
ax2.bar(centers, collapsed_binned, width=widths, align='center',
        color=range_frequency_colour, edgecolor='none', zorder=1)

ax2.bar(centers, single_binned, width=widths, align='center',
        color=single_frequency_colour, edgecolor='white', zorder=2)

ax2.bar(centers, multiple_binned, bottom=single_binned, width=widths, align='center',
        color=multiple_frequency_colour, edgecolor='white', zorder=3)

# --- Log x-axis, major ticks only
ax2.set_xscale('log')
ax2.set_xlim(1, 60000)
ticks  = [1, 10, 100, 1000, 10000, 60000]
labels = ['1 ms', '10 ms', '100 ms', '1 s', '10 s', '1 min']
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels)
ax2.xaxis.set_minor_locator(ticker.NullLocator())  # remove minor ticks

# --- y-axis and style
ax2.set_ylim(0, 160)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Reports (#)')
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)

# --- Median for single values
micro_median = single_values.median()  # keep in ms
ax2.axvline(x=micro_median, color='#d8dcd6', linestyle='--', linewidth=1)
ax2.scatter(micro_median, 120, s=100,
            facecolors='white', edgecolors=single_frequency_colour,
            linewidths=5, zorder=5)

# --- Midpoint and interval medians for ranged data
split_vals = macro_range['Timespan'].str.split('-', expand=True).apply(pd.to_numeric)
macro_range['Midpoint'] = split_vals.mean(axis=1)
macro_range['Interval'] = split_vals[1] - split_vals[0]

macro_midpoint_median = macro_range['Midpoint'].median()
macro_interval_median = macro_range['Interval'].median()

#ax2.hlines(y=125,
#           xmin=micro_midpoint_median - micro_interval_median/2,
#           xmax=micro_midpoint_median + micro_interval_median/2,
#           color=range_frequency_colour, linewidth=5)

ax2.scatter(macro_midpoint_median, 120, s=100, facecolors='white',
            edgecolors=range_frequency_colour, linewidths=5, zorder=5)

ax2.axvline(x=macro_midpoint_median, color='#d8dcd6', linestyle='--', linewidth=1)

# Save output as a .png and .svg file
plt.tight_layout()

output_path = '../outputs/'
plt.savefig(output_path + 'Figure9B.svg', format='svg', dpi=600)
plt.savefig(output_path + 'Figure9B.png', format='png', dpi=600)

plt.show()
