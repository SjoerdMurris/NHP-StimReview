# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:41:02 2024

@author: Sjoerd Murris
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils import create_binary_array

"(0) specify some of the parameters"
# Specify the colours used for the plots
single_frequency_colour   = 'royalblue'
multiple_frequency_colour = 'darkviolet'
range_frequency_colour    = 'crimson'
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
fig = plt.figure(figsize=(8.5, 4), dpi = 600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(1, 3)  # Top rows shorter than bottom

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Left
ax2 = fig.add_subplot(gs[0, 1])  # Middle
ax3 = fig.add_subplot(gs[0, 2])  # Right

ax1.pie(sizes, colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, startangle=90)
ax1.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)

# Add subplot pie charts: for micro-electrode data
# Specify micro-electrode statistics
micro_single   = (df_single['Electrode_type'] == 'Microelectrode').sum()
micro_multiple = (df_multiple['Electrode_type'] == 'Microelectrode').sum()
micro_range    = (df_range['Electrode_type'] == 'Microelectrode').sum()
micro_sp       = (df_singlepulses['Electrode_type'] == 'Microelectrode').sum()
micro_sec      = (df_seconds['Electrode_type'] == 'Microelectrode').sum()
micro_long     = (df_long['Electrode_type'] == 'Microelectrode').sum()
micro_unclear  = (df_unclear['Electrode_type'] == 'Microelectrode').sum()

micro_sizes    = [micro_single, micro_multiple, micro_range, micro_sp, micro_sec, micro_long, micro_unclear]

inset_ax = ax1.inset_axes([-0.15, -0.15, 0.4, 0.4])  # [x, y, width, height] relative to the subplot
inset_ax.pie(
    micro_sizes,
    colors=colors,
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
)

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
inset_ax.add_artist(centre_circle)

# Add subplot pie charts: for macro-electrode data
# Specify macro-electrode statistics
macro_single   = (df_single['Electrode_type'] == 'Macroelectrode').sum()
macro_multiple = (df_multiple['Electrode_type'] == 'Macroelectrode').sum()
macro_range    = (df_range['Electrode_type'] == 'Macroelectrode').sum()
macro_sp       = (df_singlepulses['Electrode_type'] == 'Macroelectrode').sum()
macro_sec      = (df_seconds['Electrode_type'] == 'Macroelectrode').sum()
macro_long     = (df_long['Electrode_type'] == 'Macroelectrode').sum()
macro_unclear  = (df_unclear['Electrode_type'] == 'Macroelectrode').sum()

macro_sizes    = [macro_single, macro_multiple, macro_range, macro_sp, macro_sec, macro_long, macro_unclear]

inset_ax2 = ax1.inset_axes([0.225, -0.15, 0.4, 0.4])  # [x, y, width, height] relative to the subplot
inset_ax2.pie(
    macro_sizes,
    colors=colors,
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
)

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
inset_ax2.add_artist(centre_circle)

# Add subplot pie charts: for optogenetics data
# Specify optogenetic statistics
opto_single   = (df_single['Type_stimulation'] == 'optogenetic').sum()
opto_multiple = (df_multiple['Type_stimulation'] == 'optogenetic').sum()
opto_range    = (df_range['Type_stimulation'] == 'optogenetic').sum()
opto_sp       = (df_singlepulses['Type_stimulation'] == 'optogenetic').sum()
opto_sec      = (df_seconds['Type_stimulation'] == 'optogenetic').sum()
opto_long     = (df_long['Type_stimulation'] == 'optogenetic').sum()
opto_unclear  = (df_unclear['Type_stimulation'] == 'optogenetic').sum()

opto_sizes    = [opto_single, opto_multiple, opto_range, opto_sp, opto_sec, opto_long, opto_unclear]

inset_ax3 = ax1.inset_axes([0.6, -0.15, 0.4, 0.4])  # [x, y, width, height] relative to the subplot
inset_ax3.pie(
    opto_sizes,
    colors=colors,
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
)

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
inset_ax3.add_artist(centre_circle)

"(6) Bar graph with counts -- per interval"
# Set default bar_width size
ms_x_interval = [-100, 1050]
s_x_interval  = [-4900, 66000]
y_interval    = [0, 65] 

# Create binary arrays
single_values = df_single['Timespan'].explode()
single_values = single_values.astype(int)
value_counts  = single_values.value_counts()

# Extract all of the multiple timespan values
# Split the column values by ',' and flatten them into a single array
multiple_values = df_multiple['Timespan'].str.split(',').explode()
multiple_values = multiple_values.astype(int)

# Use pd.Series to apply value_counts
multi_counts = multiple_values.value_counts()

## Make sure single and multiple timespan have the same lenght
key_range                  = range(1, 60000+1) 
multiple_timespan_count    = {key: multi_counts.get(key, 0) for key in key_range}
single_timespan_count      = {key: value_counts.get(key, 0) for key in key_range}

multi_array_x = np.array(list(multiple_timespan_count.keys()))
multi_array_y = np.array(list(multiple_timespan_count.values()))

single_array_x = np.array(list(single_timespan_count.keys()))
single_array_y = np.array(list(single_timespan_count.values()))

# Use masks to only plot the first 1000 ms
mask = single_array_x <= 1000

# Filter both arrays using the mask
single_filtered_values = single_array_x[mask]
single_filtered_counts = single_array_y[mask]
multi_filtered_counts  = multi_array_y[mask]

# Get the mean/median values for single pulses
df_single['Timespan'] = df_single['Timespan'].astype(int)
df_single['Timespan'].median()
mean_single   = df_single['Timespan'].mean()
median_single = df_single['Timespan'].median()

micro_df     = df_single[df_single['Electrode_type'] == 'Microelectrode']
micro_mean   = micro_df['Timespan'].mean()
micro_median = micro_df['Timespan'].median()

macro_df     = df_single[df_single['Electrode_type'] == 'Macroelectrode']
macro_mean   = macro_df['Timespan'].mean()
macro_median = macro_df['Timespan'].median()

opto_df     = df_single[df_single['Type_stimulation'] == 'optogenetic']
opto_mean   = opto_df['Timespan'].mean()
opto_median = opto_df['Timespan'].median()

# Plot the range values
df_range['Binary_ms'] = df_range['Timespan_Range'].apply(create_binary_array, start_value=0, end_value=1000)
collapsed_array       = np.sum(np.array(df_range['Binary_ms'].tolist()), axis=0)

ax_inv = ax2.twinx()  # Create twin axes sharing the x-axis
ax_inv.bar(np.arange(0,1000+1,step=1), collapsed_array, color='mistyrose', edgecolor='none', width=10, alpha=0.4, zorder=1)
ax_inv.set_ylim([0, y_interval[1]*2])

# Plot single timespan data and the median
ax2.bar(single_filtered_values, single_filtered_counts, width=20, color=single_frequency_colour, alpha=1, edgecolor='none', zorder=2)
ax2.axvline(x=micro_median, color='black', linestyle='--', zorder=5)
ax2.axvline(x=macro_median, color='black', linestyle='--', zorder=5)

# Plot the multiple timespan data on top of the single timespan data
ax2.bar(single_filtered_values, multi_filtered_counts, bottom=single_filtered_counts, width=20, color=multiple_frequency_colour, alpha=1, zorder=2)

# Plot the single pulse values as a single bar
ax2.bar(ms_x_interval[0]/2, len(df_singlepulses), color=pulse_frequency_colour, edgecolor='white', width=20, zorder=3)

# Adjust the z-order of the axes
ax2.set_zorder(ax_inv.get_zorder() + 1)  # Bring primary axis to the front
ax2.patch.set_visible(False)          # Make the primary axis background transparent

# Bar graph parameters
ax2.set_xlim(ms_x_interval)
ax2.set_xticks([0, 250, 500, 750, 1000])
ax2.set_xticklabels(["0", "250", "500", "750", "1000"])
ax2.set_ylim(y_interval)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlabel('Timespan (ms)')
ax2.set_ylabel('Reports (#)', color=single_frequency_colour)

ax_inv.yaxis.set_visible(False)
ax_inv.spines['right'].set_visible(False)
ax_inv.spines['top'].set_visible(False)


"(7) Plot the second bar graph, with data > 1s"
# Use masks to plot everything above 1s
mask2 = single_array_x > 1000

# Filter both arrays using the mask
single_filtered_values = single_array_x[mask2]
single_filtered_counts = single_array_y[mask2]
multi_filtered_counts  = multi_array_y[mask2]

ax3.bar(single_filtered_values, single_filtered_counts, width=2000, color=single_frequency_colour, alpha=1, zorder=2)
ax3.bar(single_filtered_values, multi_filtered_counts, bottom=single_filtered_counts, width=2000, color=multiple_frequency_colour, alpha=1, zorder=2)

# Plot the range values
df_range['Binary_s'] = df_range['Timespan_Range'].apply(create_binary_array, start_value=1001, end_value=60000)
collapsed_array      = np.sum(np.array(df_range['Binary_s'].tolist()), axis=0)

ax3_inv = ax3.twinx()  # Create twin axes sharing the x-axis
ax3_inv.bar(np.arange(1000,60000,step=1), collapsed_array, color='mistyrose', edgecolor='none', width=10, alpha=0.4, zorder=1)
ax3_inv.set_ylim([0, y_interval[1]*2])
ax3_inv.set_ylabel('Reports (#)', color=range_frequency_colour)

# Adjust the z-order of the axes
ax3.set_zorder(ax3_inv.get_zorder() + 1)  # Bring primary axis to the front
ax3.patch.set_visible(False)          # Make the primary axis background transparent

# Plot the seconds as a single bar
ax3.bar(s_x_interval[0]/2, len(df_seconds), color=seconds_frequency_colour, edgecolor='white', width=2000)
ax3.bar(63000, len(df_long), color=long_frequency_colour, edgecolor='white', width=2000)
ax3.set_ylim(y_interval)
ax3.set_xlim(s_x_interval)
ax3.set_xticks([1000, 10000, 20000, 40000, 60000])
ax3.set_xticklabels(["1", "10", "20", "40", "60"])
ax3.set_xlabel('Timespan (s)')

# Hide some of the axes
ax3.yaxis.set_visible(False)  # Hide y-axis ticks and labels
ax3.spines['left'].set_visible(False)  # Hide the left spine (y-axis line)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3_inv.spines['top'].set_visible(False)
ax3_inv.spines['left'].set_visible(False)

# Change axis colours
ax2.tick_params(axis='y', labelcolor=single_frequency_colour)
ax2.spines['left'].set_color(single_frequency_colour)
ax3_inv.tick_params(axis='y', labelcolor=range_frequency_colour)
ax3_inv.spines['right'].set_color(range_frequency_colour)

# Save output as a .png and .svg file
plt.tight_layout()

output_path = '../outputs/'
plt.savefig(output_path + 'Figure5.svg', format='svg', dpi=600)
plt.savefig(output_path + 'Figure5.png', format='png', dpi=600)

plt.show()
