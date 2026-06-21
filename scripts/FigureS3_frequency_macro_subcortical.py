# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:32:11 2024

@author: Sjoerd Murris
"""

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import re
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation

from utils import create_binary_array


"(0) Specify general parameters for the figures"
data_label = 'subcortical' # Specify cortical or subcortical to plot

# Specify the colours used for the plots
single_frequency_colour   = 'royalblue'
multiple_frequency_colour = 'darkviolet'
range_frequency_colour    = 'indianred'
pulse_frequency_colour    = 'gold'
unclear_frequency_colour  = 'silver'

"(1) read in the data from the excel file"
# Turn off errors related to copy warnings in python
pd.options.mode.chained_assignment = None

file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Study_Selection')

"(2) parse the frequency stimulation data"
 # First sort by electrical/optogenetic stimulation
elec_df      = df[df['Type_stimulation'] == 'electrical']
cort_df      = elec_df[elec_df['Cortical_Subcortical'] == data_label]

# Exclude macro-electrode instances
micro_df     = cort_df[cort_df['Electrode_type'] == 'Macroelectrode']
sing_df      = micro_df[micro_df['Frequency'] == 'single pulses']

# Remove instances with single pulses or 'unclear'
df_filtered = micro_df[~micro_df['Frequency'].astype(str).str.contains('single pulses', case=False)]
df_filtered = df_filtered[~df_filtered['Frequency'].astype(str).str.contains('unclear', case=False)]

# Separate single values from ranges and multiple values
df_singular = df_filtered[~df_filtered['Frequency'].astype(str).str.contains(r'[,-]', regex=True)]
df_singular['Frequency_Numbers'] = df_singular['Frequency'].str.extractall(r'(\d+)').groupby(level=0).agg(lambda x: '.'.join(x))

# Separate range and multiple values
df_multiple = df_filtered[df_filtered['Frequency'].astype(str).str.contains(r'[,]', regex=True)]
df_range    = df_filtered[df_filtered['Frequency'].astype(str).str.contains(r'[-]', regex=True)]
df_range    = df_range[~df_range['Frequency'].astype(str).str.contains(r'[,]', regex=True)]
df_unclear  = micro_df[micro_df['Frequency'].astype(str).str.contains('unclear', case=True)]

# Create a pie chart showing distribution of reporting frequencies
# Add labels and title
sizes    = [len(df_singular), len(df_multiple), len(df_range), len(sing_df), len(df_unclear)]
labels   = ['Single Frequency', 'Multiple Frequencies', 'Frequency Range', 'Single Pulses', 'Unclear']

colors = [single_frequency_colour, multiple_frequency_colour, range_frequency_colour, pulse_frequency_colour, unclear_frequency_colour]

# Create a figure
fig = plt.figure(figsize=(10.5, 8), dpi = 600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(2, 2, height_ratios=[6, 16], width_ratios=[1,3])  # Top rows shorter than bottom

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Top left
ax2 = fig.add_subplot(gs[0, 1])  # Top right
ax3 = fig.add_subplot(gs[1, 0])  # Bottom left
ax4 = fig.add_subplot(gs[1, 1])  # Bottom right

ax1.pie(sizes, colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'}, startangle=90)
ax1.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)

"(4) Create a plot that has frequency data from the four categories"
# Add a column that contains Frequency values for plots
df_singular['Frequency_Singular']  = df_singular['Frequency'].apply(lambda s: float(re.findall(r'\d+\.\d+|\d+', s)[0]) if re.findall(r'\d+\.\d+|\d+', s) else float('nan'))
df_multiple['Frequency_Multiple']  = df_multiple['Frequency'].str.findall(r'\d+')
df_range[['Upper', 'Lower']]       = df_range['Frequency'].str.extract(r'(\d+)-(\d+)')
df_range['Frequency_Range']        = df_range[['Upper', 'Lower']].apply(lambda x: [int(num) for num in x if pd.notna(num)], axis=1)
sing_df['Frequency_Single']        = -50 # arbitrary value

# Concatonate and group by brain area
sub_combined = pd.concat([df_singular, df_multiple, df_range, sing_df], ignore_index=True)
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
cort_miss_labels = ['LPons','MMid','PrT','EpiThal','LVPal']

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
    
        # Include frequency measure if not empty (NaN)
        if filtered_df['Frequency_Singular'].notna().any():
            value_counts = filtered_df['Frequency_Singular'].value_counts()
            for value, count in value_counts.items():
                ax4.scatter(value, y_counter, s=15*count, label=None, edgecolors='none', zorder=3, color=single_frequency_colour)  # Set label=None to avoid duplicate labels
           
                # Data to add for bar graphs
                row_graph = {'Index': y_counter, 'Category': 'Singular', 'Brain_Parc': label, 'Frequency': value, 'Count': count}
                graph_data.append(row_graph)
            
        # Include multiple frequency measure if not empty (NaN)
        if filtered_df['Frequency_Multiple'].notna().any():
            indices = filtered_df[filtered_df['Frequency_Multiple'].notna()]
            values = np.concatenate(indices['Frequency_Multiple'].values)
            unique_values, counts = np.unique(values, return_counts=True)
        
            # Data to add for bar graphs (per area)
            row_graph = {'Index': y_counter, 'Category': 'Multiple_Compiled', 'Brain_Parc': label, 'Frequency': 'unkown', 'Count': len(indices['Frequency_Multiple'])}
            graph_data.append(row_graph)
        
            for value, count in zip(unique_values, counts):
                ax4.scatter(int(value), y_counter, s=15*count, label=None, edgecolors='none', zorder=4, color=multiple_frequency_colour)  # Set label=None to avoid duplicate labels
            
                # Data to add for bar graphs
                row_graph = {'Index': y_counter, 'Category': 'Multiple', 'Brain_Parc': label, 'Frequency': int(value), 'Count': count}
                graph_data.append(row_graph)
            
        # Include frequency range measure if not empty (NaN)
        if filtered_df['Frequency_Range'].notna().any():
            indices = filtered_df[filtered_df['Frequency_Range'].notna()]
            values = np.concatenate(indices['Frequency_Range'].values)
            unique_values, counts = np.unique(values, return_counts=True)
        
            # Data to add for bar graphs (per area)
            row_graph = {'Index': y_counter, 'Category': 'Range_Compiled', 'Brain_Parc': label, 'Frequency': 'unkown', 'Count': len(indices['Frequency_Range'])}
            graph_data.append(row_graph)
               
            # Draw the unique points (scale with counts)
            for value, count in zip(unique_values, counts):
                    ax4.scatter(int(value), y_counter, s=15*count, label=None, edgecolors='none', zorder=4, color=range_frequency_colour)  # Set label=None to avoid duplicate labels
               
            # Draw opaque lines between the ranges that were tested
            for i, row in enumerate (indices['Frequency_Range']):
                    ax4.plot(row, [y_counter, y_counter], color=range_frequency_colour, alpha=0.3, linewidth=2)
        
            row_graph = {'Index': y_counter, 'Category': 'Range', 'Brain_Parc': label, 'Frequency': np.arange(row[0], row[1] + 1), 'Count': count}
            graph_data.append(row_graph)           

        # Include single pulses data if not empty (NaN)    
        if filtered_df['Frequency_Single'].notna().any():
            count = filtered_df['Frequency_Single'].count()
            values = filtered_df['Frequency_Single'].dropna().unique()
            ax4.scatter(values[0], y_counter, s=15*count, label=None, edgecolors='none', zorder=4, color=pulse_frequency_colour)
        
            row_graph = {'Index': y_counter, 'Category': 'Single', 'Brain_Parc': label, 'Frequency': values[0], 'Count': count}
            graph_data.append(row_graph)
        
    y_counter += 2

# Set labels
ax4.set_xlabel('Frequency (Hz)')
ax4.set_xlim(-100, 1100)

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


# Customize the length of the ticks
#tick_lengths_or  = [0, 20, 5, 5, 5, 5, 20, 20, 5, 5, 5, 5, 5, 5, 5, 
#                    20, 20, 5, 5, 5, 5, 20, 5, 5, 20, 
#                    5, 5, 5, 20, 20, 5, 5, 5, 
#                    20, 5, 5, 20, 5, 20, 5, 5, 20, 
#                    20, 5, 5, 20, 20, 5, 5, 5, 20, 0]

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

# Parcellation -- single frequency
df_singular             = bar_df[bar_df['Category'] == 'Singular']
singular_brain          = df_singular.groupby('Index')['Count'].sum()
singular_frequency      = df_singular.groupby('Frequency')['Count'].sum()

# Parcellation -- multiple frequencies
df_multiple1            = bar_df[bar_df['Category'] == 'Multiple_Compiled']
multiple_brain          = df_multiple1.groupby('Index')['Count'].sum()

df_multiple2            = bar_df[bar_df['Category'] == 'Multiple']
multiple_frequency      = df_multiple2.groupby('Frequency')['Count'].sum()

# Parcellation -- range of frequencies
df_range1               = bar_df[bar_df['Category'] == 'Range_Compiled']
range_brain             = df_range1.groupby('Index')['Count'].sum()

df_range2                   = bar_df[bar_df['Category'] == 'Range']
df_range2['BinaryArrays']   = df_range2['Frequency'].apply(create_binary_array, start_value=-100, end_value=1099)
collapsed_array             = np.sum(np.array(df_range2['BinaryArrays'].tolist()), axis=0)

# Parcellation -- single pulses
df_single               = bar_df[bar_df['Category'] == 'Single']
single_brain            = df_single.groupby('Index')['Count'].sum()
single_frequency        = df_single.groupby('Frequency')['Count'].sum()

## Make sure datasets are the same size
# Add zeros if keys are empty
key_range   = range(1, len(brain_area_labels) * 2 + 1)  # From 1 to 4 (inclusive)
f_key_range = range(-100, 1100) 

singular_brain_count        = {key: singular_brain.get(key, 0) for key in key_range}
singular_frequency_count    = {key: singular_frequency.get(key, 0) for key in f_key_range}
multiple_brain_count        = {key: multiple_brain.get(key,0) for key in key_range}
multiple_frequency_count    = {key: multiple_frequency.get(key, 0) for key in f_key_range}
range_brain_count           = {key: range_brain.get(key, 0) for key in key_range}
range_frequency_count       = collapsed_array
single_brain_count          = {key: single_brain.get(key,0) for key in key_range}
single_frequency_count      = {key: single_frequency.get(key, 0) for key in f_key_range}

## Create bar graph per brain area
# Set default bar_width size
bar_width = 1

ax3.barh(list(singular_brain_count.keys()), list(singular_brain_count.values()), color=single_frequency_colour, edgecolor='white', height=bar_width)
ax3.barh(list(multiple_brain_count.keys()), list(multiple_brain_count.values()), left=list(singular_brain_count.values()), color=multiple_frequency_colour, edgecolor='white', height=bar_width)

# Plot range frequencies bar
combined_array = np.array(list(singular_brain_count.values())) + np.array(list(multiple_brain_count.values()))
ax3.barh(list(range_brain_count.keys()), list(range_brain_count.values()), left=combined_array, color=range_frequency_colour, edgecolor='white', height=bar_width)

# Plot single pulses bar
combined_array2 = np.array(combined_array + np.array(list(range_brain_count.values())))
ax3.barh(list(single_brain_count.keys()), list(single_brain_count.values()), left=combined_array2, color=pulse_frequency_colour, edgecolor='white', height=bar_width)

# Invert the x-axis
ax3.invert_xaxis()

ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.yaxis.set_visible(False)
ax3.set_xlabel('Reports (#)')

ax3.set_ylim(0,len(new_brain_numbers)*2 - 1)

"(6) Bar graph with counts -- per frequency"
## Create bar graph per frequency
ax2.bar(singular_frequency_count.keys(), range_frequency_count, color=range_frequency_colour, edgecolor='none', width=10, alpha=0.2, zorder=1)
ax2.bar(singular_frequency_count.keys(), singular_frequency_count.values(), color=single_frequency_colour, edgecolor='white', width=15, zorder=2)

combined_array3 = range_frequency_count + np.array(list(singular_frequency_count.values()))
ax2.bar(list(multiple_frequency_count.keys()), list(multiple_frequency_count.values()), bottom=list(singular_frequency_count.values()), color=multiple_frequency_colour, edgecolor='white', width=15, zorder=2)

ax2.bar(list(single_frequency_count.keys()), list(single_frequency_count.values()), color=pulse_frequency_colour, edgecolor='white', width=15, zorder=2)

ax2_ex = ax2.twinx()
ax2_ex.set_ylim(ax2.get_ylim())  # Match y-limits
ax2_ex.set_yticks(ax2.get_yticks())  # Match y-ticks
ax2_ex.set_yticklabels(ax2.get_yticklabels())  # Match y-tick labels
ax2_ex.set_ylabel('Reports (#)')
ax2.yaxis.set_visible(False)

# Remove the left and top spines
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2_ex.spines['top'].set_visible(False)
ax2_ex.spines['left'].set_visible(False)
ax2.set_xlim(-100, 1100)
ax2.set_xlabel('Frequency (Hz)')

median_value = np.median(df_singular['Frequency'])
ax2.axvline(x=median_value, color='#d8dcd6', linestyle='--', linewidth=1)

# Save output as a .png and .svg file
plt.tight_layout()

output_path = '../outputs/'
plt.savefig(output_path + 'FigureSupp3.svg', format='svg', dpi=600)
plt.savefig(output_path + 'FigureSupp3.png', format='png', dpi=600)

plt.show()



