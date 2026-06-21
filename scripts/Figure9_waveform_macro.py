# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:02:20 2024

@author: Sjoerd Murris
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from utils import convert_pulse_width

"(0) Specify general parameters for the figures"
data_label = 'all' # Specify cortical or subcortical to plot

"(1) read in the data from the excel file"
file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Study_Selection')

"(3) parse the waveform stimulation data"
# Convert to cortical or subcortical
if data_label == 'all':
    cort_df      = df
else:
    cort_df      = df[df['Cortical_Subcortical'] == data_label]

# First sort by electrical/optogenetic stimulation
elec_df      = cort_df[cort_df['Type_stimulation'] == 'electrical']
opt_df       = cort_df[cort_df['Type_stimulation'] == 'optogenetic']

elec_df      = elec_df[elec_df['Electrode_type'] == 'Macroelectrode']

## Check instances of both mono- and biphasic waveforms
# Find rows with ', ' in both 'Column1' and 'Column2'
mask = elec_df['Cathode_anode'].str.contains(', ') & elec_df['Waveform'].str.contains(', ') 

# Create a list to store the new rows
new_rows = []

for index, row in elec_df[mask].iterrows():
    # Split the values before and after ', ' in both columns
    col1_vals = row['Waveform'].split(', ')
    col2_vals = row['Cathode_anode'].split(', ')
    
    # Append two new rows with split values and retain other column values
    new_rows.append({**row, 'Waveform': col1_vals[0], 'Cathode_anode': col2_vals[0]})
    new_rows.append({**row, 'Waveform': col1_vals[1], 'Cathode_anode': col2_vals[1]})

# Drop the original rows with ', ' in both columns
elec_df = elec_df.drop(elec_df[mask].index)

# Append the new rows to the DataFrame
elec_df = pd.concat([elec_df, pd.DataFrame(new_rows)], ignore_index=True)

## Create separate dataframes for biphasic and monophasic
biphasic_df     = elec_df[elec_df['Waveform'] == 'biphasic']
monophasic_df   = elec_df[elec_df['Waveform'] == 'monophasic']
unclear_df      = elec_df[elec_df['Waveform'] == 'unclear']

## Find cathode first, anode first (biphasic)
cath_first_df    = biphasic_df[biphasic_df['Cathode_anode'] == 'cathode first']
anod_first_df    = biphasic_df[biphasic_df['Cathode_anode'] == 'anode first']
uncl_first_df    = biphasic_df[biphasic_df['Cathode_anode'] == 'unclear']
both_first_df    = biphasic_df[biphasic_df['Cathode_anode'] == 'both']

## Find cathodal and anodal
cathodal_df      = monophasic_df[monophasic_df['Cathode_anode'] == 'cathodal']
anodal_df        = monophasic_df[monophasic_df['Cathode_anode'] == 'anodal']
mono_both_df     = monophasic_df[monophasic_df['Cathode_anode'] == 'both']
mono_uncl_df     = monophasic_df[monophasic_df['Cathode_anode'] == 'unclear']

"(4) make a bargraph showing the distribution of the waveforms"
# Create a main figure
fig = plt.figure(figsize=(8.5, 3), dpi = 600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(1, 3)

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Left
ax2 = fig.add_subplot(gs[0, 1])  # Middle Left
ax3 = fig.add_subplot(gs[0, 2])  # Middle Right

# Data: specify the lengths of each sub-bar
piece_lengths = [len(biphasic_df), len(monophasic_df), len(unclear_df)]
piece_labels  = ['Biphasic', 'Monophasic', 'Unclear']  # Labels for each segment
colors = ['salmon', '#82cafc', 'silver']  # Optional colors for each segment

## Make a pie chart for the different waveforms
explode = [0, 0, 0]
labels  = ['biphasic', 'monophasic', 'unclear']

# Plot the main pie chart
ax1.pie(
    piece_lengths,
    explode=explode,
    startangle=90,
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    colors=colors
)

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_artist(centre_circle)

"(5) parse the pulse width stimulation data"
# Parse the 'pulse width' column
pulse_df = elec_df[~elec_df['Pulse_width'].str.contains('unclear', na=False)]
pulse_df['Pulse_width_calibrated'] = pulse_df['Pulse_width'].apply(convert_pulse_width)

# Check the difference between mono and biphasic
mono_pulse_df = pulse_df[pulse_df['Waveform'] == 'monophasic']
bi_pulse_df   = pulse_df[pulse_df['Waveform'] == 'biphasic']

# Divide the biphasic pulses by two
bi_pulse_df['Pulse_width_calibrated'] = bi_pulse_df['Pulse_width_calibrated']/2

pulse_value_counts = pulse_df['Pulse_width_calibrated'].value_counts() 
mono_value_counts = mono_pulse_df['Pulse_width_calibrated'].value_counts()
bi_value_counts   = bi_pulse_df['Pulse_width_calibrated'].value_counts()

# Sort the counts by key (numerical values) to ensure correct x-axis placement
pulse_value_counts = pulse_value_counts.sort_index()
mono_value_counts  = mono_value_counts.sort_index()
bi_value_counts    = bi_value_counts.sort_index()

# Generate x and y values (integers)
x_mono = np.array(mono_value_counts.keys())
y_mono = np.array(mono_value_counts)

x_bi   = np.array(bi_value_counts.keys())
y_bi   = np.array(bi_value_counts)

mean_pulse   = pulse_df['Pulse_width_calibrated'].mean() 
median_pulse = pulse_df['Pulse_width_calibrated'].median() 

median_mono  = mono_pulse_df['Pulse_width_calibrated'].median() 
median_bi    = bi_pulse_df['Pulse_width_calibrated'].median()

# Create a union of x-values (all unique x-values)
all_x = np.union1d(x_bi, x_mono)

# Initialize aligned y-values with zeros
y_mono_aligned = np.zeros_like(all_x, dtype=float)
y_bi_aligned = np.zeros_like(all_x, dtype=float)

# Populate aligned arrays using broadcasting
y_mono_aligned[np.isin(all_x, x_mono)] = y_mono[np.searchsorted(x_mono, all_x[np.isin(all_x, x_mono)])]
y_bi_aligned[np.isin(all_x, x_bi)] = y_bi[np.searchsorted(x_bi, all_x[np.isin(all_x, x_bi)])]

ax2.bar(all_x, y_bi_aligned, width=20, color='salmon')
#ax2.bar(all_x, y_mono_aligned, bottom=y_bi_aligned, width=0.02, color='#82cafc')
ax2.axvline(x=median_bi, ymin=0, ymax=350, color='#d8dcd6', linestyle='--', linewidth=1)
ax2.scatter(median_bi, 160, s=100, facecolors='white', edgecolors='salmon', linewidths=5, zorder=5)

ax2.set_xlim(0, 1100)
ax2.set_ylim(0, 170)
ax2.set_xlabel('Pulse width (μs)')
ax2.set_ylabel('Reports (#)')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Create an inset axis for the pie chart within the first subplot
pie_sizes  = [len(cath_first_df), len(anod_first_df), len(both_first_df), len(uncl_first_df)]
pie_colors = ['salmon', '#ffc5cb', '#a90308','silver']

inset_ax = ax2.inset_axes([0.5, 0.5, 0.5, 0.5])  # [x, y, width, height] relative to the subplot
inset_ax.pie(
    pie_sizes,
    colors=pie_colors,
    explode=[0, 0, 0, 0],
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
)

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
inset_ax.add_artist(centre_circle)

ax3.bar(all_x, y_mono_aligned, width=20, color='#82cafc')
#ax3.bar(all_x, y_mono_aligned, bottom=y_bi_aligned, width=0.02, color='#82cafc')
ax3.axvline(x=median_mono, ymin=0, ymax=350, color='#d8dcd6', linestyle='--', linewidth=1)
ax3.scatter(median_mono, 160, s=100, facecolors='white', edgecolors='#82cafc', linewidths=5, zorder=5)

ax3.set_xlim(0, 1100)
ax3.set_ylim(0, 170)
ax3.set_xlabel('Pulse width (μs)')
ax3.set_ylabel('')

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.set_yticks([])
ax3.set_yticklabels([])

# Create an inset axis for the pie chart within the first subplot
pie_sizes  = [len(cathodal_df), len(anodal_df), len(mono_both_df), len(mono_uncl_df)]
pie_colors = ['#82cafc','#bdf6fe','#0485d1','silver']

inset_ax2 = ax3.inset_axes([0.5, 0.5, 0.5, 0.5])  # [x, y, width, height] relative to the subplot
inset_ax2.pie(
    pie_sizes,
    colors=pie_colors,
    explode=[0, 0, 0, 0],
    startangle=90,
    wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
)

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
inset_ax2.add_artist(centre_circle)

# Save output as a .png and .svg file
plt.tight_layout()

output_path = '../outputs/'
plt.savefig(output_path + 'Figure4_macro.svg', format='svg', dpi=600)
plt.savefig(output_path + 'Figure4_macro.png', format='png', dpi=600)

plt.show()


import pandas as pd
import statsmodels.api as sm

# Step 1: Create a binary column: 1 if 'biphasic', 0 otherwise
elec_df['is_biphasic'] = (elec_df['Waveform'] == 'biphasic').astype(int)

# Step 2: Drop rows with missing or non-numeric 'Year'
elec_df = elec_df.dropna(subset=['Year'])
elec_df['Year'] = pd.to_numeric(elec_df['Year'], errors='coerce')

# Step 3: Define predictor and response
X = sm.add_constant(elec_df['Year'])  # adds intercept term
y = elec_df['is_biphasic']

# Step 4: Fit logistic regression model
model = sm.Logit(y, X).fit()

# Step 5: Print summary
print(model.summary())



## Statistical test for publication year
# Count per year and type
from scipy.stats import chi2_contingency

# Contingency table
contingency_table = pd.crosstab(pulse_df['Year'], pulse_df['Waveform'])

# Chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2: {chi2}, p-value: {p}")

combined_df = pd.concat([bi_pulse_df, mono_pulse_df], axis=0)
# Chi-square test
contingency_table = pd.crosstab(combined_df['Year'], combined_df['Waveform'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi2: {chi2}, p-value: {p}")

N = contingency_table.values.sum()
r, c = contingency_table.shape
cramers_v = np.sqrt(chi2 / (N * (min(r - 1, c - 1))))