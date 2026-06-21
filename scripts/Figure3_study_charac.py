# -*- coding: utf-8 -*-
"""
Stimulation Review - Generate general figures about characteristics of studies
By: Sjoerd Murris
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

"(1) read in the data from the excel file"
file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Study_Selection')

"(2) get the species, sex, brain regions, number and year per study as an output"
"Species"
df_species      = df.groupby('Title')['Species'].unique()
df_species      = df_species.apply(tuple)
species_counts  = df_species.value_counts()

"Sex (male, female, unclear)"
df_sex          = df.groupby('Title')['Sex'].unique()
df_sex          = df_sex.apply(tuple)
sex_counts      = df_sex.value_counts()

"Brain regions (cortical or subcortical)"
df_brain        = df.groupby('Title')['Cortical_Subcortical']
df_brain        = df_brain.apply(tuple)
brain_counts    = df_brain.value_counts() 

"Number of animals per study"
df_number       = df.groupby('Title')['Number_animals']
df_number       = df_number.apply(tuple)
number_counts   = df_number.value_counts()

"Journal"
df_journal      = df.groupby('Title', as_index=False).agg({'Journal': 'first'})
journal_counts  = df_journal['Journal'].value_counts()

"Publication year"
df_year         = df.groupby('Title', as_index=False).agg({'Year': 'first', 'Type_stimulation': 'first'})
df_elec         = df_year[df_year['Type_stimulation'] == 'electrical']
df_opto         = df_year[df_year['Type_stimulation'] == 'optogenetic']
elec_counts     = df_elec['Year'].value_counts().sort_index()
opto_counts     = df_opto['Year'].value_counts().sort_index()

"(3) Create the main figure with specifications"
# Create a main figure
fig = plt.figure(figsize=(3.13, 2), dpi=600)  # Adjust the overall figure size

# Create a GridSpec object
gs = gridspec.GridSpec(2, 4)

# Create subplots with the specified grid sizes
ax1 = fig.add_subplot(gs[0, 0])  # Top left
ax2 = fig.add_subplot(gs[0, 1])  # Top middle
ax3 = fig.add_subplot(gs[0, 2])  # Top right
ax4 = fig.add_subplot(gs[1, 0])  # Bottom left
ax5 = fig.add_subplot(gs[1, 1])  # Bottom middle
ax6 = fig.add_subplot(gs[1, 2])  # Bottom right

"(4) Plot into pie charts"
# First graph: studies overview
elec_counts.index = elec_counts.index.astype(int) 
opto_counts.index = opto_counts.index.astype(int) 

elec_keys   = elec_counts.index.to_numpy()
elec_values = elec_counts.to_numpy()

opt_keys   = opto_counts.index.to_numpy()
opt_values = opto_counts.to_numpy()

# Create a dictionary mapping opt_keys to opt_values
opt_dict = dict(zip(opt_keys, opt_values))

# Align opt_values with elec_keys, using 0 if missing
aligned_opt_values = np.array([opt_dict.get(k, 0) for k in elec_keys])

# Plot stacked bars
ax1.bar(elec_keys, aligned_opt_values, color='cyan', width=0.5)
ax1.bar(elec_keys, elec_values, bottom=aligned_opt_values, color='black', width=0.5)
ax1.set_xlabel('Year of publication', fontsize=6)
ax1.set_ylabel('Studies (#)',fontsize=6)
ax1.set_xlim(1940, 2025)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', labelsize=6)  # 8 is the font size

# Second graph: Journals pie chart
labels = ['Journal of Neurophysiology', 'Journal of Neuroscience', 'Experimental Brain Research', 'Brain Research', 'Neuron', 'Other']
sizes  = [journal_counts[0], journal_counts[1], journal_counts[2], journal_counts[3], journal_counts[4], journal_counts[5::].sum()]
colors = ['saddlebrown', 'chocolate', 'sandybrown', 'peachpuff','linen','silver']

ax2.pie(sizes, colors=colors, startangle=90,
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax2.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax2.add_artist(centre_circle)

## Pie chart for the species distribution
list_of_strings = [str(element) for element in species_counts.keys()]

# Break down for each of the species
mulatta_indices = [i for i, item in enumerate(list_of_strings) if 'mulatta' in item]
mulatta_numb    = sum(species_counts[mulatta_indices])

fasc_indices    = [i for i, item in enumerate(list_of_strings) if 'fascicularis' in item]
fasc_numb       = sum(species_counts[fasc_indices])

fusc_indices    = [i for i, item in enumerate(list_of_strings) if 'fuscata' in item]
fusc_numb       = sum(species_counts[fusc_indices])

nemestrina_indices = [i for i, item in enumerate(list_of_strings) if 'nemestrina' in item]
nemestrina_numb    = sum(species_counts[nemestrina_indices])

radiata_indices = [i for i, item in enumerate(list_of_strings) if 'radiata' in item]
radiata_numb    = sum(species_counts[radiata_indices])

arc_indices     = [i for i, item in enumerate(list_of_strings) if 'arctoides' in item]
arc_numb        = sum(species_counts[arc_indices])

# Creating pie species chart figure
labels = ['M. mulatta', 'M. fascicularis', 'M. nemestrina', 'M. fuscata', 'M. radiata', 'M. arctoides', 'unclear']
sizes = [mulatta_numb, fasc_numb, fusc_numb, nemestrina_numb, radiata_numb, arc_numb, species_counts[4]]
colors = ['springgreen', 'mediumseagreen', 'seagreen', 'limegreen','green','darkgreen','silver']

ax3.pie(sizes, colors=colors, startangle=90, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax3.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax3.add_artist(centre_circle)


# Bar graph for the number of animals
# Initialize a list to store the highest numbers
highest_numbers = []

# Loop through each key in the 'Number_animals' column
for key in number_counts.keys():
    if isinstance(key, tuple) and all(isinstance(x, (int, float)) for x in key):
        highest_numbers.append(max(key))
    else:
        highest_numbers.append(None)


results = {}
numbers_to_process = set(highest_numbers)

for numbs in numbers_to_process:
    # Find indices where the value in highest_numbers matches numbs
    indices = [i for i, x in enumerate(highest_numbers) if x == numbs]
    
    # Sum the counts at the found indices
    total_count = sum(number_counts[i] for i in indices)
    
    # Store the result
    results[numbs] = total_count

new_results = results
if None in new_results:
    new_results.pop(None)

# Expand the dictionary into a list
expanded_list = [key for key, count in new_results.items() for _ in range(count)]

# Calculate mean
mean_value = np.mean(expanded_list)

# Calculate median
median_value = np.median(expanded_list)

# Combine the studies with large numbers together
keys = list(results.keys())
last_three_keys = keys[-3:]

combined_key = 15
combined_value = sum(results[key] for key in last_three_keys)

# Update the dictionary
# Remove the old keys
for key in last_three_keys:
    results.pop(key)

results[combined_key] = combined_value

# Plot the results
ax4.bar(results.keys(), results.values(), color='silver')
ax4.set_xlabel('Animals (#)', fontsize=6)
ax4.set_ylabel('Studies (#)', fontsize=6)
ax4.set_xticks([0, 5, 10, 15])  # Ensure all numbers are shown on the x-axis
ax4.axvline(x=median_value, color='black', linestyle='--', linewidth=0.5, label=f'Mean: {mean_value:.2f}')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.set_yticks([0, 100, 200, 300])
ax4.tick_params(axis='both', labelsize=6)  # 12 is the font size

# Fifth graph: Sex distribution
new_counts = sex_counts[0:3]
new_counts['both_sexes'] = sum(sex_counts[3::])

# Pie chart for the sex distribution
labels = ['male only', 'female only', 'both sexes', 'unclear']
sizes = [new_counts[0], new_counts[2], new_counts[3], new_counts[1]]
colors = ['deepskyblue', 'violet', 'lavender', 'silver']

ax5.pie(sizes, colors=colors, startangle=90, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax5.axis('equal')

# Sixth graph: Pie chart for the cortical-subcortical distribution
list_of_strings = [str(element) for element in brain_counts.keys()]
substrings = ['cortical', 'subcortical']

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax5.add_artist(centre_circle)

# Dictionary to store counts for each index
index_counts = {}

# Loop through the list and count occurrences of each substring at each index
for index, string in enumerate(list_of_strings):
    index_counts[index] = {substring: string.count(substring) for substring in substrings}

# Find the indices that only have subcortical targets and only cortical targets
equal_count_indices = [index for index, counts in index_counts.items() 
                       if len(set(counts.values())) == 1]
indices_with_zeros = [index for index, counts in index_counts.items() 
                      if 0 in counts.values()]

# divide into cortical, subcortical, multiple targets
cort_numb    = brain_counts[0]
sub_numb     = brain_counts[1]
m_cort_numb  = sum(brain_counts[indices_with_zeros[1::]])
m_sub_numb   = sum(brain_counts[equal_count_indices[1::]])
other_numb   = sum(brain_counts) - (cort_numb + sub_numb + m_cort_numb + m_sub_numb)

# Creating pie chart figure
labels = ['Cortical', 'Cortical (multiple regions)', 'Subcortical', 'Subcortical (multiple regions)', 'Cortical & subcortical regions']
sizes = [cort_numb, m_cort_numb, sub_numb, m_sub_numb, other_numb]
colors = ['paleturquoise', 'darkcyan','lightcoral', 'firebrick','indigo']

ax6.pie(sizes, colors=colors, startangle=90, wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
ax6.axis('equal')

# Add a white circle in the middle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax6.add_artist(centre_circle)

# Save output as a .png and .svg file
#plt.tight_layout()

output_path = '../outputs/'
plt.savefig(output_path + 'Figure1.png', format='png', dpi=600)
plt.savefig(output_path + 'Figure1.svg', format='svg', dpi=600)

# Show the plot
plt.show()


## Add statistics about sex over time
import statsmodels.formula.api as smf

# Filter out 'unclear' from df for the first model
df_reported = df[df['Sex'] != 'unclear'].copy()

# Create female_included variable on filtered data
df_reported['female_included'] = df_reported['Sex'].apply(lambda x: 1 if x in ['female', 'male, female'] else 0)

# Binary variable 2: Sex reported? (Yes if NOT 'unclear')
df['sex_reported'] = df['Sex'].apply(lambda x: 0 if x == 'unclear' else 1)

# Logistic regression on filtered data
model_female = smf.logit('female_included ~ Year', data=df_reported).fit()
print(model_female.summary())

# Logistic regression: Does sex reporting improve over years?
model_reported = smf.logit('sex_reported ~ Year', data=df).fit()
print(model_reported.summary())
