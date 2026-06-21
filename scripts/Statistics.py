# -*- coding: utf-8 -*-
"""
Statistical analysis - correlations

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"(1) read in the data from the csv file"
file_path = '../data/Pubmed_Overview_Final_StudySelection.csv'
df = pd.read_csv(file_path)

"(2) get all the amplitude values for each of the studies"
"Only using midpoints"
# First sort by electrical/optogenetic stimulation
elec_df      = df[df['Type_stimulation'] == 'electrical'].copy()

# Coerce amplitude columns to numeric (non-numeric entries become NaN)
elec_df['Amplitude_lower']  = pd.to_numeric(elec_df['Amplitude_lower'],  errors='coerce')
elec_df['Amplitude_higher'] = pd.to_numeric(elec_df['Amplitude_higher'], errors='coerce')

# Convert milli-ampere values to micro-ampere
elec_df.loc[elec_df['Amplitude_unit'] == 'mA', 'Amplitude_lower'] *= 1000
elec_df.loc[elec_df['Amplitude_unit'] == 'mA', 'Amplitude_higher'] *= 1000

# Separate A, V and unclear
mask = elec_df['Amplitude_unit'].astype(str).str.contains(r'[A]', regex=True, na=False)
elec_df.loc[~mask, 'Amplitude_lower'] = np.nan
elec_df.loc[~mask, 'Amplitude_higher'] = np.nan

elec_df['Amplitude_midpoint'] = (elec_df['Amplitude_lower'] + elec_df['Amplitude_higher']) / 2

"(3) get the frequency values"
"ranges are turned into midpoints, multiples into median"
# First excludes 'single pulses' and 'unclear'
mask = elec_df['Frequency'].astype(str).str.contains(r'single pulses|unclear', case=False, na=False)
elec_df.loc[mask, 'Frequency'] = np.nan

df_filtered = elec_df

# Separate single values from ranges and multiple values
def process_frequency(freq_str):
    # If freq_str is NaN or None, return NaN directly
    if pd.isna(freq_str):
        return np.nan

    # Ensure it's a string
    freq_str = str(freq_str)

    # Remove 'Hz' and any whitespace
    freq_str = freq_str.replace('Hz', '').strip()

    # Case 1: range with dash
    if '-' in freq_str:
        parts = freq_str.split('-')
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        except:
            return np.nan  # in case of unexpected format

    # Case 2: list with commas
    elif ',' in freq_str:
        parts = freq_str.split(',')
        try:
            nums = [float(p.strip()) for p in parts]
            return np.median(nums)
        except:
            return np.nan

    # Case 3: single number
    else:
        try:
            return float(freq_str)
        except:
            return np.nan

# Apply the function to create the new column
df_filtered['Frequency_midpoints'] = df_filtered['Frequency'].apply(process_frequency)

"(4) get duration values"
"ranges are turned into midpoints, multiples into median"
def convert_to_ms(timespan_str):
    # Remove whitespace and lowercase
    s = str(timespan_str).strip().lower()
    
    # Ignore non-numeric or unclear values
    if s in ['unclear', 'variable', 'depends on RT', 'depends on monkey', 'continuous']:
        return np.nan
    
    # Determine multiplier for unit
    if 'ms' in s:
        multiplier = 1
        s = s.replace('ms', '')
    elif 's' in s:
        multiplier = 1000
        s = s.replace('s', '')
    elif 'min' in s:
        multiplier = 60000
        s = s.replace('min', '')
    else:
        return np.nan  # Unknown unit
    
    s = s.strip()
    
    # Case 1: range with dash
    if '-' in s:
        parts = s.split('-')
        try:
            return ((float(parts[0].strip()) + float(parts[1].strip())) / 2) * multiplier
        except:
            return np.nan
    
    # Case 2: list with commas
    elif ',' in s:
        parts = s.split(',')
        try:
            nums = [float(p.strip()) for p in parts]
            return np.median(nums) * multiplier
        except:
            return np.nan
    
    # Case 3: single number
    else:
        try:
            return float(s) * multiplier
        except:
            return np.nan
        
df_filtered['Timespan_ms'] = df_filtered['Timespan'].apply(convert_to_ms)
df_filtered['Pulse_width'] = df_filtered['Pulse_width'].apply(convert_to_ms)

# Drop rows where 'Timespan_ms' is NaN
#df_filtered = df_filtered.dropna(subset=['Timespan_ms'])

"(5) get waveform details"
"only use monophasic and biphasic"
# Keep only biphasic and monophasic rows
mask = ~df_filtered['Waveform'].isin(['biphasic', 'monophasic'])
df_filtered.loc[mask, 'Waveform'] = np.nan

"(6) remove instances where the monkey species is unclear"
mask = ~df_filtered['Species_cat'].isin(['rhesus', 'non_rhesus'])
df_filtered.loc[mask, 'Species_cat'] = np.nan

"(7) collapse over targets -> back to studies"
# Example: df is your DataFrame
# Group by 'Title' and combine other columns
def collapse_duplicates(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = []  # columns for which to compute medians

    def collapse_column(x, colname):
        # If all values are NaN, return NaN
        if x.isna().all():
            return np.nan

        unique_vals = x.dropna().unique()

        # If only one unique value, return it directly
        if len(unique_vals) == 1:
            return unique_vals[0]

        # For numeric columns → compute median if possible
        if colname in numeric_cols:
            numeric_values = []
            for val in unique_vals:
                if isinstance(val, str):
                    for part in val.split(','):
                        try:
                            numeric_values.append(float(part.strip()))
                        except ValueError:
                            pass
                else:
                    try:
                        numeric_values.append(float(val))
                    except ValueError:
                        pass
            if len(numeric_values) > 0:
                return np.median(numeric_values)
            else:
                return np.nan
        else:
            # For non-numeric columns with multiple unique values, just return the first non-NaN
            return unique_vals[0]
        
        # For non-numeric columns → join unique values
        return ', '.join(map(str, unique_vals))

    # Collapse all other columns
    df_collapsed = df.groupby('Title', as_index=False).agg(
        lambda x: collapse_column(x, x.name)
    )

    # Add a column counting how many rows were collapsed
    target_counts = df['Title'].value_counts().rename_axis('Title').reset_index(name='Targets')
    df_collapsed = df_collapsed.merge(target_counts, on='Title', how='left')

    return df_collapsed

# Example usage:
numeric_columns = ['Amplitude_midpoint', 'Frequency_midpoints', 'Timespan_ms', 'Number_animals', 'Pulse_width']
df_collapsed = collapse_duplicates(df_filtered, numeric_cols=numeric_columns)

"(8) minor modifications to collapsed data"
# Standardize and replace combinations in 'Cortical_Subcortical'
# Create a mask for rows that are NOT 'cortical' or 'subcortical'
mask = ~df_collapsed['Cortical_Subcortical'].isin(['cortical', 'subcortical'])

# Replace those entries with NaN
df_collapsed.loc[mask, 'Cortical_Subcortical'] = np.nan

df_collapsed['Electrode_type'] = df_collapsed['Electrode_type'].apply(lambda x: 'Both' if isinstance(x, str) and ',' in x else x)
df_collapsed['Electrode_type'] = df_collapsed['Electrode_type'].replace('Both', np.nan)

import statsmodels.formula.api as smf
from pygam import LinearGAM, s, f
import warnings

# Ensure categorical variables are treated as such
df_collapsed['Electrode_type'] = df_collapsed['Electrode_type'].astype('category')
df_collapsed['Waveform'] = df_collapsed['Waveform'].astype('category')
df_collapsed['Amplitude_midpoint'] = pd.to_numeric(df_collapsed['Amplitude_midpoint'], errors='coerce')
df_collapsed['Number_animals'] = pd.to_numeric(df_collapsed['Number_animals'], errors='coerce')
df_collapsed['Cortical_Subcortical'] = df_collapsed['Cortical_Subcortical'].astype('category')
df_collapsed['Journal_type'] = df_collapsed['Journal_type'].astype('category')
df_collapsed['Species_cat'] = df_collapsed['Species_cat'].astype('category')

df_collapsed['log_Timespan'] = np.log10(df_collapsed['Timespan_ms'])

# ------------------------------------------------------------------ #
# OLS — combined model with all predictors
# ------------------------------------------------------------------ #
formula = 'Year ~ Amplitude_midpoint + Frequency_midpoints + log_Timespan + Targets + Number_animals + Pulse_width + C(Electrode_type) + C(Waveform) + C(Cortical_Subcortical) + C(Journal_type) + C(Species_cat)'

model = smf.ols(formula=formula, data=df_collapsed).fit()
print(model.summary())

# ------------------------------------------------------------------ #
# GAM — same predictors, smooth terms for continuous, factors for categorical
# ------------------------------------------------------------------ #
cat_cols  = ['Electrode_type', 'Waveform', 'Cortical_Subcortical', 'Journal_type', 'Species_cat']
cont_cols = ['Amplitude_midpoint', 'Frequency_midpoints', 'log_Timespan', 'Targets', 'Number_animals', 'Pulse_width']
gam_cols  = cont_cols + cat_cols

df_gam = df_collapsed[gam_cols + ['Year']].copy()
for col in cat_cols:
    df_gam[col] = df_gam[col].cat.codes  # -1 for NaN
df_gam = df_gam.replace(-1, np.nan).dropna()

X = df_gam[gam_cols].values
y = df_gam['Year'].values

terms = (
    s(0) + s(1) + s(2) + s(3) + s(4) + s(5) +
    f(6) + f(7) + f(8) + f(9) + f(10)
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    gam = LinearGAM(terms).fit(X, y)

def sig_stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))

print("\n" + "="*60)
print("GAM — Year ~ s(continuous) + f(categorical)")
print(f"N = {len(y)}  |  R² = {gam.statistics_['pseudo_r2']['explained_deviance']:.3f}  |  AIC = {gam.statistics_['AIC']:.1f}")
print("="*60)
print(f"{'Predictor':<30} {'p-value':>10}  {'':>5}")
print("-"*50)
for name, pval in zip(gam_cols, gam.statistics_['p_values']):
    print(f"{name:<30} {pval:>10.4f}  {sig_stars(pval)}")
print("="*60)

"(6) plot the amplitude versus the frequency"
plt.scatter(df_filtered['Frequency_midpoints'], df_filtered['Amplitude_midpoint'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (µA)')
plt.title('Amplitude vs. Frequency')
plt.show()

plt.scatter(df_filtered['Frequency_midpoints'], df_filtered['Timespan_ms'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Timespan (ms)')
plt.title('Timespan vs. Frequency')
plt.show()

plt.scatter(df_filtered['Year'], df_filtered['Amplitude_midpoint'])
plt.xlabel('Year')
plt.ylabel('Amplitude (mA)')
plt.title('Amplitude vs. Year')
plt.show()
 