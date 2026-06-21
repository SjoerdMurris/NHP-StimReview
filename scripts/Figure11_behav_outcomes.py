#!/usr/bin/env python3
"""
Figures for brain stimulation outcome analysis.
Requires df_model to be defined with columns:
    Effect, Electrode_type, Waveform, Frequency_log, Amplitude_log, Brain_Parc
And fitted models: model_indirect, model_no, model_mixed (mixedlm results)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Read data
data_path   = str(Path(__file__).resolve().parent.parent / 'data') + '/'
output_path = str(Path(__file__).resolve().parent.parent / 'outputs') + '/'
file_path = data_path + 'Pubmed_Overview_Final_StudySelection.csv'
df = pd.read_csv(file_path)

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
    import numpy as np
    try:
        lower = float(row['Amplitude_lower'])
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
   'Brain_Parc', 'Amplitude_log']].copy()

df_model['Waveform'] = df_model['Waveform'].str.strip()
df_model.loc[df_model['Waveform'].str.contains(',|and|/', na=False), 'Waveform'] = 'both'
df_model = df_model[df_model['Waveform'].isin(['monophasic', 'biphasic'])]
df_model = df_model.dropna().copy()
df_model['outcome_binary'] = (df_model['Effect'] == 'direct').astype(int)

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

# ------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------ #
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi':        150,
})

COLORS = {
    'direct':   '#2C7BB6',
    'indirect': '#ABD9E9',
    'no':       '#D7191C',
    'mixed':    '#FDAE61',
}
COMPARISONS = ['indirect', 'mixed', 'no']
COMP_LABELS  = ['Direct vs Indirect', 'Direct vs Mixed', 'Direct vs No effect']
PREDICTORS   = [
    ('C(Electrode_type)[T.Microelectrode]', 'Microelectrode'),
    ('C(Waveform)[T.monophasic]',           'Monophasic waveform'),
    ('Amplitude_log',                        'Amplitude (log)'),
    ('Frequency_log',                        'Frequency (log)'),
]

# ------------------------------------------------------------------ #
# Fit models
# ------------------------------------------------------------------ #
models = {}

for other in COMPARISONS:
    df_sub = df_model[df_model['Effect'].isin(['direct', other])].copy()
    df_sub['outcome_binary'] = (df_sub['Effect'] == 'direct').astype(int)

    models[other] = smf.mixedlm(
        'outcome_binary ~ C(Electrode_type) + C(Waveform) + Amplitude_log + Frequency_log',
        data=df_sub,
        groups=df_sub['Brain_Parc']
    ).fit(reml=True)

# ------------------------------------------------------------------ #
# Single-region check: does the frequency-outcome relationship hold
# within M1 alone? (addresses reviewer comment on region/task heterogeneity)
# ------------------------------------------------------------------ #

models_M1 = {}

for other in COMPARISONS:
    df_sub = df_model[df_model['Effect'].isin(['direct', other])].copy()
    df_sub['outcome_binary'] = (df_sub['Effect'] == 'direct').astype(int)

    df_m1 = df_sub[df_sub['Brain_Parc'] == 'M1'].copy()
    n_total = len(df_m1)
    n_direct = (df_m1['Effect'] == 'direct').sum()
    n_other = (df_m1['Effect'] == other).sum()

    print(f"\n{'='*60}")
    print(f"M1 only: direct vs {other} (N={n_total}, direct={n_direct}, {other}={n_other})")
    print(f"{'='*60}")

    if n_total < 10 or n_direct == 0 or n_other == 0:
        print(f"Insufficient data for M1-only model (need both outcome classes represented).")
        models_M1[other] = None
        continue

    try:
        models_M1[other] = smf.logit(
            'outcome_binary ~ C(Electrode_type) + C(Waveform) + Amplitude_log + Frequency_log',
            data=df_m1
        ).fit(disp=False)
        print(models_M1[other].summary())
    except Exception as e:
        print(f"M1-only model failed to fit for comparison '{other}': {e}")
        models_M1[other] = None

# ------------------------------------------------------------------ #
# Single-region check: DMid
# ------------------------------------------------------------------ #

models_DMid = {}

for other in COMPARISONS:
    df_sub = df_model[df_model['Effect'].isin(['direct', other])].copy()
    df_sub['outcome_binary'] = (df_sub['Effect'] == 'direct').astype(int)

    df_dmid = df_sub[df_sub['Brain_Parc'] == 'DMid'].copy()
    n_total = len(df_dmid)
    n_direct = (df_dmid['Effect'] == 'direct').sum()
    n_other = (df_dmid['Effect'] == other).sum()

    print(f"\n{'='*60}")
    print(f"DMid only: direct vs {other} (N={n_total}, direct={n_direct}, {other}={n_other})")
    print(f"{'='*60}")

    if n_total < 10 or n_direct == 0 or n_other == 0:
        print(f"Insufficient data for DMid-only model (need both outcome classes represented).")
        models_DMid[other] = None
        continue

    try:
        models_DMid[other] = smf.logit(
            'outcome_binary ~ C(Electrode_type) + C(Waveform) + Amplitude_log + Frequency_log',
            data=df_dmid
        ).fit(disp=False)
        print(models_DMid[other].summary())
    except Exception as e:
        print(f"DMid-only model failed to fit for comparison '{other}': {e}")
        models_DMid[other] = None

# ------------------------------------------------------------------ #
# Combined Figure
# ------------------------------------------------------------------ #
MIN_N = 10
roi_counts = df_model['Brain_Parc'].value_counts()
rois_keep  = roi_counts[roi_counts >= MIN_N].index.tolist()
df_roi     = df_model[df_model['Brain_Parc'].isin(rois_keep)].copy()

props = (df_roi.groupby(['Brain_Parc', 'Effect'])
               .size()
               .unstack(fill_value=0))
for col in ['direct', 'indirect', 'mixed', 'no']:
    if col not in props.columns:
        props[col] = 0
props      = props[['direct', 'indirect', 'mixed', 'no']]
props_norm = props.div(props.sum(axis=1), axis=0)
props_norm = props_norm.sort_values('direct', ascending=True)

fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(2, 2, figure=fig,
               width_ratios=[1, 1.2],
               height_ratios=[1, 1],
               hspace=0.4, wspace=0.35)

ax_roi  = fig.add_subplot(gs[:, 0])
ax_freq = fig.add_subplot(gs[0, 1])
ax_coef = fig.add_subplot(gs[1, 1])

# --- Left: ROI proportions ---
bottoms = np.zeros(len(props_norm))
for col in ['direct', 'indirect', 'mixed', 'no']:
    vals = props_norm[col].values
    ax_roi.barh(range(len(props_norm)), vals, left=bottoms,
                color=COLORS[col], label=col.capitalize(), height=0.7)
    bottoms += vals

for i, roi in enumerate(props_norm.index):
    n = roi_counts[roi]
    ax_roi.text(1.01, i, f'N={n}', va='center', fontsize=7.5, color='gray')

ax_roi.set_yticks(range(len(props_norm)))
ax_roi.set_yticklabels(props_norm.index, fontsize=8)
ax_roi.set_xlabel('Proportion of studies')
ax_roi.set_xlim(0, 1)
ax_roi.legend(loc='lower right', fontsize=8, frameon=False,
              bbox_to_anchor=(1.18, -0.06))
ax_roi.axvline(0.5, color='gray', lw=0.6, ls='--', alpha=0.5)

# --- Top right: Frequency violin ---
order  = ['direct', 'indirect', 'mixed', 'no']
labels = ['Direct', 'Indirect', 'Mixed', 'No effect']
data   = [df_model[df_model['Effect'] == o]['Frequency_log'].dropna().values
          for o in order]

parts = ax_freq.violinplot(data, positions=range(len(order)),
                           showmedians=True, showextrema=False)
for i, (pc, o) in enumerate(zip(parts['bodies'], order)):
    pc.set_facecolor(COLORS[o])
    pc.set_alpha(0.75)
    pc.set_edgecolor('white')
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(1.5)

for i, (d, o) in enumerate(zip(data, order)):
    jitter = np.random.uniform(-0.08, 0.08, len(d))
    ax_freq.scatter(i + jitter, d, color=COLORS[o], alpha=0.3, s=8, zorder=3)

for i, d in enumerate(data):
    ax_freq.text(i, df_model['Frequency_log'].min() - 0.1,
                 f'N={len(d)}', ha='center', va='top', fontsize=8, color='gray')

ax_freq.set_xticks(range(len(order)))
ax_freq.set_xticklabels(labels)
ax_freq.set_ylabel('Frequency (log Hz)')

# --- Bottom right: Coefficient plot ---
n_pred  = len(PREDICTORS)
n_comp  = len(COMPARISONS)
offsets = np.linspace(0.2, -0.2, n_comp)  # indirect on top, no effect on bottom
comp_colors = ['#ABD9E9', '#FDAE61', '#D7191C']

for ci, (other, label, color) in enumerate(zip(COMPARISONS, COMP_LABELS, comp_colors)):
    res = models[other]
    for pi, (param, _) in enumerate(PREDICTORS):
        coef  = res.params[param]
        ci_lo = res.conf_int().loc[param, 0]
        ci_hi = res.conf_int().loc[param, 1]
        y     = pi + offsets[ci]
        ax_coef.plot([ci_lo, ci_hi], [y, y], color=color, lw=1.8,
                     solid_capstyle='round')
        if res.pvalues[param] < 0.05:
            # Significant: filled diamond
            ax_coef.plot(coef, y, marker='D', color=color, ms=9,
                         markeredgecolor=color, markeredgewidth=1.0,
                         zorder=5)
        else:
            # Not significant: open circle
            ax_coef.plot(coef, y, marker='o', color='none', ms=9,
                         markeredgecolor=color, markeredgewidth=1.5,
                         zorder=5)

ax_coef.axvline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax_coef.set_yticks(range(n_pred))
ax_coef.set_yticklabels([p[1] for p in PREDICTORS])
ax_coef.set_xlabel('Coefficient (linear probability)')

legend_handles = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(comp_colors, COMP_LABELS)]
legend_handles += [
    plt.Line2D([0], [0], marker='D', color='gray', ls='none', ms=9,
               markeredgecolor='gray', label='p < 0.05'),
    plt.Line2D([0], [0], marker='o', color='none', ls='none', ms=9,
               markeredgecolor='gray', markeredgewidth=1.5, label='p ≥ 0.05'),
]
ax_coef.legend(handles=legend_handles, fontsize=7.5, frameon=False,
               loc='upper left', bbox_to_anchor=(1.02, 1))

plt.savefig(output_path + 'behav_fig_combined.svg', format='svg', bbox_inches='tight')
plt.savefig(output_path + 'behav_fig_combined.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved combined figure as .svg and .png")