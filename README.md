# StimReview

Analysis and visualization code for a systematic review of brain stimulation studies in non-human primates (macaques).

The review covers electrical microstimulation and optogenetic stimulation studies, mapping stimulation targets and outcomes onto the NMT v2.0 symmetric macaque brain atlas using the CHARM (cortical) and SARM (subcortical) parcellation schemes.

## Repository structure

```
StimReview/
├── scripts/
│   ├── Figure3_study_charac.py               # Fig 3 — study characteristics (species, journals, sex, year)
│   ├── Figure4_atlas_generate.py             # Fig 4 — prerequisite: map stimulation targets to atlas ROIs, save NIfTI outputs
│   ├── Figure4_atlas_plot.py                 # Fig 4 — render task-overlay brain slices on NMT template
│   ├── Figure4_atlas_plot_corsub.py          # Fig 4 — cortical/subcortical overlay slice visualization
│   ├── Figure5_amplitude_cortical.py         # Fig 5 — electrical stimulation amplitudes in cortical areas
│   ├── Figure6_amplitude_subcortical.py      # Fig 6 — electrical stimulation amplitudes in subcortical areas
│   ├── Figure7_frequency_cortical.py         # Fig 7 — microstimulation frequencies in cortical areas
│   ├── Figure8_frequency_subcortical.py      # Fig 8 — microstimulation frequencies in subcortical areas
│   ├── Figure9_waveform_macro.py             # Fig 9 — waveforms and pulse widths (macroelectrodes)
│   ├── Figure9_waveform_micro.py             # Fig 9 — waveforms and pulse widths (microelectrodes)
│   ├── Figure10_trainlength.py               # Fig 10 — train length across all electrode types
│   ├── Figure10_trainlength_macro.py         # Fig 10 — train length for macroelectrodes
│   ├── Figure10_trainlength_micro.py         # Fig 10 — train length for microelectrodes
│   ├── Figure11_behav_outcomes.py            # Fig 11 — stimulation parameters and behavioral outcomes
│   ├── Figure12_optogenetics.py              # Fig 12 — optogenetic stimulation parameters
│   ├── FigureS2_frequency_macro_cortical.py  # Supp S2 — macrostimulation frequencies in cortical areas
│   ├── FigureS3_frequency_macro_subcortical.py # Supp S3 — macrostimulation frequencies in subcortical areas
│   ├── Statistics.py                         # OLS regression of stimulation parameters vs year
│   ├── Region_Analysis.py                    # Per-region descriptive breakdown (exploratory)
│   ├── utils.py                              # Shared helper functions
│   └── exploratory/                          # Scripts not included in the final paper
│       ├── TaskFigure.py                     # Brain maps colored by task category (NIfTI)
│       ├── TaskPlotting.py                   # Render task-colored brain maps
│       ├── EffectFigure.py                   # Brain maps colored by stimulation effect (NIfTI)
│       ├── EffectPlotting.py                 # Render effect-colored brain maps
│       ├── EffectorFigure.py                 # Brain maps colored by anatomical effector (NIfTI)
│       ├── EffectorPlotting.py               # Render effector-colored brain maps
│       └── ReviewPapers_Figures.py           # Early overview histogram (superseded by Figure3_study_charac.py)
├── data/
│   ├── SupplementaryTable_S1_Murris_et_al.xlsx   # Manuscript Supp. Table S1 — study screening (PRISMA)
│   ├── SupplementaryTable_S2_Murris_et_al.xlsx   # Manuscript Supp. Table S2 — full included-studies database
│   ├── Pubmed_Overview_Final.xlsx            # Working database (multi-sheet; used by scripts)
│   ├── Pubmed_Overview_Final_StudySelection.csv  # CSV export of study selection data
│   ├── Pubmed_Overview_Final_Tasks.xlsx      # Working file with task categorization
│   ├── tables_CHARM/                         # CHARM atlas region labels and color maps
│   ├── tables_SARM/                          # SARM atlas region labels and color maps
│   └── NMT_v2.0_sym/                         # NMT atlas (NOT included — see below)
├── figures/                                  # Final publication figures (PNG and SVG)
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

### 2. Download the NMT v2.0 atlas

The NMT v2.0 symmetric macaque brain atlas is required but too large to include in this repository (~4 GB). Download it from the AFNI/NIMH resource:

> **Seidlitz et al. (2018)** — NMT: A structural MRI atlas of the macaque brain with toolbox for surface and volumetric analyses.  
> Download: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman_primates/docs/nmt.html

Extract the archive so that the atlas files are located at:

```
data/NMT_v2.0_sym/
├── NMT_v2.0_sym_SS.nii.gz          # Skull-stripped anatomical template
├── supplemental_CHARM/              # CHARM cortical parcellations (levels 1–6)
└── supplemental_SARM/               # SARM subcortical parcellations (levels 1–6)
```

## Running the scripts

Most scripts read from `data/` and save figures to `outputs/`. Run the Figure 4 pipeline first (it generates NIfTI files that the atlas plot scripts depend on); all other figure scripts are independent.

### Figure 4 — anatomical atlas overview (run in order)

```bash
python scripts/Figure4_atlas_generate.py     # Step 1: map stimulation targets to CHARM/SARM ROIs, save NIfTI count maps
python scripts/Figure4_atlas_plot.py         # Step 2: render task-overlay slices on NMT template
python scripts/Figure4_atlas_plot_corsub.py  # Step 2: cortical/subcortical overlay slices
```

### Figure 3 and Figures 5–12

Each script is self-contained and can be run in any order:

```bash
python scripts/Figure3_study_charac.py
python scripts/Figure5_amplitude_cortical.py
python scripts/Figure6_amplitude_subcortical.py
python scripts/Figure7_frequency_cortical.py
python scripts/Figure8_frequency_subcortical.py
python scripts/Figure9_waveform_macro.py
python scripts/Figure9_waveform_micro.py
python scripts/Figure10_trainlength.py
python scripts/Figure10_trainlength_macro.py
python scripts/Figure10_trainlength_micro.py
python scripts/Figure11_behav_outcomes.py
python scripts/Figure12_optogenetics.py
```

### Supplementary figures

```bash
python scripts/FigureS2_frequency_macro_cortical.py
python scripts/FigureS3_frequency_macro_subcortical.py
```

## Data

### Supplementary data files (published with the manuscript)

- **`SupplementaryTable_S1_Murris_et_al.xlsx`** — study screening table (PRISMA). Lists all 2688 screened papers with inclusion/exclusion decisions and rejection reasons. Corresponds to the data underlying Figure 1.
- **`SupplementaryTable_S2_Murris_et_al.xlsx`** — full database of 1030 included stimulation reports, with 28 columns covering title, authors, year, journal, stimulation type, electrode type, brain area (CHARM/SARM parcellation), species, and stimulation parameters. This is the primary published dataset.

### Working data files (used by the scripts)

- **`Pubmed_Overview_Final.xlsx`** — multi-sheet working database. The `Study_Selection` sheet contains the data used by all figure scripts; the `Optogenetic_Selection` sheet is used by `Figure12_optogenetics.py`. Corresponds to `SupplementaryTable_S2` with additional working columns.
- **`Pubmed_Overview_Final_StudySelection.csv`** — CSV export of the study selection data, used by `Figure11_behav_outcomes.py`.
- **`Pubmed_Overview_Final_Tasks.xlsx`** — working file with task categorization, used by `Figure4_atlas_plot.py`.
- **`tables_CHARM/` and `tables_SARM/`** — lookup tables mapping atlas region indices to hierarchical anatomical labels.

## Citation

If you use this code or data, please cite the associated review paper (citation to be added upon publication) and the NMT v2.0 atlas:

> Seidlitz J, Sponheim C, Glen D, Ye FQ, Saleem KS, Leopold DA, Ungerleider L, Messinger A (2018). A population MRI brain template and analysis tools for the macaque. *NeuroImage*, 170, 121–131.
