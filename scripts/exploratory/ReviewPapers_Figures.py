#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:41:29 2024

@author: sjoerdmurris
"""
# Summarize data for review paper
# Import tabular data

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

study_overview = str(Path(__file__).resolve().parent.parent.parent / 'data' / 'Pubmed_Study_7.xlsx')
df = pd.read_excel(study_overview, sheet_name='Study_Summary')

# Parse whether target is cortical or subcortical
curr_selec = df[0:122] # grab the current selection
cor_values = curr_selec['Cortical_Subcortical'].value_counts()

plt.pie(cor_values, labels=["Cortical", "Subcortical"])

# Clean stimulation frequency values
curr_selec.Frequency

df['Year'] = df.Author.str.extract('(\d+)')
pd.unique(df.Included)

# Get rid of the NaN indices at the bottom
df = df.dropna(subset=['Year'])
df['Year'] = pd.to_numeric(df['Year'])

# Isolate YES indications -> studies that are relevant
yf = df[(df['Included'] == 'YES') | (df['Included'] == 'YES ')]

# Only include electrical microstimulation
ef = yf[['elec' in s for s in yf.Type]]

# Only include optogenetic stimulation
of = yf[['opto' in s for s in yf.Type]]

# Make a histogram based on stimulation method and year
year_column = ef['Year'].to_numpy()
year_column_all = df['Year'].to_numpy()
year_column_opt = of['Year'].to_numpy()

unique, counts = np.unique(year_column, return_counts=True)
unique2, counts2 = np.unique(year_column_all, return_counts=True)
unique3, counts3 = np.unique(year_column_opt, return_counts=True)

plt.bar(unique2,counts2,color='gray')
plt.bar(unique,counts,color='black')
plt.bar(unique3,counts3,color='blue')

plt.figure().set_figwidth(15)
plt.figure().set_figheight(2)

plt.show()



