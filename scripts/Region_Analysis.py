# -*- coding: utf-8 -*-
"""
Script that outputs descriptions of each anatomical region

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"(1) read in the data from the excel file"
file_path = '../data/Pubmed_Overview_Final.xlsx'
df = pd.read_excel(file_path, sheet_name='Study_Selection')

"(2) Get the main categories"
main_outputs = df['Main_output'].value_counts()
main_tasks   = df['Task_category'].value_counts()
main_effect  = df['Effect'].value_counts()

output_categories = main_outputs.keys() 
output_colours    = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#8c564b']

task_categories   = main_tasks.keys()
task_colours      = ['#7f7f7f', '#1f77b4', '#17becf', '#ff7f0e', 'indianred', '#9467bd', '#2ca02c']

# Get a new dataframe for a specific region of interest
df_region = df[df['Brain_Parc'] == 'ACC']

study_numb = len(df_region['Title'])
study_perc = study_numb / len(df['Title']) * 100

year_range  = [df_region['Year'].min(), df_region['Year'].max()]
year_median = df_region['Year'].median()

df_region['Effect'].value_counts()

output_counts = df_region['Main_output'].value_counts()

# Reindex counts to include all categories, missing ones get 0
output_counts = output_counts.reindex(output_categories, fill_value=0)

# Plot
# Create pie chart
plt.figure(figsize=(8,8))
plt.pie(output_counts, labels=output_counts.keys(), autopct='%1.1f%%', startangle=140, colors=output_colours)
plt.title('Distribution of Task Categories')
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
plt.show()

task_counts = df_region['Task_category'].value_counts()

# Reindex counts to include all categories, missing ones get 0
task_counts = task_counts.reindex(task_categories, fill_value=0)

# Plot
plt.figure(figsize=(10,6))
task_counts.plot(kind='bar', color=task_colours)
plt.xlabel('Task Category')
plt.ylabel('Count')
plt.title('Counts per Task Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()









description = ' ; '.join(df_region['Behavioural_description'].astype(str))

