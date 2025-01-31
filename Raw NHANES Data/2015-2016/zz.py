import pandas as pd
import numpy as np

# Load datasets with corrected names
datasets = {
    'DEMO': pd.read_sas('DEMO_I.xpt'),
    'BMX': pd.read_sas('BMX_I.xpt'),
    'BPX': pd.read_sas('BPX_I.xpt'),
    'BPQ': pd.read_sas('BPQ_I.xpt'),
    'DIQ': pd.read_sas('DIQ_I.xpt'),
    'GLU': pd.read_sas('GLU_I.xpt'),
    'HDL': pd.read_sas('HDL_I.xpt'),
    'MCQ': pd.read_sas('MCQ_I.xpt'),  # Corrected from MCO to MCQ
    'RXQ': pd.read_sas('RXQ_RX_I.xpt'),
    'SMQ': pd.read_sas('SMQ_I.xpt'),   # Corrected from SMO to SMQ
    'TCHOL': pd.read_sas('TCHOL_I.xpt'),
    'TRIGLY': pd.read_sas('TRIGLY_I.xpt')
}

# Extract variables
variables = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
    'BMX': ['SEQN', 'BMXBMI'],
    'BPX': ['SEQN', 'BPXSY1'],
    'BPQ': ['SEQN', 'BPQ020', 'BPQ090D'],
    'DIQ': ['SEQN', 'DIQ010'],
    'GLU': ['SEQN', 'LBXGLU'],
    'HDL': ['SEQN', 'LBDHDD'],
    'MCQ': ['SEQN', 'MCQ160E', 'MCQ160C'],
    'RXQ': ['SEQN', 'RXDRSC1'],
    'SMQ': ['SEQN', 'SMQ020'],  # Corrected SMQ variables
    'TCHOL': ['SEQN', 'LBXTC'],
    'TRIGLY': ['SEQN', 'LBDLDL']
}

# Create subset DataFrames
dfs = {name: datasets[name][cols].copy() for name, cols in variables.items()}

# Merge datasets
merged = dfs['DEMO']
for name in ['BMX', 'BPX', 'BPQ', 'DIQ', 'GLU', 'HDL',
            'MCQ', 'RXQ', 'SMQ', 'TCHOL', 'TRIGLY']:
    merged = merged.merge(dfs[name], on='SEQN', how='left')

# Convert formats
merged['RIAGENDR'] = merged['RIAGENDR'].replace({1: 'Male', 2: 'Female'})

# Derived variables
merged['Hypertension'] = np.where(
    (merged['BPXSY1'] >= 140) |
    (merged['BPQ020'] == 1) |
    (merged['BPQ090D'] == 1), 1, 0
)

merged['Statin'] = merged['RXDRSC1'].apply(
    lambda x: 1 if x == 242 else 0 if pd.notnull(x) else np.nan)

merged['HeartDisease'] = np.where(
    (merged['MCQ160E'] == 1) | (merged['MCQ160C'] == 1), 1, 0)

# Binary conversions
binary_vars = {'DIQ010': 1, 'SMQ020': 1}
for var, yes_code in binary_vars.items():
    merged[var] = merged[var].replace({yes_code: 1, 2: 0, 7: np.nan, 9: np.nan})

# Final selection and renaming
final_df = merged[[
    'SEQN', 'RIDAGEYR', 'RIAGENDR',
    'LBXTC', 'LBDHDD', 'LBDLDL',
    'LBXGLU', 'BMXBMI', 'BPXSY1',
    'Hypertension', 'DIQ010', 'SMQ020',
    'Statin', 'HeartDisease'
]].rename(columns={
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex',
    'LBXTC': 'TotalCholesterol',
    'LBDHDD': 'HDL',
    'LBDLDL': 'LDL',
    'LBXGLU': 'FastingGlucose',
    'BMXBMI': 'BMI',
    'BPXSY1': 'SystolicBP',
    'DIQ010': 'Diabetes',
    'SMQ020': 'Smoking'
})

# Clean data (only remove rows missing target/LDL)
final_df = final_df.dropna(subset=['HeartDisease'])

# Save to CSV
final_df.to_csv('heart_disease_2015_2016.csv', index=False)

print(f"Dataset saved with {len(final_df)} participants")