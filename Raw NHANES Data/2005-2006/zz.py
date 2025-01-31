import pandas as pd
import numpy as np

# Load 2005-2006 datasets (suffix = _D.xpt)
datasets = {
    'DEMO': pd.read_sas('DEMO_D.xpt'),
    'BMX': pd.read_sas('BMX_D.xpt'),
    'BPX': pd.read_sas('BPX_D.xpt'),
    'BPQ': pd.read_sas('BPQ_D.xpt'),
    'DIQ': pd.read_sas('DIQ_D.xpt'),  # Diabetes questionnaire
    'GLU': pd.read_sas('GLU_D.xpt'),
    'HDL': pd.read_sas('HDL_D.xpt'),
    'MCQ': pd.read_sas('MCQ_D.xpt'),
    'RXQ': pd.read_sas('RXQ_RX_D.xpt'),  # Prescription medications
    'SMQ': pd.read_sas('SMQ_D.xpt'),
    'TCHOL': pd.read_sas('TCHOL_D.xpt'),
    'TRIGLY': pd.read_sas('TRIGLY_D.xpt')  # Contains LDL (LBDLDL)
}

# Extract variables (2005-2006 compatible)
variables = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
    'BMX': ['SEQN', 'BMXBMI'],
    'BPX': ['SEQN', 'BPXSY1'],
    'BPQ': ['SEQN', 'BPQ020', 'BPQ090D'],
    'DIQ': ['SEQN', 'DIQ010'],
    'GLU': ['SEQN', 'LBXGLU'],
    'HDL': ['SEQN', 'LBDHDD'],
    'MCQ': ['SEQN', 'MCQ160E', 'MCQ160C'],
    'RXQ': ['SEQN', 'RXDDRUG', 'RXDDRGID'],  # Drug name/code for statins
    'SMQ': ['SEQN', 'SMQ020'],
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

# --- Statin Identification ---
# Multum Lexicon codes for statins (consistent across cycles)
statin_codes = [
    'd00973',  # Atorvastatin
    'd00974',  # Simvastatin
    'd00975',  # Pravastatin
    'd00976',  # Fluvastatin
    'd00977',  # Lovastatin
    'd00978',  # Rosuvastatin
]

merged['Statin'] = np.where(
    merged['RXDDRGID'].isin(statin_codes), 1, 0
)

# --- Remaining Variables ---
merged['Hypertension'] = np.where(
    (merged['BPXSY1'] >= 140) |
    (merged['BPQ020'] == 1) |
    (merged['BPQ090D'] == 1), 1, 0
)

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

# Clean data
final_df = final_df.dropna(subset=['HeartDisease'])
final_df.to_csv('heart_disease_2005_2006.csv', index=False)

print(f"Dataset saved with {len(final_df)} participants")