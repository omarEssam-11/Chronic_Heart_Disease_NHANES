import pandas as pd
import numpy as np

# Load datasets
datasets = {
    'DEMO': pd.read_sas('P_DEMO.xpt'),
    'BMX': pd.read_sas('P_BMX.xpt'),
    'BPX': pd.read_sas('P_BPXO.xpt'),
    'TCHOL': pd.read_sas('P_TCHOL.xpt'),
    'HDL': pd.read_sas('P_HDL.xpt'),
    'TRIGLY': pd.read_sas('P_TRIGLY.xpt'),  # Contains LDL (LBDLDL)
    'GLU': pd.read_sas('P_GLU.xpt'),
    'MCQ': pd.read_sas('P_MCQ.xpt'),
    'DIQ': pd.read_sas('P_DIQ.xpt'),
    'SMQ': pd.read_sas('P_SMQ.xpt'),
    'BPQ': pd.read_sas('P_BPQ.xpt'),
    'RXQ': pd.read_sas('P_RXQ_RX.xpt')  # Keep statin data
}

# Extract key variables
variables = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
    'BMX': ['SEQN', 'BMXBMI'],
    'BPX': ['SEQN', 'BPXOSY1'],
    'TCHOL': ['SEQN', 'LBXTC'],
    'HDL': ['SEQN', 'LBDHDD'],
    'TRIGLY': ['SEQN', 'LBDLDL'],  # Pre-calculated LDL
    'GLU': ['SEQN', 'LBXGLU'],
    'MCQ': ['SEQN', 'MCQ160E', 'MCQ160C'],
    'DIQ': ['SEQN', 'DIQ010'],
    'SMQ': ['SEQN', 'SMQ020'],
    'BPQ': ['SEQN', 'BPQ020', 'BPQ090D'],
    'RXQ': ['SEQN', 'RXDRSC1']  # Statin variable
}

# Create subset DataFrames
dfs = {}
for name, cols in variables.items():
    dfs[name] = datasets[name][cols].copy()

# Merge all datasets
merged = dfs['DEMO']
for name in ['BMX', 'BPX', 'TCHOL', 'HDL', 'TRIGLY', 'GLU',
            'MCQ', 'DIQ', 'SMQ', 'BPQ', 'RXQ']:
    merged = merged.merge(dfs[name], on='SEQN', how='left')

# Convert SAS formats to Python types
merged['RIAGENDR'] = merged['RIAGENDR'].replace({1: 'Male', 2: 'Female'})

# Derived Variables -----------------------------------------------------------
# 1. Hypertension Status
merged['Hypertension'] = np.where(
    (merged['BPXOSY1'] >= 140) |
    (merged['BPQ020'] == 1) |
    (merged['BPQ090D'] == 1), 1, 0
)

# 2. Statin Use (keep this)
merged['Statin'] = merged['RXDRSC1'].apply(
    lambda x: 1 if x == 242 else 0 if pd.notnull(x) else np.nan)

# 3. Target Variable: Heart Disease
merged['HeartDisease'] = np.where(
    (merged['MCQ160E'] == 1) | (merged['MCQ160C'] == 1), 1, 0)

# 4. Convert Yes/No variables to binary
binary_vars = {'DIQ010': 1, 'SMQ020': 1}  # Diabetes, Smoking
for var, yes_code in binary_vars.items():
    merged[var] = merged[var].replace({yes_code: 1, 2: 0, 7: np.nan, 9: np.nan})

# Final Feature Selection ------------------------------------------------------
final_columns = [
    'SEQN', 'RIDAGEYR', 'RIAGENDR',   # Demographics
    'LBXTC', 'LBDHDD', 'LBDLDL',      # Cholesterol (pre-calculated LDL)
    'LBXGLU',                         # Glucose
    'BMXBMI',                         # BMI
    'BPXOSY1', 'Hypertension',         # Blood Pressure
    'DIQ010',                         # Diabetes
    'SMQ020',                         # Smoking
    'Statin',                         # Statin Use (retained)
    'HeartDisease'                    # Target
]

final_df = merged[final_columns].copy()

# Rename columns
column_names = {
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex',
    'LBXTC': 'TotalCholesterol',
    'LBDHDD': 'HDL',
    'LBDLDL': 'LDL',
    'LBXGLU': 'FastingGlucose',
    'BMXBMI': 'BMI',
    'BPXOSY1': 'SystolicBP',
    'DIQ010': 'Diabetes',
    'SMQ020': 'Smoking'
}

final_df = final_df.rename(columns=column_names)

# Clean missing values
final_df = final_df.dropna(subset=['HeartDisease', 'LDL'])  # Critical variables

print(f"Final dataset shape: {final_df.shape}")

final_df.to_csv('aaaaaaaaaaaaaaaaaaaaaa.csv', index=False)