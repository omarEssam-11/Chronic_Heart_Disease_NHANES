import pandas as pd
import numpy as np

# Load datasets (use your downloaded files)
datasets = {
    'DEMO': pd.read_sas('P_DEMO.xpt'),
    'BMX': pd.read_sas('P_BMX.xpt'),
    'BPX': pd.read_sas('P_BPXO.xpt'),
    'TCHOL': pd.read_sas('P_TCHOL.xpt'),
    'HDL': pd.read_sas('P_HDL.xpt'),
    'TRIGLY': pd.read_sas('P_TRIGLY.xpt'),  # Contains pre-calculated LDL
    'GLU': pd.read_sas('P_GLU.xpt'),
    'MCQ': pd.read_sas('P_MCQ.xpt'),
    'DIQ': pd.read_sas('P_DIQ.xpt'),
    'SMQ': pd.read_sas('P_SMQ.xpt'),
    'BPQ': pd.read_sas('P_BPQ.xpt')
}

# Extract key variables (no longer need RXQ dataset)
variables = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
    'BMX': ['SEQN', 'BMXBMI'],
    'BPX': ['SEQN', 'BPXOSY1'],
    'TCHOL': ['SEQN', 'LBXTC'],
    'HDL': ['SEQN', 'LBDHDD'],
    'TRIGLY': ['SEQN', 'LBDLDL'],  # Use existing LDL
    'GLU': ['SEQN', 'LBXGLU'],
    'MCQ': ['SEQN', 'MCQ160E', 'MCQ160C'],
    'DIQ': ['SEQN', 'DIQ010'],
    'SMQ': ['SEQN', 'SMQ020'],
    'BPQ': ['SEQN', 'BPQ020', 'BPQ090D']
}

# Create subset DataFrames
dfs = {}
for name, cols in variables.items():
    dfs[name] = datasets[name][cols].copy()

# Merge datasets
merged = dfs['DEMO']
for name in ['BMX', 'BPX', 'TCHOL', 'HDL', 'TRIGLY', 'GLU',
            'MCQ', 'DIQ', 'SMQ', 'BPQ']:
    merged = merged.merge(dfs[name], on='SEQN', how='left')

# Convert SAS formats to Python types
merged['RIAGENDR'] = merged['RIAGENDR'].replace({1: 'Male', 2: 'Female'})

# Derived Variables -----------------------------------------------------------
# 1. Hypertension Status (no longer need statin)
merged['Hypertension'] = np.where(
    (merged['BPXOSY1'] >= 140) |
    (merged['BPQ020'] == 1) |
    (merged['BPQ090D'] == 1), 1, 0
)

# 2. Target Variable: Heart Disease
merged['HeartDisease'] = np.where(
    (merged['MCQ160E'] == 1) | (merged['MCQ160C'] == 1), 1, 0)

# 3. Convert Yes/No variables to binary
binary_vars = {'DIQ010': 1, 'SMQ020': 1}  # Diabetes, Smoking
for var, yes_code in binary_vars.items():
    merged[var] = merged[var].replace({yes_code: 1, 2: 0, 7: np.nan, 9: np.nan})

# Final Feature Selection ------------------------------------------------------
final_columns = [
    'SEQN', 'RIDAGEYR', 'RIAGENDR',   # Demographics
    'LBXTC', 'LBDHDD', 'LBDLDL',      # Cholesterol (LDL now from TRIGLY)
    'LBXGLU',                         # Glucose
    'BMXBMI',                         # BMI
    'BPXOSY1', 'Hypertension',         # Blood Pressure
    'DIQ010',                         # Diabetes
    'SMQ020',                         # Smoking
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
# final_df = final_df.dropna(thresh=0.8*len(final_df.columns))



final_df.to_csv('heart_disease_2017_2020.csv', index=False)
print(f"Final dataset shape: {final_df.shape}")


import pandas as pd
import numpy as np

# Load all datasets using the filenames from your image
datasets = {
    'DEMO': pd.read_sas('P_DEMO.xpt'),
    'BMX': pd.read_sas('P_BMX.xpt'),
    'BPX': pd.read_sas('P_BPXO.xpt'),
    'TCHOL': pd.read_sas('P_TCHOL.xpt'),
    'HDL': pd.read_sas('P_HDL.xpt'),
    'TRIGLY': pd.read_sas('P_TRIGLY.xpt'),
    'GLU': pd.read_sas('P_GLU.xpt'),
    'MCQ': pd.read_sas('P_MCQ.xpt'),
    'DIQ': pd.read_sas('P_DIQ.xpt'),
    'SMQ': pd.read_sas('P_SMQ.xpt'),
    'BPQ': pd.read_sas('P_BPQ.xpt'),
    'RXQ': pd.read_sas('P_RXQ_RX.xpt')
}

# Extract key variables from each dataset
variables = {
    'DEMO': ['SEQN', 'RIDAGEYR', 'RIAGENDR'],
    'BMX': ['SEQN', 'BMXBMI'],
    'BPX': ['SEQN', 'BPXOSY1'],
    'TCHOL': ['SEQN', 'LBXTC'],
    'HDL': ['SEQN', 'LBDHDD'],
    'TRIGLY': ['SEQN', 'LBXTR'],
    'GLU': ['SEQN', 'LBXGLU'],
    'MCQ': ['SEQN', 'MCQ160E', 'MCQ160C'],
    'DIQ': ['SEQN', 'DIQ010'],
    'SMQ': ['SEQN', 'SMQ020'],
    'BPQ': ['SEQN', 'BPQ020', 'BPQ090D'],
    'RXQ': ['SEQN', 'RXDDRUG', 'RXDRSC1']
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

# 1. Calculate LDL Cholesterol (Friedewald equation)
merged['LDL'] = merged['LBXTC'] - merged['LBDHDD'] - (merged['LBXTR']/5)
merged['LDL'] = merged['LDL'].where(merged['LBXTR'] < 400, np.nan)  # Validity check

# 2. Hypertension Status
merged['Hypertension'] = np.where(
    (merged['BPXOSY1'] >= 140) |
    (merged['BPQ020'] == 1) |
    (merged['BPQ090D'] == 1), 1, 0
)

# 3. Statin Use
merged['Statin'] = merged['RXDRSC1'].apply(
    lambda x: 1 if x == 242 else 0 if pd.notnull(x) else np.nan)

# 4. Target Variable: Heart Disease
merged['HeartDisease'] = np.where(
    (merged['MCQ160E'] == 1) | (merged['MCQ160C'] == 1), 1, 0)

# 5. Convert Yes/No variables to binary (1/0)
binary_vars = {
    'DIQ010': 1,  # Diabetes
    'SMQ020': 1,  # Smoking
}

for var, yes_code in binary_vars.items():
    merged[var] = merged[var].replace({yes_code: 1, 2: 0, 7: np.nan, 9: np.nan})

# Final Feature Selection ------------------------------------------------------
final_columns = [
    'SEQN', 'RIDAGEYR', 'RIAGENDR',         # Demographics
    'LBXTC', 'LBDHDD', 'LDL',               # Cholesterol
    'LBXGLU',                               # Glucose
    'BMXBMI',                               # BMI
    'BPXOSY1', 'Hypertension',               # Blood Pressure
    'DIQ010',                               # Diabetes
    'SMQ020',                               # Smoking
    'BPQ090D', 'Statin',                    # Medications
    'HeartDisease'                          # Target
]

final_df = merged[final_columns].copy()

# Rename columns for clarity
column_names = {
    'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex',
    'LBXTC': 'TotalCholesterol',
    'LBDHDD': 'HDL',
    'LBXGLU': 'FastingGlucose',
    'BMXBMI': 'BMI',
    'BPXOSY1': 'SystolicBP',
    'DIQ010': 'Diabetes',
    'SMQ020': 'Smoking',
    'BPQ090D': 'BP_Medication'
}

final_df = final_df.rename(columns=column_names)

# Clean missing values (adjust threshold as needed)
final_df = final_df.dropna(subset=['HeartDisease'])  # Remove rows missing target
# final_df = final_df.dropna(thresh=0.8*len(final_df.columns))  # Keep rows with ≥80% data

final_df.to_csv('heart_disease_2017_2020.csv', index=False)
print(f"Final dataset shape: {final_df.shape}")

