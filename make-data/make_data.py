'''
Description
---
This function does feature extraction and drops superfluous columns.
'''

# Import required modules
import pandas as pd
import os
from metaparameters import metaparameters
from preprocessing import feature_extraction, drop_superfluous_columns, finalize_data

# Specify the study parameters
print("Specifying the metaparameters")
metaparameters()

# Read study specification
print("Reading study specification")
metaparameters = pd.read_json("metaparameters.json")

# Only read data if we have not done feature extraction yet
print("Reading the raw data")
log_data = pd.read_csv(metaparameters["log_data_path"][0], index_col = 0)
esm_data = pd.read_csv(metaparameters["esm_data_path"][0], index_col = 0, low_memory = False, usecols = metaparameters["self_report_columns"][0])

# Drop missingness in the targets
print("Dropping missingness in the targets")
esm_data.dropna(subset = metaparameters["targets"][0], inplace = True)

# Extract features
if "data.csv" not in os.listdir():
    print("Currently doing feature extraction.")
    feature_extraction(esm_data, log_data, metaparameters.iloc[0,:])
else:
    print("I see you've got yourself some data.csv here already, so I'm skipping feature extraction for now.")

# Remove superfluous columns
print("Dropping columns.")
drop_superfluous_columns()

# Finalize the dataset
print("Finalizing dataset")
finalize_data()