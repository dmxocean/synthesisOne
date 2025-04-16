import re
import os
import sys
import pandas as pd

# Setup paths
path_base = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(path_base)
data_raw_dir = os.path.join(path_base, "data", "raw")

# Load all CSV files
df_clients = pd.read_csv(os.path.join(data_raw_dir, "clients.csv"))
df_data = pd.read_csv(os.path.join(data_raw_dir, "data.csv"))
df_schedules = pd.read_csv(os.path.join(data_raw_dir, "schedules.csv"))
df_translators = pd.read_csv(os.path.join(data_raw_dir, "translatorsCostPairs.csv"))

# Clients
df_clients = df_clients.drop_duplicates()
df_clients = df_clients.dropna()
df_clients.to_csv(os.path.join(data_raw_dir, "clients.csv"), index=False)

# Data
df_data = df_data.drop_duplicates()
df_data.to_csv(os.path.join(data_raw_dir, "data.csv"), index=False)

# Schedules
time_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}$")  # Define a regex pattern for HH:MM:SS format

# Remove the "YYYY-MM-DD " part from START and END columns
df_schedules["START"] = df_schedules["START"].str.split(" ").str[-1]
df_schedules["END"] = df_schedules["END"].str.split(" ").str[-1]

df_schedules.to_csv(os.path.join(data_raw_dir, "schedules.csv"), index=False)

# Translators Cost Pairs
df_translators = df_translators.drop_duplicates()
df_translators = df_translators.dropna()
df_translators.to_csv(os.path.join(data_raw_dir, "translatorsCostPairs.csv"), index=False)