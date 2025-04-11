#!/usr/bin/env python

import os
import re
import unicodedata
import pandas as pd

print("Starting Excel to CSV conversion...")

# Get paths
path_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Base path of the repository
path_file = os.path.join(path_base, "data", "raw") # Path to the raw folder
path_output = os.path.join(path_base, "data", "raw")
input_file = "data.xlsx" # Input file name

# Check if input file exists
input_path = os.path.join(path_file, input_file)
if not os.path.exists(input_path):
    print(f"ERROR: Input file not found at {input_path}")
    exit(1)

# Load all sheets from Excel file
excel_data = pd.read_excel(input_path, sheet_name=None)

def clean_sheet_name(sheet_name):
    normalized = unicodedata.normalize("NFKD", sheet_name) # Remove accents
    normalized = "".join([c for c in normalized if not unicodedata.combining(c)])
    clean_name = re.sub(r"[^a-zA-Z0-9]", "", normalized) # Remove special characters
    if clean_name:
        clean_name = clean_name[0].lower() + clean_name[1:] # Make just the first letter lowercase
    return clean_name

# Clean sheet names
excel_data = {clean_sheet_name(sheet_name): df for sheet_name, df in excel_data.items()}

for sheet_name, df in excel_data.items():
    print(f"\nProcessing sheet: {sheet_name} ({len(df)} rows)")
    
    if sheet_name == "data":
        if "HOURS" in df.columns:
            df.rename(columns={"HOURS": "FORECAST"}, inplace=True) # Rename HOURS to FORECAST
        if "START" in df.columns:
            # Convert START column to datetime to ensure proper sorting
            df["START"] = pd.to_datetime(df["START"], errors="coerce")
            df.sort_values(by="START", ascending=False, inplace=True) # Order by START column, recent first
    
    elif sheet_name == "clients":
        if "CLIENT_NAME" in df.columns:
            df.sort_values(by="CLIENT_NAME", ascending=True, inplace=True) # Order alphabetically
    
    elif sheet_name == "schedules":
        if "NAME" in df.columns:
            df.sort_values(by="NAME", ascending=True, inplace=True) # Order alphabetically
    
    elif sheet_name == "translatorsCostPairs":
        if "TRANSLATOR" in df.columns:
            df.sort_values(by="TRANSLATOR", ascending=True, inplace=True) # Order alphabetically
    
    # Save sorted dataframe to CSV
    output_file = os.path.join(path_output, f"{sheet_name}.csv")
    df.to_csv(
        output_file,
        index=False,
        sep=","
    )
    print(f"  Saved to {output_file}")