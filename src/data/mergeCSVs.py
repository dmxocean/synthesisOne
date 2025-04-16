import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

path_base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Base path of the repository
sys.path.append(path_base)  # Add base path to system path

from utils.logging import logger_setup, log_section
from utils.merge import log_dataframe_info, log_merge_stats

pd.set_option("display.max_columns", None) # Show all columns

path_raw_data = os.path.join(path_base, "data", "raw")
path_interim_data = os.path.join(path_base, "data", "interim")

os.makedirs(path_interim_data, exist_ok=True) # Ensure the interim directory exists

path_logs = os.path.join(path_interim_data, "logs") # Set up log directory
os.makedirs(path_logs, exist_ok=True)

# Set up our main logger
main_logger = logger_setup("InfoMerge", path_logs, "infoMerge.log")
main_logger.info("Starting data merge process")
main_logger.info(f"Reading data from: {path_raw_data}")
main_logger.info(f"Saving results to: {path_interim_data}")

def check_missing_threshold(df, key_columns, df_name, threshold=0.05):
    """
    Check if missing values in key columns exceed the threshold
        - If exceed threshold, will return False and we should stop processing
        - If not exceeded threshold, will return True and we can proceed
    
    Params:
        - df (DataFrame): DataFrame to check for missing values
        - key_columns (list): List of columns to check for missing values
        - df_name (str): Name of the DataFrame for logging purposes
        - threshold (float): Threshold for missing value percentage
    
    Returns:
        - bool: True if missing percentage is below threshold, False otherwise
        - float: Percentage of missing values
        - DataFrame: Rows with missing values in key columns
    """
    
    total_rows = len(df)
    rows_with_missing = df[df[key_columns].isnull().any(axis=1)]
    missing_count = len(rows_with_missing)
    missing_percentage = missing_count / total_rows if total_rows > 0 else 0
    
    main_logger.info(f"")
    main_logger.info(f"Checking missing values in {df_name} key columns: {key_columns}")
    main_logger.info(f"  Total rows in {df_name}: {total_rows}")
    main_logger.info(f"  Rows with missing values in key columns: {missing_count}")
    main_logger.info(f"  Missing percentage: {missing_percentage:.4f} (threshold: {threshold:.4f})")
    
    return missing_percentage < threshold, missing_percentage, rows_with_missing

# Load all data files
path_data = os.path.join(path_raw_data, "data.csv")
path_translator_pairs = os.path.join(path_raw_data, "translatorsCostPairs.csv")
path_schedules = os.path.join(path_raw_data, "schedules.csv")
path_clients = os.path.join(path_raw_data, "clients.csv")

df_data = pd.read_csv(path_data)  # Historical data containing project information
df_translator_pairs = pd.read_csv(path_translator_pairs)  # Current translator rates
df_schedules = pd.read_csv(path_schedules)  # Translator schedules
df_clients = pd.read_csv(path_clients)  # Client information

main_logger.info(f"Loaded {len(df_data)} rows from data.csv")
main_logger.info(f"Loaded {len(df_translator_pairs)} rows from translatorsCostPairs.csv")
main_logger.info(f"Loaded {len(df_schedules)} rows from schedules.csv")
main_logger.info(f"Loaded {len(df_clients)} rows from clients.csv")

# Data overview
main_logger.info("Data Overview:")
main_logger.info(f"- data.csv: {df_data.shape[0]} rows, {df_data.shape[1]} columns")
main_logger.info(f"- translatorsCostPairs.csv: {df_translator_pairs.shape[0]} rows, {df_translator_pairs.shape[1]} columns")
main_logger.info(f"- schedules.csv: {df_schedules.shape[0]} rows, {df_schedules.shape[1]} columns")
main_logger.info(f"- clients.csv: {df_clients.shape[0]} rows, {df_clients.shape[1]} columns")

# Missing values analysis
missing_data = df_data.isnull().sum() # Analyze missing values in data.csv
main_logger.info(f"\nMissing values in data.csv: \n\n{missing_data}\n")

missing_translator_pairs = df_translator_pairs.isnull().sum() # Analyze missing values in translatorsCostPairs.csv
main_logger.info(f"\nMissing values in translatorsCostPairs.csv: \n\n{missing_translator_pairs}\n")

missing_schedules = df_schedules.isnull().sum() # Analyze missing values in schedules.csv
main_logger.info(f"\nMissing values in schedules.csv: \n\n{missing_schedules}\n")

missing_clients = df_clients.isnull().sum() # Analyze missing values in clients.csv
main_logger.info(f"\nMissing values in clients.csv: \n\n{missing_clients}\n")

# Check and clean missing values in key columns for data.csv
data_key_columns = ["TRANSLATOR", "SOURCE_LANG", "TARGET_LANG", "MANUFACTURER"]
can_proceed_data, missing_pct_data, rows_deletion = check_missing_threshold(
    df_data, data_key_columns, "data.csv"
)

# Log details of rows with missing key values
if len(rows_deletion) > 0:
    main_logger.info("  Rows with missing values in key columns for data.csv:")
    for idx, row in rows_deletion.iterrows():
        missing_cols = [col for col in data_key_columns if pd.isnull(row[col])]
        missing_cols_str = ", ".join(missing_cols)
        main_logger.info(f"    - Row {idx}: Missing values in {missing_cols_str}")

# Check translatorsCostPairs.csv
translator_key_columns = ["TRANSLATOR", "SOURCE_LANG", "TARGET_LANG"]
can_proceed_translator, missing_pct_translator, rows_to_remove_translator = check_missing_threshold(
    df_translator_pairs, translator_key_columns, "translatorsCostPairs.csv"
)

# Check schedules.csv
schedules_key_columns = ["NAME"]
can_proceed_schedules, missing_pct_schedules, rows_to_remove_schedules = check_missing_threshold(
    df_schedules, schedules_key_columns, "schedules.csv"
)

# Check clients.csv
clients_key_columns = ["CLIENT_NAME"]
can_proceed_clients, missing_pct_clients, rows_to_remove_clients = check_missing_threshold(
    df_clients, clients_key_columns, "clients.csv"
)

# Determine if we can proceed with the merge
can_proceed_overall = can_proceed_data and can_proceed_translator and can_proceed_schedules and can_proceed_clients

if not can_proceed_overall:
    main_logger.error("DATA QUALITY ERROR: Missing values in key columns exceed the 5% threshold")
    
    if not can_proceed_data:
        main_logger.error(f"data.csv: {missing_pct_data:.2%} rows missing values in {data_key_columns}")
    
    if not can_proceed_translator:
        main_logger.error(f"translatorsCostPairs.csv: {missing_pct_translator:.2%} rows missing values in {translator_key_columns}")
    
    if not can_proceed_schedules:
        main_logger.error(f"schedules.csv: {missing_pct_schedules:.2%} rows missing values in {schedules_key_columns}")
    
    if not can_proceed_clients:
        main_logger.error(f"clients.csv: {missing_pct_clients:.2%} rows missing values in {clients_key_columns}")
    
    main_logger.error("TERMINATING MERGE PROCESS... DUE TO DATA QUALITY ISSUES CHECK THEM BEFORE MERGING")
    raise SystemExit("Exiting due to data quality errors in key columns")
else:
    main_logger.info("")
    main_logger.info("PROCEEDING WITH DATA CLEANING... (ALL KEY COLUMNS HAVE ACCEPTABLE MISSING VALUE RATES)")
    main_logger.info("")

# Get all rows that need to be removed from data.csv
rows_deletion = rows_deletion.drop_duplicates()
rows_before_total = len(df_data)

# Drop rows with missing values
df_data.dropna(inplace=True)
rows_after_total = len(df_data)

# Calculate the number of rows removed due to missing values
deletion_total = rows_before_total - rows_after_total

main_logger.info(f"Removed {deletion_total} rows with missing values from data.csv")
main_logger.info(f"Remaining rows after cleaning: {rows_after_total} ({(rows_after_total/rows_before_total)*100:.5f}% of original)")

# Additional: Remove duplicate rows based on the TASK_ID column
rows_before_task_id_dedup = len(df_data)
df_data = df_data.drop_duplicates(subset=['TASK_ID'])
rows_after_task_id_dedup = len(df_data)

# Number of rows removed due to duplicate TASK_ID
task_id_dedup_total = rows_before_task_id_dedup - rows_after_task_id_dedup

main_logger.info(f"Removed {task_id_dedup_total} duplicate rows based on TASK_ID from data.csv")
main_logger.info(f"Remaining rows after TASK_ID deduplication: {rows_after_task_id_dedup} ({(rows_after_task_id_dedup/rows_before_task_id_dedup)*100:.5f}% of original)")


# Show details of removed rows
if deletion_total > 0:
    main_logger.info(f"\nDetails of removed rows: \n\n{rows_deletion[['PROJECT_ID', 'TRANSLATOR', 'SOURCE_LANG', 'TARGET_LANG', 'MANUFACTURER']].to_string()}\n")

# Clean the translator pairs dataframe if needed
if len(rows_to_remove_translator) > 0:
    translator_rows_before = len(df_translator_pairs)
    df_translator_pairs = df_translator_pairs.dropna(subset=translator_key_columns)
    translator_rows_after = len(df_translator_pairs)
    translator_removed = translator_rows_before - translator_rows_after
    
    main_logger.info(f"Removed {translator_removed} rows with missing values from translatorsCostPairs.csv")

# Clean the schedules dataframe if needed
if len(rows_to_remove_schedules) > 0:
    schedules_rows_before = len(df_schedules)
    df_schedules = df_schedules.dropna(subset=schedules_key_columns)
    schedules_rows_after = len(df_schedules)
    schedules_removed = schedules_rows_before - schedules_rows_after
    
    main_logger.info(f"Removed {schedules_removed} rows with missing values from schedules.csv")

# Clean the clients dataframe if needed
if len(rows_to_remove_clients) > 0:
    clients_rows_before = len(df_clients)
    df_clients = df_clients.dropna(subset=clients_key_columns)
    clients_rows_after = len(df_clients)
    clients_removed = clients_rows_before - clients_rows_after
    
    main_logger.info(f"Removed {clients_removed} rows with missing values from clients.csv")

main_logger.info(f"\nSample of data.csv (after cleaning): \n\n{df_data.head().to_string()}\n")

# Logger for the translator merge
translator_logger = logger_setup("TranslatorMerge", path_logs, "translatorsCostPairsMerge.log")
log_section(translator_logger, "PROCESSING TRANSLATOR COST PAIRS MERGE")

log_dataframe_info(translator_logger, df_data, "Cleaned data")
log_dataframe_info(translator_logger, df_translator_pairs, "Translator pairs")

# Composite key for identifying unique translator-language pairs
df_data["PAIR_KEY"] = df_data.apply(
    lambda row: (row["TRANSLATOR"], row["SOURCE_LANG"], row["TARGET_LANG"]), 
    axis=1
)
df_translator_pairs["PAIR_KEY"] = df_translator_pairs.apply(
    lambda row: (row["TRANSLATOR"], row["SOURCE_LANG"], row["TARGET_LANG"]), 
    axis=1
)

set_existing_pairs = set(df_data["PAIR_KEY"]) # Find existing pairs in historical data
set_translator_pairs = set(df_translator_pairs["PAIR_KEY"]) # Find pairs in translator cost pairs
set_missing_pairs = set_translator_pairs - set_existing_pairs

# Log missing pairs information
translator_logger.info(f"Total pairs in data.csv: {len(set_existing_pairs)}")
translator_logger.info(f"Total pairs in translatorsCostPairs.csv: {len(set_translator_pairs)}")
translator_logger.info(f"Total missing pairs: {len(set_missing_pairs)}")


if len(set_missing_pairs) > 0:
    translator_logger.info("Missing Pairs (sample):")
    missing_pairs_list = []
    for pair in list(set_missing_pairs)[:10]: # Log first top missing pairs
        translator, source, target = pair  # Direct tuple unpacking without splitting
        missing_pairs_list.append({
            "TRANSLATOR": translator,
            "SOURCE_LANG": source,
            "TARGET_LANG": target
        })
        translator_logger.info(f"  TRANSLATOR: {translator}, SOURCE_LANG: {source}, TARGET_LANG: {target}")
    
    df_missing_pairs = pd.DataFrame(missing_pairs_list)
    translator_logger.info(f"\nSample of Missing Pairs: {df_missing_pairs.to_string()}\n")

# Build dictionary structures for easy querying
dict_translator_langs = {} # Dictionary by translator
dict_source_langs = {} # Dictionary by source languages
dict_target_langs = {} # Dictionary by target languages

translator_logger.info("Building language dictionaries...")

# Populate the dictionaries
for _, row in df_translator_pairs.iterrows():
    str_translator = row["TRANSLATOR"]
    str_source = row["SOURCE_LANG"]
    str_target = row["TARGET_LANG"]
    num_rate = row["HOURLY_RATE"]
    
    if str_translator not in dict_translator_langs: # Add to translator dictionary
        dict_translator_langs[str_translator] = {}
    if str_source not in dict_translator_langs[str_translator]: 
        dict_translator_langs[str_translator][str_source] = {}
    dict_translator_langs[str_translator][str_source][str_target] = num_rate
    
    if str_source not in dict_source_langs: # Add to source language dictionary
        dict_source_langs[str_source] = {}
    if str_target not in dict_source_langs[str_source]: 
        dict_source_langs[str_source][str_target] = {}
    dict_source_langs[str_source][str_target][str_translator] = num_rate
    
    if str_target not in dict_target_langs: # Add to target language dictionary
        dict_target_langs[str_target] = {}
    if str_source not in dict_target_langs[str_target]:
        dict_target_langs[str_target][str_source] = {}
    dict_target_langs[str_target][str_source][str_translator] = num_rate

translator_logger.info("Dictionary Structure Summary:")
translator_logger.info(f"  - Translators: {len(dict_translator_langs)}")
translator_logger.info(f"  - Source languages: {len(dict_source_langs)}")
translator_logger.info(f"  - Target languages: {len(dict_target_langs)}")

df_merged = df_data.copy() # Create a new dataframe with optimized structure


translator_logger.info("Adding structured language pairs data...") # Add the structured language pairs column


# TODO - Do not consider this columns since the dictionaries are faster and the only purpose is to check the newest information of a translator
#
# def extract_translator_langs(str_translator_name):
#     """
#     Extracts language pairs for a given translator from the dictionary structure
#    
#     Params:
#         - str_translator_name (str): Name of the translator to extract pairs for
#   
#     Returns:
#         - list: List of dictionaries containing source, target languages and hourly rates
#     """
#     if str_translator_name not in dict_translator_langs:
#         return []  # Return empty list if translator not found
#    
#     list_pairs = []
#     dict_sources = dict_translator_langs[str_translator_name]
#    
#     for str_source, dict_targets in dict_sources.items():
#         for str_target, num_rate in dict_targets.items():
#             list_pairs.append({
#                 "source_lang": str_source,
#                 "target_lang": str_target,
#                 "hourly_rate": num_rate,
#                 "in_historical_data": f"{str_translator_name}||{str_source}||{str_target}" in set_existing_pairs
#             })
#    
#     return list_pairs
# df_merged["TRANSLATOR_LANGUAGE_PAIRS"] = df_merged["TRANSLATOR"].apply(extract_translator_langs)
# df_merged["TRANSLATOR_LANGUAGE_PAIRS_JSON"] = df_merged["TRANSLATOR_LANGUAGE_PAIRS"].apply(json.dumps) # Convert to JSON for easier storage/viewing


def get_specific_pair_rate(row): # Function to get specific pair rates
    str_translator = row["TRANSLATOR"]
    str_source = row["SOURCE_LANG"]
    str_target = row["TARGET_LANG"]
    
    # Check if the translator exists in the dictionary    
    if (str_translator in dict_translator_langs and 
        str_source in dict_translator_langs[str_translator] and
        str_target in dict_translator_langs[str_translator][str_source]):
        return dict_translator_langs[str_translator][str_source][str_target]
    
    return None

translator_logger.info("Adding latest hourly rates...")
df_merged["TRANSLATOR_HOURLY_RATE_LATEST"] = df_merged.apply(get_specific_pair_rate, axis=1)
df_merged["DISCREPANCY_HOURLY_RATE"] = df_merged["HOURLY_RATE"] - df_merged["TRANSLATOR_HOURLY_RATE_LATEST"] # Compare with the original hourly rate

# Log rate discrepancies
discrepancies = df_merged[df_merged["DISCREPANCY_HOURLY_RATE"].notnull() & 
                         (df_merged["DISCREPANCY_HOURLY_RATE"] != 0)]

translator_logger.info(f"Rate Discrepancies Found: {len(discrepancies)}")

# Additional analysis for notebook
main_logger.info(f"Rate Discrepancies Found: {len(discrepancies)}")
if len(discrepancies) > 0:
    main_logger.info(f"Sample discrepancies: \n\n{discrepancies[['TRANSLATOR', 'SOURCE_LANG', 'TARGET_LANG', 'HOURLY_RATE', 'TRANSLATOR_HOURLY_RATE_LATEST', 'DISCREPANCY_HOURLY_RATE']].head().to_string()}\n")

log_section(translator_logger, "TRANSLATOR COST PAIRS MERGE COMPLETE")
main_logger.info("Translator cost pairs merge completed")

day_mappings = {
    "MON": "MONDAY",
    "TUES": "TUESDAY",
    "WED": "WEDNESDAY", 
    "THURS": "THURSDAY",
    "FRI": "FRIDAY",
    "SAT": "SATURDAY",
    "SUN": "SUNDAY"
}

# Logger for the schedules merge
schedules_logger = logger_setup("SchedulesMerge", path_logs, "schedulesMerge.log")
log_section(schedules_logger, "PROCESSING SCHEDULES MERGE")

log_dataframe_info(schedules_logger, df_merged, "Merged data (with translator pairs)")
log_dataframe_info(schedules_logger, df_schedules, "Schedules data")

schedules_logger.info("Renaming columns for schedules merge...")
df_schedules_renamed = df_schedules.copy()

# Add prefix to schedule columns to rename them
for col in df_schedules_renamed.columns:
    if col != "NAME": # Keep NAME for merging
        if col in day_mappings:
            new_name = f"SCHEDULE_{day_mappings[col]}"
            df_schedules_renamed.rename(columns={col: new_name}, inplace=True)
        else:
            df_schedules_renamed.rename(columns={col: f"SCHEDULE_{col}"}, inplace=True)

df_schedules_renamed.rename(columns={ # Rename availability dates
    "SCHEDULE_START": "SCHEDULE_START_AVAILABLE",
    "SCHEDULE_END": "SCHEDULE_END_AVAILABLE"
}, inplace=True)

df_merged.rename(columns={ # Rename task dates in merged data
    "START": "START_TASK",
    "END": "END_TASK",
    "DELIVERED": "DELIVERED_TASK",
    "READY": "READY_TASK",
    "WORKING": "WORKING_TASK",
    "RECEIVED": "RECEIVED_TASK",
    "CLOSE": "CLOSE_TASK",
}, inplace=True)

# Log column renaming results
schedules_logger.info("Column Renaming Results:")
schedules_logger.info(f"  - Original schedule columns: {list(df_schedules.columns)}")
schedules_logger.info(f"  - Renamed schedule columns: {list(df_schedules_renamed.columns)}")
schedules_logger.info(f"  - Data date columns renamed: START -> START_TASK, END -> END_TASK")

# Check for missing translators
set_data_translators = set(df_merged["TRANSLATOR"])
set_schedule_translators = set(df_schedules_renamed["NAME"])
set_missing_schedules = set_data_translators - set_schedule_translators

schedules_logger.info(f"Translators in data.csv: {len(set_data_translators)}")
schedules_logger.info(f"Translators in schedules.csv: {len(set_schedule_translators)}")
schedules_logger.info(f"Translators without schedules: {len(set_missing_schedules)}")

if len(set_missing_schedules) > 0:
    schedules_logger.info("Sample of translators without schedule information:")
    df_missing_schedules = pd.DataFrame(list(set_missing_schedules), columns=["TRANSLATOR"])
    main_logger.info(f"\nSample of translators without schedule information: {df_missing_schedules.to_string()}")

# Merge Dataframes 
schedules_logger.info("Performing merge operation...")
df_merged_with_schedules = pd.merge(
    df_merged,
    df_schedules_renamed,
    left_on="TRANSLATOR", # Left join to keep all rows from data.csv
    right_on="NAME",
    how="left"
)

log_merge_stats( # Log merge statistics
    schedules_logger, 
    df_merged, 
    df_schedules_renamed,
    df_merged_with_schedules,
    "TRANSLATOR",
    "NAME"
)

# Calculate merge statistics
num_total_rows = len(df_merged_with_schedules)
num_with_schedules = df_merged_with_schedules["NAME"].notnull().sum()
num_without_schedules = num_total_rows - num_with_schedules

schedules_logger.info(f"Detailed Merge Results:")
schedules_logger.info(f"  - Total rows after merge: {num_total_rows}")
schedules_logger.info(f"  - Rows with schedule data: {num_with_schedules}")
schedules_logger.info(f"  - Rows without schedule data: {num_without_schedules}")
schedules_logger.info(f"  - Percentage with schedule data: {(num_with_schedules/num_total_rows)*100:.2f}%")


df_merged_with_schedules.drop(columns=["NAME"], inplace=True, errors="ignore") # Drop duplicate NAME column

schedules_logger.info("Dropped duplicate NAME column from merged result")
main_logger.info("Dropped duplicate NAME column from merged result")

log_section(schedules_logger, "SCHEDULES MERGE COMPLETE")
main_logger.info("Schedules merge completed")

# Logger for the clients merge
clients_logger = logger_setup("ClientsMerge", path_logs, "clientsMerge.log")
log_section(clients_logger, "PROCESSING CLIENTS MERGE")

log_dataframe_info(clients_logger, df_merged_with_schedules, "Merged data (with schedules)")
log_dataframe_info(clients_logger, df_clients, "Clients data")

clients_logger.info("Renaming columns for clients merge...")
df_clients_renamed = df_clients.copy()
df_clients_renamed.rename(columns={ # Rename specific client columns
    "SELLING_HOURLY_PRICE": "CLIENT_HOURLY_PRICE",
    "MIN_QUALITY": "CLIENT_MIN_QUALITY",
    "WILDCARD": "CLIENT_WILDCARD"
}, inplace=True)

# Add prefix to remaining client columns
for col in df_clients_renamed.columns:
    if col == "CLIENT_NAME" or col.startswith("CLIENT_"):  # Keep CLIENT_NAME for merging
        continue
    df_clients_renamed.rename(columns={col: f"CLIENT_{col}"}, inplace=True)

# Log column renaming results
clients_logger.info("Column Renaming Results:")
clients_logger.info(f"  - Original client columns: {list(df_clients.columns)}")
clients_logger.info(f"  - Renamed client columns: {list(df_clients_renamed.columns)}")

# Check for missing manufacturers
set_manufacturers = set(df_merged_with_schedules["MANUFACTURER"])
set_clients = set(df_clients_renamed["CLIENT_NAME"])
set_missing_clients = set_manufacturers - set_clients

clients_logger.info(f"Manufacturers in data.csv: {len(set_manufacturers)}")
clients_logger.info(f"Clients in clients.csv: {len(set_clients)}")
clients_logger.info(f"Manufacturers without client information: {len(set_missing_clients)}")

if len(set_missing_clients) > 0:
    df_missing_clients = pd.DataFrame(list(set_missing_clients)[:10], columns=["Manufacturer"])
    clients_logger.info(f"\nSample of manufacturers without client information: \n\n{df_missing_clients.to_string()}\n")

# Merge Dataframes 
clients_logger.info("Performing merge operation...")
df_final = pd.merge(
    df_merged_with_schedules,
    df_clients_renamed,
    left_on="MANUFACTURER", # Left join to keep all rows from merged data
    right_on="CLIENT_NAME",
    how="left"
)

log_merge_stats( # Log merge statistics
    clients_logger, 
    df_merged_with_schedules, 
    df_clients_renamed,
    df_final,
    "MANUFACTURER",
    "CLIENT_NAME"
)

# Calculate merge statistics
num_total_rows = len(df_final)
num_with_clients = df_final["CLIENT_NAME"].notnull().sum()
num_without_clients = num_total_rows - num_with_clients

clients_logger.info(f"Detailed Merge Results:")
clients_logger.info(f"  - Total rows after merge: {num_total_rows}")
clients_logger.info(f"  - Rows with client data: {num_with_clients}")
clients_logger.info(f"  - Rows without client data: {num_without_clients}")
clients_logger.info(f"  - Percentage with client data: {(num_with_clients/num_total_rows)*100:.2f}%")

# Add quality comparison
clients_logger.info("Checking client quality requirements...")
df_final["MEETS_CLIENT_QUALITY"] = pd.NA  # Use pandas NA which respects dtypes
mask = df_final["CLIENT_MIN_QUALITY"].notnull() & df_final["QUALITY_EVALUATION"].notnull()
# Create the comparison result as a series first, then assign it
comparison_result = df_final.loc[mask, "QUALITY_EVALUATION"] >= df_final.loc[mask, "CLIENT_MIN_QUALITY"]
df_final.loc[mask, "MEETS_CLIENT_QUALITY"] = comparison_result

# Count quality issues
quality_issues = df_final[(df_final["MEETS_CLIENT_QUALITY"] == False)].shape[0]
clients_logger.info(f"Quality Assessment:")
clients_logger.info(f"  - Tasks not meeting client minimum quality: {quality_issues}")

if quality_issues > 0:
    clients_logger.info(f"  - Percentage of quality issues: {(quality_issues/num_with_clients)*100:.2f}%")
    
    quality_data = df_final[mask].copy()
    quality_data["Quality_Status"] = quality_data["MEETS_CLIENT_QUALITY"].map({True: "Meets", False: "Below"})
    
    # Show quality by client
    quality_summary = quality_data.groupby("CLIENT_NAME").agg(
        Total=("CLIENT_NAME", "count"),
        MeetsQuality=("MEETS_CLIENT_QUALITY", lambda x: sum(x)),
        BelowQuality=("MEETS_CLIENT_QUALITY", lambda x: sum(~x))
    ).sort_values("Total", ascending=False)
    
    # Display quality summary
    clients_logger.info(f"\nQuality issues by client: \n\n{quality_summary.head().to_string()}\n")

# After all logging and analysis is complete
clients_logger.info("Removing redundant CLIENT_NAME column (same as MANUFACTURER)...")
df_final = df_final.drop(columns=["CLIENT_NAME"])

log_section(clients_logger, "CLIENTS MERGE COMPLETE")
main_logger.info("Clients merge completed")

new_column_order = [
    "PROJECT_ID", "TASK_ID", # Project Identification Columns
    "PM", "TASK_TYPE", # Management Information
    "START_TASK", "END_TASK", "ASSIGNED", "READY_TASK", "WORKING_TASK", # Timeline Information
    "DELIVERED_TASK", "RECEIVED_TASK", "CLOSE_TASK",
    "SOURCE_LANG", "TARGET_LANG", "PAIR_KEY", # Language Pair Information
    "TRANSLATOR", "QUALITY_EVALUATION", # Translator Information
    "SCHEDULE_START_AVAILABLE", "SCHEDULE_END_AVAILABLE", # Translator Schedule Information
    "SCHEDULE_MONDAY", "SCHEDULE_TUESDAY", "SCHEDULE_WEDNESDAY", 
    "SCHEDULE_THURSDAY", "SCHEDULE_FRIDAY", "SCHEDULE_SATURDAY", "SCHEDULE_SUNDAY",
    "FORECAST", "HOURLY_RATE", "COST", "TRANSLATOR_HOURLY_RATE_LATEST", # Financial Information
    "DISCREPANCY_HOURLY_RATE", "CLIENT_HOURLY_PRICE",
    "MANUFACTURER", "MANUFACTURER_SECTOR", "MANUFACTURER_INDUSTRY_GROUP", # Client Information
    "MANUFACTURER_INDUSTRY", "MANUFACTURER_SUBINDUSTRY", 
    "CLIENT_MIN_QUALITY", "CLIENT_WILDCARD",
    "MEETS_CLIENT_QUALITY" # Analytical Columns
]

df_final = df_final[new_column_order]

# No need to print the first few rows in a script

main_logger.info("Saving merge results...")

path_merged_csv = os.path.join(path_interim_data, "mergedAll.csv")
df_final.to_csv(path_merged_csv, index=False) # Save the final merged CSV
main_logger.info(f"Saved merged data to {path_merged_csv}")


# Save the dictionary structures to JSON for reference
path_translator_langs_json = os.path.join(path_interim_data, "translatorLanguages.json")
with open(path_translator_langs_json, "w", encoding="utf-8") as f:
    json.dump(dict_translator_langs, f, indent=4, ensure_ascii=False)
main_logger.info(f"Saved translator languages dictionary to {path_translator_langs_json}")

path_source_langs_json = os.path.join(path_interim_data, "sourceLanguages.json")
with open(path_source_langs_json, "w", encoding="utf-8") as f:
    json.dump(dict_source_langs, f, indent=4, ensure_ascii=False)
main_logger.info(f"Saved source languages dictionary to {path_source_langs_json}")

path_target_langs_json = os.path.join(path_interim_data, "targetLanguages.json")
with open(path_target_langs_json, "w", encoding="utf-8") as f:
    json.dump(dict_target_langs, f, indent=4, ensure_ascii=False)
main_logger.info(f"Saved target languages dictionary to {path_target_langs_json}")

# Summary statistics
log_section(main_logger, "PROCESSING COMPLETE")
main_logger.info(f"Original data rows before cleaning: {rows_before_total}")
main_logger.info(f"Rows removed due to missing values: {deletion_total}")
main_logger.info(f"Data rows after cleaning: {rows_after_total}")
main_logger.info(f"Final merged rows: {len(df_final)}")
main_logger.info(f"Final columns count: {len(df_final.columns)}")
main_logger.info(f"Translators: {len(set_data_translators)}")
main_logger.info("Languages covered:")
main_logger.info(f"  - Source languages: {len(dict_source_langs)}")
main_logger.info(f"  - Target languages: {len(dict_target_langs)}")
main_logger.info(f"Manufacturers/clients: {len(set_manufacturers)}")
main_logger.info("Check the logs directory for detailed merge information.")

# Additional notebook analysis
main_logger.info("Data Merge Summary:")
main_logger.info(f"  - Original data rows before cleaning: {rows_before_total}")
main_logger.info(f"  - Rows removed due to missing values: {deletion_total}")
main_logger.info(f"  - Data rows after cleaning: {rows_after_total}")
main_logger.info(f"  - Final merged rows: {len(df_final)}")
main_logger.info(f"  - Final columns count: {len(df_final.columns)}")
main_logger.info(f"  - Translators: {len(set_data_translators)}")
main_logger.info(f"  - Languages covered:")
main_logger.info(f"    - Source languages: {len(dict_source_langs)}")
main_logger.info(f"    - Target languages: {len(dict_target_langs)}")
main_logger.info(f"  - Manufacturers/clients: {len(set_manufacturers)}")


# Dataframe of final schema
final_schema_df = pd.DataFrame({
    "Column": df_final.columns,
    "Type": df_final.dtypes,
    "Non-Null Count": df_final.count(),
    "Null Count": df_final.isnull().sum(),
    "Null Percentage": (df_final.isnull().sum() / len(df_final) * 100).round(2)
})

# Log the final schema using the main logger
main_logger.info(f"\nFinal Schema: \n\n{final_schema_df.to_string()}\n")
main_logger.info("DATA MERGE PROCESS COMPLETED SUCCESSFULLY")