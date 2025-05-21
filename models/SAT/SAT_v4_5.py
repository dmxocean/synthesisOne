"""
Single Output Per Task
CP-SAT based Task Assignment with Wildcards, using data.csv as the primary source,
integrating client constraints and translator costs/profiles.
(Corrected fillna/astype usage)
"""

from ortools.sat.python import cp_model
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).resolve().parents[2]  # Go up two levels from current file
sys.path.insert(0, str(project_root))


from src.data import specialties_json_queries 
from src.data.data_constants import JsonKeys

# --- Configuration ---

# 1. File Paths and Project Structure Assumptions
# Assume the script is in src/models/
# Navigate two levels up to get the project root
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH_BASE = os.path.dirname(os.path.dirname(SCRIPT_DIR))
except NameError:
    # Fallback if __file__ is not defined (e.g., running interactively)
    PATH_BASE = os.path.abspath(".") # Use current working directory
    SCRIPT_DIR = os.path.join(PATH_BASE, "src", "models") # Assume standard location

PATH_INTERIM_DATA = os.path.join(PATH_BASE, "data", "interim")

# --- !! Using Provided Filenames !! ---
DATA_CSV = os.path.join(PATH_INTERIM_DATA, "data.csv")
CLIENTS_CSV = os.path.join(PATH_INTERIM_DATA, "clients.csv")
TRANSLATORS_COST_CSV = os.path.join(PATH_INTERIM_DATA, "translatorsCostPairs.csv")
# schedules.csv seems irrelevant for assignment logic based on columns, ignore for now


#TODO Column constants should be standardized at the project level (WIP)
# 2. Column Names Constants - !! Adjusted based on user input !!

# data.csv (Primary Task & History Source)
COL_PROJECT_ID = "PROJECT_ID"
COL_TASK_ID = "TASK_ID"
COL_TASK_TYPE = "TASK_TYPE"
COL_SOURCE_LANG = "SOURCE_LANG"
COL_TARGET_LANG = "TARGET_LANG"
COL_TRANSLATOR_ASSIGNED = "TRANSLATOR" # Translator who did the task (for history AND to check if task needs assignment)
COL_QUALITY_EVALUATION = "QUALITY_EVALUATION" # Quality score for the task (for history)
COL_MANUFACTURER = "MANUFACTURER" # Client Name (used to link to clients.csv)
COL_MANUFACTURER_SECTOR = "MANUFACTURER_SECTOR"
COL_MANUFACTURER_INDUSTRY_GROUP = "MANUFACTURER_INDUSTRY_GROUP"
COL_MANUFACTURER_INDUSTRY = "MANUFACTURER_INDUSTRY"
COL_MANUFACTURER_SUBINDUSTRY = "MANUFACTURER_SUBINDUSTRY"

# clients.csv (Client Constraints)
COL_CLIENT_NAME = "CLIENT_NAME" # Key to join with data.csv's MANUFACTURER
COL_CLIENT_HOURLY_PRICE = "SELLING_HOURLY_PRICE" # Max price client wants to pay (applied to tasks)
COL_CLIENT_MIN_QUALITY = "MIN_QUALITY" # Min quality client wants (applied to tasks)
COL_CLIENT_WILDCARD = "WILDCARD" # Wildcard preference

# translatorsCostPairs.csv (Translator Rates)
COL_COST_TRANSLATOR = "TRANSLATOR" # Translator identifier
COL_COST_SOURCE_LANG = "SOURCE_LANG" # Translator's source language
COL_COST_TARGET_LANG = "TARGET_LANG" # Translator's target language
COL_COST_RATE = "HOURLY_RATE" # Translator's rate for this pair

# Translator Profile (Generated Columns)
COL_TRANSLATOR = "TRANSLATOR" # Consistent name
COL_PAIR_KEY = "PAIR_KEY" # Tuple: (TranslatorName, SourceLang, TargetLang) - Generated
COL_TRANSLATOR_HOURLY_RATE_LATEST = "TRANSLATOR_HOURLY_RATE_LATEST" # From cost file
COL_AVG_QUALITY = "AVG_QUALITY" # Taken from specialties json
#COL_MANUFACTURER_SPECIALTIES = "MANUFACTURER_SPECIALTIES" # Dict, calculated from data.csv history
#COL_TASK_TYPE_SPECIALTIES = "TASK_TYPE_SPECIALTIES" # Dict, calculated from data.csv history

# 3. Wildcard Values Constants
# Ensure these strings exactly match the values used in the WILDCARD column of clients.csv
WC_HOURLY_RATE = "Price"   # Assumed value for Rate wildcard
WC_MIN_QUALITY = "Quality" # Assumed value for Quality wildcard
WC_DEADLINE = "Deadline"   # Keep for potential future use
# WC_MANUFACTURER = "Manufacturer" # Example if manufacturer match could be wildcarded
# WC_TASK_TYPE = "TaskType"      # Example if task type match could be wildcarded
WC_NONE = None             # Represents no wildcard applied (used internally)

# 4. Required Columns Lists (Adjusted based on new understanding)
# Base columns needed from data.csv to define a task instance
TASK_BASE_COLUMNS = [
    COL_PROJECT_ID, COL_TASK_ID, COL_TASK_TYPE,
    COL_SOURCE_LANG, COL_TARGET_LANG, COL_MANUFACTURER, # Manufacturer is the link key
    COL_MANUFACTURER_SECTOR, COL_MANUFACTURER_INDUSTRY_GROUP,
    COL_MANUFACTURER_INDUSTRY, COL_MANUFACTURER_SUBINDUSTRY,
    # Also need the translator column to check for assignment status
    COL_TRANSLATOR_ASSIGNED
]
# Columns needed from clients.csv to add constraints
CLIENT_CONSTRAINT_COLUMNS = [
    COL_CLIENT_NAME, COL_CLIENT_HOURLY_PRICE, COL_CLIENT_MIN_QUALITY, COL_CLIENT_WILDCARD
]
# Columns needed from data.csv for historical profiling
HISTORY_PROFILING_COLUMNS = [
    COL_TRANSLATOR_ASSIGNED, COL_SOURCE_LANG, COL_TARGET_LANG, COL_QUALITY_EVALUATION,
    COL_TASK_TYPE, COL_MANUFACTURER_SECTOR, COL_MANUFACTURER_INDUSTRY_GROUP,
    COL_MANUFACTURER_INDUSTRY, COL_MANUFACTURER_SUBINDUSTRY
]
# Columns needed from translatorsCostPairs.csv
TRANSLATOR_COST_COLUMNS = [
    COL_COST_TRANSLATOR, COL_COST_SOURCE_LANG, COL_COST_TARGET_LANG, COL_COST_RATE
]

# 5. Scoring Weights (Adjust if needed)
W_MANUFACTURER = 1
W_TASK_TYPE = 2
W_RATE = 10
W_RATE_WILD = 2 # Lower weight when RATE is wildcarded

# 6. Global DataFrame Variables
DATA_DF = None
CLIENTS_DF = None
TRANSLATORS_COST_DF = None
# 7. Default Values
DEFAULT_QUALITY = 7.0 # Default quality for translators with no history
DEFAULT_NUM_OUTPUT_TRANSLATORS=50 # Default number of translators assigned per task
# --- Utility Functions ---

specialties= specialties_json_queries.QueryFunctions()

# TODO Probably best if this is moved to a separate file, so that this one is the SAT by itself
def create_dummy_data():
    """Creates dummy CSV files with correct columns if they don't exist."""
    print("Attempting to create dummy data files...")
    created_any = False
    os.makedirs(PATH_INTERIM_DATA, exist_ok=True) # Ensure directory exists

    # Dummy data.csv
    dummy_data_data = {
        COL_PROJECT_ID: ['P1', 'P1', 'P2', 'P3', 'P4', 'P1', 'P5'],
        'PM': ['Anna', 'Anna', 'Bob', 'Carla', 'Bob', 'Anna', 'Carla'],
        COL_TASK_ID: ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'], # T6, T7 are new tasks
        'START': [pd.NaT] * 7, 'END': [pd.NaT] * 7,
        COL_TASK_TYPE: ['Translation', 'ProofReading', 'Translation', 'Translation', 'Translation', 'Editing', 'Translation'],
        COL_SOURCE_LANG: ['English', 'English', 'English', 'Spanish (LA)', 'French', 'English', 'English'],
        COL_TARGET_LANG: ['Spanish (LA)', 'French', 'Spanish (LA)', 'English', 'English', 'Spanish (LA)', 'Basque'],
        COL_TRANSLATOR_ASSIGNED: ['Aaron', 'Charlie', 'Betty', 'Aaron', 'Charlie', np.nan, np.nan], # T6, T7 unassigned
        'ASSIGNED': [1, 1, 1, 1, 1, 0, 0], 'READY': [1, 1, 1, 1, 1, 1, 1],
        'WORKING': [0, 0, 0, 0, 0, 0, 0], 'DELIVERED': [1, 1, 1, 1, 1, 0, 0],
        'RECEIVED': [1, 1, 1, 1, 1, 0, 0], 'CLOSE': [1, 1, 1, 1, 1, 0, 0],
        'FORECAST': [pd.NaT] * 7,
        'HOURLY_RATE': [24, 25, 25, 24, 26, np.nan, np.nan],
        'COST': [48, 25, 50, 72, 52, np.nan, np.nan],
        COL_QUALITY_EVALUATION: [9, 7, 8, 8.5, 9.5, np.nan, np.nan],
        COL_MANUFACTURER: ['ClientA', 'ClientB', 'ClientA', 'ClientC', 'ClientB', 'ClientA', 'ClientA'], # T7 is ClientA
        COL_MANUFACTURER_SECTOR: ['Tech', 'Finance', 'Tech', 'Health', 'Finance', 'Tech', 'Tech'],
        COL_MANUFACTURER_INDUSTRY_GROUP: ['Software', 'Banking', 'Software', 'Pharma', 'Banking', 'Software', 'Software'],
        COL_MANUFACTURER_INDUSTRY: ['AppDev', 'Investment', 'AppDev', 'Research', 'Investment', 'AppDev', 'AppDev'],
        COL_MANUFACTURER_SUBINDUSTRY: ['Mobile', 'Global', 'Mobile', 'Clinical', 'Global', 'Mobile', 'Mobile'],
    }
    if not os.path.exists(DATA_CSV):
        try:
            pd.DataFrame(dummy_data_data).to_csv(DATA_CSV, index=False)
            print(f"Created dummy data: {DATA_CSV}")
            created_any = True
        except Exception as e:
            print(f"Error creating dummy {DATA_CSV}: {e}")

    # Dummy clients.csv
    dummy_clients_data = {
        COL_CLIENT_NAME: ['ClientA', 'ClientB', 'ClientC', 'ClientD'],
        COL_CLIENT_HOURLY_PRICE: [30, 35, 28, 40],
        COL_CLIENT_MIN_QUALITY: [7, 8, 6.5, 7.5],
        COL_CLIENT_WILDCARD: [WC_MIN_QUALITY, '', WC_HOURLY_RATE, 'None'] # Mix of None, empty, actual
    }
    if not os.path.exists(CLIENTS_CSV):
        try:
            pd.DataFrame(dummy_clients_data).to_csv(CLIENTS_CSV, index=False)
            print(f"Created dummy data: {CLIENTS_CSV}")
            created_any = True
        except Exception as e:
            print(f"Error creating dummy {CLIENTS_CSV}: {e}")

    # Dummy translatorsCostPairs.csv
    dummy_costs_data = {
        COL_COST_TRANSLATOR: ['Aaron', 'Aaron', 'Betty', 'Charlie', 'Charlie', 'Diana', 'Aaron', 'Betty'], # Added Eng->Basque
        COL_COST_SOURCE_LANG: ['English', 'Spanish (LA)', 'English', 'English', 'French', 'English', 'English', 'English'],
        COL_COST_TARGET_LANG: ['Spanish (LA)', 'English', 'Spanish (LA)', 'French', 'English', 'Spanish (LA)', 'Basque', 'Basque'],
        COL_COST_RATE: [24, 24, 25, 25, 26, 23, 24, 25]
    }
    if not os.path.exists(TRANSLATORS_COST_CSV):
        try:
            pd.DataFrame(dummy_costs_data).to_csv(TRANSLATORS_COST_CSV, index=False)
            print(f"Created dummy data: {TRANSLATORS_COST_CSV}")
            created_any = True
        except Exception as e:
            print(f"Error creating dummy {TRANSLATORS_COST_CSV}: {e}")

    return created_any

# TODO Conatins pre-processing, either remove if duplicate or move to the correct file
def load_data():
    """Loads data from data.csv, clients.csv, translatorsCostPairs.csv."""
    global DATA_DF, CLIENTS_DF, TRANSLATORS_COST_DF

    files_exist = all([
        os.path.exists(DATA_CSV),
        os.path.exists(CLIENTS_CSV),
        os.path.exists(TRANSLATORS_COST_CSV)
    ])

    if not files_exist:
        print("Warning: One or more required data files not found. Attempting to create dummy data.")
        if not create_dummy_data():
             print("Error: Failed to create necessary dummy data files. Cannot proceed.")
             return False
        files_exist = all([
            os.path.exists(DATA_CSV), os.path.exists(CLIENTS_CSV), os.path.exists(TRANSLATORS_COST_CSV)
        ])
        if not files_exist:
            print("Error: Still missing data files after attempting dummy creation. Cannot proceed.")
            return False

    try:
        # Define dtypes for potentially mixed-type columns to avoid warnings
        data_dtypes = {
            COL_PROJECT_ID: str, COL_TASK_ID: str, COL_TASK_TYPE: str,
            COL_SOURCE_LANG: str, COL_TARGET_LANG: str, COL_TRANSLATOR_ASSIGNED: str,
            COL_MANUFACTURER: str, COL_MANUFACTURER_SECTOR: str,
            COL_MANUFACTURER_INDUSTRY_GROUP: str, COL_MANUFACTURER_INDUSTRY: str,
            COL_MANUFACTURER_SUBINDUSTRY: str,
            'PM': str, # Example other string column
            # Let pandas infer numeric types like QUALITY_EVALUATION, HOURLY_RATE, COST etc.
        }
        client_dtypes = {COL_CLIENT_NAME: str, COL_CLIENT_WILDCARD: str}
        cost_dtypes = {COL_COST_TRANSLATOR: str, COL_COST_SOURCE_LANG: str, COL_COST_TARGET_LANG: str}


        DATA_DF = pd.read_csv(DATA_CSV, dtype=data_dtypes, low_memory=False)
        print(f"Successfully loaded main data from: {DATA_CSV} ({len(DATA_DF)} rows)")
        if DATA_DF.empty: print("Warning: Main data file is empty.")

        CLIENTS_DF = pd.read_csv(CLIENTS_CSV, dtype=client_dtypes, low_memory=False)
        print(f"Successfully loaded client constraint data from: {CLIENTS_CSV} ({len(CLIENTS_DF)} rows)")
        if CLIENTS_DF.empty: print("Warning: Client data file is empty.")

        TRANSLATORS_COST_DF = pd.read_csv(TRANSLATORS_COST_CSV, dtype=cost_dtypes, low_memory=False)
        print(f"Successfully loaded translator cost data from: {TRANSLATORS_COST_CSV} ({len(TRANSLATORS_COST_DF)} rows)")
        if TRANSLATORS_COST_DF.empty: print("Warning: Translator cost data file is empty.")

        # --- Basic Column Validation ---
        required_data_cols = TASK_BASE_COLUMNS + HISTORY_PROFILING_COLUMNS
        for col in set(required_data_cols): # Use set to avoid duplicate checks
            if col not in DATA_DF.columns:
                 if col == COL_QUALITY_EVALUATION:
                     print(f"Warning: Quality column '{col}' not found in {DATA_CSV}. Using default quality.")
                     DATA_DF[col] = np.nan # Add dummy column if missing for history processing
                 else:
                     # Raise error for other missing essential columns
                     raise ValueError(f"Missing required column '{col}' in {DATA_CSV}")
        for col in CLIENT_CONSTRAINT_COLUMNS:
             if col not in CLIENTS_DF.columns: raise ValueError(f"Missing required column '{col}' in {CLIENTS_CSV}")
        for col in TRANSLATOR_COST_COLUMNS:
             if col not in TRANSLATORS_COST_DF.columns: raise ValueError(f"Missing required column '{col}' in {TRANSLATORS_COST_CSV}")

        # --- Process Wildcards in CLIENTS_DF ---
        # Ensure the wildcard column exists (already checked above)
        # Replace various 'None' representations and empty strings with the internal WC_NONE constant
        replace_map = {'None': WC_NONE, 'none': WC_NONE, '': WC_NONE, np.nan: WC_NONE}
        # Use fillna first to handle actual np.nan, then replace strings
        CLIENTS_DF[COL_CLIENT_WILDCARD] = CLIENTS_DF[COL_CLIENT_WILDCARD].fillna(np.nan).replace(replace_map)

        # Validate remaining wildcard values
        valid_wildcards = {WC_HOURLY_RATE, WC_MIN_QUALITY, WC_DEADLINE, WC_NONE} # Add others if defined
        invalid_mask = ~CLIENTS_DF[COL_CLIENT_WILDCARD].isin(valid_wildcards)
        if invalid_mask.any():
            invalid_wcs = CLIENTS_DF.loc[invalid_mask, COL_CLIENT_WILDCARD].unique()
            print(f"Warning: Unexpected values found in '{COL_CLIENT_WILDCARD}' column: {list(invalid_wcs)}. Treating them as No Wildcard.")
            CLIENTS_DF.loc[invalid_mask, COL_CLIENT_WILDCARD] = WC_NONE

        print("Wildcard processing complete.")
        return True

    except FileNotFoundError as e:
        print(f"Error loading data: File not found - {e}.")
        return False
    except ValueError as e:
         # Catch the specific error from the user log and provide guidance
         if "Must specify a fill 'value' or 'method'" in str(e):
              print(f"Error loading data: Encountered pandas fill error - {e}. Please check data cleaning steps (fillna/astype).")
         else:
              print(f"Error loading data: Missing columns or data issue - {e}.")
         return False
    except Exception as e:
        print(f"Error loading data from CSV files: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_tasks():
    """
    Get tasks needing assignment from DATA_DF, merged with client constraints.
    Returns(DataFrame): DF of tasks ready for assignment, or None on error.
    """
    if DATA_DF is None or CLIENTS_DF is None:
        print("Error: Cannot get tasks, DATA_DF or CLIENTS_DF not loaded.")
        return None
    try:
        print("Preparing tasks for assignment...")
        # Identify Tasks needing assignment (where assigned TRANSLATOR is NaN or empty string)
        tasks_to_assign_idx = DATA_DF[COL_TRANSLATOR_ASSIGNED].isna() | (DATA_DF[COL_TRANSLATOR_ASSIGNED] == '')
        if not tasks_to_assign_idx.any():
            print("No tasks found needing assignment (based on TRANSLATOR being empty/NaN).")
            # Return empty DF with expected columns for downstream steps
            expected_cols = list(set(TASK_BASE_COLUMNS + [COL_CLIENT_HOURLY_PRICE, COL_CLIENT_MIN_QUALITY, COL_CLIENT_WILDCARD]))
            return pd.DataFrame(columns=expected_cols)

        tasks_to_assign = DATA_DF[tasks_to_assign_idx].copy()
        print(f"Found {len(tasks_to_assign)} tasks potentially needing assignment.")

        # --- Merge with Client Constraints ---
        print(f"Merging task data with client constraints...")
        if COL_MANUFACTURER not in tasks_to_assign.columns or COL_CLIENT_NAME not in CLIENTS_DF.columns:
             raise ValueError(f"Missing key columns ('{COL_MANUFACTURER}' or '{COL_CLIENT_NAME}') for merging tasks and client constraints.")

        tasks_with_constraints = pd.merge(
            tasks_to_assign,
            CLIENTS_DF[CLIENT_CONSTRAINT_COLUMNS],
            left_on=COL_MANUFACTURER,
            right_on=COL_CLIENT_NAME,
            how='left' # Keep all tasks
        )

        # Handle tasks whose client wasn't found in clients.csv
        missing_clients_mask = tasks_with_constraints[COL_CLIENT_NAME].isna()
        if missing_clients_mask.any():
            missing_client_names = tasks_with_constraints.loc[missing_clients_mask, COL_MANUFACTURER].unique()
            print(f"Warning: Client constraints not found for: {list(filter(None, missing_client_names))}. Using defaults (High Price=9999, Low Quality=0, No Wildcard).")
            tasks_with_constraints.loc[missing_clients_mask, COL_CLIENT_HOURLY_PRICE] = 9999
            tasks_with_constraints.loc[missing_clients_mask, COL_CLIENT_MIN_QUALITY] = 0
            tasks_with_constraints.loc[missing_clients_mask, COL_CLIENT_WILDCARD] = WC_NONE # Ensure WC_NONE is assigned

        # --- Select Final Columns and Clean ---
        final_task_columns = list(set(TASK_BASE_COLUMNS + [
            COL_CLIENT_HOURLY_PRICE, COL_CLIENT_MIN_QUALITY, COL_CLIENT_WILDCARD
        ]))

        # Ensure all required columns are present
        if not all(col in tasks_with_constraints.columns for col in final_task_columns):
            missing_cols = [col for col in final_task_columns if col not in tasks_with_constraints.columns]
            raise ValueError(f"Internal Error: Columns missing after merging: {missing_cols}")

        tasks_df = tasks_with_constraints[final_task_columns].copy()

        # Data Type Conversion/Validation
        # Convert price/quality to numeric, coercing errors to NaN first
        tasks_df[COL_CLIENT_HOURLY_PRICE] = pd.to_numeric(tasks_df[COL_CLIENT_HOURLY_PRICE], errors='coerce')
        tasks_df[COL_CLIENT_MIN_QUALITY] = pd.to_numeric(tasks_df[COL_CLIENT_MIN_QUALITY], errors='coerce')

        # Fill NaNs resulting from coercion or missing defaults
        tasks_df.fillna({COL_CLIENT_HOURLY_PRICE:9999}, inplace=True) # Fill with default high price
        tasks_df.fillna({COL_CLIENT_MIN_QUALITY:0}, inplace=True)    # Fill with default low quality
        tasks_df.fillna({COL_CLIENT_WILDCARD:np.nan}).replace({np.nan:WC_NONE},inplace=True) # Ensure wildcard is WC_NONE if it became NaN somehow

        # Convert identifier/category columns to string AFTER handling NaNs
        str_cols = [COL_PROJECT_ID, COL_TASK_ID, COL_TASK_TYPE, COL_SOURCE_LANG, COL_TARGET_LANG,
                    COL_MANUFACTURER, COL_MANUFACTURER_SECTOR, COL_MANUFACTURER_INDUSTRY_GROUP,
                    COL_MANUFACTURER_INDUSTRY, COL_MANUFACTURER_SUBINDUSTRY]
        for col in str_cols:
            if col in tasks_df.columns:
                 # Fill potential remaining NaNs (e.g., from original data) with empty string BEFORE type conversion
                 tasks_df[col] = tasks_df[col].fillna('').astype(str)

        print(f"Prepared {len(tasks_df)} tasks with constraints for assignment.")
        return tasks_df

    except (KeyError, ValueError) as e:
         print(f"Error preparing tasks: {e}. Check column constants and CSV data integrity.")
         return None
    except Exception as e:
        print(f"An unexpected error occurred in get_tasks: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_translators():
    """
    Processes DATA_DF (for history) and TRANSLATORS_COST_DF (for rates)
    to build enriched translator profiles. Returns list of profile dicts.
    """
    if DATA_DF is None or TRANSLATORS_COST_DF is None:
        print("Error: Cannot get translators, DATA_DF or TRANSLATORS_COST_DF not loaded.")
        return []
    try:
        print("Processing translator profiles...")
        # --- Step 1: Prepare Historical Data ---
        # Filter for records with assigned translator and quality score
        history_idx = DATA_DF[COL_TRANSLATOR_ASSIGNED].notna() & \
                      (DATA_DF[COL_TRANSLATOR_ASSIGNED] != '') & \
                      DATA_DF[COL_QUALITY_EVALUATION].notna()

        history_df = DATA_DF[history_idx].copy()

        # Ensure required history columns exist
        if not all(col in history_df.columns for col in HISTORY_PROFILING_COLUMNS):
            missing = [col for col in HISTORY_PROFILING_COLUMNS if col not in history_df.columns]
            # Allow quality to be missing (handled by default), but error on others
            missing_critical = [m for m in missing if m != COL_QUALITY_EVALUATION]
            if missing_critical:
                 raise ValueError(f"Missing required history columns in data.csv: {missing_critical}")
            if COL_QUALITY_EVALUATION not in history_df.columns:
                 print(f"Adding missing '{COL_QUALITY_EVALUATION}' column with NaNs for history processing.")
                 history_df[COL_QUALITY_EVALUATION] = np.nan


        historical_data = history_df[HISTORY_PROFILING_COLUMNS].copy()
        print(f"Using {len(historical_data)} historical records for profiling.")

        # Clean data before aggregation
        historical_data[COL_QUALITY_EVALUATION] = pd.to_numeric(historical_data[COL_QUALITY_EVALUATION], errors='coerce')
        # Define essential columns for grouping/aggregation
        essential_hist_cols = [COL_TRANSLATOR_ASSIGNED, COL_SOURCE_LANG, COL_TARGET_LANG, COL_QUALITY_EVALUATION]
        # Drop rows where any essential column (incl. quality after coercion) is NaN
        historical_data.dropna(subset=essential_hist_cols, inplace=True)

        # Convert key grouping/category columns to string AFTER dropping NaNs in essential ones
        str_hist_cols = [COL_TRANSLATOR_ASSIGNED, COL_SOURCE_LANG, COL_TARGET_LANG, COL_TASK_TYPE,
                         COL_MANUFACTURER_SECTOR, COL_MANUFACTURER_INDUSTRY_GROUP,
                         COL_MANUFACTURER_INDUSTRY, COL_MANUFACTURER_SUBINDUSTRY]
        for col in str_hist_cols:
            if col in historical_data.columns:
                 historical_data[col] = historical_data[col].fillna('').astype(str)

        print(f"Historical data reduced to {len(historical_data)} valid rows after cleaning.")

        # --- Step 2: Prepare Translator Rates ---
        cost_data = TRANSLATORS_COST_DF[TRANSLATOR_COST_COLUMNS].copy()
        cost_data.dropna(subset=TRANSLATOR_COST_COLUMNS, inplace=True)
        cost_data = cost_data.drop_duplicates(subset=[COL_COST_TRANSLATOR, COL_COST_SOURCE_LANG, COL_COST_TARGET_LANG], keep='first')
        # Convert identifier/lang columns to string AFTER dropping NaNs
        cost_data[COL_COST_TRANSLATOR] = cost_data[COL_COST_TRANSLATOR].fillna('').astype(str)
        cost_data[COL_COST_SOURCE_LANG] = cost_data[COL_COST_SOURCE_LANG].fillna('').astype(str)
        cost_data[COL_COST_TARGET_LANG] = cost_data[COL_COST_TARGET_LANG].fillna('').astype(str)
        print(f"Processed {len(cost_data)} unique translator-pair rates.")
        rate_lookup = cost_data.set_index([COL_COST_TRANSLATOR, COL_COST_SOURCE_LANG, COL_COST_TARGET_LANG])[COL_COST_RATE].to_dict()

        # --- Step 3: Calculate Quality and Specialties ---
        translator_profiles = {}
        if not historical_data.empty:
            group_cols = [COL_TRANSLATOR_ASSIGNED, COL_SOURCE_LANG, COL_TARGET_LANG]
            # Ensure grouping columns are present before grouping
            if not all(col in historical_data.columns for col in group_cols):
                 missing_group_cols = [col for col in group_cols if col not in historical_data.columns]
                 raise ValueError(f"Cannot group historical data, missing columns: {missing_group_cols}")

            grouped_history = historical_data.groupby(group_cols)
            print(f"Aggregating historical data for {len(grouped_history)} translator-language pairs...")

            for (translator_name, src_lang, tgt_lang), group in grouped_history:
                # Ensure keys are strings
                pair_key = (str(translator_name), str(src_lang), str(tgt_lang))

                # 1. Calculate Average Quality (already dropped NaNs)
                #DEPRECATED
                """
                avg_quality = group[COL_QUALITY_EVALUATION].mean()
                """
                avg_quality=specialties.get_quality_for_translator(translator_name,src_lang,tgt_lang)
                # 2. Calculate Manufacturer Specialties
                #DEPRECATED
                """ 
                manufacturer_specialties = defaultdict(int)
                manuf_cols = [COL_MANUFACTURER_SECTOR, COL_MANUFACTURER_INDUSTRY_GROUP, COL_MANUFACTURER_INDUSTRY, COL_MANUFACTURER_SUBINDUSTRY]
                for col in manuf_cols:
                     if col in group.columns:
                          # Count non-empty string values
                          counts = group[group[col] != ''][col].value_counts()
                          for manuf_cat, count in counts.items():
                              manufacturer_specialties[str(manuf_cat)] += count # Ensure category is string
                """
                # 3. Calculate Task Type Specialties
                #TODO: Check, we're no longer doing this right?
                """
                task_type_specialties = defaultdict(int)
                if COL_TASK_TYPE in group.columns:
                     # Count non-empty string values
                     counts = group[group[COL_TASK_TYPE] != ''][COL_TASK_TYPE].value_counts()
                     for task_type, count in counts.items():
                          task_type_specialties[str(task_type)] += count # Ensure type is string
                """
                # 4. Combine with Rate
                latest_rate = rate_lookup.get(pair_key)
                if latest_rate is not None:
                     translator_profiles[pair_key] = {
                         COL_TRANSLATOR: translator_name, # Keep original type if needed, though keys are strings
                         COL_PAIR_KEY: pair_key, # String tuple key
                         COL_TRANSLATOR_HOURLY_RATE_LATEST: latest_rate,
                         COL_AVG_QUALITY: avg_quality, 
                        #COL_MANUFACTURER_SPECIALTIES: {}, #DEPRECATED, no?
                        #COL_TASK_TYPE_SPECIALTIES: {}, #DEPRECATED, no?
                     }
                # else: # Don't warn here, handled later
                #    pass

        else:
            print("No valid historical data available to calculate quality/specialties.")

        # --- Step 4: Add Translators from Cost file only (if they had no history) ---
        added_from_cost = 0
        for (translator_name, src_lang, tgt_lang), rate in rate_lookup.items():
             # Ensure keys used for lookup match the string format from history grouping
             pair_key = (str(translator_name), str(src_lang), str(tgt_lang))
             if pair_key not in translator_profiles:
                 translator_profiles[pair_key] = {
                     COL_TRANSLATOR: translator_name,
                     COL_PAIR_KEY: pair_key,
                     COL_TRANSLATOR_HOURLY_RATE_LATEST: rate,
                     COL_AVG_QUALITY: DEFAULT_QUALITY, 
                     #COL_MANUFACTURER_SPECIALTIES: {}, # No history
                     #COL_TASK_TYPE_SPECIALTIES: {}      # No history
                 }
                 added_from_cost +=1
        if added_from_cost > 0: print(f"Added {added_from_cost} profiles based only on rate data (using default quality {DEFAULT_QUALITY}).")

        translator_list = list(translator_profiles.values())
        print(f"Generated {len(translator_list)} unique translator profiles.")
        return translator_list

    except (KeyError, ValueError) as e:
        print(f"Error processing translator data: {e}.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in get_translators: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Main Execution ---

def main():
    # === Load Data ===
    if not load_data():
        print("Failed to load data. Exiting.")
        return

    tasks_df = get_tasks()      # DataFrame of tasks needing assignment
    translators = get_translators() # List of translator profile dictionaries

    # --- Input Validation ---
    if tasks_df is None: print("Task preparation failed. Exiting."); return
    if tasks_df.empty: print("No tasks require assignment. Exiting."); return
    if not translators: print("No translator profiles generated. Cannot assign tasks. Exiting."); return

    num_tasks = len(tasks_df)
    num_translators = len(translators)
    print(f"\nAttempting to assign {num_tasks} tasks using {num_translators} translator profiles.")

    # Create CP-SAT Model
    model = cp_model.CpModel()

    # === Create Decision Variables & Constraints ===
    x = {} # x[i, j] = 1 if task i is assigned to translator j
    valid_candidate = {} # Store initial validity based on hard constraints

    print("Building model constraints...")
    for i in range(num_tasks):
        try:
            task = tasks_df.iloc[i] # Get task data as Pandas Series
            wildcard = task.get(COL_CLIENT_WILDCARD, WC_NONE) # Get wildcard, default to None
            task_constraints_met_by_any = False

            # Ensure task languages are strings for comparison
            task_src_lang = str(task.get(COL_SOURCE_LANG, ''))
            task_tgt_lang = str(task.get(COL_TARGET_LANG, ''))
            # Ensure client price/quality are numeric
            client_max_price = float(task.get(COL_CLIENT_HOURLY_PRICE, 9999))
            client_min_qual = float(task.get(COL_CLIENT_MIN_QUALITY, 0))

            for j in range(num_translators):
                translator = translators[j] # Translator profile dict
                var = model.NewBoolVar(f"x_{i}_{j}")
                x[(i, j)] = var
                is_valid = True

                # 1. Language Matching (Mandatory)
                translator_pair_key = translator[COL_PAIR_KEY] # Tuple: (Name, Source, Target) - already strings
                if translator_pair_key[1] != task_src_lang or translator_pair_key[2] != task_tgt_lang:
                    model.Add(var == 0)
                    is_valid = False
                    valid_candidate[(i, j)] = False
                    continue # Skip other checks if language doesn't match

                # 2. Rate Constraint (Apply if Rate is NOT the wildcard)
                if wildcard != WC_HOURLY_RATE:
                    translator_rate = translator.get(COL_TRANSLATOR_HOURLY_RATE_LATEST)
                    # Check rate validity and compare
                    if pd.isna(translator_rate) or float(translator_rate) > client_max_price:
                         model.Add(var == 0)
                         is_valid = False
                         valid_candidate[(i, j)] = False
                         continue

                # 3. Quality Constraint (Apply if Quality is NOT the wildcard)
                if wildcard != WC_MIN_QUALITY:
                    translator_qual = translator.get(COL_AVG_QUALITY) # Calculated or default
                    # Check quality validity and compare
                    if pd.isna(translator_qual) or float(translator_qual) < client_min_qual:
                        model.Add(var == 0)
                        is_valid = False
                        valid_candidate[(i, j)] = False
                        continue

                # If all checks passed
                valid_candidate[(i, j)] = True
                if is_valid: # Double check, though continue should handle it
                     task_constraints_met_by_any = True

        except Exception as e:
            print(f"Error processing constraints for task index {i}: {e}")
            print(f"Task Data: {task.to_dict() if 'task' in locals() else 'N/A'}")
            # Decide how to handle - skip task, exit, etc. Here, we'll just print and continue.
            # Mark all candidates for this task as invalid to prevent assignment
            for j_err in range(num_translators):
                 valid_candidate[(i, j_err)] = False
                 if (i, j_err) in x: model.Add(x[(i, j_err)] == 0) # Add constraint if var exists
            task_constraints_met_by_any = False # Ensure warning triggers


        # Warning if no translator meets hard constraints after loop
        if not task_constraints_met_by_any:
             # Check if the task object exists before accessing it
             task_id_info = f"'{task.get(COL_PROJECT_ID, 'UNK_PROJ')}' / '{task.get(COL_TASK_ID, 'UNK_TASK')}'" if 'task' in locals() else f"Index {i}"
             print(f"Warning: No translator initially meets hard constraints for Task {task_id_info}.")


    # === Assignment Uniqueness Constraints ===
    print("Adding assignment uniqueness constraints...")
    # Each task assigned to AT MOST one translator
    for i in range(num_tasks):
        model.Add(sum(x[(i, j)] for j in range(num_translators) if (i,j) in x) <= 1)
    # Each translator assigned to AT MOST one task
    for j in range(num_translators):
        model.Add(sum(x[(i, j)] for i in range(num_tasks) if (i,j) in x) <= 1)

    # === Build Objective Function ===
    print("Building objective function...")
    objective_terms = []
    scores = {} # Store calculated scores for reporting
    SCORE_SCALE_FACTOR = 100 # Scale scores to integers for CP-SAT

    for i in range(num_tasks):
        try:
            task = tasks_df.iloc[i]
            wildcard = task.get(COL_CLIENT_WILDCARD, WC_NONE)
            task_src_lang = str(task.get(COL_SOURCE_LANG, ''))
            task_tgt_lang = str(task.get(COL_TARGET_LANG, ''))
            client_price = float(task.get(COL_CLIENT_HOURLY_PRICE, 0)) # Default to 0 if missing/invalid after clean
            task_type_str = str(task.get(COL_TASK_TYPE, ''))
            task_manuf_dims = [str(task.get(dim, '')) for dim in [COL_MANUFACTURER_SECTOR, COL_MANUFACTURER_INDUSTRY_GROUP,
                                                               COL_MANUFACTURER_INDUSTRY, COL_MANUFACTURER_SUBINDUSTRY]]

            for j in range(num_translators):
                translator = translators[j]
                translator_pair_key = translator[COL_PAIR_KEY]

                # Initialize scores for this pair
                score_value = -float('inf')
                norm_manufacturer = 0
                norm_task_type = 0
                normalized_rate = -float('inf')
                cost_component = -float('inf')

                # Only calculate score if languages match (basic check)
                if translator_pair_key[1] == task_src_lang and translator_pair_key[2] == task_tgt_lang:

                    # Manufacturer Proficiency Score
                    manf_data=specialties.get_manufacturer_data_for_translator(translator_pair_key[0],translator_pair_key[1],translator_pair_key[2])
                    
                    total_manufacturer=sum(manf_data[JsonKeys.MANUFACTURERS].values())
                    total_manufacturer+=sum(manf_data[JsonKeys.SECTORS].values())
                    print(task[COL_MANUFACTURER])

                    match_manufacturer = manf_data[JsonKeys.MANUFACTURERS].get(task.get(COL_MANUFACTURER),0) 
                    match_manufacturer += manf_data[JsonKeys.SECTORS].get(task.get(COL_MANUFACTURER_SECTOR),0) 
                    norm_manufacturer = (100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0

                    #DEPRECATED, should we use this?
                    # Task Type Proficiency Score
                    # ttypes = translator.get(COL_TASK_TYPE_SPECIALTIES, {})
                    # total_task_type = sum(ttypes.values())
                    # match_task_type = ttypes.get(task_type_str, 0) if task_type_str else 0
                    # norm_task_type = (100 * match_task_type / total_task_type) if total_task_type > 0 else 0

                    # Normalized Cost (Rate) Score Component
                    translator_rate = translator.get(COL_TRANSLATOR_HOURLY_RATE_LATEST)
                    if client_price > 0 and pd.notna(translator_rate):
                        normalized_rate = 100 * (client_price - float(translator_rate)) / client_price
                    # else normalized_rate remains -inf

                    # Apply wildcard weight
                    effective_rate_weight = W_RATE_WILD if wildcard == WC_HOURLY_RATE else W_RATE
                    if normalized_rate > -float('inf'):
                         cost_component = normalized_rate * effective_rate_weight
                    # else cost_component remains -inf

                    # Final Score Calculation
                    if cost_component > -float('inf'):
                        score_value = (W_MANUFACTURER * norm_manufacturer +
                                       W_TASK_TYPE * norm_task_type +
                                       cost_component)
                    # else score_value remains -inf

                # Store calculated scores
                scores[(i, j)] = {
                    'total': score_value, 'norm_manuf': norm_manufacturer, 'norm_task': norm_task_type,
                    'norm_rate': normalized_rate if normalized_rate > -float('inf') else 0,
                    'cost_comp': cost_component if cost_component > -float('inf') else 0
                }

                # Add term to objective ONLY if candidate was valid *initially* and score is valid
                if valid_candidate.get((i, j), False) and score_value > -float('inf'):
                    int_score = int(score_value * SCORE_SCALE_FACTOR)
                    objective_terms.append(x[(i, j)] * int_score)

        except Exception as e:
            print(f"Error calculating score for task index {i}: {e}")
            # Mark score as invalid for all translators for this task
            for j_err in range(num_translators):
                 scores[(i, j_err)] = {'total': -float('inf')} # Ensure fallback doesn't pick it


    # Set objective function
    if objective_terms:
        model.Maximize(sum(objective_terms))
        print(f"Objective function created with {len(objective_terms)} terms.")
    else:
        print("\nWarning: No valid, scorable translator-task pairings found. Cannot optimize.")

    # === Solve the Model ===
    solver = cp_model.CpSolver()
    print("\nStarting CP-SAT solver...")
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    print(f"Solver finished with status: {status_name}")

    # === Process Results ===
    print("\n" + "="*30)
    print("Assignment Results")
    print("="*30)

    assigned_tasks_count = 0
    results_data = [] # Store results for potential output

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        try:
            objective_value = solver.ObjectiveValue() / SCORE_SCALE_FACTOR
            print(f"Total Objective Value (Sum of Scores): {objective_value:.2f}\n")
        except RuntimeError:
            print("No objective value available.\n")

        for i in range(num_tasks):
            task = tasks_df.iloc[i]
            wildcard = task.get(COL_CLIENT_WILDCARD, WC_NONE)
            task_name = f"Task '{task.get(COL_PROJECT_ID, 'UNK')}' / '{task.get(COL_TASK_ID, 'UNK')}'"
            client_name = task.get(COL_MANUFACTURER, 'UNK')
            wildcard_str = f"Wildcard: {wildcard if wildcard else 'None'}"

            print(f"{task_name} ({client_name}) [{wildcard_str}]")
            print(f"  Task Langs: {task.get(COL_SOURCE_LANG,'?')} -> {task.get(COL_TARGET_LANG,'?')}, Type: {task.get(COL_TASK_TYPE,'?')}")

            assigned_translator_idx = -1
            # Find assigned translator
            for j in range(num_translators):
                if (i,j) in x and solver.Value(x[(i, j)]) > 0.5:
                    assigned_translator_idx = j
                    break

            # Prepare result row structure
            result_row = {
                'ProjectID': task.get(COL_PROJECT_ID), 'TaskID': task.get(COL_TASK_ID),
                'AssignedTranslator': None, 'AssignmentScore': np.nan,
                'TranslatorRate': np.nan, 'TranslatorAvgQuality': np.nan,
                'Reason': 'Not Assigned'
            }

            if assigned_translator_idx != -1:
                # --- Process Assigned Translator ---
                assigned_tasks_count += 1
                translator = translators[assigned_translator_idx]
                # Retrieve scores calculated earlier
                score_details = scores.get((i, assigned_translator_idx), {'total': np.nan}) # Use .get for safety

                print(f"  -> Assigned Translator: {translator.get(COL_TRANSLATOR, 'UNK_NAME')}")
                rate_str = f"{translator.get(COL_TRANSLATOR_HOURLY_RATE_LATEST, ''):.2f}" if pd.notna(translator.get(COL_TRANSLATOR_HOURLY_RATE_LATEST)) else "N/A"
                qual_str = f"{translator.get(COL_AVG_QUALITY, ''):.2f}" if pd.notna(translator.get(COL_AVG_QUALITY)) else "N/A"
                max_price_str = f"{task.get(COL_CLIENT_HOURLY_PRICE, ''):.2f}"
                min_qual_str = f"{task.get(COL_CLIENT_MIN_QUALITY, '')}"

                print(f"     Rate: {rate_str} (Task Max: {max_price_str}) [Enforced: {wildcard != WC_HOURLY_RATE}]")
                print(f"     Quality: {qual_str} (Task Min: {min_qual_str}) [Enforced: {wildcard != WC_MIN_QUALITY}]")
                if 'norm_manuf' in score_details: # Check if score details exist
                    print(f"     NormProf: Manuf={score_details.get('norm_manuf', 0):.0f}%, Task={score_details.get('norm_task', 0):.0f}%")
                    print(f"     Score Details: NormRate={score_details.get('norm_rate', 0):.1f}%, CostComp={score_details.get('cost_comp', 0):.2f}, Total={score_details.get('total', np.nan):.2f}")
                else:
                    print("     Score details not available.")

                result_row.update({
                    'AssignedTranslator': translator.get(COL_TRANSLATOR),
                    'AssignmentScore': score_details.get('total'),
                    'TranslatorRate': translator.get(COL_TRANSLATOR_HOURLY_RATE_LATEST),
                    'TranslatorAvgQuality': translator.get(COL_AVG_QUALITY),
                    'Reason': 'Assigned by Solver'
                })

            else:
                # --- Process Unassigned Task - Suggest Best Fallback ---
                print("  -> No translator assigned by the solver.")
                lang_candidates = []
                task_src_lang_str = str(task.get(COL_SOURCE_LANG, ''))
                task_tgt_lang_str = str(task.get(COL_TARGET_LANG, ''))

                for j in range(num_translators):
                     t = translators[j]
                     pair_key = t[COL_PAIR_KEY]
                     # Check language match and if score is valid number
                     if pair_key[1] == task_src_lang_str and pair_key[2] == task_tgt_lang_str:
                          score_info = scores.get((i, j), {'total': -float('inf')})
                          if pd.notna(score_info['total']) and score_info['total'] > -float('inf'):
                               lang_candidates.append(j)

                if lang_candidates:
                    # Find candidate with highest score among language matches
                    best_j = max(lang_candidates, key=lambda j: scores.get((i, j), {'total': -float('inf')})['total'])
                    t = translators[best_j]
                    s = scores.get((i, best_j), {'total': np.nan})

                    # Check if this 'best' candidate actually met the original hard constraints
                    met_constraints = valid_candidate.get((i, best_j), False)
                    validity_str = "(Met hard constraints)" if met_constraints else "(Violated hard constraints)"

                    print(f"     Suggestion (Best Score): {t.get(COL_TRANSLATOR, 'UNK')} {validity_str}")
                    rate_str = f"{t.get(COL_TRANSLATOR_HOURLY_RATE_LATEST, ''):.2f}" if pd.notna(t.get(COL_TRANSLATOR_HOURLY_RATE_LATEST)) else "N/A"
                    qual_str = f"{t.get(COL_AVG_QUALITY, ''):.2f}" if pd.notna(t.get(COL_AVG_QUALITY)) else "N/A"
                    print(f"       Score: {s.get('total', np.nan):.2f}, Rate: {rate_str}, Quality: {qual_str}")

                    reason_str = f"Suggested: {t.get(COL_TRANSLATOR, 'UNK')} (Score: {s.get('total', np.nan):.2f})"
                    if not met_constraints:
                         violations = []
                         # Check specific constraint violations
                         if wildcard != WC_HOURLY_RATE and (pd.isna(t.get(COL_TRANSLATOR_HOURLY_RATE_LATEST)) or float(t.get(COL_TRANSLATOR_HOURLY_RATE_LATEST, np.inf)) > float(task.get(COL_CLIENT_HOURLY_PRICE, -np.inf))): violations.append("Rate")
                         if wildcard != WC_MIN_QUALITY and (pd.isna(t.get(COL_AVG_QUALITY)) or float(t.get(COL_AVG_QUALITY, -np.inf)) < float(task.get(COL_CLIENT_MIN_QUALITY, np.inf))): violations.append("Quality")
                         # Add language violation check if needed, though covered by loop filter
                         print(f"       Constraint violation(s): {'; '.join(violations) if violations else 'Language/Other'}")
                         if violations: reason_str += f" Violates: {'; '.join(violations)}"

                    result_row['Reason'] = reason_str

                else:
                    print("     No suitable translators found for this language pair (or scores were invalid).")
                    result_row['Reason'] = 'No suitable language candidates'

            results_data.append(result_row)
            print("-" * 20) # Separator for tasks

        print(f"\nSummary: Assigned {assigned_tasks_count} out of {num_tasks} tasks by the solver.")

        # --- Output Results ---
        results_df = pd.DataFrame(results_data)
        print("\nAssignment Summary DataFrame:")
        display_cols = ['ProjectID', 'TaskID', 'AssignedTranslator', 'AssignmentScore', 'Reason', 'TranslatorRate', 'TranslatorAvgQuality']
        # Ensure columns exist before trying to display
        display_cols = [col for col in display_cols if col in results_df.columns]
        if display_cols:
             print(results_df[display_cols].to_string(index=False, float_format="%.2f"))
        else:
             print("Result columns not available for display.")


        # Optional: Save results to CSV in data/processed/SAT
        try:
            results_output_dir = os.path.join(PATH_BASE, "data", "processed", "SAT")
            os.makedirs(results_output_dir, exist_ok=True)
            results_output_path = os.path.join(results_output_dir, "task_assignments.csv")
            results_df.to_csv(results_output_path, index=False, float_format="%.2f")
            print(f"\nAssignment results saved to: {results_output_path}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")


    elif status == cp_model.INFEASIBLE:
        print("Solver status: INFEASIBLE")
        print("No assignment possible that satisfies all constraints.")
    elif status == cp_model.MODEL_INVALID:
         print("Solver status: MODEL_INVALID")
         print("The CP-SAT model is invalid. Check constraints and objective function.")
         # print(model.Validate()) # May provide details but can be verbose
    else:
        print(f"Solver status: {status_name}")
        print("Could not find an optimal or feasible solution.")


if __name__ == '__main__':
    main()