# src/data/preprocess.py

"""
Data Preprocessing Pipeline for Translator Assignment System

Converts raw Excel data into clean CSV files for ranking model pipeline
Validates translator assignments and normalizes data formats

IMPORTANT: Enforces PRIMORDIAL RULE that only translators in cost pairs sheet are valid
"""

import os
import sys
import re
import unicodedata
import pandas as pd
import numpy as np

from datetime import datetime
import logging

# Path configuration
PATH_SCRIPT = os.path.dirname(__file__)
PATH_ROOT = os.path.dirname(os.path.dirname(PATH_SCRIPT))
PATH_SRC = os.path.dirname(PATH_SCRIPT)
sys.path.append(PATH_SRC)

from utils.utils import setup_logger, ensure_dir, log_dataframe_info

# Configure paths
PATH_INPUT_EXCEL = "data.xlsx"
PATH_DATA = os.path.join(PATH_ROOT, "data", "raw")
PATH_OUTPUT = os.path.join(PATH_ROOT, "data", "interim")
PATH_LOGS = os.path.join(PATH_ROOT, "logs")

logger = setup_logger(
    "data_preprocessor",
    "preprocessing.log"
)

# Data quality thresholds
MISSING_THRESHOLD = 0.05  # Maximum 5% missing values allowed in key columns

# Create required directories
os.makedirs(PATH_DATA, exist_ok=True)
os.makedirs(PATH_OUTPUT, exist_ok=True)
os.makedirs(PATH_LOGS, exist_ok=True)

# Key columns for validation - missing these breaks downstream processing
LIST_DATA_KEY_COLUMNS = [
    "TRANSLATOR",
    "SOURCE_LANG",
    "TARGET_LANG",
    "MANUFACTURER"
]
LIST_TRANSLATOR_KEY_COLUMNS = [
    "TRANSLATOR",
    "SOURCE_LANG",
    "TARGET_LANG"
]
LIST_SCHEDULES_KEY_COLUMNS = [
    "NAME"
]
LIST_CLIENTS_KEY_COLUMNS = [
    "CLIENT_NAME"
]

# Status workflow columns
LIST_STATUS_COLUMNS = [
    "ASSIGNED",
    "READY",
    "WORKING",
    "DELIVERED",
    "RECEIVED",
    "CLOSE"
]

# Standard sheet name mappings
DICT_SHEET_MAPPINGS = {
    "data": "data",
    "clients": "clients",
    "schedules": "schedules",
    "translatorscostpairs": "translatorsCostPairs"
}


def clean_sheet_name(str_sheet_name: str) -> str:
    """
    Normalize Excel sheet names to standard format

    Args:
        str_sheet_name: Original sheet name

    Returns:
        Standardized sheet name in camelCase
    """

    if not str_sheet_name:
        return "unnamed_sheet"

    try:
        # Remove unicode diacritics for consistency
        str_normalized = unicodedata.normalize("NFKD", str(str_sheet_name))
        str_normalized = "".join([c for c in str_normalized if not unicodedata.combining(c)])

        # Keep only alphanumeric characters
        str_clean_name = re.sub(r"[^a-zA-Z0-9]", "", str_normalized)

        # Apply camelCase convention
        if str_clean_name:
            str_clean_name = str_clean_name[0].lower() + str_clean_name[1:]

        # Map to standard names if available
        str_lower_name = str_clean_name.lower()
        for str_key, str_value in DICT_SHEET_MAPPINGS.items():
            if str_lower_name == str_key.lower():
                return str_value

        return str_clean_name or f"unnamed_{hash(str_sheet_name)}"
    except Exception as e:
        logger.warning(f"Error cleaning sheet name '{str_sheet_name}': {e}")
        return f"sheet_{hash(str(str_sheet_name))}"  # Fallback to prevent crash


def check_missing_threshold(
    df_input: pd.DataFrame,
    list_columns: list,
    str_df_name: str,
    float_threshold: float = MISSING_THRESHOLD
) -> tuple:
    """
    Check if missing values exceed threshold in key columns

    Args:
        df_input: DataFrame to check
        list_columns: Columns to validate
        str_df_name: Name for logging
        float_threshold: Maximum missing percentage

    Returns:
        (bool_below_threshold, float_missing_pct, df_rows_with_missing)
    """

    # Identify which key columns actually exist
    list_columns_present = [col for col in list_columns if col in df_input.columns]
    list_columns_missing = [col for col in list_columns if col not in df_input.columns]

    if list_columns_missing:
        logger.warning(f"Key columns missing in '{str_df_name}': {list_columns_missing}")

        if not list_columns_present:  # No key columns found - critical error
            logger.error(f"No key columns present in '{str_df_name}'")
            return False, 1.0, pd.DataFrame()

    # Calculate missing value statistics
    int_total_rows = len(df_input)
    df_rows_with_missing = df_input[df_input[list_columns_present].isnull().any(axis=1)]
    int_missing_count = len(df_rows_with_missing)
    float_missing_pct = int_missing_count / int_total_rows if int_total_rows > 0 else 0

    logger.info(f"Missing values in {str_df_name}: {float_missing_pct:.4f} (threshold: {float_threshold:.4f})")

    bool_below_threshold = float_missing_pct < float_threshold

    return bool_below_threshold, float_missing_pct, df_rows_with_missing


def log_missing_values(df_input: pd.DataFrame, str_name: str) -> None:
    """Log missing value statistics by column"""

    if df_input.empty:
        logger.warning(f"{str_name} is empty")
        return

    dict_missing = df_input.isnull().sum()
    dict_missing = dict_missing[dict_missing > 0]

    if not dict_missing.empty:
        logger.info(f"Missing values in {str_name}:")
        for str_col, int_count in dict_missing.items():
            float_pct = (int_count / len(df_input)) * 100
            logger.info(f"  - {str_col}: {int_count} missing ({float_pct:.2f}%)")


def load_excel_data(path_file: str) -> dict:
    """
    Load all sheets from Excel file with normalized names

    Args:
        path_file: Path to Excel file

    Returns:
        Dictionary of DataFrames with cleaned sheet names
    """

    logger.info(f"Loading Excel file: {path_file}")

    try:
        dict_excel_data = pd.read_excel(
            path_file,
            sheet_name=None
        )
        logger.info(f"Loaded {len(dict_excel_data)} sheets")

        # Normalize sheet names for consistent processing
        dict_cleaned_data = {}
        for str_original_name, df in dict_excel_data.items():
            str_clean_name = clean_sheet_name(str_original_name)
            dict_cleaned_data[str_clean_name] = df
            logger.info(f"Sheet: '{str_original_name}' → '{str_clean_name}'")

        return dict_cleaned_data

    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        raise


def process_data_sheet(
    df_input: pd.DataFrame,
    df_translators: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Process and clean the main data sheet

    IMPORTANT: Validates translators against cost pairs (PRIMORDIAL RULE)

    Args:
        df_input: Original data DataFrame
        df_translators: Valid translator cost pairs

    Returns:
        Cleaned DataFrame with validated translators
    """

    logger.info("Processing data sheet")
    df_clean = df_input.copy()

    # Standardize column names
    if "HOURS" in df_clean.columns:
        df_clean.rename(
            columns={
                "HOURS": "FORECAST"
            },
            inplace=True
        )

    # Date conversions for temporal analysis
    list_date_columns = ["START", "END"] + LIST_STATUS_COLUMNS
    for str_col in list_date_columns:
        if str_col in df_clean.columns:
            df_clean[str_col] = pd.to_datetime(
                df_clean[str_col],
                errors='coerce'
            )

    # Numeric conversions for calculations
    for col in ["QUALITY_EVALUATION", "COST", "HOURLY_RATE", "FORECAST"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(
                df_clean[col],
                errors="coerce"
            )

    # ID conversions for consistent matching
    for str_col in ["PROJECT_ID", "TASK_ID"]:
        if str_col in df_clean.columns:
            df_clean[str_col] = df_clean[str_col].astype(str)

    # PRIMORDIAL RULE: Only translators with defined costs are valid
    if df_translators is not None and not df_translators.empty:
        if "TRANSLATOR" in df_clean.columns and "TRANSLATOR" in df_translators.columns:
            set_valid_translators = set(df_translators["TRANSLATOR"].dropna().unique())
            logger.info(f"Found {len(set_valid_translators)} valid translators")

            int_rows_before = len(df_clean)
            df_clean = df_clean[df_clean["TRANSLATOR"].isin(set_valid_translators)]
            logger.info(f"Removed {int_rows_before - len(df_clean)} rows with invalid translators")

    # Remove incomplete temporal data
    for str_col in list_date_columns:
        if str_col in df_clean.columns:
            int_rows_before = len(df_clean)
            df_clean = df_clean[df_clean[str_col].notna()]
            logger.info(f"Removed {int_rows_before - len(df_clean)} rows with missing {str_col}")

    # Data deduplication
    int_rows_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    logger.info(f"Removed {int_rows_before - len(df_clean)} duplicate rows")

    # Ensure unique tasks
    if "TASK_ID" in df_clean.columns:
        int_rows_before = len(df_clean)
        df_clean = df_clean.drop_duplicates(
            subset=["TASK_ID"],
            keep="first"
        )
        logger.info(f"Removed {int_rows_before - len(df_clean)} duplicate tasks")

    # Validate PROJECT_ID format
    if "PROJECT_ID" in df_clean.columns:
        int_rows_before = len(df_clean)
        bool_numeric_mask = df_clean["PROJECT_ID"].str.match(r"^\d+$")  # Only numeric IDs
        df_clean = df_clean[bool_numeric_mask]
        logger.info(f"Removed {int_rows_before - len(df_clean)} non-integer PROJECT_IDs")

    # Temporal consistency checks
    if "START" in df_clean.columns and "END" in df_clean.columns:
        int_rows_before = len(df_clean)
        bool_valid_mask = (
            df_clean["START"].isna()
            |
            df_clean["END"].isna()
            |
            (df_clean["END"] >= df_clean["START"])  # END must be after START
        )
        df_clean = df_clean[bool_valid_mask]
        logger.info(f"Removed {int_rows_before - len(df_clean)} time inconsistencies")

    # Manufacturer data integrity
    list_manufacturer_cols = [
        "MANUFACTURER",
        "MANUFACTURER_SECTOR",
        "MANUFACTURER_INDUSTRY_GROUP",
        "MANUFACTURER_INDUSTRY",
        "MANUFACTURER_SUBINDUSTRY"
    ]
    list_existing_cols = [col for col in list_manufacturer_cols if col in df_clean.columns]
    if list_existing_cols:
        bool_missing_mask = df_clean[list_existing_cols].isnull().any(axis=1)
        if bool_missing_mask.sum() > 0:
            df_clean = df_clean[~bool_missing_mask]
            logger.info(f"Removed {bool_missing_mask.sum()} rows with missing manufacturer info")

    # Memory optimization with categorical types
    list_categorical_cols = [
        "SOURCE_LANG",
        "TARGET_LANG",
        "TRANSLATOR",
        "TASK_TYPE",
        "MANUFACTURER",
        "MANUFACTURER_SECTOR",
        "PM"
    ]
    for col in list_categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("category")

    # Sort for temporal analysis
    if "START" in df_clean.columns:
        df_clean = df_clean.sort_values(
            by="START",
            ascending=False,
            na_position='last'
        )

    logger.info(f"Final data sheet: {len(df_clean)} rows")
    return df_clean


def process_clients_sheet(df_input: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the clients sheet"""

    logger.info("Processing clients sheet")

    if df_input is None or df_input.empty:
        logger.warning("Clients sheet is empty")
        return pd.DataFrame()

    df_clean = df_input.copy()

    # Remove duplicates
    int_rows_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    logger.info(f"Removed {int_rows_before - len(df_clean)} duplicate clients")

    # Remove incomplete entries
    int_rows_before = len(df_clean)
    df_clean.dropna(inplace=True)
    logger.info(f"Removed {int_rows_before - len(df_clean)} clients with missing values")

    # Alphabetical sorting for consistency
    if "CLIENT_NAME" in df_clean.columns:
        df_clean.sort_values(
            by="CLIENT_NAME",
            ascending=True,
            inplace=True
        )

    logger.info(f"Final clients sheet: {len(df_clean)} rows")
    return df_clean


def process_schedules_sheet(df_input: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the schedules sheet"""

    logger.info("Processing schedules sheet")

    if df_input is None or df_input.empty:
        logger.warning("Schedules sheet is empty")
        return pd.DataFrame()

    df_clean = df_input.copy()

    # Validate time format
    regex_time_pattern = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")  # HH:MM:SS format

    for str_col in ["START", "END"]:
        if str_col in df_clean.columns:
            # Extract time component
            df_clean[str_col] = df_clean[str_col].apply(lambda x: str(x) if pd.notna(x) else None)
            df_clean[str_col] = df_clean[str_col].str.split(" ").str[-1]

            # Flag invalid time formats
            df_invalid_times = df_clean[~df_clean[str_col].str.match(regex_time_pattern, na=False) & df_clean[str_col].notna()]
            if not df_invalid_times.empty:
                logger.warning(f"Found {len(df_invalid_times)} non-standard times in '{str_col}'")

    # Sort by translator name
    if "NAME" in df_clean.columns:
        df_clean.sort_values(
            by="NAME",
            ascending=True,
            inplace=True
        )

    logger.info(f"Final schedules sheet: {len(df_clean)} rows")
    return df_clean


def process_translators_sheet(df_input: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the translator cost pairs sheet"""

    logger.info("Processing translator cost pairs sheet")

    if df_input is None or df_input.empty:
        logger.warning("Translator cost pairs sheet is empty")
        return pd.DataFrame()

    df_clean = df_input.copy()

    # Remove duplicate cost entries
    int_rows_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    logger.info(f"Removed {int_rows_before - len(df_clean)} duplicate translator pairs")

    # Remove incomplete cost data
    int_rows_before = len(df_clean)
    df_clean.dropna(inplace=True)
    logger.info(f"Removed {int_rows_before - len(df_clean)} translator pairs with missing values")

    # Sort for consistent processing
    if "TRANSLATOR" in df_clean.columns:
        df_clean.sort_values(
            by="TRANSLATOR",
            ascending=True,
            inplace=True
        )

    logger.info(f"Final translator cost pairs sheet: {len(df_clean)} rows")
    return df_clean


def validate_data_quality(
    df_data: pd.DataFrame,
    df_clients: pd.DataFrame,
    df_schedules: pd.DataFrame,
    df_translators: pd.DataFrame
) -> bool:
    """
    Validate data quality across all DataFrames

    Returns:
        True if all quality checks pass
    """

    logger.info("Validating data quality")

    # Check critical columns in each dataset
    bool_data_ok, _, _ = check_missing_threshold(
        df_data,
        LIST_DATA_KEY_COLUMNS,
        "data"
    )
    bool_translator_ok, _, _ = check_missing_threshold(
        df_translators,
        LIST_TRANSLATOR_KEY_COLUMNS,
        "translatorsCostPairs"
    )
    bool_schedules_ok, _, _ = check_missing_threshold(
        df_schedules,
        LIST_SCHEDULES_KEY_COLUMNS,
        "schedules"
    )
    bool_clients_ok, _, _ = check_missing_threshold(
        df_clients,
        LIST_CLIENTS_KEY_COLUMNS,
        "clients"
    )

    bool_all_checks_pass = bool_data_ok and bool_translator_ok and bool_schedules_ok and bool_clients_ok

    if not bool_all_checks_pass:
        logger.error("DATA QUALITY ERROR: Missing values exceed threshold")
    else:
        logger.info("ALL DATA QUALITY CHECKS PASSED")

    return bool_all_checks_pass


def save_processed_data(dict_dfs: dict, path_output_dir: str) -> None:
    """Save processed DataFrames to CSV files"""

    logger.info(f"Saving processed data to {path_output_dir}")

    for str_name, df in dict_dfs.items():
        if df.empty:
            logger.warning(f"Skipping empty DataFrame: {str_name}")
            continue

        path_output_file = os.path.join(
            path_output_dir,
            f"{str_name}.csv"
        )

        try:
            df.to_csv(
                path_output_file,
                index=False
            )
            logger.info(f"Saved {str_name}.csv: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error saving {str_name}.csv: {e}")


def main():
    """Main execution function"""
    
    logger.info("Starting Excel data preprocessing")
    logger.info(f"Project root: {PATH_ROOT}")

    # Verify input file exists
    PATH_INPUT_FILE = os.path.join(PATH_DATA, PATH_INPUT_EXCEL)

    if not os.path.exists(PATH_INPUT_FILE):
        logger.error(f"Input Excel file not found: {PATH_INPUT_FILE}")
        return 1

    dict_excel_data = load_excel_data(PATH_INPUT_FILE)

    # Process sheets in dependency order
    df_clients = process_clients_sheet(dict_excel_data.get("clients", pd.DataFrame()))
    log_dataframe_info(
        logger,
        df_clients,
        "clients"
    )

    df_schedules = process_schedules_sheet(dict_excel_data.get("schedules", pd.DataFrame()))
    log_dataframe_info(
        logger,
        df_schedules,
        "schedules"
    )

    df_translators = process_translators_sheet(dict_excel_data.get("translatorsCostPairs", pd.DataFrame()))
    log_dataframe_info(
        logger,
        df_translators,
        "translatorsCostPairs"
    )

    # Process data sheet with translator validation (PRIMORDIAL RULE applied here)
    df_data = process_data_sheet(
        dict_excel_data.get("data", pd.DataFrame()),
        df_translators
    )
    log_dataframe_info(
        logger,
        df_data,
        "data"
    )

    # Quality validation
    bool_quality_ok = validate_data_quality(
        df_data,
        df_clients,
        df_schedules,
        df_translators
    )

    if not bool_quality_ok:
        logger.error("Data quality validation failed")
        return 1

    # Bundle and save outputs
    dict_processed_dfs = {
        "data": df_data,
        "clients": df_clients,
        "schedules": df_schedules,
        "translatorsCostPairs": df_translators
    }

    save_processed_data(dict_processed_dfs, PATH_OUTPUT)

    logger.info("Data preprocessing completed successfully")


if __name__ == "__main__":
    main()