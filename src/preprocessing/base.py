# src/preprocessing/base.py

"""
Base Preprocessing module for the Translator Assignment System
"""

import os
import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# Path configuration
PATH_PREPROCESSING = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_PREPROCESSING)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)

# Path constants
PATH_DATA_DIR = os.path.join(PATH_ROOT, "data", "interim")
PATH_DATA_CSV = os.path.join(PATH_DATA_DIR, "data.csv")
PATH_TRANSLATORS_COST_PAIRS_CSV = os.path.join(PATH_DATA_DIR, "translatorsCostPairs.csv")
PATH_SCHEDULES_CSV = os.path.join(PATH_DATA_DIR, "schedules.csv")
PATH_CLIENTS_CSV = os.path.join(PATH_DATA_DIR, "clients.csv")
PATH_PROCESSED_DIR = os.path.join(PATH_ROOT, "data", "processed")
PATH_BASE_DIR = os.path.join(PATH_PROCESSED_DIR, "base")
PATH_ARTIFACTS_DIR = os.path.join(PATH_BASE_DIR, "artifacts")
PATH_TRAIN_PARQUET = os.path.join(PATH_BASE_DIR, "train.parquet")
PATH_VAL_PARQUET = os.path.join(PATH_BASE_DIR, "val.parquet")
PATH_TEST_PARQUET = os.path.join(PATH_BASE_DIR, "test.parquet")

# Project modules
from src.utils.utils import *

# Initialize logger
logger = setup_logger("base_preprocessing", "preprocessing.log", logging.INFO)

# Global Constants
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42
MAX_QUALITY_SCORE = 10.0


def create_output_dirs() -> None:
    """
    Create the necessary directory structure for base processed data

    Creates directories for:
    - Processed data root
    - Base data
    - Artifacts
    """

    logger.info("Creating output directories")

    ensure_dir(PATH_PROCESSED_DIR)
    ensure_dir(PATH_BASE_DIR)
    ensure_dir(PATH_ARTIFACTS_DIR)

    logger.info("Output directories created successfully")


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load all raw data from CSV sources into pandas DataFrames

    Loads four data sources:
    - Main historical task data with timestamps
    - Translator-language-cost pairings
    - Translator availability schedules
    - Client requirements and preferences

    Returns:
        Dictionary containing DataFrames for main data, translators cost pairs,
        schedules, and clients information
    """
    logger.info("Loading raw data files")

    df_data = pd.read_csv(
        PATH_DATA_CSV,
        parse_dates=[
            "START",
            "END",
            "ASSIGNED",
            "READY",
            "WORKING",
            "DELIVERED",
            "RECEIVED",
            "CLOSE"
        ]
    )
    logger.info(
        f"Loaded main data: {df_data.shape[0]} rows, {df_data.shape[1]} columns"
    )

    df_translators_cost_pairs = pd.read_csv(PATH_TRANSLATORS_COST_PAIRS_CSV)
    logger.info(
        "Loaded translators cost pairs:"
        f" {df_translators_cost_pairs.shape[0]} rows"
    )

    df_schedules = pd.read_csv(PATH_SCHEDULES_CSV)
    logger.info(f"Loaded schedules: {df_schedules.shape[0]} rows")

    df_clients = pd.read_csv(PATH_CLIENTS_CSV)
    logger.info(f"Loaded clients: {df_clients.shape[0]} rows")

    dict_dataframes = {
        "df_data": df_data,
        "df_translators_cost_pairs": df_translators_cost_pairs,
        "df_schedules": df_schedules,
        "df_clients": df_clients
    }

    for name, df in dict_dataframes.items():
        log_dataframe_info(logger, df, name)

    return dict_dataframes


def create_translator_mapping(df_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Create bidirectional mappings between translator names and unique integer IDs

    These mappings enable efficient storage and retrieval of translator information
    throughout the system, particularly in models and databases

    Args:
        df_data: Main historical dataset with translator information

    Returns:
        Dictionary with translator_to_id and id_to_translator bidirectional mappings
    """

    logger.info("Creating translator mapping")

    list_translators = sorted(
        df_data["TRANSLATOR"].unique()
    )  # Sort for deterministic results

    dict_translator_to_id = {
        translator: idx for idx, translator in enumerate(list_translators)
    }
    dict_id_to_translator = {
        str(idx): translator for idx, translator in enumerate(list_translators)
    }

    dict_translator_mapping = {
        "translator_to_id": dict_translator_to_id,
        "id_to_translator": dict_id_to_translator
    }

    logger.info(
        "Created translator mapping for"
        f" {len(list_translators)} translators"
    )

    return dict_translator_mapping


def create_translator_metrics(df_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive performance metrics for each translator

    Aggregates historical performance data including quality ratings, delivery rates,
    client experience, industry sector experience, and task type specialization
    Also computes global averages for comparative analysis

    Args:
        df_data: Main historical dataset with translation tasks

    Returns:
        Dictionary with metrics per translator and global system averages
    """

    logger.info("Creating translator metrics")

    dict_translator_metrics = {}

    # Calculate global averages for reference
    float_global_avg_quality = df_data["QUALITY_EVALUATION"].mean()
    float_global_avg_cost = df_data["COST"].mean()
    float_global_avg_forecast = df_data["FORECAST"].mean()
    float_global_task_count = df_data.shape[0] / df_data["TRANSLATOR"].nunique()

    # Calculate on-time delivery rate globally
    df_data["ONTIME"] = (df_data["DELIVERED"] <= df_data["END"]).astype(int)
    float_global_ontime_rate = df_data["ONTIME"].mean()

    dict_translator_metrics["__global_average__"] = {
        "task_count": float_global_task_count,
        "avg_cost": float_global_avg_cost,
        "avg_forecast": float_global_avg_forecast,
        "avg_quality": float_global_avg_quality,
        "ontime_rate": float_global_ontime_rate,
    }

    for translator in df_data["TRANSLATOR"].unique():
        df_translator = df_data[df_data["TRANSLATOR"] == translator]

        task_count = df_translator.shape[0]
        avg_quality = df_translator["QUALITY_EVALUATION"].mean()
        avg_cost = df_translator["COST"].mean()
        avg_forecast = df_translator["FORECAST"].mean()
        ontime_rate = df_translator["ONTIME"].mean()

        dict_client_history = df_translator["MANUFACTURER"].value_counts().to_dict()
        dict_sector_history = (
            df_translator["MANUFACTURER_SECTOR"].value_counts().to_dict()
        )
        dict_task_type_history = (
            df_translator["TASK_TYPE"].value_counts().to_dict()
        )

        dict_translator_metrics[translator] = {
            "task_count": task_count,
            "avg_cost": avg_cost,
            "avg_forecast": avg_forecast,
            "avg_quality": avg_quality,
            "ontime_rate": ontime_rate,
            "client_history": dict_client_history,
            "sector_history": dict_sector_history,
            "task_type_history": dict_task_type_history,
        }

    logger.info(
        "Created metrics for"
        f" {len(dict_translator_metrics) - 1} translators"
    )

    return dict_translator_metrics


def create_language_pair_metrics(
    df_data: pd.DataFrame
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate statistical aggregates for each language pair in the system

    Computes metrics such as frequency, average quality, average cost,
    number of capable translators, and average time estimates. These
    metrics help in understanding language pair difficulty and resource needs

    Args:
        df_data: Main historical dataset with translation tasks

    Returns:
        Nested dictionary structure with metrics by source language, target language
        and global averages
    """

    logger.info("Creating language pair metrics")

    dict_language_pair_metrics = {}

    # Calculate global averages
    float_global_avg_quality = df_data["QUALITY_EVALUATION"].mean()
    float_global_avg_cost = df_data["COST"].mean()
    float_global_avg_forecast = df_data["FORECAST"].mean()
    float_global_translator_count = (
        df_data.groupby(["SOURCE_LANG", "TARGET_LANG"])["TRANSLATOR"]
        .nunique()
        .mean()
    )
    float_global_task_count = (
        df_data.groupby(["SOURCE_LANG", "TARGET_LANG"]).size().mean()
    )

    dict_language_pair_metrics["__global_average__"] = {
        "task_count": float_global_task_count,
        "avg_cost": float_global_avg_cost,
        "avg_quality": float_global_avg_quality,
        "translator_count": float_global_translator_count,
        "avg_forecast": float_global_avg_forecast,
    }

    # Process each language pair
    for source_lang in df_data["SOURCE_LANG"].unique():
        dict_language_pair_metrics[source_lang] = {}
        df_source = df_data[df_data["SOURCE_LANG"] == source_lang]

        for target_lang in df_source["TARGET_LANG"].unique():
            df_pair = df_source[df_source["TARGET_LANG"] == target_lang]

            task_count = df_pair.shape[0]
            avg_quality = df_pair["QUALITY_EVALUATION"].mean()
            avg_cost = df_pair["COST"].mean()
            translator_count = df_pair["TRANSLATOR"].nunique()
            avg_forecast = df_pair["FORECAST"].mean()

            dict_language_pair_metrics[source_lang][target_lang] = {
                "task_count": task_count,
                "avg_cost": avg_cost,
                "avg_quality": avg_quality,
                "translator_count": translator_count,
                "avg_forecast": avg_forecast,
            }

    logger.info(
        "Created metrics for language pairs across"
        f" {len(dict_language_pair_metrics) - 1} source languages"
    )

    return dict_language_pair_metrics


def create_translator_capabilities(
    df_translators_cost_pairs: pd.DataFrame
) -> Dict[str, Dict[str, List[str]]]:
    """
    Map each translator to their supported language pairs

    Creates a capability matrix indicating which source and target language
    combinations each translator can handle. This is essential for
    translator assignment and filtering candidates for tasks

    Args:
        df_translators_cost_pairs: Dataset with translator language capabilities

    Returns:
        Nested dictionary mapping translators to their source and target language capabilities
    """

    logger.info("Creating translator capabilities")

    dict_translator_capabilities = {}

    for translator in df_translators_cost_pairs[
        "TRANSLATOR"
    ].unique():  # Get unique translators
        df_translator = df_translators_cost_pairs[
            df_translators_cost_pairs["TRANSLATOR"] == translator
        ]  # Filter by translator
        dict_translator_capabilities[translator] = {}  # Initialize capabilities

        for source_lang in df_translator["SOURCE_LANG"].unique():
            df_source = df_translator[
                df_translator["SOURCE_LANG"] == source_lang
            ]
            list_target_langs = df_source["TARGET_LANG"].tolist()
            dict_translator_capabilities[translator][source_lang] = (
                list_target_langs
            )

    logger.info(
        "Created capabilities for"
        f" {len(dict_translator_capabilities)} translators"
    )

    return dict_translator_capabilities


def create_translator_hourly_rates(
    df_translators_cost_pairs: pd.DataFrame
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Map each translator to their hourly rates for specific language pairs

    Creates a comprehensive rate card for each translator across all their
    supported language combinations. These rates are used for cost calculations
    and optimization in the assignment process

    Args:
        df_translators_cost_pairs: Dataset with translator rates by language pair

    Returns:
        Nested dictionary mapping translators to their hourly rates by language pair
    """

    logger.info("Creating translator hourly rates")

    dict_translator_hourly_rates = {}

    for translator in df_translators_cost_pairs["TRANSLATOR"].unique():
        df_translator = df_translators_cost_pairs[
            df_translators_cost_pairs["TRANSLATOR"] == translator
        ]
        dict_translator_hourly_rates[translator] = {}

        for source_lang in df_translator["SOURCE_LANG"].unique():
            df_source = df_translator[
                df_translator["SOURCE_LANG"] == source_lang
            ]
            dict_translator_hourly_rates[translator][source_lang] = {}

            for _, row in df_source.iterrows():
                target_lang = row["TARGET_LANG"]
                hourly_rate = row["HOURLY_RATE"]
                dict_translator_hourly_rates[translator][source_lang][
                    target_lang
                ] = hourly_rate

    logger.info(
        "Created hourly rates for"
        f" {len(dict_translator_hourly_rates)} translators"
    )

    return dict_translator_hourly_rates


def create_clients_data(
    df_clients: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Process client-specific quality requirements and pricing information

    Extracts key client parameters including minimum quality thresholds,
    selling hourly prices, and special preferences (wildcards). These settings
    guide the translator selection and prioritization for each client

    Args:
        df_clients: Dataset with client requirements and preferences

    Returns:
        Dictionary with quality thresholds, hourly prices, and wildcards by client
    """

    logger.info("Creating clients data")

    dict_clients_data = {}

    for _, row in df_clients.iterrows():
        client_name = row["CLIENT_NAME"]
        selling_hourly_price = row["SELLING_HOURLY_PRICE"]
        min_quality = row["MIN_QUALITY"]
        wildcard = row["WILDCARD"]

        dict_clients_data[client_name] = {
            "selling_hourly_price": selling_hourly_price,
            "min_quality": min_quality,
            "wildcard": wildcard
        }

    logger.info(f"Created data for {len(dict_clients_data)} clients")

    return dict_clients_data


def create_translator_schedule_metrics(
    df_schedules: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """
    Process translator availability and workload data from schedules

    Parses schedule information to determine when each translator is available
    and their total working capacity. This information is crucial for workload
    balancing and deadline feasibility assessment in assignment decisions

    Args:
        df_schedules: Dataset with translator availability schedules

    Returns:
        Dictionary with weekly hours and hourly availability by translator
    """

    logger.info("Creating translator schedule metrics")

    logger.info(f"Sample START times: {df_schedules['START'].head(3).tolist()}")
    logger.info(f"Sample END times: {df_schedules['END'].head(3).tolist()}")

    dict_translator_schedule_metrics = {}
    list_days = ["MON", "TUES", "WED", "THURS", "FRI", "SAT", "SUN"]

    for idx, row in df_schedules.iterrows():
        translator = row["NAME"]

        # Parse working hours
        start_str = str(row["START"]).strip()
        end_str = str(row["END"]).strip()

        start_time = datetime.strptime(start_str, "%H:%M:%S").time()
        end_time = datetime.strptime(end_str, "%H:%M:%S").time()

        # Calculate daily working hours
        if end_time > start_time:
            daily_hours = (
                datetime.combine(datetime.today(), end_time)
                - datetime.combine(datetime.today(), start_time)
            ).seconds / 3600
        else:
            # Handle overnight shifts
            daily_hours = (
                datetime.combine(datetime.today(), end_time)
                - datetime.combine(datetime.today(), start_time)
            ).seconds / 3600 + 24

        # Calculate weekly capacity
        weekly_hours = daily_hours * sum(row[day] for day in list_days)

        # Create availability schedule
        dict_availability = {}
        for day in list_days:
            if row[day] == 1:  # Translator works on this day
                dict_hours = {}

                start_hour = start_time.hour
                end_hour = end_time.hour

                if end_hour <= start_hour:
                    end_hour += 24  # Adjust for overnight shifts

                # Mark hourly availability
                for hour in range(start_hour, end_hour + 1):
                    hour_key = str(hour % 24)  # Convert back to 24-hour format
                    dict_hours[hour_key] = 1

                dict_availability[day] = dict_hours

        dict_translator_schedule_metrics[translator] = {
            "weekly_hours": weekly_hours,
            "availability": dict_availability
        }

    logger.info(
        "Created schedule metrics for"
        f" {len(dict_translator_schedule_metrics)} translators"
    )

    return dict_translator_schedule_metrics


def create_translator_efficiency_metrics(
    df_data: pd.DataFrame,
    df_translators_cost_pairs: pd.DataFrame
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Calculate detailed performance metrics for each translator by language pair

    Generates granular efficiency metrics showing how each translator performs
    with specific language combinations

    These metrics include task counts, average costs, time estimates, and quality ratings

    Args:
        df_data: Main historical dataset with translation tasks
        df_translators_cost_pairs: Dataset with translator language capabilities

    Returns:
        Nested dictionary with performance metrics by translator and language pair
    """

    logger.info("Creating translator efficiency metrics")

    dict_translator_efficiency_metrics = {}
    list_translators = df_translators_cost_pairs["TRANSLATOR"].unique()

    for translator in list_translators:
        # Get historical data for this translator
        df_translator = df_data[df_data["TRANSLATOR"] == translator]

        if df_translator.empty:
            continue  # Skip if no historical data

        dict_translator_efficiency_metrics[translator] = {}

        # Get language capabilities for this translator
        df_capabilities = df_translators_cost_pairs[
            df_translators_cost_pairs["TRANSLATOR"] == translator
        ]

        for source_lang in df_capabilities["SOURCE_LANG"].unique():
            dict_translator_efficiency_metrics[translator][source_lang] = {}
            df_source = df_capabilities[
                df_capabilities["SOURCE_LANG"] == source_lang
            ]

            for target_lang in df_source["TARGET_LANG"].unique():
                # Find historical performance for this language pair
                df_pair = df_translator[
                    (df_translator["SOURCE_LANG"] == source_lang)
                    & (df_translator["TARGET_LANG"] == target_lang)
                ]

                if df_pair.empty:
                    continue  # Skip if no historical data for this pair

                # Calculate performance metrics
                task_count = df_pair.shape[0]
                avg_cost = df_pair["COST"].mean()
                avg_forecast = df_pair["FORECAST"].mean()
                avg_quality = df_pair["QUALITY_EVALUATION"].mean()

                dict_translator_efficiency_metrics[translator][source_lang][
                    target_lang
                ] = {
                    "task_count": task_count,
                    "avg_cost": avg_cost,
                    "avg_forecast": avg_forecast,
                    "avg_quality": avg_quality,
                }

    logger.info(
        "Created efficiency metrics for"
        f" {len(dict_translator_efficiency_metrics)} translators"
    )

    return dict_translator_efficiency_metrics


def split_data(df_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split data into train, validation, and test sets using chronological partitioning

    Uses time-based splitting to create non-overlapping datasets that maintain
    temporal patterns in the data. This approach follows Google Vertex AI's
    recommended practice for time-series data, ensuring realistic evaluation

    Args:
        df_data: Main historical dataset with translation tasks

    Returns:
        Dictionary containing training, validation, and test dataframes
    """

    logger.info(
        "Splitting data chronologically into train, validation, and test sets"
    )

    # Verify split proportions
    float_test_size = 1 - TRAIN_SIZE - VAL_SIZE
    if abs(float_test_size - TEST_SIZE) > 0.001:
        logger.warning(
            f"Test size computed as {float_test_size},"
            f" but TEST_SIZE is {TEST_SIZE}"
        )

    # Sort chronologically by completion date
    df_sorted = df_data.sort_values(by="CLOSE").reset_index(drop=True)

    # Determine split points
    n_samples = len(df_sorted)
    train_end_idx = int(n_samples * TRAIN_SIZE)
    val_end_idx = train_end_idx + int(n_samples * VAL_SIZE)

    # Create splits
    df_train = df_sorted.iloc[:train_end_idx]
    df_val = df_sorted.iloc[train_end_idx:val_end_idx]
    df_test = df_sorted.iloc[val_end_idx:]

    dict_splits = {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test
    }

    # Log split statistics
    logger.info(
        "Data split:"
        f" {df_train.shape[0]} train, {df_val.shape[0]} validation,"
        f" {df_test.shape[0]} test samples"
    )
    logger.info(
        "Unique translators:"
        f" {df_train['TRANSLATOR'].nunique()} train,"
        f" {df_val['TRANSLATOR'].nunique()} validation,"
        f" {df_test['TRANSLATOR'].nunique()} test"
    )
    logger.info(
        "Train period:"
        f" {df_train['CLOSE'].min()} to {df_train['CLOSE'].max()}"
    )
    logger.info(
        "Validation period:"
        f" {df_val['CLOSE'].min()} to {df_val['CLOSE'].max()}"
    )
    logger.info(
        "Test period:"
        f" {df_test['CLOSE'].min()} to {df_test['CLOSE'].max()}"
    )

    # Analyze translator overlap
    train_translators = set(df_train["TRANSLATOR"].unique())
    val_translators = set(df_val["TRANSLATOR"].unique())
    test_translators = set(df_test["TRANSLATOR"].unique())

    logger.info(
        "Train-val translator overlap:"
        f" {len(train_translators.intersection(val_translators))}"
    )
    logger.info(
        "Train-test translator overlap:"
        f" {len(train_translators.intersection(test_translators))}"
    )
    logger.info(
        "Val-test translator overlap:"
        f" {len(val_translators.intersection(test_translators))}"
    )

    return dict_splits


def save_processed_data(dict_splits: Dict[str, pd.DataFrame]) -> None:
    """
    Save processed datasets to parquet files for efficient storage and access

    Persists the training, validation, and test datasets in optimized
    parquet format for downstream modeling tasks

    Args:
        dict_splits: Dictionary containing training, validation, and test dataframes
    """

    logger.info("Saving processed data")

    save_parquet(dict_splits["df_train"], PATH_TRAIN_PARQUET)
    save_parquet(dict_splits["df_val"], PATH_VAL_PARQUET)
    save_parquet(dict_splits["df_test"], PATH_TEST_PARQUET)

    logger.info(f"Saved processed data to {PATH_BASE_DIR}")


def save_artifacts(dict_artifacts: Dict[str, Any]) -> None:
    """
    Save generated artifacts in both JSON and pickle formats

    Persists all system artifacts for use by downstream modules. JSON format
    provides human-readability and interoperability, while pickle format
    preserves Python object structures

    Args:
        dict_artifacts: Dictionary containing all artifact dictionaries
    """

    logger.info("Saving artifacts")

    for artifact_name, artifact_data in dict_artifacts.items():
        path_json = os.path.join(
            PATH_ARTIFACTS_DIR,
            f"{artifact_name}.json"
        )
        path_pkl = os.path.join(
            PATH_ARTIFACTS_DIR,
            f"{artifact_name}.pkl"
        )

        save_json(artifact_data, path_json)
        save_pickle(artifact_data, path_pkl)

    # Save metadata for tracking
    metadata = {
        "preprocessing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "artifacts": list(dict_artifacts.keys())
    }
    path_metadata = os.path.join(PATH_ARTIFACTS_DIR, "metadata.json")
    save_json(metadata, path_metadata)

    logger.info(f"Saved artifacts to {PATH_ARTIFACTS_DIR}")


def process_base_data() -> None:
    """
    Orchestrate the complete base preprocessing pipeline

    This function coordinates the entire workflow from data loading through
    artifact generation to data splitting and persistence. It forms the
    foundation for all downstream modeling and analysis tasks
    """

    logger.info("Starting base preprocessing")

    # Create directory structure
    create_output_dirs()

    # Load raw datasets
    dict_dataframes = load_raw_data()
    df_data = dict_dataframes["df_data"]
    df_translators_cost_pairs = dict_dataframes["df_translators_cost_pairs"]
    df_schedules = dict_dataframes["df_schedules"]
    df_clients = dict_dataframes["df_clients"]

    # Generate system artifacts
    dict_translator_mapping = create_translator_mapping(df_data)
    dict_translator_metrics = create_translator_metrics(df_data)
    dict_language_pair_metrics = create_language_pair_metrics(df_data)
    dict_translator_capabilities = create_translator_capabilities(
        df_translators_cost_pairs
    )
    dict_translator_hourly_rates = create_translator_hourly_rates(
        df_translators_cost_pairs
    )
    dict_clients_data = create_clients_data(df_clients)
    dict_translator_schedule_metrics = create_translator_schedule_metrics(
        df_schedules
    )
    dict_translator_efficiency_metrics = (
        create_translator_efficiency_metrics(df_data, df_translators_cost_pairs)
    )

    dict_artifacts = {
        "translator_mapping": dict_translator_mapping,
        "translator_metrics": dict_translator_metrics,
        "language_pair_metrics": dict_language_pair_metrics,
        "translator_capabilities": dict_translator_capabilities,
        "translator_hourly_rates": dict_translator_hourly_rates,
        "clients_data": dict_clients_data,
        "translator_schedule_metrics": dict_translator_schedule_metrics,
        "translator_efficiency_metrics": dict_translator_efficiency_metrics,
    }

    # Split data for modeling
    dict_splits = split_data(df_data)

    # Persist processed data and artifacts
    save_processed_data(dict_splits)
    save_artifacts(dict_artifacts)

    logger.info("Base preprocessing completed successfully")


def main():
    """
    Entry point for the base preprocessing pipeline with error handling
    """
    
    try:
        process_base_data()
    except Exception as e:
        logger.error(f"Error in base preprocessing: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()