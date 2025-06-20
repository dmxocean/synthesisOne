# src/preprocessing/ranking.py

"""
Preprocessing for LGBMRanker model in the Translator Assignment System
"""

import os
import sys
import json
import logging
import pickle
import pandas as pd
import numpy as np

from typing import (
    Dict,
    Any,
    List,
    Tuple,
    Optional,
)

from sklearn.preprocessing import OrdinalEncoder
import multiprocessing as mp
from functools import partial
import time
import yaml

# Path configuration
PATH_PREPROCESSING = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_PREPROCESSING)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)

# Path constants
PATH_DATA_PROCESSED = os.path.join(PATH_ROOT, "data", "processed")
PATH_BASE_PROCESSED = os.path.join(PATH_DATA_PROCESSED, "base")
PATH_RANKING_PROCESSED = os.path.join(PATH_DATA_PROCESSED, "ranking")
PATH_ARTIFACTS_DIR = os.path.join(PATH_RANKING_PROCESSED, "artifacts")
PATH_CONFIG = os.path.join(PATH_ROOT, "config", "ranking_config.yaml")

# Input file paths
PATH_TRAIN_PARQUET = os.path.join(PATH_BASE_PROCESSED, "train.parquet")
PATH_VAL_PARQUET = os.path.join(PATH_BASE_PROCESSED, "val.parquet")
PATH_TEST_PARQUET = os.path.join(PATH_BASE_PROCESSED, "test.parquet")
PATH_BASE_ARTIFACTS = os.path.join(PATH_BASE_PROCESSED, "artifacts")

# Output file paths
PATH_X_TRAIN_PARQUET = os.path.join(PATH_RANKING_PROCESSED, "X_train.parquet")
PATH_GROUPS_TRAIN_NPY = os.path.join(PATH_RANKING_PROCESSED, "groups_train.npy")
PATH_RELEVANCE_TRAIN_NPY = os.path.join(PATH_RANKING_PROCESSED, "relevance_train.npy")
PATH_TASK_IDS_TRAIN_NPY = os.path.join(PATH_RANKING_PROCESSED, "task_ids_train.npy")
PATH_X_VAL_PARQUET = os.path.join(PATH_RANKING_PROCESSED, "X_val.parquet")
PATH_GROUPS_VAL_NPY = os.path.join(PATH_RANKING_PROCESSED, "groups_val.npy")
PATH_RELEVANCE_VAL_NPY = os.path.join(PATH_RANKING_PROCESSED, "relevance_val.npy")
PATH_TASK_IDS_VAL_NPY = os.path.join(PATH_RANKING_PROCESSED, "task_ids_val.npy")
PATH_X_TEST_PARQUET = os.path.join(PATH_RANKING_PROCESSED, "X_test.parquet")
PATH_GROUPS_TEST_NPY = os.path.join(PATH_RANKING_PROCESSED, "groups_test.npy")
PATH_RELEVANCE_TEST_NPY = os.path.join(PATH_RANKING_PROCESSED, "relevance_test.npy")
PATH_TASK_IDS_TEST_NPY = os.path.join(PATH_RANKING_PROCESSED, "task_ids_test.npy")

# Artifact file paths
PATH_ENCODERS_PKL = os.path.join(PATH_ARTIFACTS_DIR, "encoders.pkl")
PATH_FEATURE_COLUMNS_JSON = os.path.join(PATH_ARTIFACTS_DIR, "feature_columns.json")
PATH_METADATA_JSON = os.path.join(PATH_ARTIFACTS_DIR, "metadata.json")

# Project modules
from src.utils.utils import *

# Initialize logger
logger = setup_logger(
    "ranking_preprocessing",
    "preprocessing.log",
    logging.INFO
)

# Constants for data processing
CATEGORICAL_COLS = [
    "SOURCE_LANG",
    "TARGET_LANG",
    "LANGUAGE_PAIR",
    "TASK_TYPE",
    "TRANSLATOR",
    "MANUFACTURER",
    "MANUFACTURER_SECTOR",
    "WILDCARD",
]
NUMERICAL_COLS = [
    "FORECAST",
    "TASK_DURATION_HOURS",
    "URGENCY_RATIO",
]

# Define core model features
CORE_FEATURES = [
    "CAPABILITY_MATCH",
    "RATE_COST_RATIO",
    "HISTORICAL_QUALITY",
    # "CLIENT_EXPERIENCE",
    "SECTOR_EXPERIENCE",
    "TASK_TYPE_EXPERIENCE",
    "LANGUAGE_PAIR_RARITY",
    "EXPERIENCE_LEVEL",
    "ONTIME_RATE",
    "TASK_COMPLEXITY",
    "URGENCY_COMPATIBILITY",
]
ENCODED_FEATURES = [
    "SOURCE_LANG_ENCODED",
    "TARGET_LANG_ENCODED",
    "LANGUAGE_PAIR_ENCODED",
    "TASK_TYPE_ENCODED",
    "MANUFACTURER_ENCODED",
    "MANUFACTURER_SECTOR_ENCODED",
]
BASE_NUMERICAL = [
    "FORECAST",
    "URGENCY_RATIO",
]
BATCH_SIZE = 1000  # Avoid memory issues with large datasets
MIN_QUALITY_THRESHOLD = 6.0

# Language pair frequency thresholds for group sizing
COMMON_LANGUAGE_PAIR_GROUP_SIZE = 25  # For language pairs with >1000 tasks
MEDIUM_LANGUAGE_PAIR_GROUP_SIZE = 10  # For language pairs with >100 tasks
RARE_LANGUAGE_PAIR_GROUP_SIZE = 5  # For all other language pairs

# Group composition percentages for relevance scores
CAPABLE_TRANSLATORS_PERCENTAGE = 0.50
INCAPABLE_TRANSLATORS_PERCENTAGE = 1.0 - CAPABLE_TRANSLATORS_PERCENTAGE

# Distribution within capable translators
STRONG_MATCHES_PERCENTAGE = 0.3
BASIC_MATCHES_PERCENTAGE = 1.0 - STRONG_MATCHES_PERCENTAGE

# Task complexity weights
TASK_TYPE_WEIGHTS = {
    "Translation": 1.0,
    "Proofreading": 0.7,
    "Editing": 0.8,
    "Review": 0.6,
    "QA": 0.5,
    "Transcription": 0.9,
    "Post-Editing": 0.8,
}

# Urgency thresholds
VERY_URGENT_THRESHOLD = 0.8
MODERATELY_URGENT_THRESHOLD = 0.5

# Experience normalization factor
EXPERIENCE_NORM_FACTOR = 10.0  # Caps log-transformed experience at this value
SECTOR_EXPERIENCE_NORM_FACTOR = 5.0
TASK_TYPE_EXPERIENCE_NORM_FACTOR = 5.0

# Relevance thresholds
HIGH_QUALITY_THRESHOLD = 7.0


def create_output_dirs() -> None:
    """Create necessary directory structure for LGBMRanker processed data"""

    logger.info("Creating output directories")
    ensure_dir(PATH_RANKING_PROCESSED)
    ensure_dir(PATH_ARTIFACTS_DIR)
    ensure_dir(os.path.join(PATH_ROOT, "models", "ranking"))
    ensure_dir(os.path.join(PATH_ROOT, "results", "ranking", "plots"))
    logger.info("Output directories created successfully")


def load_base_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load base processed data from parquet files"""

    logger.info("Loading base processed data")
    df_train = load_parquet(PATH_TRAIN_PARQUET)
    df_val = load_parquet(PATH_VAL_PARQUET)
    df_test = load_parquet(PATH_TEST_PARQUET)
    logger.info(
        f"Loaded {len(df_train)} training samples, {len(df_val)} validation samples, {len(df_test)} test samples"
    )
    log_dataframe_info(
        logger,
        df_train,
        "Training data"
    )

    return df_train, df_val, df_test


def load_artifacts() -> Dict[str, Any]:
    """Load artifacts from base preprocessing"""

    logger.info("Loading base preprocessing artifacts")
    dict_artifacts = {
        "translator_mapping": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "translator_mapping.pkl")),
        "translator_metrics": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "translator_metrics.pkl")),
        "language_pair_metrics": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "language_pair_metrics.pkl")),
        "translator_capabilities": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "translator_capabilities.pkl")),
        "translator_hourly_rates": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "translator_hourly_rates.pkl")),
        "translator_efficiency_metrics": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "translator_efficiency_metrics.pkl")),
        "clients_data": load_pickle(os.path.join(PATH_BASE_ARTIFACTS, "clients_data.pkl")),
    }
    logger.info("Base preprocessing artifacts loaded successfully")

    return dict_artifacts


def create_derived_features(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate derived features from the main dataset for the ranking model

    Creates and transforms several important features:
    - LANGUAGE_PAIR: Combines source and target languages in that order
    - TASK_DURATION_HOURS: Converts task duration from timestamps to hours
        IMPORTANT: higher values means longer tasks
    - URGENCY_RATIO: Calculates and normalizes the time pressure ratio
        IMPORTANT: higher means more urgent tasks
    - FORECAST: Applies logarithmic transformation to normalize distribution, higher means longer time needed for the task

    Returns:
        DataFrame with added and transformed columns ready for model training
    """

    logger.info("Creating derived features for ranking model")
    if "LANGUAGE_PAIR" not in df_data.columns:
        df_data["LANGUAGE_PAIR"] = df_data["SOURCE_LANG"] + " > " + df_data["TARGET_LANG"]  # Concatenate languages with separator
        logger.info(f"Created LANGUAGE_PAIR feature: {df_data['LANGUAGE_PAIR'].nunique()} unique pairs")

    if "TASK_DURATION_HOURS" not in df_data.columns:
        df_data["TASK_DURATION_HOURS"] = (df_data["END"] - df_data["START"]).dt.total_seconds() / 3600  # Convert seconds to hours
        logger.info(f"Created TASK_DURATION_HOURS feature: range [{df_data['TASK_DURATION_HOURS'].min():.2f}, {df_data['TASK_DURATION_HOURS'].max():.2f}] hours")

    logger.info(f"Raw FORECAST range: [{df_data['FORECAST'].min():.2f}, {df_data['FORECAST'].max():.2f}]")

    if "URGENCY_RATIO" not in df_data.columns:
        df_data["URGENCY_RATIO"] = df_data["TASK_DURATION_HOURS"] / df_data["FORECAST"]  # Ratio of actual time to forecast time

        df_data["URGENCY_RATIO"] = df_data["URGENCY_RATIO"].replace(
            [np.inf, -np.inf],
            np.nan
        )  # Replace infinite values
        df_data["URGENCY_RATIO"] = df_data["URGENCY_RATIO"].fillna(df_data["URGENCY_RATIO"].median())  # Fill missing with median

        logger.info(f"Raw URGENCY_RATIO range: [{df_data['URGENCY_RATIO'].min():.2f}, {df_data['URGENCY_RATIO'].max():.2f}]")

        log_ratio = np.log1p(df_data["URGENCY_RATIO"])  # Log transform to handle skewness

        log_mean = np.mean(log_ratio)
        log_std = np.std(log_ratio)
        z_score = (log_ratio - log_mean) / log_std  # Standardize to z-scores

        df_data["URGENCY_RATIO"] = 1 / (1 + np.exp(-z_score))  # Sigmoid transformation to [0,1] range

        logger.info(f"Created normalized URGENCY_RATIO feature: range [{df_data['URGENCY_RATIO'].min():.2f}, {df_data['URGENCY_RATIO'].max():.2f}]")

    df_data["FORECAST"] = np.log1p(df_data["FORECAST"])  # Log transform forecast values

    logger.info(f"Normalized FORECAST range: [{df_data['FORECAST'].min():.2f}, {df_data['FORECAST'].max():.2f}]")

    return df_data


def check_language_capability(
    str_translator: str,
    str_source_lang: str,
    str_target_lang: str,
    dict_translator_capabilities: Dict[str, Dict],
) -> bool:
    """
    Verify if a translator has capability for a specific language pair

    Checks dictionary structure to determine if a translator
    is qualified to translate from source language to target language
    based on their recorded capabilities

    All historical translators are capable, this is relevant for the pool generation

    Returns:
        Boolean indicating capability match status
    """

    try:
        if str_source_lang in dict_translator_capabilities[str_translator]:  # Check if source language exists
            if str_target_lang in dict_translator_capabilities[str_translator][str_source_lang]:  # Check target language
                return True
        return False
    except KeyError:
        return False  # Translator or language not found


def determine_group_size(
    str_source_lang: str,
    str_target_lang: str,
    dict_language_pair_metrics: Dict[str, Dict[str, Dict[str, Any]]],
) -> int:
    """
    Adjusts the translator pool size dynamically depending on how common
    or rare a language pair is within the historical data:
    - Common language pairs use larger groups
    - Medium language pairs use medium-sized groups
    - Rare language pairs use smaller groups

    Returns:
        Target group size for constructing translator pools
    """

    try:
        int_task_count = dict_language_pair_metrics[str_source_lang][str_target_lang]["task_count"]  # Get historical count
    except (KeyError, TypeError):
        int_task_count = dict_language_pair_metrics["__global_average__"]["task_count"]  # Fallback to global average

    if int_task_count > 1000:
        return COMMON_LANGUAGE_PAIR_GROUP_SIZE  # Larger pools for common languages
    elif int_task_count > 100:
        return MEDIUM_LANGUAGE_PAIR_GROUP_SIZE  # Medium pools for moderately common languages
    else:
        return RARE_LANGUAGE_PAIR_GROUP_SIZE  # Smaller pools for rare language pairs


def calculate_rate_cost_ratio(
    str_translator: str,
    str_source_lang: str,
    str_target_lang: str,
    float_cost: float,
    dict_language_pair_metrics: Dict[str, Dict],
    dict_translator_efficiency_metrics: Dict[str, Dict],
    dict_translator_metrics: Dict[str, Dict] = None,
) -> float:
    """
    Evaluates translator cost efficiency by comparing their rates to his historical performance

    Uses a priority approach:
    - First tries specific language pair costs from efficiency metrics
    - Falls back to global averages if specific data unavailable
    - Falls back to translator overall average cost as last resort

    IMPORTANT: Higher values indicate better cost efficiency

    Returns:
        Cost efficiency score where higher values represent better cost efficiency
    """

    try:  # Specific language pair costs
        global_avg_cost = dict_language_pair_metrics[str_source_lang][str_target_lang]["avg_cost"]
        translator_avg_cost = dict_translator_efficiency_metrics[str_translator][str_source_lang][str_target_lang]["avg_cost"]

        if translator_avg_cost <= 0:
            return 0.5  # Default value for invalid cost

        ratio = global_avg_cost / translator_avg_cost  # Lower cost = higher ratio = better efficiency
        return min(1.0, max(0.0, ratio))  # Clamp between 0 and 1

    except (KeyError, TypeError, ZeroDivisionError):
        try:  # Global average costs
            global_avg_cost = dict_language_pair_metrics["__global_average__"]["avg_cost"]
            translator_avg_cost = dict_translator_efficiency_metrics[str_translator][str_source_lang][str_target_lang]["avg_cost"]

            if translator_avg_cost <= 0:
                return 0.5  # Default value for invalid cost

            ratio = global_avg_cost / translator_avg_cost
            return min(1.0, max(0.0, ratio))

        except (KeyError, TypeError, ZeroDivisionError):
            if dict_translator_metrics and str_translator in dict_translator_metrics:  # Translator's overall average cost
                try:
                    global_avg_cost = dict_language_pair_metrics["__global_average__"]["avg_cost"]
                    translator_avg_cost = dict_translator_metrics[str_translator]["avg_cost"]

                    if translator_avg_cost <= 0:
                        return 0.5  # Default value for invalid cost

                    ratio = global_avg_cost / translator_avg_cost
                    return min(1.0, max(0.0, ratio))
                except (KeyError, TypeError, ZeroDivisionError):
                    pass

            return 0.5  # Default when all lookups fail


def calculate_sector_experience(
    str_translator: str,
    str_sector: str,
    dict_translator_metrics: Dict[str, Dict],
) -> float:
    """
    Calculate sector experience score based on translator history

    Measures translator expertise in a specific industry sector
    using logarithmic transformation of historical task counts

    IMPORTANT: Higher values indicate more experience in the sector

    Returns:
        Sector experience score between zero and one
    """

    try:
        translator_metrics = dict_translator_metrics[str_translator]

        try:
            sector_history = translator_metrics["sector_history"]
            tasks_for_sector = 0
            if str_sector in sector_history:
                tasks_for_sector = sector_history[str_sector]  # Get count of tasks in this sector
        except KeyError:
            tasks_for_sector = 0

        return min(
            1.0,
            np.log1p(tasks_for_sector) / SECTOR_EXPERIENCE_NORM_FACTOR
        )  # Normalize with diminishing returns
    except KeyError:
        return 0.0  # No experience if translator not found


def calculate_task_type_experience(
    str_translator: str,
    str_task_type: str,
    dict_translator_metrics: Dict[str, Dict],
) -> float:
    """
    Calculate task type experience score based on translator history

    Measures translator expertise with a specific task type
    such as translation, editing, proofreading, or review

    Uses logarithmic transformation to represent experience level
    with diminishing returns for very high task counts

    IMPORTANT: Higher values indicate more extensive experience with the task type

    Returns:
        Task type experience score between zero and one
    """

    try:
        translator_metrics = dict_translator_metrics[str_translator]

        try:
            task_type_history = translator_metrics["task_type_history"]
            tasks_for_type = 0
            if str_task_type in task_type_history:
                tasks_for_type = task_type_history[str_task_type]  # Get count of tasks of this type
        except KeyError:
            tasks_for_type = 0

        return min(
            1.0,
            np.log1p(tasks_for_type) / TASK_TYPE_EXPERIENCE_NORM_FACTOR
        )  # Log transform with cap
    except KeyError:
        return 0.0  # No experience if translator not found


def calculate_language_pair_rarity(
    str_source_lang: str,
    str_target_lang: str,
    dict_language_pair_metrics: Dict[str, Dict],
) -> float:
    """
    Calculate language pair rarity score with improved distribution

    Measures how rare or common a language pair is compared to
    the global average across all language combinations

    Uses logarithmic scaling and sigmoid normalization for better
    distribution of scores across the frequency spectrum

    IMPORTANT: Higher values indicate rarer language pairs

    Returns:
        Rarity score where higher values represent rarer language pairs
    """

    try:
        task_count = dict_language_pair_metrics[str_source_lang][str_target_lang]["task_count"]  # Historical frequency

        if task_count <= 0:
            return 1.0  # Very rare if no historical data

        global_avg_task_count = dict_language_pair_metrics["__global_average__"]["task_count"]
        log_ratio = np.log1p(global_avg_task_count) / np.log1p(task_count)  # Log ratio for better scaling
        normalized_rarity = 2 / (1 + np.exp(-log_ratio + 1)) - 1  # Modified sigmoid for better distribution

        return min(1.0, max(0.0, normalized_rarity))  # Ensure range [0,1]
    except KeyError:
        return 1.0  # Default to rarest if language pair not found


def calculate_ontime_rate(str_translator: str, dict_translator_metrics: Dict[str, Dict]) -> float:
    """
    Calculate on-time rate based on translator historical performance

    Retrieves the proportion of tasks completed on time by a translator
    from their performance metrics history

    Falls back to global average if translator-specific data is unavailable

    IMPORTANT: Higher values indicate better punctuality

    Returns:
        On-time delivery rate between zero and one
    """

    try:
        if str_translator in dict_translator_metrics:
            ontime_rate = dict_translator_metrics[str_translator]["ontime_rate"]  # Direct metric lookup
            return float(ontime_rate)

        # Fallback to global average
        return float(
            dict_translator_metrics["__global_average__"]["ontime_rate"]
        )  # Global fallback

    except (KeyError, TypeError):
        return 0.5  # Default value if calculation fails


def calculate_task_complexity(float_forecast: float, str_task_type: str) -> float:
    """
    Calculate task complexity score based on forecast time and task type

    Combines two factors to estimate task difficulty:
    - Estimated duration which correlates with content complexity
    - Task type which affects the cognitive load on translators

    IMPORTANT: Higher values indicate more complex tasks

    Returns:
        Task complexity score where higher values represent more complex tasks
    """

    task_weight = 1.0
    if str_task_type in TASK_TYPE_WEIGHTS:
        task_weight = TASK_TYPE_WEIGHTS[str_task_type]  # Different task types have different complexity weights

    forecast_norm = min(1.0, float_forecast / 5)  # Normalize forecast time, cap at 5 units

    return (0.7 * forecast_norm) + (0.3 * task_weight)  # Weighted combination of factors


def calculate_urgency_compatibility(
    str_translator: str,
    float_urgency_ratio: float,
    dict_translator_metrics: Dict[str, Dict],
) -> float:
    """
    Calculate urgency compatibility based on translator on-time performance

    Evaluates how suitable a translator is for a task with specific
    urgency requirements based on their historical punctuality

    Different scoring approaches applied based on urgency level:
    - For very urgent tasks, score depends heavily on on-time rate
    - For moderately urgent tasks, requirements are more relaxed
    - For non-urgent tasks, most translators receive good scores

    IMPORTANT: Higher values indicate better compatibility with urgency

    Returns:
        Urgency compatibility score where higher values represent better match
    """

    try:
        translator_metrics = dict_translator_metrics[str_translator]
        ontime_rate = 0.5
        if "ontime_rate" in translator_metrics:
            ontime_rate = translator_metrics["ontime_rate"]  # Get historical on-time rate

        urgency_norm = min(1.0, float_urgency_ratio)  # Normalize urgency

        if urgency_norm > VERY_URGENT_THRESHOLD:
            return ontime_rate  # Very urgent tasks need reliable translators
        elif urgency_norm > MODERATELY_URGENT_THRESHOLD:
            return 0.5 + (0.5 * ontime_rate)  # Moderate urgency is more forgiving
        else:
            return 0.8  # Non-urgent tasks are compatible with most translators
    except KeyError:
        return 0.5  # Default value if translator not found


def create_task_translator_pairs(
    df_task: pd.DataFrame,
    dict_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create task-translator pairs for a single task with relevance scoring

    Generates balanced sets of translator candidates for each task, including:
    - The actual translator who performed the task historically
    - Strong matches with relevant experience and high quality
    - Basic matches who have capability but less optimal fit
    - Incapable translators who cannot handle the language pair

    Assigns relevance scores to create training data for the ranking model

    Returns:
        DataFrame with expanded task-translator combinations and relevance scores
    """

    dict_translator_capabilities = dict_artifacts["translator_capabilities"]
    dict_translator_metrics = dict_artifacts["translator_metrics"]
    dict_language_pair_metrics = dict_artifacts["language_pair_metrics"]
    dict_translator_hourly_rates = dict_artifacts["translator_hourly_rates"]

    global_avg_metrics = dict_translator_metrics["__global_average__"]

    str_task_id = df_task["TASK_ID"].iloc[0]
    str_source_lang = df_task["SOURCE_LANG"].iloc[0]
    str_target_lang = df_task["TARGET_LANG"].iloc[0]
    str_language_pair = df_task["LANGUAGE_PAIR"].iloc[0]
    str_true_translator = df_task["TRANSLATOR"].iloc[0]  # The translator who actually did the task
    str_manufacturer = df_task["MANUFACTURER"].iloc[0]

    target_group_size = determine_group_size(str_source_lang, str_target_lang, dict_language_pair_metrics)

    capable_count = int(target_group_size * CAPABLE_TRANSLATORS_PERCENTAGE)  # Using the new global variable
    incapable_count = target_group_size - capable_count  # Remainder are incapable translators

    import random

    random.seed(42)  # Fixed seed for reproducible results

    # Identify translators by capability
    list_capable_translators = []
    for translator in dict_translator_capabilities.keys():
        if check_language_capability(translator, str_source_lang, str_target_lang, dict_translator_capabilities):
            list_capable_translators.append(translator)

    # Ensure true translator is included
    if str_true_translator not in list_capable_translators and str_true_translator in dict_translator_capabilities:
        list_capable_translators.append(str_true_translator)  # Force inclusion of actual translator

    list_incapable_translators = [translator for translator in dict_translator_capabilities.keys() if translator not in list_capable_translators]

    # Categorize capable translators by relevance
    list_relevance_perfect = [str_true_translator]  # Perfect match - Actual translator
    list_remaining_capable = [t for t in list_capable_translators if t != str_true_translator]
    list_relevance_strong = []  # Good match - High quality + Client experience
    list_relevance_basic = []  # Basic match - Capable but less optimal

    # Evaluate translators
    for translator in list_remaining_capable:
        try:
            translator_data = dict_translator_metrics[translator]

            try:
                client_history = translator_data["client_history"]
                task_count_for_client = 0
                if str_manufacturer in client_history:
                    task_count_for_client = client_history[str_manufacturer]  # Previous work with client
            except KeyError:
                task_count_for_client = 0

            try:
                avg_quality = translator_data["avg_quality"]
            except KeyError:
                avg_quality = global_avg_metrics["avg_quality"]  # Use global average if missing

            # IMPORTANT: Strong relevance if worked with client AND high quality
            if task_count_for_client > 0 and avg_quality > HIGH_QUALITY_THRESHOLD:
                list_relevance_strong.append(translator)  # Previous client experience with good quality
            else:
                list_relevance_basic.append(translator)  # Lower relevance but still capable

        except KeyError:
            avg_quality = global_avg_metrics["avg_quality"]
            list_relevance_basic.append(translator)

    # Determine distribution of translators
    additional_capable_needed = capable_count - 1  # -1 for true translator
    target_relevance_strong_count = min(len(list_relevance_strong), int(additional_capable_needed * STRONG_MATCHES_PERCENTAGE))  # Using global variable
    target_relevance_basic_count = additional_capable_needed - target_relevance_strong_count  # Remaining are basic matches

    # Shuffle for randomness obtaining a diverse pool
    random.shuffle(list_relevance_strong)
    random.shuffle(list_relevance_basic)
    random.shuffle(list_incapable_translators)

    selected_relevance_strong = list_relevance_strong[:target_relevance_strong_count]
    selected_relevance_basic = list_relevance_basic[:target_relevance_basic_count]

    # Handle insufficiency in categories to meet target counts
    if len(selected_relevance_basic) < target_relevance_basic_count and len(list_relevance_strong) > target_relevance_strong_count:
        additional_needed = target_relevance_basic_count - len(selected_relevance_basic)
        additional_from_strong = list_relevance_strong[
            target_relevance_strong_count : target_relevance_strong_count + additional_needed
        ]
        selected_relevance_basic.extend(additional_from_strong)  # Use extra strong matches as basic

    if len(selected_relevance_strong) < target_relevance_strong_count and len(list_relevance_basic) > target_relevance_basic_count:
        additional_needed = target_relevance_strong_count - len(selected_relevance_strong)
        additional_from_basic = list_relevance_basic[target_relevance_basic_count : target_relevance_basic_count + additional_needed]
        selected_relevance_strong.extend(additional_from_basic)  # Promote basic matches if needed

    selected_relevance_none = list_incapable_translators[:incapable_count]  # Select incapable translators
    list_selected_translators = list_relevance_perfect + selected_relevance_strong + selected_relevance_basic + selected_relevance_none  # Combine all selected translators

    if len(list_selected_translators) != target_group_size:  # Ensure exact group size
        if len(list_selected_translators) > target_group_size:
            excess = len(list_selected_translators) - target_group_size
            if len(selected_relevance_basic) >= excess:
                selected_relevance_basic = selected_relevance_basic[:-excess]  # Trim basic matches first
            else:
                remaining_excess = excess - len(selected_relevance_basic)
                selected_relevance_none = selected_relevance_none[: (incapable_count - remaining_excess)]
                selected_relevance_basic = []

            list_selected_translators = list_relevance_perfect + selected_relevance_strong + selected_relevance_basic + selected_relevance_none
        else:
            shortage = target_group_size - len(list_selected_translators)
            additional_incapable = list_incapable_translators[incapable_count : incapable_count + shortage]
            selected_relevance_none.extend(additional_incapable)  # Add more incapable if needed

            list_selected_translators = list_relevance_perfect + selected_relevance_strong + selected_relevance_basic + selected_relevance_none

    # Create rows for each selected translator to obtain the pool
    list_rows = []
    for translator in list_selected_translators:
        row = df_task.iloc[0].to_dict()  # Copy task data
        row["TRANSLATOR"] = translator  # Set candidate translator

        # Assign relevance score based on category
        if translator in list_relevance_perfect:
            row["RELEVANCE"] = 3  # Perfect match (actual historical assignment)
        elif translator in selected_relevance_strong:
            row["RELEVANCE"] = 2  # Strong match
        elif translator in selected_relevance_basic:
            row["RELEVANCE"] = 1  # Basic match
        else:
            row["RELEVANCE"] = 0  # Incapable translators

        list_rows.append(row)

    return pd.DataFrame(list_rows)


def process_task_batch(df_batch: pd.DataFrame, dict_artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Process a batch of tasks to create task-translator pairs

    Handles multiple tasks in batch mode to improve performance,
    processing each task separately and combining results

    Returns:
        Combined DataFrame with task-translator pairs for all tasks in batch
    """

    list_pair_dfs = []

    for task_id, task_df in df_batch.groupby("TASK_ID"):  # Process each task individually
        try:
            df_task_pairs = create_task_translator_pairs(
                task_df,
                dict_artifacts
            )  # Create pairs for this task
            list_pair_dfs.append(df_task_pairs)
        except Exception as e:
            logger.error(
                f"Error processing task {task_id}: {str(e)}"
            )  # Log error but continue with other tasks

    if list_pair_dfs:
        return pd.concat(list_pair_dfs, ignore_index=True)  # Combine all results
    else:
        return pd.DataFrame()  # Return empty dataframe if all tasks failed


def create_task_translator_dataset(
    df_data: pd.DataFrame,
    dict_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create task-translator combinations for ranking with batch processing

    Processes tasks in batches to generate translator candidates for each task,
    assigning relevance scores to create training data for the ranking model

    Returns:
        DataFrame with all valid task-translator combinations and relevance scores
    """

    logger.info("Creating task-translator dataset")

    unique_task_ids = df_data["TASK_ID"].unique()
    num_tasks = len(unique_task_ids)

    logger.info(f"Processing {num_tasks} unique tasks")

    num_batches = (num_tasks + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division for batch count
    list_all_pairs = []

    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min(
            (i + 1) * BATCH_SIZE,
            num_tasks
        )  # Prevent overflow on final batch
        batch_task_ids = unique_task_ids[start_idx:end_idx]

        logger.info(f"Processing batch {i+1}/{num_batches} with {len(batch_task_ids)} tasks")

        df_batch = df_data[df_data["TASK_ID"].isin(batch_task_ids)]  # Filter data for current batch
        df_batch_pairs = process_task_batch(
            df_batch,
            dict_artifacts
        )

        if not df_batch_pairs.empty:
            list_all_pairs.append(df_batch_pairs)

        logger.info(f"Batch {i+1}/{num_batches} processed, generated {len(df_batch_pairs)} pairs")

    if list_all_pairs:
        df_all_pairs = pd.concat(list_all_pairs, ignore_index=True)
        logger.info(f"Generated a total of {len(df_all_pairs)} task-translator pairs")
        return df_all_pairs
    else:
        logger.error("No valid task-translator pairs were generated")
        return pd.DataFrame()


def try_get_dict_value(dictionary, key1, key2, default_value):
    """
    Safely retrieve nested dictionary value

    Returns:
        Retrieved value if successful, default value otherwise
    """

    try:
        return dictionary[key1][key2]  # Attempt to access nested dict
    except (KeyError, TypeError):
        return default_value  # Return default if any key is missing


def try_calculate_experience_level(translator, dict_translator_metrics):
    """
    Calculate experience level safely

    Determines translator experience level using logged transformation
    of total task count with proper error handling

    Returns:
        Experience level score or zero if calculation fails
    """

    try:
        task_count = dict_translator_metrics[translator]["task_count"]  # Get total task count
        return np.log1p(task_count) / EXPERIENCE_NORM_FACTOR  # Log transform and normalize
    except (KeyError, TypeError):
        return 0.0  # Default to no experience if data missing


def engineer_ranking_features(
    df_pairs: pd.DataFrame,
    dict_artifacts: Dict[str, Any],
) -> pd.DataFrame:
    """
    Engineer feature set for the LGBMRanker model

    Creates a comprehensive set of features measuring translator-task compatibility
    across multiple dimensions including:
    - Language expertise and rarity
    - Cost efficiency and quality metrics
    - Client and sector experience
    - Task type familiarity and complexity
    - Urgency and deadline compatibility

    Returns:
        DataFrame with complete ranking feature set ready for model training
    """

    logger.info("Engineering ranking features")

    df_features = df_pairs.copy()

    dict_translator_metrics = dict_artifacts["translator_metrics"]
    dict_language_pair_metrics = dict_artifacts["language_pair_metrics"]
    dict_translator_capabilities = dict_artifacts["translator_capabilities"]
    dict_translator_hourly_rates = dict_artifacts["translator_hourly_rates"]
    dict_translator_efficiency_metrics = dict_artifacts["translator_efficiency_metrics"]

    # Calculate LANGUAGE_PAIR_RARITY
    logger.info("Starting LANGUAGE_PAIR_RARITY calculation")
    df_features["LANGUAGE_PAIR_RARITY"] = df_features.apply(
        lambda row: calculate_language_pair_rarity(
            row["SOURCE_LANG"],
            row["TARGET_LANG"],
            dict_language_pair_metrics
        ),
        axis=1,
    )
    logger.info("Finished LANGUAGE_PAIR_RARITY calculation")

    # Calculate or use existing CAPABILITY_MATCH
    logger.info("Starting CAPABILITY_MATCH calculation")
    if "CAPABILITY_MATCH" not in df_features.columns:
        logger.info("CAPABILITY_MATCH not found, calculating it")
        df_features["CAPABILITY_MATCH"] = df_features.apply(
            lambda row: 1
            if check_language_capability(
                row["TRANSLATOR"],
                row["SOURCE_LANG"],
                row["TARGET_LANG"],
                dict_translator_capabilities
            )
            else 0,
            axis=1,
        )
    else:
        logger.info(f"CAPABILITY_MATCH already exists with {df_features['CAPABILITY_MATCH'].sum()} capable pairs")
    logger.info("Finished CAPABILITY_MATCH calculation")

    # Calculate RATE_COST_RATIO
    logger.info("Starting RATE_COST_RATIO calculation")
    df_features["RATE_COST_RATIO"] = df_features.apply(
        lambda row: calculate_rate_cost_ratio(
            row["TRANSLATOR"],
            row["SOURCE_LANG"],
            row["TARGET_LANG"],
            row["COST"] if "COST" in row else 0,
            dict_language_pair_metrics,
            dict_translator_efficiency_metrics,
            dict_translator_metrics,
        ),
        axis=1,
    )
    logger.info("Finished RATE_COST_RATIO calculation")

    # Calculate HISTORICAL_QUALITY
    logger.info("Starting HISTORICAL_QUALITY calculation")
    df_features["HISTORICAL_QUALITY"] = df_features.apply(
        lambda row: try_get_dict_value(
            dict_translator_metrics,
            row["TRANSLATOR"],
            "avg_quality",
            0
        )
        / 10.0,
        axis=1,
    )
    logger.info("Finished HISTORICAL_QUALITY calculation")

    # Calculate SECTOR_EXPERIENCE
    logger.info("Starting SECTOR_EXPERIENCE calculation")
    df_features["SECTOR_EXPERIENCE"] = df_features.apply(
        lambda row: calculate_sector_experience(
            row["TRANSLATOR"],
            row["MANUFACTURER_SECTOR"],
            dict_translator_metrics
        ),
        axis=1,
    )
    logger.info("Finished SECTOR_EXPERIENCE calculation")

    # Calculate TASK_TYPE_EXPERIENCE
    logger.info("Starting TASK_TYPE_EXPERIENCE calculation")
    df_features["TASK_TYPE_EXPERIENCE"] = df_features.apply(
        lambda row: calculate_task_type_experience(
            row["TRANSLATOR"],
            row["TASK_TYPE"],
            dict_translator_metrics
        ),
        axis=1,
    )
    logger.info("Finished TASK_TYPE_EXPERIENCE calculation")

    # Calculate EXPERIENCE_LEVEL
    logger.info("Starting EXPERIENCE_LEVEL calculation")
    df_features["EXPERIENCE_LEVEL"] = df_features.apply(
        lambda row: try_calculate_experience_level(
            row["TRANSLATOR"],
            dict_translator_metrics
        ),
        axis=1,
    )
    logger.info("Finished EXPERIENCE_LEVEL calculation")

    logger.info("Starting ONTIME_RATE calculation")
    df_features["ONTIME_RATE"] = df_features.apply(
        lambda row: calculate_ontime_rate(
            row["TRANSLATOR"],
            dict_translator_metrics
        ),
        axis=1,
    )
    logger.info("Finished ONTIME_RATE calculation")

    # Calculate TASK_COMPLEXITY
    logger.info("Starting TASK_COMPLEXITY calculation")
    df_features["TASK_COMPLEXITY"] = df_features.apply(
        lambda row: calculate_task_complexity(
            row["FORECAST"],
            row["TASK_TYPE"]
        ),
        axis=1,
    )
    logger.info("Finished TASK_COMPLEXITY calculation")

    # Calculate URGENCY_COMPATIBILITY
    logger.info("Starting URGENCY_COMPATIBILITY calculation")
    df_features["URGENCY_COMPATIBILITY"] = df_features.apply(
        lambda row: calculate_urgency_compatibility(
            row["TRANSLATOR"],
            row["URGENCY_RATIO"],
            dict_translator_metrics
        ),
        axis=1,
    )
    logger.info("Finished URGENCY_COMPATIBILITY calculation")

    logger.info(f"Feature engineering completed, generated {len(df_features.columns) - len(df_pairs.columns)} additional features")

    return df_features


def filter_model_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to include only features used by the model

    Selects relevant subset of columns needed by the ranking model,
    keeping only core features, encoded features, and base numerical features

    Preserves task identifier when available for group formation

    Returns:
        DataFrame with selected model-relevant features
    """

    logger.info("Filtering dataframe to include only model features")

    # Core features
    model_features = CORE_FEATURES + ENCODED_FEATURES + BASE_NUMERICAL  # All features needed for the model
    available_features = [col for col in model_features if col in df_features.columns]  # Only keep existing columns

    logger.info(f"Selected {len(available_features)} features for model: {', '.join(available_features)}")

    if "TASK_ID" in df_features.columns:
        return df_features[["TASK_ID"] + available_features]  # Keep task ID for grouping
    else:
        return df_features[available_features]  # Just the feature columns


def generate_task_groups(df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate group indicators for LGBMRanker training

    Creates grouping arrays that identify which samples belong to the same task,
    allowing the ranker to compare only translators within the same task

    Returns:
        Tuple containing group sizes array and task IDs array
    """

    logger.info("Generating task groups")

    grouped = df_features.groupby("TASK_ID")  # Group by task ID

    groups = []
    task_ids = []

    for task_id, group in grouped:
        groups.append(len(group))  # Store group size for LGBMRanker
        task_ids.append(task_id)  # Store task ID for reference

    logger.info(f"Generated {len(groups)} groups with {sum(groups)} total samples")

    return np.array(groups), np.array(task_ids)


def encode_categorical_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    list_categorical_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features to numeric form for model training

    Transforms categorical columns into numeric features using OrdinalEncoder,
    fitting on training data and applying same transformation to validation and test

    Handles unknown values with encoded placeholders and preserves encoders
    for later inference stage usage

    Returns:
        Tuple containing encoded datasets and dictionary of fitted encoders
    """

    logger.info("Encoding categorical features")

    dict_encoders = {}

    df_train_encoded = df_train.copy()
    df_val_encoded = df_val.copy()
    df_test_encoded = df_test.copy()

    for col in list_categorical_cols:
        if col in df_train.columns:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )  # Handle unseen values
            encoder.fit(df_train[[col]])  # Fit only on training data

            dict_encoders[col] = encoder  # Save for inference

            df_train_encoded[f"{col}_ENCODED"] = encoder.transform(df_train[[col]])  # Transform training set
            df_val_encoded[f"{col}_ENCODED"] = encoder.transform(df_val[[col]])  # Transform validation set
            df_test_encoded[f"{col}_ENCODED"] = encoder.transform(df_test[[col]])  # Transform test set

            df_train_encoded = df_train_encoded.drop(col, axis=1)  # Remove original categorical column
            df_val_encoded = df_val_encoded.drop(col, axis=1)
            df_test_encoded = df_test_encoded.drop(col, axis=1)

    logger.info(f"Categorical features encoded: {', '.join(list_categorical_cols)}")

    return df_train_encoded, df_val_encoded, df_test_encoded, dict_encoders


def save_metadata(
    df_train_final: pd.DataFrame,
    df_val_final: pd.DataFrame,
    df_test_final: pd.DataFrame,
    arr_groups_train: np.ndarray,
    arr_groups_val: np.ndarray,
    arr_groups_test: np.ndarray,
    arr_relevance_train: np.ndarray,
    arr_task_ids_train: np.ndarray,
    arr_task_ids_val: np.ndarray,
    arr_task_ids_test: np.ndarray,
    list_categorical_cols: List[str],
    dict_encoders: Dict[str, Any],
) -> None:
    """
    Save metadata about the preprocessing workflow

    Creates a JSON file with comprehensive information about the processed datasets,
    including sample counts, feature details, group characteristics, and
    relevance score distribution

    Records timestamp and configuration details for reproducibility
    """

    logger.info("Saving metadata")

    dict_metadata = {
        "preprocessing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Record when processing was done
        "num_train_samples": len(df_train_final),  # Total training samples
        "num_val_samples": len(df_val_final),  # Validation samples
        "num_test_samples": len(df_test_final),  # Test samples
        "num_train_groups": len(arr_groups_train),  # Number of training tasks
        "num_val_groups": len(arr_groups_val),  # Validation tasks
        "num_test_groups": len(arr_groups_test),  # Test tasks
        "feature_count": len(df_train_final.columns),  # Total feature count
        "categorical_features": list_categorical_cols,  # List of categorical features
        "relevance_levels": [int(x) for x in np.unique(arr_relevance_train)],  # Available relevance scores
        "task_ids_saved": True,  # Flag indicating task IDs were preserved
        "unique_tasks_count": {
            "train": len(arr_task_ids_train),
            "val": len(arr_task_ids_val),
            "test": len(arr_task_ids_test)
        },
        "group_composition": {
            "capable_translators_percentage": CAPABLE_TRANSLATORS_PERCENTAGE,
            "incapable_translators_percentage": INCAPABLE_TRANSLATORS_PERCENTAGE,
            "strong_matches_percentage": STRONG_MATCHES_PERCENTAGE,
            "basic_matches_percentage": BASIC_MATCHES_PERCENTAGE,
        },
        "group_sizes": {
            "common_language_pair": COMMON_LANGUAGE_PAIR_GROUP_SIZE,
            "medium_language_pair": MEDIUM_LANGUAGE_PAIR_GROUP_SIZE,
            "rare_language_pair": RARE_LANGUAGE_PAIR_GROUP_SIZE,
        },
    }

    save_json(dict_metadata, PATH_METADATA_JSON)  # Save as JSON for readability
    logger.info("Metadata saved successfully")


def process_ranking_data() -> None:
    """
    Execute the complete preprocessing pipeline for the LGBMRanker model

    Orchestrates end-to-end workflow:
    - Creates directory structure
    - Loads and transforms base data
    - Creates task-translator pairs with relevance scores
    - Engineers features for ranking
    - Encodes categorical features
    - Organizes data into LGBMRanker compatible format
    - Saves outputs and artifacts

    This function serves as the main pipeline coordinator
    """

    create_output_dirs()  # Create directories

    df_train, df_val, df_test = load_base_data()  # Load base preprocessed data

    df_train = create_derived_features(df_train)  # Generate derived features
    df_val = create_derived_features(df_val)
    df_test = create_derived_features(df_test)

    dict_artifacts = load_artifacts()  # Load preprocessing artifacts from base step

    logger.info("Processing training data...")
    df_train_pairs = create_task_translator_dataset(
        df_train,
        dict_artifacts
    )  # Generate translator candidates

    logger.info("Processing validation data...")
    df_val_pairs = create_task_translator_dataset(
        df_val,
        dict_artifacts
    )

    logger.info("Processing test data...")
    df_test_pairs = create_task_translator_dataset(
        df_test,
        dict_artifacts
    )

    logger.info("Engineering features for training data...")
    df_train_features = engineer_ranking_features(
        df_train_pairs,
        dict_artifacts
    )  # Create ranking features

    logger.info("Engineering features for validation data...")
    df_val_features = engineer_ranking_features(
        df_val_pairs,
        dict_artifacts
    )

    logger.info("Engineering features for test data...")
    df_test_features = engineer_ranking_features(
        df_test_pairs,
        dict_artifacts
    )

    arr_groups_train, arr_task_ids_train = generate_task_groups(df_train_features)  # Generate group indicators
    arr_groups_val, arr_task_ids_val = generate_task_groups(df_val_features)
    arr_groups_test, arr_task_ids_test = generate_task_groups(df_test_features)

    arr_relevance_train = df_train_features["RELEVANCE"].values  # Extract relevance scores
    arr_relevance_val = df_val_features["RELEVANCE"].values
    arr_relevance_test = df_test_features["RELEVANCE"].values

    list_categorical_cols = [col for col in CATEGORICAL_COLS if col in df_train_features.columns]  # Find actual categorical columns
    df_train_encoded, df_val_encoded, df_test_encoded, dict_encoders = encode_categorical_features(
        df_train_features,
        df_val_features,
        df_test_features,
        list_categorical_cols
    )

    df_train_final = filter_model_features(df_train_encoded)  # Keep only model-relevant features
    df_val_final = filter_model_features(df_val_encoded)
    df_test_final = filter_model_features(df_test_encoded)

    logger.info("Saving processed data...")

    save_parquet(df_train_final, PATH_X_TRAIN_PARQUET)  # Save feature matrices
    save_parquet(df_val_final, PATH_X_VAL_PARQUET)
    save_parquet(df_test_final, PATH_X_TEST_PARQUET)

    np.save(PATH_GROUPS_TRAIN_NPY, arr_groups_train)  # Save group indicators
    np.save(PATH_GROUPS_VAL_NPY, arr_groups_val)
    np.save(PATH_GROUPS_TEST_NPY, arr_groups_test)

    np.save(PATH_TASK_IDS_TRAIN_NPY, arr_task_ids_train)  # Save task IDs
    np.save(PATH_TASK_IDS_VAL_NPY, arr_task_ids_val)
    np.save(PATH_TASK_IDS_TEST_NPY, arr_task_ids_test)

    np.save(PATH_RELEVANCE_TRAIN_NPY, arr_relevance_train)  # Save target variables
    np.save(PATH_RELEVANCE_VAL_NPY, arr_relevance_val)
    np.save(PATH_RELEVANCE_TEST_NPY, arr_relevance_test)

    logger.info("Saving artifacts...")

    save_pickle(dict_encoders, PATH_ENCODERS_PKL)  # Save encoders for inference
    save_json(df_train_final.columns.tolist(), PATH_FEATURE_COLUMNS_JSON)  # Save feature names

    save_metadata(
        df_train_final,
        df_val_final,
        df_test_final,
        arr_groups_train,
        arr_groups_val,
        arr_groups_test,
        arr_relevance_train,
        arr_task_ids_train,
        arr_task_ids_val,
        arr_task_ids_test,
        list_categorical_cols,
        dict_encoders,
    )
    logger.info("Data preprocessing completed successfully")


def main():
    """
    Executes the preprocessing workflow
    """
    
    try:  # Main execution
        process_ranking_data()
    except Exception as e:  # Error handling
        logger.error(f"Ranking preprocessing failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()