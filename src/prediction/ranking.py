# src/prediction/ranking.py

"""
Machine Learning Ranking Model for Translator Assignment System

LGBMRanker-based translator recommendation and ranking inference
Handles feature engineering, candidate generation, and model prediction
Provides production-ready translator rankings for translation tasks

IMPORTANT: Core ML component for data-driven translator selection
"""

import os
import sys
import pandas as pd
import numpy as np

from typing import (
    Dict,
    Any,
    List,
    Tuple,
    Optional,
)

import logging

# Path configuration
PATH_PREDICTION = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_PREDICTION)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)

# Path constants
PATH_MODEL_DIR = os.path.join(PATH_ROOT, "models", "ranking")
PATH_MODEL_PKL = os.path.join(PATH_MODEL_DIR, "lgbm_ranker_model.pkl")
PATH_RANKING_PROCESSED = os.path.join(PATH_ROOT, "data", "processed", "ranking")
PATH_BASE_ARTIFACTS = os.path.join(PATH_ROOT, "data", "processed", "base", "artifacts")

# Project modules
from src.utils.utils import setup_logger, load_pickle, load_json
from src.preprocessing.ranking import create_derived_features, engineer_ranking_features, generate_task_groups, check_language_capability
from src.models.ranking import load_artifacts

# Initialize logger
logger = setup_logger(
    "ranking_predictor",
    "prediction.log",
    logging.INFO
)

# Required columns for inference
REQUIRED_INPUT_COLUMNS = [
    "TASK_ID",
    "SOURCE_LANG",
    "TARGET_LANG",
    "TASK_TYPE",
    "FORECAST",
    "START",
    "END",
    "MANUFACTURER",
    "MANUFACTURER_SECTOR",
    # Add any other absolutely required columns
]


def validate_input_columns(df_tasks: pd.DataFrame) -> None:
    """
    Validate that all required columns are present in input dataframe

    Args:
        df_tasks: Input tasks dataframe to validate

    Raises:
        ValueError: If any required columns are missing

    Notes:
        Required columns defined in REQUIRED_INPUT_COLUMNS constant
        Critical for ensuring feature engineering pipeline can execute successfully
    """

    missing_columns = [col for col in REQUIRED_INPUT_COLUMNS if col not in df_tasks.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {', '.join(missing_columns)} "
            f"The following columns are required: {', '.join(REQUIRED_INPUT_COLUMNS)}"
        )


def create_candidate_translators(
    df_tasks: pd.DataFrame,
    dict_artifacts: Dict[str, Any],
    max_candidates: int = 50
) -> Dict[str, List[str]]:
    """
    Create candidate translator lists for each task based on language capabilities

    Args:
        df_tasks: DataFrame containing task information with source and target languages
        dict_artifacts: Dictionary containing translator capabilities and metrics
        max_candidates: Maximum number of candidates to consider per task (default 50)

    Returns:
        Dict[str, List[str]]: Dictionary mapping task IDs to lists of eligible translator IDs

    Notes:
        Filters translators based on language pair capability
        Sorts candidates by quality metrics when available
        Handles rare language pairs by returning all available candidates
        Critical for managing inference complexity and performance
    """

    logger.info(f"Creating up to {max_candidates} candidate translators per task")

    # Extract capabilities
    dict_translator_capabilities = dict_artifacts[
        "translator_capabilities"
    ] if "translator_capabilities" in dict_artifacts else {}
    dict_translator_metrics = dict_artifacts["translator_metrics"] if "translator_metrics" in dict_artifacts else {}

    if not dict_translator_capabilities:
        logger.error("Translator capabilities not found in artifacts")
        return {}

    # Create dictionary to store candidates
    dict_candidates = {}

    # Process each task
    for _, task_row in df_tasks.iterrows():
        task_id = task_row["TASK_ID"]
        source_lang = task_row["SOURCE_LANG"]
        target_lang = task_row["TARGET_LANG"]

        # Find eligible translators with language capabilities
        list_eligible = [
            translator for translator in dict_translator_capabilities
            if check_language_capability(
                translator,
                source_lang,
                target_lang,
                dict_translator_capabilities
            )
        ]

        if dict_translator_metrics and list_eligible:
            list_eligible = sorted(  # Sort by quality if available
                list_eligible,
                key=lambda t: dict_translator_metrics[t]["avg_quality"] if t in dict_translator_metrics and "avg_quality" in dict_translator_metrics[t] else 0,
                reverse=True,
            )

        # Limit to max_candidates
        if len(list_eligible) > max_candidates:
            list_eligible = list_eligible[:max_candidates]

        dict_candidates[task_id] = list_eligible  # Store candidates for this task
        logger.info(f"Task {task_id}: Found {len(list_eligible)} eligible translators")

    return dict_candidates


def preprocess_for_inference(
    df_tasks: pd.DataFrame,
    dict_artifacts: Dict[str, Any],
    dict_candidate_translators: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[str]], List[str]]:
    """
    Preprocess tasks and create features for model inference

    Args:
        df_tasks: DataFrame containing task information
        dict_artifacts: Dictionary containing necessary artifacts for feature engineering
        dict_candidate_translators: Dictionary mapping task IDs to candidate translator lists

    Returns:
        Tuple containing:
            - pd.DataFrame: Feature matrix ready for model prediction
            - np.ndarray: Group sizes array for ranking structure
            - Dict[str, List[str]]: Task ID to translator list mapping
            - List[str]: Ordered list of task IDs

    Notes:
        Creates task-translator pairs for each candidate
        Engineers all necessary features matching training pipeline
        Handles categorical encoding using saved encoders
        Validates features match training set columns
    """

    logger.info("Preprocessing tasks for inference")

    # Create derived features
    df_tasks_prep = create_derived_features(df_tasks.copy())

    # Create task-translator pairs
    list_task_pairs = []
    dict_task_translators = {}
    list_task_ids = []

    for _, task_row in df_tasks_prep.iterrows():
        task_id = task_row["TASK_ID"]

        if task_id not in dict_candidate_translators:  # Skip if no candidates for this task
            logger.warning(f"No candidates found for task {task_id}")
            continue

        # Create a row for each candidate translator
        dict_task_translators[task_id] = []
        for translator in dict_candidate_translators[task_id]:
            pair_row = task_row.copy()
            pair_row["TRANSLATOR"] = translator
            pair_row["CAPABILITY_MATCH"] = 1  # All candidates should be capable

            list_task_pairs.append(pair_row)
            dict_task_translators[task_id].append(translator)

        list_task_ids.append(task_id)

    if not list_task_pairs:
        logger.error("No valid task-translator pairs for inference")
        return pd.DataFrame(), np.array([]), {}, []

    df_pairs = pd.DataFrame(list_task_pairs)

    # Engineer features
    df_features = engineer_ranking_features(df_pairs, dict_artifacts)

    # Generate groups
    arr_groups, _ = generate_task_groups(df_features)

    # Prepare feature columns for model
    encoders = dict_artifacts["encoders"] if "encoders" in dict_artifacts else {}

    # Encode categorical features if encoders are available
    if encoders:
        for col, encoder in encoders.items():
            if col in df_features.columns:
                df_features[f"{col}_ENCODED"] = encoder.transform(df_features[[col]])
                logger.debug(f"Encoded {col} using encoder")
                df_features = df_features.drop(col, axis=1)

    # Get feature columns used during training
    feature_columns = None
    feature_columns_path = os.path.join(PATH_RANKING_PROCESSED, "artifacts", "feature_columns.json")
    if os.path.exists(feature_columns_path):
        feature_columns = load_json(feature_columns_path)

    # Filter to match training columns if available
    if feature_columns:
        # Keep only columns that were used in training
        available_columns = [col for col in feature_columns if col in df_features.columns]
        missing_feature_columns = [col for col in feature_columns if col not in df_features.columns]

        if missing_feature_columns:
            raise ValueError(f"Failed to generate required model features: {', '.join(missing_feature_columns)} ")

        df_features = df_features[available_columns]
    
    # WORKAROUND: The model was trained with TASK_ID as a feature (which is incorrect)
    # We need to keep it but convert to numeric to avoid dtype error
    if 'TASK_ID' in df_features.columns:
        # Convert TASK_ID to numeric (just use row index as dummy value)
        df_features['TASK_ID'] = range(len(df_features))

    logger.info(f"Preprocessed {len(df_features)} task-translator pairs for inference")

    return df_features, arr_groups, dict_task_translators, list_task_ids


def predict_with_lgbm_ranker(
    model: Any,
    X: pd.DataFrame,
    groups: np.ndarray,
    dict_task_translators: Dict[str, List[str]],
    list_task_ids: List[str],
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Generate translator rankings using trained LGBMRanker model

    IMPORTANT: This function creates the final rankings that determine which translators
    are recommended for each task in production

    Args:
        model: Trained LGBMRanker model
        X: Feature matrix for prediction
        groups: Array indicating group sizes for each task
        dict_task_translators: Mapping of task IDs to translator lists
        list_task_ids: Ordered list of task IDs matching groups
        top_k: Number of top translators to return per task (default 10)

    Returns:
        Dict[str, List[Tuple[str, float]]]: Dictionary mapping task IDs to ranked lists
        of (translator_id, score) tuples

    Notes:
        Scores represent ranking confidence not probabilities
        Handles rare language pairs by returning fewer than top_k if necessary
        Maintains group structure to ensure proper task-translator mapping
        Critical for production recommendation quality
    """

    logger.info("Generating predictions with LGBMRanker model")

    predictions = model.predict(X)  # Make predictions

    # Create results dictionary
    dict_results = {}
    start_idx = 0

    for i, group_size in enumerate(groups):
        if i >= len(list_task_ids):
            logger.warning(f"More groups than task IDs")
            break

        task_id = list_task_ids[i]  # Get task ID

        end_idx = start_idx + group_size  # Get predictions for this group
        group_pred = predictions[start_idx:end_idx]

        translators = dict_task_translators[task_id] if task_id in dict_task_translators else []  # Get translators for this group

        if len(translators) != group_size:
            logger.warning(f"Task {task_id}: Translator count mismatch ({len(translators)} vs {group_size})")
            start_idx = end_idx
            continue

        # Create (translator, score) tuples
        translator_scores = list(zip(translators, group_pred))

        translator_scores.sort(
            key=lambda x: x[1],
            reverse=True
        )  # Sort by score descending
        dict_results[task_id] = translator_scores[:top_k]  # Take top-k

        start_idx = end_idx

    logger.info(f"Generated predictions for {len(dict_results)} tasks")

    return dict_results


def run_inference(
    df_tasks: pd.DataFrame,
    model_path: str = None,
    artifacts_path: str = None,
    top_k: int = 10
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Main inference pipeline for ranking translators on new tasks

    IMPORTANT: This is the primary entry point for production predictions orchestrating
    the entire inference workflow from raw tasks to ranked translator recommendations

    Args:
        df_tasks: DataFrame containing task information to predict on
        model_path: Path to trained model file (uses default if None)
        artifacts_path: Path to preprocessing artifacts (uses default if None)
        top_k: Number of top translators to return per task (default 10)

    Returns:
        Dict[str, List[Tuple[str, float]]]: Dictionary mapping task IDs to ranked lists
        of (translator_id, score) tuples

    Notes:
        Validates input columns before processing
        Creates candidate lists based on language capabilities
        Engineers features matching training pipeline
        Returns empty dict if any step fails
        Handles rare language pairs by returning all available candidates
    """

    logger.info("Running inference for new tasks")
    validate_input_columns(df_tasks)  # Validate required input columns

    # Set default paths if not provided
    if model_path is None:
        model_path = PATH_MODEL_PKL

    if artifacts_path is None:
        artifacts_path = PATH_BASE_ARTIFACTS

    model = load_pickle(model_path)  # Load trained model
    logger.info(f"Loaded model from {model_path}")

    dict_artifacts = load_artifacts()  # Load necessary artifacts
    logger.info("Loaded artifacts successfully")

    # Create candidate translators
    dict_candidates = create_candidate_translators(
        df_tasks,
        dict_artifacts,
        max_candidates=100
    )

    if not dict_candidates:
        logger.error("Failed to generate candidate translators")
        return {}

    # Preprocess for inference
    X, groups, dict_task_translators, list_task_ids = preprocess_for_inference(
        df_tasks,
        dict_artifacts,
        dict_candidates
    )

    if X.empty:
        logger.error("Failed to generate features for inference")
        return {}

    # Generate predictions
    dict_results = predict_with_lgbm_ranker(
        model,
        X,
        groups,
        dict_task_translators,
        list_task_ids,
        top_k
    )
    logger.info(f"Inference completed for {len(dict_results)} tasks")
    return dict_results