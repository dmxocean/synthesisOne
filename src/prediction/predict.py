# src/prediction/predict.py

"""
Prediction Pipeline for Translator Assignment System

Orchestrates multiple models to generate translator recommendations for translation tasks
"""

import os
import sys
import pandas as pd
import numpy as np
import logging

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Any,
)

from datetime import datetime

# Path configuration
PATH_PREDICTION = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_PREDICTION)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)

# Project modules
from src.utils.utils import setup_logger, log_dataframe_info

from src.prediction.ranking import run_inference as run_ranking_inference

# from src.prediction.sat import run_inference as run_sat_inference

# Initialize logger
logger = setup_logger(
    "prediction_runner",
    "prediction.log",
    logging.INFO
)

# Constants for model configuration
DEFAULT_TOP_K = 10
ENSEMBLE_WEIGHTS = {
    "ranking": 0.6,
    "sat": 0.4
}  # LGBMRanker weight in ensemble  # SAT model weight

# Model availability flags
MODELS_AVAILABLE = {
    "ranking": True,
    "sat": False
}  # LGBMRanker is implemented  # SAT is implemented

# Expected input columns from PM's CSV
EXPECTED_INPUT_COLUMNS = [
    "PROJECT_ID",
    "PM",
    "TASK_ID",
    "START",
    "END",
    "TASK_TYPE",
    "SOURCE_LANG",
    "TARGET_LANG",
    "FORECAST",
    "COST",
    "MANUFACTURER",
    "MANUFACTURER_SECTOR"
]

# Columns required by models
MODEL_REQUIRED_COLUMNS = [
    "TASK_ID",
    "SOURCE_LANG",
    "TARGET_LANG",
    "TASK_TYPE",
    "FORECAST",
    "START",
    "END",
    "MANUFACTURER",
    "MANUFACTURER_SECTOR"
]


def validate_task_data(df_tasks: pd.DataFrame) -> bool:
    """
    Validate input task data for required columns and data types

    Ensures all necessary fields are present and properly formatted
    before running predictions to prevent runtime errors

    IMPORTANT: Each row in df_tasks represents one translation task

    Args:
        df_tasks: DataFrame where each row is a task to predict

    Returns:
        Boolean indicating whether the data is valid
    """
    
    missing_columns = []
    for col in MODEL_REQUIRED_COLUMNS:
        if col not in df_tasks.columns:
            missing_columns.append(col)  # Track missing columns

    if missing_columns:
        logger.error(f"Missing required columns for model: {missing_columns}")
        return False

    # Check for empty dataframe
    if len(df_tasks) == 0:  # No tasks to process
        logger.error("Empty dataframe provided")
        return False

    try:  # Verify data types for critical columns
        if not pd.api.types.is_numeric_dtype(df_tasks["FORECAST"]):  # Check forecast is numeric
            logger.error("FORECAST column must be numeric")
            return False

        # Handle both datetime and string date formats
        for date_col in ["START", "END"]:
            if not pd.api.types.is_datetime64_any_dtype(df_tasks[date_col]):  # Check if already datetime
                try:
                    pd.to_datetime(df_tasks[date_col])  # Test if convertible to datetime
                except Exception:
                    logger.error(f"{date_col} column must be datetime or convertible to datetime")
                    return False

    except Exception as e:
        logger.error(f"Error validating data types: {str(e)}")
        return False

    logger.info(f"Validated {len(df_tasks)} tasks for prediction")
    return True


def prepare_task_data(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare task data for prediction by ensuring proper data types

    Converts date columns to datetime and performs any necessary
    preprocessing before model inference

    Args:
        df_tasks: Raw DataFrame from CSV upload

    Returns:
        DataFrame with properly formatted columns
    """

    df_prepared = df_tasks.copy()

    # Convert date columns to datetime if needed
    date_columns = ["START", "END"]
    for col in date_columns:
        if col in df_prepared.columns:
            df_prepared[col] = pd.to_datetime(df_prepared[col])  # Convert to datetime

    # Ensure IDs are integers (matching historical data format)
    if "TASK_ID" in df_prepared.columns:
        df_prepared["TASK_ID"] = df_prepared["TASK_ID"].astype(int)  # Keep as integer

    if "PROJECT_ID" in df_prepared.columns:
        df_prepared["PROJECT_ID"] = df_prepared["PROJECT_ID"].astype(int)  # Keep as integer

    # Ensure FORECAST is numeric
    if "FORECAST" in df_prepared.columns:
        df_prepared["FORECAST"] = pd.to_numeric(
            df_prepared["FORECAST"],
            errors="coerce"
        )  # Convert to float

    # Ensure COST is numeric if present
    if "COST" in df_prepared.columns:
        df_prepared["COST"] = pd.to_numeric(
            df_prepared["COST"],
            errors="coerce"
        )  # Convert to float

    return df_prepared


def extract_model_columns(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the columns needed by the models

    Models don't need all input columns (like PROJECT_ID, PM)
    but we preserve the full dataframe for later reconstruction

    Args:
        df_tasks: Full DataFrame with all columns

    Returns:
        DataFrame with only model-required columns
    """

    model_cols = [col for col in MODEL_REQUIRED_COLUMNS if col in df_tasks.columns]
    return df_tasks[model_cols].copy()


def run_batch_predictions(
    df_tasks: pd.DataFrame,
    model_name: str,
    top_k: int = DEFAULT_TOP_K
) -> Optional[Dict[str, List[Tuple[str, float]]]]:
    """
    Run predictions for a batch of tasks using specified model

    Processes multiple tasks simultaneously for efficiency,
    where each row in df_tasks represents a distinct task

    Args:
        df_tasks: DataFrame where each row is a task requiring predictions
        model_name: Name of the model to use ('ranking' or 'sat')
        top_k: Number of top translators to return per task

    Returns:
        Dictionary mapping task IDs to lists of (translator, score) tuples
        Returns None if model is unavailable or errors occur
    """

    if not MODELS_AVAILABLE.get(model_name, False):  # Check model availability
        logger.warning(f"Model '{model_name}' is not available")
        return None

    try:
        logger.info(f"Running {model_name} predictions for {len(df_tasks)} tasks")

        # Extract only columns needed by models
        df_model_input = extract_model_columns(df_tasks)  # Filter to required columns

        if model_name == "ranking":
            return run_ranking_inference(
                df_model_input,
                top_k=top_k
            )  # LGBMRanker batch inference

        # elif model_name == 'sat':
        #     return run_sat_inference(df_model_input, top_k=top_k)  # SAT batch inference # TODO: Implement SAT model inference

        else:
            logger.error(f"Unknown model name: {model_name}")
            return None

    except Exception as e:
        logger.error(f"Error running {model_name} predictions: {str(e)}")
        return None


def ensemble_predictions(
    model_results: Dict[str, Dict[str, List[Tuple[str, float]]]],
    weights: Dict[str, float] = ENSEMBLE_WEIGHTS,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combine predictions from multiple models using weighted ensemble

    Aggregates scores from different models using specified weights
    to create a unified ranking of translators for each task

    IMPORTANT: Ensures score normalization across models for fair comparison

    Args:
        model_results: Dictionary mapping model names to their predictions
        weights: Dictionary of model weights for ensemble
        top_k: Number of top translators to return per task

    Returns:
        Dictionary mapping task IDs to ensembled translator rankings
    """

    logger.info("Ensembling predictions from multiple models")

    ensemble_results = {}

    # Get all unique task IDs across models
    all_task_ids = set()
    for model_name, results in model_results.items():
        all_task_ids.update(results.keys())  # Collect all task IDs

    for task_id in all_task_ids:
        translator_scores = {}  # Combined scores for this task
        total_weight = 0.0  # Track total weight for normalization

        for model_name, results in model_results.items():
            if task_id not in results:  # Skip if model didn't process this task
                continue

            model_weight = weights.get(model_name, 0.0)  # Get model weight
            total_weight += model_weight  # Accumulate total weight

            for translator, score in results[task_id]:
                if translator not in translator_scores:
                    translator_scores[translator] = 0.0  # Initialize score

                translator_scores[translator] += model_weight * score  # Add weighted score

        # Normalize by total weight
        if total_weight > 0:  # Avoid division by zero
            for translator in translator_scores:
                translator_scores[translator] /= total_weight  # Normalize scores

        # Sort by combined score and take top k
        sorted_translators = sorted(
            translator_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        ensemble_results[task_id] = sorted_translators  # Store ensemble results

    logger.info(f"Ensemble completed for {len(ensemble_results)} tasks")
    return ensemble_results


def run_prediction_pipeline(
    df_tasks: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    models_to_use: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Execute the complete prediction pipeline for translator assignment

    This is the main entry point called by external applications (like Streamlit)
    to get translator recommendations for a batch of tasks

    IMPORTANT: Input df_tasks should contain one row per task
    Expected columns: PROJECT_ID, PM, TASK_ID, START, END, TASK_TYPE,
                     SOURCE_LANG, TARGET_LANG, FORECAST, COST,
                     MANUFACTURER, MANUFACTURER_SECTOR

    Args:
        df_tasks: DataFrame where each row is a task requiring predictions
        top_k: Number of top translators to return per task
        models_to_use: List of model names to use (defaults to all available)

    Returns:
        Dictionary mapping task IDs to lists of recommended translators
        with their scores
    """

    logger.info("Starting prediction pipeline")
    logger.info(f"Processing batch of {len(df_tasks)} tasks")

    # Prepare and validate input data
    df_prepared = prepare_task_data(df_tasks)  # Ensure proper data types

    if not validate_task_data(df_prepared):  # Check data validity
        raise ValueError("Invalid task data provided. Please check the required columns and data types.")

    # Log task batch information
    log_dataframe_info(
        logger,
        df_prepared,
        "Input task batch"
    )
    logger.info(f"Task IDs in batch: {df_prepared['TASK_ID'].tolist()}")

    # Store original data for later reconstruction
    df_original = df_prepared.copy()  # Keep all original columns

    # Determine which models to use
    if models_to_use is None:
        models_to_use = [name for name, available in MODELS_AVAILABLE.items() if available]  # Use all available models

    logger.info(f"Using models: {', '.join(models_to_use)}")

    # Run batch predictions for each model
    model_results = {}

    for model_name in models_to_use:
        logger.info(f"Running {model_name} batch predictions")

        predictions = run_batch_predictions(
            df_prepared,
            model_name,
            top_k=top_k
        )  # Get model predictions for batch

        if predictions is not None:
            model_results[model_name] = predictions  # Store successful predictions
            logger.info(f"{model_name} predictions completed successfully")
        else:
            logger.warning(f"{model_name} predictions failed or unavailable")

    # Check if we have any results
    if not model_results:  # No models produced results
        logger.error("No models produced valid predictions")
        raise RuntimeError("All models failed to produce predictions. Please check the logs for details.")

    # Ensemble results if multiple models were used
    if len(model_results) > 1:  # Multiple models available
        logger.info("Ensembling predictions from multiple models")
        final_results = ensemble_predictions(
            model_results,
            top_k=top_k
        )  # Combine predictions
    else:
        # Single model result
        model_name = list(model_results.keys())[0]  # Get single model name
        final_results = model_results[model_name]  # Use single model results
        logger.info(f"Using predictions from single model: {model_name}")

    logger.info("Prediction pipeline completed successfully")
    return final_results


def format_prediction_output(predictions: Dict[str, List[Tuple[str, float]]], df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Format prediction results into a structured DataFrame for output

    Reconstructs the original input columns and adds translator recommendations
    Each row represents one translator recommendation for a task

    IMPORTANT: Preserves all original columns (PROJECT_ID, PM, etc.) and adds
    RANK, TRANSLATOR, and SCORE columns

    Args:
        predictions: Dictionary mapping task IDs to translator recommendations
        df_tasks: Original task DataFrame with all columns

    Returns:
        DataFrame with all original columns plus recommendation details
    """

    rows = []

    # Create a copy to avoid modifying original
    df_tasks = df_tasks.copy()

    for task_id_key, recommendations in predictions.items():
        # Handle both string and integer task ID keys from predictions
        try:
            task_id = int(task_id_key) if isinstance(task_id_key, str) else task_id_key  # Convert to int if string
        except ValueError:
            task_id = task_id_key  # Keep as is if conversion fails

        # Find corresponding task row
        task_mask = df_tasks["TASK_ID"] == task_id  # Direct comparison with integer
        if not task_mask.any():
            logger.warning(f"Task ID {task_id} not found in original data")
            continue

        task_info = df_tasks[task_mask].iloc[0]  # Get task details

        for rank, (translator, score) in enumerate(recommendations, 1):
            row = task_info.to_dict()  # Start with all original columns

            # Add recommendation details
            row.update(
                {
                    "RANK": rank,
                    "TRANSLATOR": translator,
                    "SCORE": round(score, 4)
                }
            )  # Round score for readability

            rows.append(row)

    result_df = pd.DataFrame(rows)

    # Define column order
    column_order = EXPECTED_INPUT_COLUMNS + ["RANK", "TRANSLATOR", "SCORE"]
    available_columns = [col for col in column_order if col in result_df.columns]
    result_df = result_df[available_columns]

    return result_df


if __name__ == "__main__":
    """
    Test section to run the prediction pipeline with sample data
    """

    print("Testing prediction module with sample data...")

    # Create sample data
    sample_tasks = pd.DataFrame(
        {
            "PROJECT_ID": [220830, 220830, 220831],
            "PM": ["KMT", "KMT", "JDoe"],
            "TASK_ID": [11230024, 11230025, 11230026],
            "START": ["2023-05-05 09:00:00", "2023-05-05 14:00:00", "2023-05-06 10:00:00"],
            "END": ["2023-05-06 17:00:00", "2023-05-05 18:00:00", "2023-05-07 18:00:00"],
            "TASK_TYPE": ["Translation", "Proofreading", "Translation"],
            "SOURCE_LANG": ["English", "Spanish (Iberian)", "French"],
            "TARGET_LANG": ["Spanish (Iberian)", "French", "English"],
            "FORECAST": [2.5, 1.0, 3.0],
            "COST": [125.00, 50.00, 150.00],
            "MANUFACTURER": ["InnovateWorks", "TechGlobal", "BioMedCorp"],
            "MANUFACTURER_SECTOR": ["Consumer Discretionary", "Information Technology", "Health Care"],
        }
    )

    try:
        # Test the prediction pipeline
        results = run_prediction_pipeline(
            sample_tasks,
            top_k=3
        )

        formatted_results = format_prediction_output(
            results,
            sample_tasks
        )
        print("\nTest Results (showing all original columns plus recommendations):")
        print(formatted_results.to_string())

        print(f"\nResult columns: {formatted_results.columns.tolist()}")
        print(f"\nData types:\n{formatted_results.dtypes}")

    except Exception as e:
        print(f"Test failed: {str(e)}")