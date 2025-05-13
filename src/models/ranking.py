# src/models/ranking.py

"""
LGBMRanker model implementation and evaluation
"""

import os
import sys
import json
import time
import yaml
import pickle
import logging
import pandas as pd
import numpy as np

import lightgbm as lgb

from typing import (
    Dict,
    Any,
    List,
    Tuple,
    Optional,
)

from sklearn.metrics import ndcg_score
from functools import partial
from datetime import datetime
from itertools import product
from lightgbm.callback import early_stopping, log_evaluation, record_evaluation

# Path configuration
PATH_MODELS = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_MODELS)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)

# Path constants
PATH_DATA_PROCESSED = os.path.join(PATH_ROOT, "data", "processed")
PATH_RANKING_PROCESSED = os.path.join(PATH_DATA_PROCESSED, "ranking")
PATH_CONFIG = os.path.join(PATH_ROOT, "config", "ranking_config.yaml")

# Input paths
PATH_BASE_PROCESSED = os.path.join(PATH_DATA_PROCESSED, "base")
PATH_BASE_ARTIFACTS = os.path.join(PATH_BASE_PROCESSED, "artifacts")
PATH_TRAIN_PARQUET = os.path.join(PATH_BASE_PROCESSED, "train.parquet")
PATH_VAL_PARQUET = os.path.join(PATH_BASE_PROCESSED, "val.parquet")
PATH_TEST_PARQUET = os.path.join(PATH_BASE_PROCESSED, "test.parquet")

# Ranking input paths
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

# Output paths
PATH_MODEL_DIR = os.path.join(PATH_ROOT, "models", "ranking")
PATH_MODEL_PKL = os.path.join(PATH_MODEL_DIR, "lgbm_ranker_model.pkl")
PATH_MODEL_TXT = os.path.join(PATH_MODEL_DIR, "lgbm_ranker_model.txt")
PATH_MODEL_METADATA = os.path.join(PATH_MODEL_DIR, "metadata.json")

PATH_RESULTS_DIR = os.path.join(PATH_ROOT, "results", "ranking")
PATH_METRICS_JSON = os.path.join(PATH_RESULTS_DIR, "metrics.json")
PATH_PLOTS_DIR = os.path.join(PATH_RESULTS_DIR, "plots")
PATH_FEATURE_IMPORTANCE_PNG = os.path.join(PATH_PLOTS_DIR, "feature_importance.png")
PATH_NDCG_CURVE_PNG = os.path.join(PATH_PLOTS_DIR, "ndcg_curve.png")
PATH_LEARNING_CURVE_PNG = os.path.join(PATH_PLOTS_DIR, "learning_curve.png")

# Grid search specific paths
PATH_GRID_SEARCH_DIR = os.path.join(PATH_RESULTS_DIR, "grid_search")

# Project modules
from src.utils.utils import (
    setup_logger,
    ensure_dir,
    save_json,
    save_pickle,
    load_json,
    load_pickle,
    load_parquet,
    load_npy
)

from src.preprocessing.ranking import (
    create_derived_features,
    engineer_ranking_features,
    generate_task_groups,
    check_language_capability,
    calculate_task_complexity,
    calculate_urgency_compatibility,
    calculate_ontime_rate,
    calculate_language_pair_rarity,
    calculate_rate_cost_ratio,
    calculate_sector_experience,
    calculate_task_type_experience
)

from src.evaluation.visualization import visualize_grid_search_results, generate_visualizations

# Initialize logger
logger = setup_logger(
    'lgbm_ranker',
    'training.log',
    logging.INFO
)

# Define parameter grids for different sizes
PARAM_GRIDS = {
    "S": {  # Small grid (fast testing)
        "learning_rate": [0.08, 0.1, 0.12],
        "num_leaves": [55, 63, 71]
    },
    "M": {  # Medium grid (reasonable time)
        "learning_rate": [0.08, 0.1, 0.12],
        "num_leaves": [55, 63, 71],
        "max_depth": [80, 120],
        "feature_fraction": [0.6, 0.9]
    },
    "L": {  # Large grid (comprehensive)
        "learning_rate": [0.07, 0.09, 0.1, 0.11, 0.13],
        "num_leaves": [50, 57, 63, 69, 75],
        "max_depth": [80, 100, 120],
        "feature_fraction": [0.6, 0.75, 0.9],
        "bagging_fraction": [0.5, 0.7, 0.9],
        "lambda_l1": [0.0, 0.1, 0.5],
        "lambda_l2": [0.0, 0.1, 0.5]
    }
}

PERFORM_GRID_SEARCH = True  # Default grid search flag
GRID_SIZE = "M"  # Default grid size for testing


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file and merge with default values"""

    default_config = { # Default configuration
        "project": {
            "name": "Translation",
            "team": "synthesisOne",
            "version": "1.0"
        },
        "model": {
            "name": "lgbm_ranker",
            "version": "1.0"
        },
        "parameters": {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [1, 3, 5, 10],
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 45,
            "max_depth": 120,
            "min_data_in_leaf": 5,
            "min_sum_hessian_in_leaf": 1e-3,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.55,
            "bagging_freq": 5,
            "lambda_l1": 0.15,
            "lambda_l2": 0.35,
            "random_seed": 42,
            "force_row_wise": True  # Multithreaded distributes data rows across threads
        },
        "training": {
            "num_boost_round": 1000,
            "early_stopping_rounds": 10,
            "verbose_eval": 10
        },
        "prediction": {
            "top_k": 10 # Top K predictions
        },
        "evaluation": {
            "metrics": ["ndcg", "precision", "mrr"],
            "k_values": [1, 3, 5, 10]
        }
    }

    config = default_config.copy()

    try:
        with open(PATH_CONFIG, 'r') as f:
            loaded_config = yaml.safe_load(f) # Load parameters from YAML file
        logger.info("Configuration loaded successfully from YAML")

        # Deep merge loaded config with defaults
        if loaded_config:
            for section in default_config:
                if section in loaded_config:
                    if isinstance(default_config[section], dict) and isinstance(loaded_config[section], dict):
                        for key in default_config[section]: # Merge dictionary sections
                            if key not in loaded_config[section]:
                                loaded_config[section][key] = default_config[section][key]
                    # Use the loaded value
                    config[section] = loaded_config[section]

    except Exception as e:
        logger.warning(f"Error loading config from {PATH_CONFIG}: {str(e)} Using defaults") # Error during loading

    return config


def load_artifacts() -> Dict[str, Any]:
    """Load base preprocessing artifacts and ranking-specific artifacts"""

    logger.info("Loading base preprocessing artifacts")

    artifact_files = {
        "translator_mapping": os.path.join(PATH_BASE_ARTIFACTS, "translator_mapping.pkl"),
        "translator_metrics": os.path.join(PATH_BASE_ARTIFACTS, "translator_metrics.pkl"),
        "language_pair_metrics": os.path.join(PATH_BASE_ARTIFACTS, "language_pair_metrics.pkl"),
        "translator_capabilities": os.path.join(PATH_BASE_ARTIFACTS, "translator_capabilities.pkl"),
        "translator_hourly_rates": os.path.join(PATH_BASE_ARTIFACTS, "translator_hourly_rates.pkl"),
        "translator_efficiency_metrics": os.path.join(PATH_BASE_ARTIFACTS, "translator_efficiency_metrics.pkl"),
        "clients_data": os.path.join(PATH_BASE_ARTIFACTS, "clients_data.pkl")
    }

    dict_artifacts = {}
    for name, path in artifact_files.items():
        dict_artifacts[name] = load_pickle(path)
        logger.debug(f"Loaded artifact: {name}")

    encoders_path = os.path.join(PATH_RANKING_PROCESSED, "artifacts", "encoders.pkl")
    if os.path.exists(encoders_path):
        dict_artifacts["encoders"] = load_pickle(encoders_path)
        logger.debug("Loaded ranking encoders")

    logger.info("Artifacts loaded successfully")
    return dict_artifacts


def load_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed ranking data from parquet and numpy files"""

    logger.info("Loading preprocessed ranking data")

    # Load feature matrices
    X_train = load_parquet(PATH_X_TRAIN_PARQUET)
    X_val = load_parquet(PATH_X_VAL_PARQUET)
    X_test = load_parquet(PATH_X_TEST_PARQUET)

    # Load groups
    groups_train = np.load(PATH_GROUPS_TRAIN_NPY)
    groups_val = np.load(PATH_GROUPS_VAL_NPY)
    groups_test = np.load(PATH_GROUPS_TEST_NPY)

    # Load task IDs
    task_ids_train = np.load(PATH_TASK_IDS_TRAIN_NPY)
    task_ids_val = np.load(PATH_TASK_IDS_VAL_NPY)
    task_ids_test = np.load(PATH_TASK_IDS_TEST_NPY)

    # Load relevance scores
    relevance_train = np.load(PATH_RELEVANCE_TRAIN_NPY)
    relevance_val = np.load(PATH_RELEVANCE_VAL_NPY)
    relevance_test = np.load(PATH_RELEVANCE_TEST_NPY)

    logger.info(f"Loaded {len(X_train)} training samples, {len(X_val)} validation samples, {len(X_test)} test samples")
    logger.info(f"Loaded {len(groups_train)} training groups, {len(groups_val)} validation groups, {len(groups_test)} test groups")

    return (
        X_train, groups_train, relevance_train,
        X_val, groups_val, relevance_val,
        X_test, groups_test, relevance_test,
        task_ids_train, task_ids_val, task_ids_test
    )


def train_lgbm_ranker(
    X_train: pd.DataFrame,
    relevance_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame,
    relevance_val: np.ndarray,
    groups_val: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """
    Train LGBMRanker model using LambdaRank objective

    IMPORTANT: This function trains the core ranking model using the LambdaRank objective
    which directly optimizes ranking metrics like NDCG rather than pointwise objectives

    Params:
        X_train: Training features dataframe
        relevance_train: Relevance scores for training data
        groups_train: Group sizes for training data
        X_val: Validation features dataframe
        relevance_val: Relevance scores for validation data
        groups_val: Group sizes for validation data
        config: Configuration dictionary containing model parameters

    Returns:
        Tuple[lgb.Booster, Dict[str, Any]]: Trained model and training history

    Notes:
        Uses early stopping based on validation performance to prevent overfitting
        Compatible with LightGBM 4.x callback system
    """

    logger.info("Starting LGBMRanker model training")

    # Create dataset objects
    dtrain = lgb.Dataset(X_train, relevance_train, group=groups_train)
    dval = lgb.Dataset(X_val, relevance_val, group=groups_val, reference=dtrain)

    params = config['parameters'] # Get parameters from config

    # Handle metric configuration for LightGBM 4.6.0
    if 'eval_at' in params and isinstance(params['eval_at'], list):
        params['eval_at'] = params['eval_at']  # Make sure eval_at is a list

    # Training control parameters
    num_boost_round = config['training']['num_boost_round']
    early_stopping_rounds = config['training']['early_stopping_rounds']
    verbose_eval = config['training']['verbose_eval']

    evals_result = {} # Dictionary to store evaluation results
    training_history = {'train': [], 'validation': []} # Track training history

    try: # Training function
        callbacks = [ # LightGBM 4.x callback system
            log_evaluation(period=verbose_eval),
            early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            record_evaluation(evals_result)  # Add this callback to record evaluation results
        ]
        logger.info(f"Using LightGBM 4.x callbacks")

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'validation'],
            callbacks=callbacks
        )
    except Exception as e:
        logger.error(f"Error in LightGBM training: {str(e)}") # Training error
        raise

    logger.info(f"Model training completed successfully with best iteration {model.best_iteration}")

    # Extract training history from evals_result
    if evals_result and 'train' in evals_result and 'validation' in evals_result:
        # Get the first metric (usually there is only one)
        metric_name = next(iter(evals_result['train'].keys()))
        training_history['train'] = evals_result['train'][metric_name]
        training_history['validation'] = evals_result['validation'][metric_name]

    return model, training_history


def evaluate_lgbm_ranker(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    relevance_test: np.ndarray,
    groups_test: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evaluate the trained LGBMRanker model on test data

    IMPORTANT: This function evaluates ranking quality using NDCG@k Precision@k and MRR metrics

    Params:
        model: Trained LGBMRanker model
        X_test: Test features dataframe
        relevance_test: Relevance scores for test data
        groups_test: Group sizes for test data
        config: Configuration dictionary containing evaluation settings

    Returns:
        Dict[str, float]: Dictionary containing metric names and values

    Notes:
        Calculates metrics for each query group separately then returns average values
        Handles cases where k is larger than the group size by using effective k
    """

    logger.info("Evaluating LGBMRanker model on test data")

    predictions = model.predict(X_test) # Make predictions

    metrics = {} # Calculate metrics
    k_values = config['evaluation']['k_values']

    for k in k_values: # Calculate NDCG@k for different k values
        ndcg_values = []
        precision_values = []
        start_idx = 0

        for group_size in groups_test:
            end_idx = start_idx + group_size

            # Get true relevance and predictions for this group
            group_y = relevance_test[start_idx:end_idx]
            group_pred = predictions[start_idx:end_idx]

            group_y = group_y.reshape(1, -1) # Reshape for ndcg_score
            group_pred = group_pred.reshape(1, -1)

            try: # Calculate NDCG
                effective_k = min(k, len(group_y[0])) # Handle case where k is larger than group size

                ndcg = ndcg_score(
                    group_y,
                    group_pred,
                    k=effective_k
                ) # Calculate NDCG@k using sklearn
                ndcg_values.append(ndcg)

                # Calculate Precision@k
                sorted_indices = np.argsort(-group_pred[0])[:effective_k]
                relevant_items = sum(group_y[0][idx] > 0 for idx in sorted_indices)
                precision = relevant_items / effective_k
                precision_values.append(precision)
            except Exception as e:
                logger.warning(f"Error calculating metrics for group: {str(e)}") # Metrics calculation issue
                pass # Skip groups with issues

            start_idx = end_idx

        # Calculate mean NDCG@k and Precision@k
        if ndcg_values:
            metrics[f'ndcg@{k}'] = np.mean(ndcg_values)
        else:
            metrics[f'ndcg@{k}'] = 0.0

        if precision_values:
            metrics[f'precision@{k}'] = np.mean(precision_values)
        else:
            metrics[f'precision@{k}'] = 0.0

    # Calculate Mean Reciprocal Rank (MRR)
    mrr_values = []
    start_idx = 0

    for group_size in groups_test:
        end_idx = start_idx + group_size

        # Get true relevance and predictions for this group
        group_y = relevance_test[start_idx:end_idx]
        group_pred = predictions[start_idx:end_idx]

        sorted_indices = np.argsort(-group_pred) # Get indices that would sort predictions in descending order

        # Find position of relevant items (relevance > 0)
        for i, idx in enumerate(sorted_indices):
            if group_y[idx] > 0:
                mrr_values.append(1.0 / (i + 1)) # MRR is 1/(rank of first relevant item)
                break
        else:
            mrr_values.append(0.0) # No relevant items found

        start_idx = end_idx

    if mrr_values:
        metrics['mrr'] = np.mean(mrr_values)
    else:
        metrics['mrr'] = 0.0

    logger.info("Evaluation completed")
    logger.info("Evaluation results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def save_model_and_metrics(
    model: lgb.Booster,
    training_history: Dict[str, List],
    metrics: Dict[str, float],
    feature_names: List[str],
    config: Dict[str, Any]
) -> None:
    """
    Save trained model evaluation metrics and metadata

    Params:
        model: Trained LGBMRanker model
        training_history: Dictionary containing training and validation history
        metrics: Dictionary containing evaluation metrics
        feature_names: List of feature names used in the model
        config: Configuration dictionary

    Notes:
        Saves model in both pickle and text formats for compatibility
        Creates comprehensive metadata including timestamps configuration and performance metrics
    """

    logger.info("Saving model, metrics and visualizations")

    save_pickle(model, PATH_MODEL_PKL) # Save model
    model.save_model(PATH_MODEL_TXT)

    model_metadata = { # Create model metadata
        "created_at": pd.Timestamp.now().isoformat(),
        "config": config,
        "features": feature_names,
        "metrics": metrics,
        "best_iteration": model.best_iteration
    }


    save_json(model_metadata, PATH_MODEL_METADATA) # Save model metadata
    save_json(metrics, PATH_METRICS_JSON) # Save evaluation metrics

    logger.info(f"Model saved to {PATH_MODEL_PKL} and {PATH_MODEL_TXT}")
    logger.info(f"Metrics saved to {PATH_METRICS_JSON}")


def perform_grid_search(
    X_train: pd.DataFrame,
    relevance_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame,
    relevance_val: np.ndarray,
    groups_val: np.ndarray,
    param_grid: Dict[str, List],
    base_params: Dict[str, Any],
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
) -> Tuple[lgb.Booster, Dict[str, Any], pd.DataFrame]:
    """
    Perform grid search for optimal hyperparameters

    IMPORTANT: This function performs grid search optimizing for a weighted combination of
    NDCG@1 and NDCG@3 which are critical for ranking performance at the top positions

    Params:
        X_train: Training features dataframe
        relevance_train: Relevance scores for training data
        groups_train: Group sizes for training data
        X_val: Validation features dataframe
        relevance_val: Relevance scores for validation data
        groups_val: Group sizes for validation data
        param_grid: Dictionary of parameter names to lists of values to try
        base_params: Base parameters that remain constant
        num_boost_round: Maximum number of boosting rounds
        early_stopping_rounds: Early stopping patience

    Returns:
        Tuple containing best model best parameters and results dataframe

    Notes:
        Uses combined score of NDCG@1 and NDCG@3 with equal weights
        Records detailed metrics and training time for each parameter combination
    """

    logger.info("Starting grid search for LGBMRanker with combined NDCG@1 and NDCG@3 evaluation")

    # Create dataset objects
    dtrain = lgb.Dataset(X_train, relevance_train, group=groups_train)
    dval = lgb.Dataset(X_val, relevance_val, group=groups_val, reference=dtrain)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = [
        dict(zip(param_names, combo))
        for combo in product(*param_values)
    ]

    logger.info(f"Generated {len(param_combinations)} parameter combinations to evaluate")

    # Initialize results tracking
    search_results = []
    best_combined_score = 0
    best_params = None
    best_model = None
    start_time = time.time()

    # Iterate through parameter combinations
    for i, params in enumerate(param_combinations):
        iteration_start = time.time()


        current_params = {**base_params, **params} # Combine base params with current parameter set
        evals_result = {} # Create and train model

        callbacks = [ # Set up callbacks for LightGBM 4.x
            early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            record_evaluation(evals_result),
            log_evaluation(period=0)  # Zero means no logging
        ]

        model = lgb.train(
            current_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'validation'],
            callbacks=callbacks
        )

        best_iteration = model.best_iteration # Extract metrics

        # Weights for combined score
        ndcg1_weight = 0.5
        ndcg3_weight = 0.5

        try: # Extract individual NDCG metrics
            val_ndcg1 = evals_result['validation']['ndcg@1'][best_iteration]
            val_ndcg3 = evals_result['validation']['ndcg@3'][best_iteration]
            val_ndcg5 = evals_result['validation']['ndcg@5'][best_iteration]


            combined_score = (ndcg1_weight * val_ndcg1) + (ndcg3_weight * val_ndcg3) # Calculate combined weighted score

            # Record all NDCG metrics
            metrics = {
                'train_ndcg@1': evals_result['train']['ndcg@1'][best_iteration],
                'val_ndcg@1': val_ndcg1,
                'train_ndcg@3': evals_result['train']['ndcg@3'][best_iteration],
                'val_ndcg@3': val_ndcg3,
                'train_ndcg@5': evals_result['train']['ndcg@5'][best_iteration],
                'val_ndcg@5': val_ndcg5,
                'combined_score': combined_score
            }

            # Add other available NDCG metrics
            for k in [10]:
                metric_key = f'ndcg@{k}'
                if metric_key in evals_result['train']:
                    metrics[f'train_{metric_key}'] = evals_result['train'][metric_key][best_iteration]
                    metrics[f'val_{metric_key}'] = evals_result['validation'][metric_key][best_iteration]

        except (KeyError, IndexError) as e:
            logger.warning(f"Error extracting metrics: {str(e)}") # Metrics extraction failed
            continue

        # Calculate training time
        train_time = time.time() - iteration_start

        result = { # Store results
            'params': params.copy(),
            'best_iteration': best_iteration,
            'train_time': train_time,
            **metrics
        }
        search_results.append(result)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_params = params.copy()
            best_model = model
            logger.info(
                f"New best: combined score {best_combined_score:.4f} "
                f"(NDCG@1={val_ndcg1:.4f}, NDCG@3={val_ndcg3:.4f}) "
                f"with params: {best_params}"
            )

        logger.info(
            f"Completed {i+1}/{len(param_combinations)}: "
            f"NDCG@1={val_ndcg1:.4f}, NDCG@3={val_ndcg3:.4f}, "
            f"Combined={combined_score:.4f}, "
            f"time={train_time:.2f}s"
        )

    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Grid search completed in {total_time:.2f}s")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best combined score: {best_combined_score:.4f}")

    results_df = pd.DataFrame(search_results) # Convert results to DataFrame

    return best_model, best_params, results_df


def train_and_evaluate(
    config: Dict[str, Any] = None,
    grid_search: bool = None,
    grid_size: str = None,
    param_grid_override: Dict[str, List] = None
) -> Tuple[lgb.Booster, Dict[str, Any], Dict[str, float]]:
    """
    Main training and evaluation pipeline with optional grid search

    IMPORTANT: This is the primary entry point for model training coordinating all training steps including
    optional hyperparameter optimization data loading model training and comprehensive evaluation

    Params:
        config: Configuration dictionary If None will load from file
        grid_search: Whether to perform grid search If None uses module default
        grid_size: Size of grid search If None uses module default
        param_grid_override: Custom parameter grid If None uses predefined grids

    Returns:
        Tuple containing trained model training history and evaluation metrics

    Notes:
        Automatically saves model metrics and visualizations upon completion
        Supports flexible configuration through multiple parameter sources
    """

    logger.info("Starting training and evaluation pipeline")

    # Use module defaults if not provided
    if grid_search is None:
        grid_search = PERFORM_GRID_SEARCH
    if grid_size is None:
        grid_size = GRID_SIZE

    # Load configuration if not provided
    if config is None:
        config = load_config()

    # Load processed data
    (X_train, groups_train, relevance_train,
     X_val, groups_val, relevance_val,
     X_test, groups_test, relevance_test,
     task_ids_train, task_ids_val, task_ids_test) = load_data()

    if grid_search:
        logger.info(f"Grid search ENABLED with {grid_size} parameter grid")

        # Use override grid or default grid
        if param_grid_override is not None:
            active_param_grid = param_grid_override
        else:
            active_param_grid = PARAM_GRIDS[grid_size]

        total_combinations = np.prod([len(v) for v in active_param_grid.values()]) # Calculate total combinations
        logger.info(f"Using grid with {total_combinations} combinations")

        base_params = {k: v for k, v in config["parameters"].items() if k not in active_param_grid} # Get base parameters from config

        # Run grid search
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_model, best_params, grid_results = perform_grid_search(
            X_train, relevance_train, groups_train,
            X_val, relevance_val, groups_val,
            active_param_grid, base_params,
            num_boost_round=config['training']['num_boost_round'],
            early_stopping_rounds=config['training']['early_stopping_rounds']
        )

        # Save grid search results
        os.makedirs(PATH_GRID_SEARCH_DIR, exist_ok=True)
        grid_results.to_csv(os.path.join(PATH_GRID_SEARCH_DIR, f"grid_search_results_{timestamp}.csv"), index=False)
        save_json(best_params, os.path.join(PATH_GRID_SEARCH_DIR, f"best_params_{timestamp}.json"))

        # Save best model from grid search
        save_pickle(best_model, os.path.join(PATH_MODEL_DIR, f"lgbm_ranker_model_gs_{timestamp}.pkl"))
        best_model.save_model(os.path.join(PATH_MODEL_DIR, f"lgbm_ranker_model_gs_{timestamp}.txt"))

        visualize_grid_search_results(grid_results, active_param_grid)

        logger.info("Grid search completed successfully!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best validation combined score: {grid_results['combined_score'].max():.4f}")

        # Update config with best parameters
        config['parameters'].update(best_params)

        model = best_model # Use the best model from grid search
        training_history = {'train': [], 'validation': []}  # Grid search doesn't return history

    else:
        logger.info("Grid search DISABLED, using configured parameters")

        # Train model with current config
        model, training_history = train_lgbm_ranker(
            X_train, relevance_train, groups_train,
            X_val, relevance_val, groups_val,
            config
        )

    # Evaluate model
    metrics = evaluate_lgbm_ranker(
        model, X_test, relevance_test, groups_test, config
    )

    # Save model and metrics
    save_model_and_metrics(
        model, training_history, metrics, X_train.columns.tolist(), config
    )

    # Generate visualizations
    generate_visualizations(
        model, training_history, metrics, X_train.columns.tolist()
    )

    logger.info("Training and evaluation completed successfully")

    return model, training_history, metrics


def main():
    """
    Main function to train and evaluate LGBMRanker model
    """

    for dir_path in [PATH_MODEL_DIR, PATH_RESULTS_DIR, PATH_PLOTS_DIR, PATH_GRID_SEARCH_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    try:
        logger.info("Starting LGBMRanker training and evaluation")

        if PERFORM_GRID_SEARCH:
            logger.info(f"Grid search is ENABLED with {GRID_SIZE} parameter grid")
        else:
            logger.info("Grid search is DISABLED, using default/configured parameters")

        model, training_history, metrics = train_and_evaluate()

        logger.info("Training completed. Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info(f"Model saved to: {PATH_MODEL_PKL}")
        logger.info(f"Results saved to: {PATH_METRICS_JSON}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()