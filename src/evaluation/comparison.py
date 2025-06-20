# src/evaluation/comparison.py

"""
Cross-model comparison tools
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ndcg_score

from typing import (
    Dict,
    List,
    Any,
    Optional,
)

import logging

# Path configuration
PATH_EVALUATION = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_EVALUATION)
PATH_ROOT = os.path.dirname(PATH_SRC)

# Path constants
PATH_RESULTS_DIR = os.path.join(PATH_ROOT, "results", "ranking")
PATH_PLOTS_DIR = os.path.join(PATH_RESULTS_DIR, "plots")
PATH_MODEL_VS_RANDOM_PNG = os.path.join(PATH_PLOTS_DIR, "model_vs_random.png")

# Project modules
import src.utils.matplotlib

from src.utils.utils import setup_logger

# Initialize logger
logger = setup_logger(
    'comparison',
    'evaluation.log',
    logging.INFO
)


def compare_with_random_baseline(
    model: Any,
    X_test: pd.DataFrame,
    relevance_test: np.ndarray,
    groups_test: np.ndarray,
    k_values: List[int] = [1, 3, 5, 10]
):
    """
    Compare trained model performance against random baseline

    IMPORTANT: Random baseline provides critical context for evaluating model effectiveness
    The baseline represents what performance would be achieved by randomly ranking translators
    without any intelligence from features or learned patterns

    Args:
        model: Trained LGBMRanker model to evaluate
        X_test: Test features dataframe
        relevance_test: Ground truth relevance scores for test data
        groups_test: Array of group sizes indicating number of translators per task
        k_values: List of k values for NDCG@k evaluation (default [1, 3, 5, 10])

    Returns:
        None but prints comparison metrics and saves visualization

    Notes:
        Random baseline uses np random with fixed seed for reproducibility
        Compares NDCG@k metrics between model predictions and random scores
        Calculates percentage improvement of model over random baseline
        Generates bar chart showing performance comparison
        Baseline helps validate that model learned meaningful patterns
    """

    if model is None:
        print("Model not available Please train the model first")
        return

    print("Comparing model performance with random baseline...")

    # Make predictions with the trained model
    model_predictions = model.predict(X_test)

    # Generate random predictions
    np.random.seed(42)  # For reproducibility
    random_predictions = np.random.rand(len(model_predictions))

    # Calculate metrics for both prediction sets
    metrics_model = {}
    metrics_random = {}

    # Calculate NDCG@k for different k values
    for k in k_values:
        # Model metrics
        ndcg_values_model = []
        # Random baseline metrics
        ndcg_values_random = []

        start_idx = 0
        for group_size in groups_test:
            end_idx = start_idx + group_size

            group_y = relevance_test[start_idx:end_idx] # Get true relevance
            group_pred_model = model_predictions[start_idx:end_idx] # Get model predictions for this group
            group_pred_random = random_predictions[start_idx:end_idx] # Get random predictions for this group

            # Reshape for ndcg_score
            group_y = group_y.reshape(1, -1)
            group_pred_model = group_pred_model.reshape(1, -1)
            group_pred_random = group_pred_random.reshape(1, -1)

            try: # Calculate NDCG for model predictions
                effective_k = min(k, len(group_y[0]))
                ndcg_model = ndcg_score(
                    group_y,
                    group_pred_model,
                    k=effective_k
                )
                ndcg_values_model.append(ndcg_model)
            except Exception as e:
                pass # Skip groups with issues

            try: # Calculate NDCG for random predictions
                effective_k = min(k, len(group_y[0]))
                ndcg_random = ndcg_score(
                    group_y,
                    group_pred_random,
                    k=effective_k
                )
                ndcg_values_random.append(ndcg_random)
            except Exception as e:
                pass # Skip groups with issues

            start_idx = end_idx

        if ndcg_values_model: # Store average metrics
            metrics_model[f'ndcg@{k}'] = np.mean(ndcg_values_model)
        else:
            metrics_model[f'ndcg@{k}'] = 0.0

        if ndcg_values_random:
            metrics_random[f'ndcg@{k}'] = np.mean(ndcg_values_random)
        else:
            metrics_random[f'ndcg@{k}'] = 0.0

    print()
    print("Performance comparison (NDCG@k):")
    print(f"{'k':<5} {'Model':<10} {'Random':<10} {'Improvement':<15}")
    print()
    for k in k_values:
        model_ndcg = metrics_model[f'ndcg@{k}']
        random_ndcg = metrics_random[f'ndcg@{k}']

        if random_ndcg > 0:
            improvement = (model_ndcg / random_ndcg - 1) * 100  # Percentage improvement
        else:
            improvement = float('inf')

        print(f"{k:<5} {model_ndcg:<10.4f} {random_ndcg:<10.4f} {improvement:>9.2f}%")

    plt.figure()

    x = range(len(k_values))
    width = 0.35

    # Create bars
    model_bars = plt.bar(
        [i - width/2 for i in x],
        [metrics_model[f'ndcg@{k}'] for k in k_values],
        width,
        label='Model'
    )
    random_bars = plt.bar(
        [i + width/2 for i in x],
        [metrics_random[f'ndcg@{k}'] for k in k_values],
        width,
        label='Random'
    )

    # Add labels and title
    plt.xlabel('k')
    plt.ylabel('NDCG@k')
    plt.title('Model vs Random Baseline (NDCG@k)')
    plt.xticks(x, k_values)
    plt.legend()

    # Add value labels
    for bar in model_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom'
        )

    for bar in random_bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f"{height:.3f}",
            ha='center',
            va='bottom'
        )

    plt.savefig(PATH_MODEL_VS_RANDOM_PNG)
    plt.show()
    plt.close()