# src/evaluation/visualization.py

"""
Visualization utilities for model evaluation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
PATH_FEATURE_IMPORTANCE_PNG = os.path.join(PATH_PLOTS_DIR, "feature_importance.png")
PATH_LEARNING_CURVE_PNG = os.path.join(PATH_PLOTS_DIR, "learning_curve.png")
PATH_NDCG_CURVE_PNG = os.path.join(PATH_PLOTS_DIR, "ndcg_curve.png")
PATH_GRID_SEARCH_DIR = os.path.join(PATH_RESULTS_DIR, "grid_search")
PATH_GRID_SEARCH_VISUALIZATION_PNG = os.path.join(PATH_GRID_SEARCH_DIR, "grid_search_results.png")

# Project modules
import src.utils.matplotlib

from src.utils.utils import setup_logger

# Initialize logger
logger = setup_logger(
    'visualization',
    'evaluation.log',
    logging.INFO
)

TOP_FEATURES_IMPORTANCE = 15  # Number of top features

def generate_feature_importance_plot(
    model: Any,
    feature_names: List[str]
) -> None:
    """
    Generate bar chart showing model feature importance scores

    Args:
        model: Trained LGBMRanker model with feature importance method
        feature_names: List of feature names corresponding to model inputs
    """

    logger.info("Generating feature importance plot")

    # Extract feature importance values using gain metric
    feature_importance = model.feature_importance(importance_type='gain')

    # Create DataFrame for easier manipulation and sorting
    feature_imp_df = pd.DataFrame(
        {
            'feature': feature_names,
            'importance': feature_importance
        }
    ).sort_values(
        by='importance',
        ascending=False
    )

    plt.figure()

    top_features = feature_imp_df.head(TOP_FEATURES_IMPORTANCE)

    plt.barh( # Create horizontal bar chart
        top_features['feature'],
        top_features['importance']
    )

    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title(f'Top {TOP_FEATURES_IMPORTANCE} Feature Importance (Gain)')

    # Add percentage labels for each bar
    total_importance = feature_imp_df['importance'].sum()
    for i, (_, row) in enumerate(top_features.iterrows()):
        pct = row['importance'] / total_importance * 100
        plt.text(
            row['importance'] + 1,
            i,
            f"{pct:.1f}%",
            va='center'
        )

    plt.savefig(PATH_FEATURE_IMPORTANCE_PNG)
    plt.show()
    plt.close()

    logger.info(f"Feature importance plot saved to {PATH_FEATURE_IMPORTANCE_PNG}")


def generate_learning_curve_plot(
    model: Any,
    training_history: Dict[str, List]
) -> None:
    """
    Generate learning curve plot showing training progression

    Visualizes training and validation metric evolution across epochs

    Args:
        model: Trained model containing best_iteration attribute
        training_history: Dictionary with 'train' and 'validation' metric lists
    """

    logger.info("Generating learning curve plot")

    # Validate training history data exists
    if (not training_history or
        'train' not in training_history or
        'validation' not in training_history or
        not training_history['train'] or
        not training_history['validation']):
        logger.warning("Training history not available, skipping learning curve visualization")
        return

    plt.figure()

    plt.plot(
        training_history['train'], # Plot training and validation curves
        label='Training',
        linewidth=2
    )
    plt.plot(
        training_history['validation'],
        label='Validation',
        linewidth=2
    )

    # Mark best iteration with vertical line
    plt.axvline(
        x=model.best_iteration,
        color='red',
        linestyle='--',
        label=f'Best Iteration: {model.best_iteration}'
    )

    plt.xlabel('Iteration')
    plt.ylabel('NDCG')
    plt.title('Learning Curve')
    plt.legend()

    plt.savefig(PATH_LEARNING_CURVE_PNG)
    plt.show()
    plt.close()

    logger.info(f"Learning curve plot saved to {PATH_LEARNING_CURVE_PNG}")


def generate_ndcg_curve_plot(
    metrics: Dict[str, float]
) -> None:
    """
    Generate NDCG@k curve showing performance at different cutoff points

    Plots NDCG scores for multiple k values to show how well the model
    ranks relevant translators at different list depths

    Args:
        metrics: Dictionary containing NDCG scores at different k values
    """

    logger.info("Generating NDCG curve plot")

    # Define k values to plot
    k_values = [1, 3, 5, 10]
    ndcg_values = [metrics[f'ndcg@{k}'] for k in k_values]

    # Initialize plot
    plt.figure()

    # Plot NDCG curve with markers
    plt.plot(
        k_values,
        ndcg_values,
        'o-',
        linewidth=2,
        markersize=8
    )

    # Configure labels and title
    plt.xlabel('k')
    plt.ylabel('NDCG@k')
    plt.title('NDCG at Different k Values')
    plt.xticks(k_values)

    # Add value annotations for each point
    for k, ndcg in zip(k_values, ndcg_values):
        plt.annotate(
            f"{ndcg:.4f}",
            (k, ndcg),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )

    # Save plot
    plt.savefig(PATH_NDCG_CURVE_PNG)
    plt.show()
    plt.close()

    logger.info(f"NDCG curve plot saved to {PATH_NDCG_CURVE_PNG}")


def generate_visualizations(
    model: Any,
    training_history: Dict[str, List],
    metrics: Dict[str, float],
    feature_names: List[str]
) -> None:
    """
    Generate all evaluation visualizations in sequence

    Orchestrates creation of feature importance, learning curve,
    and NDCG curve plots for comprehensive model evaluation

    Args:
        model: Trained LGBMRanker model
        training_history: Training metric history
        metrics: Evaluation metrics dictionary
        feature_names: List of feature names
    """

    logger.info("Generating visualizations")

    generate_feature_importance_plot(model, feature_names)
    generate_learning_curve_plot(model, training_history)
    generate_ndcg_curve_plot(metrics)

    logger.info("Visualization generation completed")


def visualize_grid_search_results(
    results_df: pd.DataFrame,
    param_grid: Dict[str, List]
) -> None:
    """
    Create comprehensive visualization of hyperparameter tuning results

    Generates multi-panel figure showing parameter importance, top configurations,
    and performance-efficiency trade-offs from grid search optimization

    Args:
        results_df: DataFrame containing grid search results and metrics
        param_grid: Dictionary of parameter ranges tested
    """

    if results_df.empty:
        logger.warning("No grid search results to visualize")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # Determine scoring metric to use
    score_column = 'combined_score' if 'combined_score' in results_df.columns else 'val_ndcg@5'

    plt.subplot(2, 2, 1)

    # Extract parameter values and calculate correlations
    params = list(param_grid.keys())
    param_values = {}
    for param in params:
        param_values[param] = []
        for _, row in results_df.iterrows():
            param_values[param].append(row['params'][param])

    # Calculate correlation with performance metric
    corr_values = []
    for param in params:
        param_series = pd.Series(param_values[param])
        if param_series.dtype != 'object':  # Skip categorical parameters
            corr = param_series.corr(results_df[score_column])
            if not pd.isna(corr):
                corr_values.append((param, abs(corr)))

    corr_values.sort(
        key=lambda x: x[1],
        reverse=True
    ) # Sort by correlation strength and plot

    if corr_values:
        param_names = [x[0] for x in corr_values]
        corrs = [x[1] for x in corr_values]
        plt.barh(param_names, corrs)
        plt.xlabel(f'Absolute Correlation with {score_column.replace("_", " ").title()}')
        plt.ylabel('Parameter')
        plt.title('Parameter Importance')
    else:
        plt.text(
            0.5,
            0.5,
            'No correlation data available',
            ha='center',
            va='center',
            transform=plt.gca().transAxes
        )

    plt.subplot(2, 2, 2)

    top_configs = results_df.sort_values(
        score_column,
        ascending=False
    ).head(10) # Get top performing configurations
    param_strings = []
    for _, row in top_configs.iterrows():
        params_str = ', '.join([f"{p}={row['params'][p]}" for p in params])
        param_strings.append(params_str)

    plt.barh(
        range(len(top_configs)),
        top_configs[score_column]
    )
    plt.yticks(
        range(len(top_configs)),
        [f"Config {i+1}" for i in range(len(top_configs))]
    )
    plt.xlabel(score_column.replace("_", " ").title())
    plt.title('Top 10 Configurations')

    for i, (params, score) in enumerate(zip(param_strings, top_configs[score_column])):
        plt.text(
            score + 0.001,
            i,
            params,
            va='center',
            fontsize=8
        )

    plt.subplot(2, 1, 2)
    plt.scatter(
        results_df['train_time'],  # Scatter plot of efficiency trade-off
        results_df[score_column],
        alpha=0.6
    )
    plt.xlabel('Training Time (seconds)')
    plt.ylabel(score_column.replace("_", " ").title())
    plt.title('Training Time vs Performance')

    # Highlight best configuration
    best_idx = results_df[score_column].idxmax()
    best_time = results_df.loc[best_idx, 'train_time']
    best_score = results_df.loc[best_idx, score_column]
    plt.scatter(
        [best_time],
        [best_score],
        color='red',
        s=100,
        zorder=5,
        label='Best Configuration'
    )
    plt.legend()

    plt.savefig(PATH_GRID_SEARCH_VISUALIZATION_PNG)
    logger.info(f"Grid search visualization saved to {PATH_GRID_SEARCH_VISUALIZATION_PNG}")
    plt.show()
    plt.close()