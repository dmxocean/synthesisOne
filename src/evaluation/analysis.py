# src/evaluation/analysis.py

"""
Model analysis tools
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
PATH_MODEL_DIR = os.path.join(PATH_ROOT, "models", "ranking")
PATH_MODEL_PKL = os.path.join(PATH_MODEL_DIR, "lgbm_ranker_model.pkl")
PATH_FEATURE_IMPORTANCE_BY_CATEGORY_PNG = os.path.join(PATH_PLOTS_DIR, "feature_importance_by_category.png")
PATH_NDCG_BY_GROUP_SIZE_PNG = os.path.join(PATH_PLOTS_DIR, "ndcg_by_group_size.png")

# Project modules
import src.utils.matplotlib

from src.utils.utils import setup_logger

# Initialize logger
logger = setup_logger(
    'analysis',
    'evaluation.log',
    logging.INFO
)


def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance and group by semantic categories

    Args:
        model: Trained LGBMRanker model with feature importance capabilities
        feature_names: List of feature names corresponding to model inputs
    """

    if model is None:
        print("Model not available Please train the model first")
        return

    feature_importance = model.feature_importance(importance_type='gain') # Get feature importance from the model

    feature_imp_df = pd.DataFrame(
        {
            'feature': feature_names,
            'importance': feature_importance
        }
    ).sort_values(
        by='importance',
        ascending=False
    )

    total_importance = feature_imp_df['importance'].sum() # Calculate percentage
    feature_imp_df['percentage'] = (feature_imp_df['importance'] / total_importance * 100)

    print("Top features by importance:") # Top features by importance
    for i, (feature, importance, pct) in enumerate(
        zip(feature_imp_df['feature'][:10],
            feature_imp_df['importance'][:10],
            feature_imp_df['percentage'][:10]), 1):
        print(f"{i}. {feature}: {importance:.2f} ({pct:.2f}%)")

    feature_categories = { # Group features by category
        'Capability': [f for f in feature_imp_df['feature'] if 'CAPABILITY' in f],
        'Experience': [f for f in feature_imp_df['feature'] if 'EXPERIENCE' in f],
        'Language': [f for f in feature_imp_df['feature'] if 'LANGUAGE' in f or 'LANG' in f],
        'Quality': [f for f in feature_imp_df['feature'] if 'QUALITY' in f],
        'Task': [f for f in feature_imp_df['feature'] if 'TASK' in f],
        'Cost': [f for f in feature_imp_df['feature'] if 'COST' in f or 'RATE' in f],
        'Other': []
    }

    all_categorized = [f for sublist in feature_categories.values() for f in sublist] # Flatten the list of categorized features
    feature_categories['Other'] = [f for f in feature_imp_df['feature'] if f not in all_categorized] # Add remaining features to category 'Other'

    # Calculate category importance
    category_importance = {}
    for category, features in feature_categories.items():
        if features:
            total_cat_importance = feature_imp_df[feature_imp_df['feature'].isin(features)]['importance'].sum()
            category_importance[category] = total_cat_importance

    # Create sorted categories for plotting
    categories_sorted = sorted(
        category_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    categories = [c[0] for c in categories_sorted]
    importances = [c[1] for c in categories_sorted]

    print("\nFeature importance by category:")
    for category, importance in categories_sorted:
        pct = importance / total_importance * 100
        print(f"{category}: {importance:.2f} ({pct:.2f}%)")

    plt.figure()
    bars = plt.bar(
        categories,
        importances,
        color='lightblue'
    )
    plt.xlabel('Feature Category')
    plt.ylabel('Total Importance (Gain)')
    plt.title('Feature Importance by Category')

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f"{height/total_importance*100:.1f}%",
            ha='center',
            va='bottom'
        )

    plt.savefig(PATH_FEATURE_IMPORTANCE_BY_CATEGORY_PNG)
    plt.show()
    plt.close()


def analyze_performance_by_group_size(
    model, X_test, relevance_test, groups_test
):
    """
    Analyze model performance across different group sizes to understand behavior for rare language pairs

    IMPORTANT: This function reveals insights about model performance for rare language pairs
    showing how ranking quality varies when fewer translators are available

    Args:
        model: Trained LGBMRanker model for making predictions
        X_test: Test features dataframe
        relevance_test: Ground truth relevance scores for test data
        groups_test: Array of group sizes indicating number of translators per task

    Notes:
        Calculates NDCG@5 for each category to measure ranking quality
    """

    if model is None:
        print("Model not available Please train the model first")
        return

    print("Analyzing model performance by group size...")

    predictions = model.predict(X_test) # Make predictions on the test set

    # Group sizes to analyze
    small_groups = []   # <= 5 translators
    medium_groups = []  # 6-15 translators
    large_groups = []   # > 15 translators

    small_ndcg, medium_ndcg, large_ndcg = [], [], [] # Metrics by group size

    # Calculate NDCG@5 for different group sizes
    start_idx = 0
    for group_size in groups_test:
        end_idx = start_idx + group_size

        # Get true relevance and predictions for this group
        group_y = relevance_test[start_idx:end_idx]
        group_pred = predictions[start_idx:end_idx]

        # Reshape for ndcg_score
        group_y = group_y.reshape(1, -1)
        group_pred = group_pred.reshape(1, -1)

        # Calculate NDCG@5
        try:
            ndcg = ndcg_score(
                group_y,
                group_pred,
                k=min(5, len(group_y[0]))
            )

            # Add to appropriate group size category
            if group_size <= 5:
                small_groups.append(group_size)
                small_ndcg.append(ndcg)
            elif group_size <= 15:
                medium_groups.append(group_size)
                medium_ndcg.append(ndcg)
            else:
                large_groups.append(group_size)
                large_ndcg.append(ndcg)
        except Exception as e:
            pass # Skip groups with issues

        start_idx = end_idx

    # Print summary statistics
    print(f"Small groups (<=5 translators): {len(small_groups)} groups, avg NDCG@5: {np.mean(small_ndcg) if small_ndcg else 0:.4f}")
    print(f"Medium groups (6-15 translators): {len(medium_groups)} groups, avg NDCG@5: {np.mean(medium_ndcg) if medium_ndcg else 0:.4f}")
    print(f"Large groups (>15 translators): {len(large_groups)} groups, avg NDCG@5: {np.mean(large_ndcg) if large_ndcg else 0:.4f}")

    plt.figure(figsize=(15, 6)) # Create box plots for each group size category
    plt.subplot(1, 2, 1)

    box_data = []
    labels = []

    if small_ndcg:
        box_data.append(small_ndcg)
        labels.append(f'Small (n={len(small_ndcg)})')

    if medium_ndcg:
        box_data.append(medium_ndcg)
        labels.append(f'Medium (n={len(medium_ndcg)})')

    if large_ndcg:
        box_data.append(large_ndcg)
        labels.append(f'Large (n={len(large_ndcg)})')

    if box_data:
        plt.boxplot(
            box_data,
            labels=labels
        )
        plt.title('NDCG@5 by Group Size')
        plt.ylabel('NDCG@5')

    plt.subplot(1, 2, 2) # Create scatter plot to see correlation between group size and NDCG
    all_groups = small_groups + medium_groups + large_groups
    all_ndcg = small_ndcg + medium_ndcg + large_ndcg

    if all_groups and all_ndcg:
        plt.scatter(
            all_groups,
            all_ndcg,
            alpha=0.5
        )
        plt.xlabel('Group Size (Number of Translators)')
        plt.ylabel('NDCG@5')
        plt.title('NDCG@5 vs Group Size')

        # Add a trend line
        if len(all_groups) > 1:
            z = np.polyfit(all_groups, all_ndcg, 1)
            p = np.poly1d(z)
            plt.plot(
                sorted(all_groups),
                p(sorted(all_groups)),
                "r--"
            )
            correlation = np.corrcoef(all_groups, all_ndcg)[0, 1]
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=plt.gca().transAxes
            )

    plt.savefig(PATH_NDCG_BY_GROUP_SIZE_PNG)
    plt.show()
    plt.close()