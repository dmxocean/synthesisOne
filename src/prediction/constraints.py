#!/usr/bin/env python3
"""
Combined Translator Prediction System
Takes ML components from predict.py and SAT components from constraints.py
Calendar/scheduling system remains separate in calender.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

# Path configuration
PATH_CURRENT = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_CURRENT)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)
sys.path.append(PATH_CURRENT)

print(f"Working directory: {PATH_CURRENT}")
print(f"Root directory: {PATH_ROOT}")

# Import SAT model (from constraints.py)
try:
    from models.SAT.SAT_v5_5 import run_sat_inference
    print("✓ SAT model imported successfully")
    SAT_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"✗ SAT model not available: {e}")
    SAT_MODEL_AVAILABLE = False

# Import ML model (EXACTLY from predict.py)
try:
    from src.prediction.ranking import run_inference as run_ranking_inference
    print("✓ ML model imported successfully")
    RANKING_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"✗ ML model not available: {e}")
    RANKING_MODEL_AVAILABLE = False

# Setup logging (from predict.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (from predict.py)
DEFAULT_TOP_K = 10
ENSEMBLE_WEIGHTS = {
    "ranking": 0.6,
    "sat": 0.4
}

# Model availability (from predict.py)
MODELS_AVAILABLE = {
    "ranking": RANKING_MODEL_AVAILABLE,
    "sat": SAT_MODEL_AVAILABLE
}

# Expected input columns (from predict.py)
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

# Columns required by models (from predict.py)
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

class TranslatorAssignmentError(Exception):
    """Custom exception for translator assignment errors"""
    pass

# ML MODEL FUNCTIONS - TAKEN DIRECTLY FROM predict.py
def validate_task_data(df_tasks: pd.DataFrame) -> bool:
    """
    Validate input task data for required columns and data types
    TAKEN FROM predict.py
    """
    
    missing_columns = []
    for col in MODEL_REQUIRED_COLUMNS:
        if col not in df_tasks.columns:
            missing_columns.append(col)

    if missing_columns:
        logger.error(f"Missing required columns for model: {missing_columns}")
        return False

    # Check for empty dataframe
    if len(df_tasks) == 0:
        logger.error("Empty dataframe provided")
        return False

    try:
        if not pd.api.types.is_numeric_dtype(df_tasks["FORECAST"]):
            logger.error("FORECAST column must be numeric")
            return False

        # Handle both datetime and string date formats
        for date_col in ["START", "END"]:
            if not pd.api.types.is_datetime64_any_dtype(df_tasks[date_col]):
                try:
                    pd.to_datetime(df_tasks[date_col])
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
    TAKEN FROM predict.py
    """

    df_prepared = df_tasks.copy()

    # Convert date columns to datetime if needed
    date_columns = ["START", "END"]
    for col in date_columns:
        if col in df_prepared.columns:
            df_prepared[col] = pd.to_datetime(df_prepared[col])

    # Ensure IDs are integers (matching historical data format)
    if "TASK_ID" in df_prepared.columns:
        df_prepared["TASK_ID"] = df_prepared["TASK_ID"].astype(int)

    if "PROJECT_ID" in df_prepared.columns:
        df_prepared["PROJECT_ID"] = df_prepared["PROJECT_ID"].astype(int)

    # Ensure FORECAST is numeric
    if "FORECAST" in df_prepared.columns:
        df_prepared["FORECAST"] = pd.to_numeric(
            df_prepared["FORECAST"],
            errors="coerce"
        )

    # Ensure COST is numeric if present
    if "COST" in df_prepared.columns:
        df_prepared["COST"] = pd.to_numeric(
            df_prepared["COST"],
            errors="coerce"
        )

    return df_prepared

def extract_model_columns(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the columns needed by the models
    TAKEN FROM predict.py
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
    TAKEN FROM predict.py with SAT model addition from constraints.py
    """

    if not MODELS_AVAILABLE.get(model_name, False):
        logger.warning(f"Model '{model_name}' is not available")
        return None

    try:
        logger.info(f"Running {model_name} predictions for {len(df_tasks)} tasks")

        if model_name == "ranking":
            # Extract only columns needed by models (from predict.py)
            df_model_input = extract_model_columns(df_tasks)
            return run_ranking_inference(
                df_model_input,
                top_k=top_k
            )

        elif model_name == 'sat':
            # SAT model preparation (from constraints.py)
            df_sat = df_tasks.copy()
            
            # Convert all ID columns to strings to avoid concatenation errors
            string_columns = ['PROJECT_ID', 'TASK_ID', 'SOURCE_LANG', 'TARGET_LANG', 'MANUFACTURER', 'MANUFACTURER_SECTOR', 'TASK_TYPE']
            for col in string_columns:
                if col in df_sat.columns:
                    df_sat[col] = df_sat[col].astype(str)
            
            return run_sat_inference(df_sat, top_k)

        else:
            logger.error(f"Unknown model name: {model_name}")
            return None

    except Exception as e:
        logger.error(f"Error running {model_name} predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def ensemble_predictions(
    model_results: Dict[str, Dict[str, List[Tuple[str, float]]]],
    weights: Dict[str, float] = ENSEMBLE_WEIGHTS,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combine predictions from multiple models using weighted ensemble
    TAKEN DIRECTLY FROM predict.py
    """

    logger.info("Ensembling predictions from multiple models")

    ensemble_results = {}

    # Get all unique task IDs across models
    all_task_ids = set()
    for model_name, results in model_results.items():
        all_task_ids.update(results.keys())

    for task_id in all_task_ids:
        translator_scores = {}
        total_weight = 0.0

        for model_name, results in model_results.items():
            if task_id not in results:
                continue

            model_weight = weights.get(model_name, 0.0)
            total_weight += model_weight

            for translator, score in results[task_id]:
                if translator not in translator_scores:
                    translator_scores[translator] = 0.0

                translator_scores[translator] += model_weight * score

        # Normalize by total weight
        if total_weight > 0:
            for translator in translator_scores:
                translator_scores[translator] /= total_weight

        # Sort by combined score and take top k
        sorted_translators = sorted(
            translator_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        ensemble_results[task_id] = sorted_translators

    logger.info(f"Ensemble completed for {len(ensemble_results)} tasks")
    return ensemble_results

def run_prediction_pipeline(
    df_tasks: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    models_to_use: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Execute the complete prediction pipeline for translator assignment
    TAKEN FROM predict.py with modifications for simple output
    """

    logger.info("Starting prediction pipeline")
    logger.info(f"Processing batch of {len(df_tasks)} tasks")

    # Prepare and validate input data (from predict.py)
    df_prepared = prepare_task_data(df_tasks)

    if not validate_task_data(df_prepared):
        raise ValueError("Invalid task data provided. Please check the required columns and data types.")

    logger.info(f"Task IDs in batch: {df_prepared['TASK_ID'].tolist()}")

    # Store original data for later reconstruction
    df_original = df_prepared.copy()

    # Determine which models to use
    if models_to_use is None:
        models_to_use = [name for name, available in MODELS_AVAILABLE.items() if available]

    logger.info(f"Using models: {', '.join(models_to_use)}")

    # Run batch predictions for each model
    model_results = {}

    for model_name in models_to_use:
        logger.info(f"Running {model_name} batch predictions")

        predictions = run_batch_predictions(
            df_prepared,
            model_name,
            top_k=top_k
        )

        if predictions is not None:
            model_results[model_name] = predictions
            logger.info(f"{model_name} predictions completed successfully")
        else:
            logger.warning(f"{model_name} predictions failed or unavailable")

    # Check if we have any results
    if not model_results:
        logger.error("No models produced valid predictions")
        raise RuntimeError("All models failed to produce predictions. Please check the logs for details.")

    # Ensemble results if multiple models were used
    if len(model_results) > 1:
        logger.info("Ensembling predictions from multiple models")
        final_results = ensemble_predictions(
            model_results,
            top_k=top_k
        )
    else:
        # Single model result
        model_name = list(model_results.keys())[0]
        final_results = model_results[model_name]
        logger.info(f"Using predictions from single model: {model_name}")

    logger.info("Prediction pipeline completed successfully")
    return final_results

# CONVERSION TO SIMPLE FORMAT
def convert_to_simple_format(predictions: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
    """
    Convert model predictions to simple format: {task_id: [translator_names]}
    """
    simple_results = {}
    
    for task_id, translator_score_pairs in predictions.items():
        # Extract just the translator names, ignore scores
        translator_names = [translator for translator, score in translator_score_pairs]
        simple_results[str(task_id)] = translator_names
    
    return simple_results

def run_prediction_system(df_tasks: pd.DataFrame, 
                         model_names: Optional[List[str]] = None, 
                         top_k: int = DEFAULT_TOP_K) -> Dict[str, List[str]]:
    """
    Main prediction function that returns simple dictionary format
    
    Args:
        df_tasks: DataFrame with task data
        model_names: List of models to use ['sat', 'ranking'] or None for all available
        top_k: Number of top translators to return per task
    
    Returns:
        Dictionary mapping task IDs to lists of translator names
        Format: {'10988490': ['Leonor', 'David', 'Gerardo', 'Roque Marlene', 'Philipp'], ...}
    """
    
    print("="*50)
    print("TRANSLATOR PREDICTION SYSTEM")
    print("="*50)
    
    # Use the prediction pipeline from predict.py
    predictions_with_scores = run_prediction_pipeline(
        df_tasks=df_tasks,
        top_k=top_k,
        models_to_use=model_names
    )
    
    # Convert to simple format
    simple_results = convert_to_simple_format(predictions_with_scores)
    
    print(f"✓ Generated predictions for {len(simple_results)} tasks")
    print("="*50)
    
    return simple_results

def create_sample_tasks() -> pd.DataFrame:
    """Create sample tasks for testing"""
    sample_data = {
        'PROJECT_ID': [220900, 220901, 220902, 220903, 220904],
        'PM': ['KMT', 'PMT', 'RMT', 'KMT', 'PMT'],
        'TASK_ID': [11240000, 11240001, 11240002, 11240003, 11240004],
        'START': ['2025-05-25 09:00:00', '2025-05-26 09:00:00', '2025-05-27 09:00:00', 
                 '2025-05-28 09:00:00', '2025-05-29 09:00:00'],
        'END': ['2025-05-27 17:00:00', '2025-05-28 17:00:00', '2025-05-29 17:00:00', 
               '2025-05-30 17:00:00', '2025-05-31 17:00:00'],
        'TASK_TYPE': ['Engineering', 'ProofReading', 'Engineering', 'ProofReading', 'Engineering'],
        'SOURCE_LANG': ['English', 'English', 'English', 'English', 'English'],
        'TARGET_LANG': ['Spanish (Iberian)', 'Spanish (Iberian)', 'Spanish (Iberian)', 
                       'Spanish (Iberian)', 'Spanish (Iberian)'],
        'FORECAST': [3.0, 7.0, 5.0, 1.5, 2.5],
        'COST': [51.0, 119.0, 85.0, 19.5, 42.5],
        'MANUFACTURER': ['InnovateWorks', 'Pinnacle Heavy Industries', 'FrontierTech', 
                        'VidaCore Biotech', 'SunTech'],
        'MANUFACTURER_SECTOR': ['Consumer Discretionary', 'Industrials', 'Information Technology', 
                               'Health Care', 'Information Technology']
    }
    
    return pd.DataFrame(sample_data)

def main():
    """Test the combined prediction system"""
    print("🚀 TESTING COMBINED TRANSLATOR PREDICTION SYSTEM")
    print("="*60)
    
    # Show system status
    print(f"\nSystem Status:")
    print(f"✓ SAT Model: {SAT_MODEL_AVAILABLE}")
    print(f"✓ ML Model: {RANKING_MODEL_AVAILABLE}")
    
    if not (SAT_MODEL_AVAILABLE or RANKING_MODEL_AVAILABLE):
        print("\n❌ No models available. Please check imports.")
        return
    
    # Create sample data
    sample_tasks = create_sample_tasks()
    print(f"\nCreated {len(sample_tasks)} sample tasks for testing")
    
    try:
        # Test different model combinations
        test_scenarios = []
        
        if SAT_MODEL_AVAILABLE:
            test_scenarios.append(("SAT Only", ["sat"]))
        
        if RANKING_MODEL_AVAILABLE:
            test_scenarios.append(("ML Only", ["ranking"]))
        
        if SAT_MODEL_AVAILABLE and RANKING_MODEL_AVAILABLE:
            test_scenarios.append(("Both Models (Ensemble)", ["sat", "ranking"]))
        
        for scenario_name, models in test_scenarios:
            print(f"\n🧪 Testing: {scenario_name}")
            print("-" * 40)
            
            results = run_prediction_system(
                sample_tasks, 
                model_names=models, 
                top_k=5
            )
            
            if results:
                print(f"✅ {scenario_name}: Generated predictions for {len(results)} tasks")
                
                # Show sample results in the requested format
                print("Sample results (requested format):")
                sample_count = 0
                for task_id, translators in results.items():
                    if sample_count < 3:  # Show first 3
                        print(f"  '{task_id}': {translators}")
                        sample_count += 1
                
                # Save full results
                output_file = f"predictions_{scenario_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json"
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4)
                print(f"📄 Full results saved to: {output_file}")
                
            else:
                print(f"❌ {scenario_name}: No predictions generated")
    
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎯 TESTING COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()