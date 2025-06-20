# src/prediction/constraints.py

"""
Combined Translator Prediction System

Integrates rule-based SAT and machine learning ranking models
Manages prediction pipeline and translator availability checking
Ensures consistent output format across all prediction models

IMPORTANT: Central orchestrator for all translator assignment predictions
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

# Path configuration
PATH_CURRENT = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_CURRENT)
PATH_ROOT = os.path.dirname(PATH_SRC)
sys.path.append(PATH_ROOT)
sys.path.append(PATH_CURRENT)

print(f"Working directory: {PATH_CURRENT}")
print(f"Root directory: {PATH_ROOT}")

# Import SAT model
try:
    from models.SAT.SAT import run_sat_inference
    print("SAT model imported successfully")
    SAT_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"SAT model not available: {e}")
    SAT_MODEL_AVAILABLE = False

try:
    from src.prediction.ranking import run_inference as run_ranking_inference
    print("ML model imported successfully")
    RANKING_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"ML model not available: {e}")
    RANKING_MODEL_AVAILABLE = False

try:
    from src.prediction.calender import (
        assign_tasks_from_model_output, 
        load_translator_schedules, 
        check_translator_availability_only, 
        load_translator_calendar,
        load_translator_calendar_from_data,
        check_calendar_status
    )
    print("Scheduling system imported successfully")
    SCHEDULER_AVAILABLE = True
except ImportError as e:
    print(f"Scheduling system not available: {e}")
    SCHEDULER_AVAILABLE = False

# Setup logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants 
DEFAULT_TOP_K = 10
ENSEMBLE_WEIGHTS = {
    "ranking": 0.6,
    "sat": 0.4
}

# Model availability 
MODELS_AVAILABLE = {
    "ranking": RANKING_MODEL_AVAILABLE,
    "sat": SAT_MODEL_AVAILABLE
}

# Expected input columns 
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

class TranslatorAssignmentError(Exception):
    """Custom exception for translator assignment errors"""
    pass


def validate_task_data(df_tasks: pd.DataFrame) -> bool:
    """
    Validate input task data for required columns and data types
    
    Args:
        df_tasks: DataFrame containing task data to validate
        
    Returns:
        bool: True if validation passes, False otherwise
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

    if not pd.api.types.is_numeric_dtype(df_tasks["FORECAST"]):
        logger.error("FORECAST column must be numeric")
        return False

    # Handle both datetime and string date formats
    for date_col in ["START", "END"]:
        if not pd.api.types.is_datetime64_any_dtype(df_tasks[date_col]):
            pd.to_datetime(df_tasks[date_col])

    logger.info(f"Validated {len(df_tasks)} tasks for prediction")
    return True

def prepare_task_data(df_tasks: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare task data for prediction by ensuring proper data types
    
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
    
    Args:
        df_tasks: Full DataFrame with all columns
        
    Returns:
        DataFrame with only model-required columns
    """
    model_cols = [col for col in MODEL_REQUIRED_COLUMNS if col in df_tasks.columns]
    result = df_tasks[model_cols].copy()
    return result

def normalize_sat_results(sat_results: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Normalize SAT model results to use consistent task ID format
    
    SAT model returns composite keys like "PROJECT_ID TASK_ID", 
    but we want just "TASK_ID" to be consistent with ranking model
    """
    normalized_results = {}
    
    for key, translators in sat_results.items():
        # Convert key to string
        key_str = str(key)
        
        # Extract just the task ID part
        if " " in key_str:
            # Split "PROJECT_ID TASK_ID" format and take just the task ID
            parts = key_str.split(" ")
            task_id = parts[-1]  # Take the last part as task ID
            print(f"SAT: Normalized '{key_str}' → '{task_id}'")
        else:
            # Already in correct format
            task_id = key_str
        
        normalized_results[task_id] = translators
    
    return normalized_results

def run_batch_predictions(
    df_tasks: pd.DataFrame,
    model_name: str,
    top_k: int = DEFAULT_TOP_K
) -> Optional[Dict[str, List[Tuple[str, float]]]]:
    """
    Run predictions for a batch of tasks using specified model
    
    Args:
        df_tasks: DataFrame containing task data
        model_name: Name of model to use ('ranking' or 'sat')
        top_k: Number of top translators to return per task
        
    Returns:
        Dictionary mapping task IDs to translator recommendations
    """

    if not MODELS_AVAILABLE.get(model_name, False):
        logger.warning(f"Model '{model_name}' is not available")
        return None

    logger.info(f"Running {model_name} predictions for {len(df_tasks)} tasks")

    if model_name == "ranking":
        # Extract only columns needed by models
        df_model_input = extract_model_columns(df_tasks)
        result = run_ranking_inference(
            df_model_input,
            top_k=top_k
        )
        print(f"Ranking model returned {len(result)} task predictions")
        return result

    elif model_name == 'sat':
        # SAT model preparation (from constraints.py)
        df_sat = df_tasks.copy()
        
        # Convert all ID columns to strings to avoid concatenation errors
        string_columns = ['PROJECT_ID', 'TASK_ID', 'SOURCE_LANG', 'TARGET_LANG', 'MANUFACTURER', 'MANUFACTURER_SECTOR', 'TASK_TYPE']
        for col in string_columns:
            if col in df_sat.columns:
                df_sat[col] = df_sat[col].astype(str)
        
        raw_result = run_sat_inference(df_sat, top_k)
        print(f"SAT model returned {len(raw_result)} task predictions (before normalization)")
        
        # FIXED: Normalize SAT results to use consistent task ID format
        normalized_result = normalize_sat_results(raw_result)
        print(f"Normalized SAT results to {len(normalized_result)} task predictions")
        
        return normalized_result

    else:
        logger.error(f"Unknown model name: {model_name}")
        return None

def ensemble_predictions(
    model_results: Dict[str, Dict[str, List[Tuple[str, float]]]],
    weights: Dict[str, float] = ENSEMBLE_WEIGHTS,
    top_k: int = DEFAULT_TOP_K
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combine predictions from multiple models using weighted ensemble
    SIMPLIFIED: No longer needs to handle different task ID formats since they're normalized
    """
    logger.info("Ensembling predictions from multiple models")
    
    ensemble_results = {}
    
    # Get all unique task IDs across models
    all_task_ids = set()
    for model_name, results in model_results.items():
        all_task_ids.update(results.keys())
        print(f"{model_name}: {len(results)} tasks")
    
    print(f"Ensembling {len(all_task_ids)} unique tasks")
    
    for task_id in all_task_ids:
        # First normalize scores within each model
        normalized_model_scores = {}
        
        for model_name, results in model_results.items():
            if task_id not in results:
                continue
            
            # Extract scores for normalization
            scores = [score for _, score in results[task_id]]
            if not scores:
                continue
            
            # Normalize scores to range [0, 1] within this model
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score
            
            if score_range > 0:
                normalized_scores = [
                    (translator, (score - min_score) / score_range)
                    for translator, score in results[task_id]
                ]
            else:
                # If all scores are equal, just use 1.0
                normalized_scores = [
                    (translator, 1.0)
                    for translator, score in results[task_id]
                ]
            
            normalized_model_scores[model_name] = normalized_scores
        
        # Now combine normalized scores with weights
        translator_scores = {}
        total_weight = 0.0
        
        for model_name, normalized_scores in normalized_model_scores.items():
            model_weight = weights.get(model_name, 0.0)
            total_weight += model_weight
            
            for translator, score in normalized_scores:
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
    
    Args:
        df_tasks: DataFrame containing task data
        top_k: Number of top translators to return per task
        models_to_use: List of model names to use (defaults to all available)
        
    Returns:
        Dictionary mapping task IDs to translator recommendations
    """
    logger.info("Starting prediction pipeline")
    logger.info(f"Processing batch of {len(df_tasks)} tasks")

    # Prepare and validate input data
    df_prepared = prepare_task_data(df_tasks)

    if not validate_task_data(df_prepared):
        raise ValueError("Invalid task data provided. Please check the required columns and data types.")

    task_ids = df_prepared['TASK_ID'].tolist()
    logger.info(f"Task IDs in batch: {task_ids}")

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
def convert_to_simple_format(
    predictions: Dict[str, List[Union[Tuple[str, float], Dict[str, Any]]]]
) -> Dict[int, List[Dict[str, float]]]:
    """
    Convert model predictions (task_id → List[(translator,score)] OR
    List[{"translator":…, "score":…}]) into simple format
    (task_id(int) → List[{translator:…, score:…}]).
    """
    print("\n CONVERTING TO SIMPLE FORMAT")
    print()
    
    simple_results: Dict[int, List[Dict[str, float]]] = {}
    for task_id_str, pairs in predictions.items():
        # 1) convert task_id to int
        try:
            tid = int(task_id_str)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid task_id '{task_id_str}'; cannot convert to int.")
        
        # 2) normalize each entry
        converted: List[Dict[str, float]] = []
        for e in pairs:
            if isinstance(e, dict):
                # already simple form? just grab keys
                translator = e.get("translator")
                score      = e.get("score")
            else:
                # assume it's a (translator, score) tuple or list
                translator, score = e  
            converted.append({"translator": translator, "score": float(score)})
        
        simple_results[tid] = converted
    
    print(f" Converted {len(simple_results)} tasks to simple format")
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
    # Use the prediction pipeline
    predictions_with_scores = run_prediction_pipeline(
        df_tasks=df_tasks,
        top_k=top_k,
        models_to_use=model_names
    )
    
    # Convert to simple format
    simple_results = convert_to_simple_format(predictions_with_scores)
    
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

def create_task_rankings_for_scheduler(df_tasks, translator_recommendations):
    """
    Convert simple translator recommendations to the format needed by the scheduler
    
    Args:
        df_tasks: Original task DataFrame
        translator_recommendations: Dictionary from run_prediction_system {task_id: [translators]}
    
    Returns:
        Dictionary mapping task_id to DataFrame of ranked translators with task details
    """
    task_rankings = {}
    
    for task_id_str, translators in translator_recommendations.items():
        try:
            task_id = int(task_id_str)
                
            # Find this task in the DataFrame
            task_row = df_tasks[df_tasks['TASK_ID'] == task_id]
            if len(task_row) == 0:
                print(f"Warning: Task ID {task_id} not found in task dataframe")
                continue
                
            task_data = task_row.iloc[0]
            
            # Create a DataFrame with all translators for this task
            rows = []
            if isinstance(translators, list) and len(translators) > 0:
                # Check if it's new format with scores or old format with just names
                if isinstance(translators[0], dict) and 'translator' in translators[0]:
                    # New format: [{"translator": name, "score": value}, ...]
                    for entry in translators:
                        rows.append({
                            'translator': entry['translator'],
                            'score': entry.get('score', 0.0),
                            'start': task_data['START'],
                            'deadline': task_data['END'],
                            'forecast': task_data['FORECAST'],
                            'source_lang': task_data.get('SOURCE_LANG', ''),
                            'target_lang': task_data.get('TARGET_LANG', ''),
                            'industry': task_data.get('MANUFACTURER_SECTOR', '')
                        })
                else:
                    # Old format: [translator_name1, translator_name2, ...]
                    for translator in translators:
                        rows.append({
                            'translator': translator,
                            'score': 0.0,  # Default score for old format
                            'start': task_data['START'],
                            'deadline': task_data['END'],
                            'forecast': task_data['FORECAST'],
                            'source_lang': task_data.get('SOURCE_LANG', ''),
                            'target_lang': task_data.get('TARGET_LANG', ''),
                            'industry': task_data.get('MANUFACTURER_SECTOR', '')
                        })
                
                # Use the original task_id (not the composite key) for the scheduler
                task_rankings[str(task_id)] = pd.DataFrame(rows)
                
        except Exception as e:
            print(f"Error processing task {task_id_str}: {e}")
            continue
    
    return task_rankings

def get_available_translators(
    df_tasks: pd.DataFrame,
    translator_recommendations: Union[str, Dict],
    calendar_path: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Check which translators are available for each task without making assignments
    
    Args:
        df_tasks: DataFrame with task data
        translator_recommendations: Either a path to JSON file with assignments or a dictionary
                                   in format {task_id: [{"translator": name, "score": value}, ...]}
        calendar_path: Path to existing translator calendar JSON (optional, will use default if None)
    
    Returns:
        Dictionary mapping task_id to lists of available translators with scores
        Format: {'11240000': [{"translator": "Lincoln", "score": 0.95}, ...], ...}
    """
    print(f"\n CHECKING TRANSLATOR AVAILABILITY")
    print(f"{'='*50}")
    
    # Load recommendations from JSON file if given as path
    if isinstance(translator_recommendations, str):
        try:
            with open(translator_recommendations, 'r', encoding='utf-8') as f:
                translator_recommendations = json.load(f)
            print(f" Loaded recommendations from file")
        except Exception as e:
            logger.error(f"Failed to load translator recommendations: {e}")
            return {}

    # Set default calendar path if none provided
    if calendar_path is None:
        default_calendar_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "src", "prediction", "translator_schedule.json")
        calendar_path = str(default_calendar_path)
        print(f"Using default calendar path: {calendar_path}")

    # Load existing calendar with better error handling
    translator_calendar = {}
    calendar_path_str = str(calendar_path)
    if os.path.exists(calendar_path_str):
        try:
            translator_calendar = load_translator_calendar(calendar_path_str)
            
            # Check calendar status
            if SCHEDULER_AVAILABLE:
                status = check_calendar_status(translator_calendar)
                print(f"Calendar loaded: {status['loaded']}")
                print(f" Total translators: {status['total_translators']}")
                print(f" Total assignments: {status['total_assignments']}")
            else:
                print(f"Calendar loaded from: {calendar_path_str}")
                
        except Exception as e:
            print(f" Failed to load calendar from {calendar_path_str}: {e}")
            translator_calendar = {}
    else:
        print(f" Calendar file not found: {calendar_path_str}")
        print(f"   Creating empty calendar (all translators will be available)")

    task_rankings: Dict[str, pd.DataFrame] = {}
    
    # Build ranking DataFrames per task
    print(f"\n Building task rankings for {len(translator_recommendations)} tasks...")
    for task_id, translators_data in translator_recommendations.items():
        # Convert task_id to int for DataFrame lookup
        try:
            task_id_int = int(task_id)
        except ValueError:
            print(f" Invalid task ID format: {task_id}")
            continue
        
        # Find task in DataFrame
        task_row = df_tasks[df_tasks['TASK_ID'] == task_id_int]
        if task_row.empty:
            print(f" Task ID {task_id_int} not found in task dataframe")
            continue
        
        task_data = task_row.iloc[0]
        
        # Build translator rows
        rows = []
        for entry in translators_data:
            if isinstance(entry, dict) and 'translator' in entry:
                # New format: {"translator": name, "score": value}
                rows.append({
                    'translator': entry['translator'],
                    'score': entry.get('score', 0.0),
                    'start': task_data['START'],
                    'deadline': task_data['END'],
                    'forecast': task_data['FORECAST'],
                    'source_lang': task_data.get('SOURCE_LANG', ''),
                    'target_lang': task_data.get('TARGET_LANG', ''),
                    'industry': task_data.get('MANUFACTURER_SECTOR', '')
                })
            else:
                # Old format: just translator name
                rows.append({
                    'translator': entry,
                    'score': 0.0,
                    'start': task_data['START'],
                    'deadline': task_data['END'],
                    'forecast': task_data['FORECAST'],
                    'source_lang': task_data.get('SOURCE_LANG', ''),
                    'target_lang': task_data.get('TARGET_LANG', ''),
                    'industry': task_data.get('MANUFACTURER_SECTOR', '')
                })
        
        if rows:
            task_rankings[str(task_id_int)] = pd.DataFrame(rows)
            print(f" Task {task_id_int}: {len(rows)} translator candidates")

    # Load weekly schedules
    translator_schedules = None
    if SCHEDULER_AVAILABLE:
        try:
            translator_schedules = load_translator_schedules()
            print(f"Loaded translator schedules for {len(translator_schedules)} translators")
        except Exception as e:
            print(f" Could not load translator schedules: {e}")

    available_translators: Dict[str, List[Dict[str, Any]]] = {}
    
    # Check availability for each task
    print(f"\n Checking availability for {len(task_rankings)} tasks...")
    for task_id_str, df_ranks in task_rankings.items():
        available_for_task = []
        total_candidates = len(df_ranks)
        
        for _, row in df_ranks.iterrows():
            rec = row.to_dict()
            rec['task_id'] = task_id_str
            
            try:
                if SCHEDULER_AVAILABLE:
                    is_available = check_translator_availability_only(
                        rec, translator_calendar, translator_schedules
                    )
                else:
                    # If scheduler not available, assume all translators are available
                    is_available = True
                    
                if is_available:
                    available_for_task.append({
                        'translator': rec['translator'], 
                        'score': rec['score']
                    })
            except Exception as e:
                print(f" Error checking availability for {rec['translator']}: {e}")
                # On error, assume not available
                continue
        
        available_translators[task_id_str] = available_for_task
        print(f"   Task {task_id_str}: {len(available_for_task)}/{total_candidates} translators available")
    
    print(f"\n Availability check completed!")
    print(f" Summary: {len(available_translators)} tasks processed")
    print(available_translators)
    available_translators = convert_to_simple_format(available_translators)
    return available_translators

def main():
    """Generate available translators JSON for each model"""
    # Show system status
    if not (SAT_MODEL_AVAILABLE or RANKING_MODEL_AVAILABLE):
        print("No prediction models available. Please check imports.")
        return
    
    if not SCHEDULER_AVAILABLE:
        print("Scheduling system not available. Cannot check translator availability.")
        return
    
    # Create sample data
    sample_tasks = create_sample_tasks()
    
    # Define the models to test
    models_to_test = []
    
    if SAT_MODEL_AVAILABLE:
        models_to_test.append(("SAT Only", ["sat"]))
    
    if RANKING_MODEL_AVAILABLE:
        models_to_test.append(("ML Only", ["ranking"]))
    
    if SAT_MODEL_AVAILABLE and RANKING_MODEL_AVAILABLE:
        models_to_test.append(("Both Models (Ensemble)", ["sat", "ranking"]))
    
    # Test each model and generate availability JSON
    for scenario_name, model_names in models_to_test:
        try:
            # Get raw predictions with scores
            raw_predictions = run_prediction_pipeline(
                df_tasks=sample_tasks,
                top_k=10,  # Get more recommendations to have a better pool
                models_to_use=model_names
            )
            
            if not raw_predictions:
                print(f"{scenario_name}: No predictions generated")
                continue
            
            # Convert to new format with scores
            predictions_with_scores = convert_to_simple_format(raw_predictions)
            
            # Check translator availability (with scores)
            available_translators = get_available_translators(
                sample_tasks,
                predictions_with_scores
            )
            
            if not available_translators:
                print(f"{scenario_name}: No availability results generated")
                continue
            
            # Save availability results
            availability_file = f"available_translators_{scenario_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.json"
            with open(availability_file, 'w') as f:
                json.dump(available_translators, f, indent=4)
            print(f"Saved to: {availability_file}")
            
        except Exception as e:
            print(f"Error processing {scenario_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print a summary of generated files
    print("\nGenerated availability files:")
    import glob
    for file in sorted(glob.glob("available_translators_*.json")):
        print(f"  - {file}")

if __name__ == '__main__':
    main()