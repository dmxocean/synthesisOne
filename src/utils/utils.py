# src/utils/utils.py

"""
Utility functions for the Translator Assignment System
"""

import os
import logging
import json
import pickle
import yaml
import numpy as np
import pandas as pd

from datetime import datetime

from typing import (
    Dict,
    Any,
    Optional,
    Union,
    List,
    Tuple,
)

# Path configuration
PATH_UTILS = os.path.dirname(os.path.abspath(__file__))
PATH_ROOT = os.path.dirname(PATH_UTILS)

# Add this at the module level
_logger_handlers_initialized = set()

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure and return a logger with specified name and level"""

    logger = logging.getLogger(name)

    # Clear any existing handlers to prevent duplication
    if logger.handlers:
        logger.handlers = []

    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        logs_dir = os.path.join(PATH_ROOT, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, log_file)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def ensure_dir(dir_path: str) -> str:
    """Create directory if it doesn't exist and return the path"""
    
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def load_pickle(file_path: str) -> Any:
    """Load a pickle file"""
    
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading pickle file {file_path}: {e}\n")
        raise

def save_pickle(obj: Any, file_path: str) -> None:
    """Save an object to a pickle file"""
    
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        logging.error(f"Error saving pickle file {file_path}: {e}\n")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file"""
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}\n")
        raise

def save_json(obj: Any, file_path: str) -> None:
    """Save an object to a JSON file with pretty formatting"""
    
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(
                obj,
                f,
                indent=2
            )
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {e}\n")
        raise

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file"""
    
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading YAML file {file_path}: {e}\n")
        raise

def save_yaml(obj: Any, file_path: str) -> None:
    """Save an object to a YAML file"""
    
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            yaml.dump(
                obj,
                f,
                default_flow_style=False
            )
    except Exception as e:
        logging.error(f"Error saving YAML file {file_path}: {e}\n")
        raise

def load_npy(file_path: str) -> np.ndarray:
    """Load a NumPy .npy file"""
    
    try:
        return np.load(file_path)
    except Exception as e:
        logging.error(f"Error loading NumPy file {file_path}: {e}\n")
        raise

def save_npy(array: np.ndarray, file_path: str) -> None:
    """Save a NumPy array to a .npy file"""
    
    try:
        ensure_dir(os.path.dirname(file_path))
        np.save(file_path, array)
    except Exception as e:
        logging.error(f"Error saving NumPy file {file_path}: {e}\n")
        raise

def load_parquet(
    file_path: str,
    as_array: bool = False,
    column_name: str = None
) -> Union[pd.DataFrame, np.ndarray]:
    """Load a parquet file into a pandas DataFrame or numpy array"""
    
    try:
        df = pd.read_parquet(file_path)

        if column_name is not None:  # Return a specific column as array if requested
            return df[column_name].values

        if as_array and len(df.columns) == 1:
            return df.squeeze().values  # Return the single column as array

        return df
    except Exception as e:
        logging.error(f"Error loading parquet file {file_path}: {e}\n")
        raise

def save_parquet(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, List],
    file_path: str,
    column_name: str = 'TARGET'
) -> None:
    """Save data to a parquet file with automatic conversion for non-DataFrame inputs"""
    
    try:
        ensure_dir(os.path.dirname(file_path))

        if not isinstance(data, pd.DataFrame):  # Convert to DataFrame if not already one
            if isinstance(data, pd.Series):
                df = data.to_frame(name=column_name)
            else:  # numpy array or list
                df = pd.DataFrame({column_name: data})
        else:
            df = data

        df.to_parquet(file_path, index=False)
    except Exception as e:
        logging.error(f"Error saving parquet file {file_path}: {e}\n")
        raise

def get_timestamp() -> str:
    """Get current timestamp string for filenames"""
    
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def log_dataframe_info(
    logger: logging.Logger,
    df: pd.DataFrame,
    name: str
) -> None:
    """Log information about a DataFrame"""
    
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {df.columns.tolist()}")
    logger.info(f"{name} dtypes:\n{df.dtypes}\n")

    missing = df.isnull().sum()  # Log missing values if any
    if missing.sum() > 0:
        logger.warning(f"{name} missing values:\n{missing[missing > 0]}\n")

def get_project_root() -> str:
    """Get absolute path to project root directory"""
    
    return PATH_ROOT

def get_absolute_path(relative_path: str) -> str:
    """Convert a path relative to the project root to an absolute path"""
    
    return os.path.join(PATH_ROOT, relative_path)