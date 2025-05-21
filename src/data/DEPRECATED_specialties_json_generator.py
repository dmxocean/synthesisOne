"""
Script to generate JSON files for efficient queries from translation data.

This script processes CSV data files and creates two JSON files:
1. translator_data.json - Main data storage with nested structure
2. lookup_indexes.json - Inverted indexes for quick lookups

Usage:
    python generate_json_files.py

The script reads from ../data/interim/data.csv and outputs JSON files to the same directory.
"""

import os
import json
import pandas as pd
import logging
from pathlib import Path
from data_constants import Columns, JsonKeys, Files

# Required columns list
REQUIRED_COLUMNS = [
    Columns.TRANSLATOR,
    Columns.SOURCE_LANG,
    Columns.TARGET_LANG,
    Columns.MANUFACTURER,
    Columns.MANUFACTURER_SECTOR,
    Columns.QUALITY_EVALUATION
]

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_DIR = DATA_DIR

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_json_files():
    """Generate JSON files for efficient querying of translator data."""
    
    # Load the CSV data
    data_path = DATA_DIR / Files.DATA_CSV
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded {len(df)} records from {Files.DATA_CSV}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Check if required columns exist
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return
    
    # Initialize data structures
    translator_data = {JsonKeys.TRANSLATORS: {}}
    lookup_indexes = {
        JsonKeys.MANUFACTURERS: {},
        JsonKeys.SECTORS: {}
    }
    
    # Process each record
    logger.info("Processing records...")
    
    for _, row in df.iterrows():
        translator = row[Columns.TRANSLATOR]
        source_lang = row[Columns.SOURCE_LANG]
        target_lang = row[Columns.TARGET_LANG]
        manufacturer = row[Columns.MANUFACTURER]
        sector = row[Columns.MANUFACTURER_SECTOR]
        quality = row[Columns.QUALITY_EVALUATION]
        
        # Skip records with missing data
        if pd.isna(translator) or pd.isna(source_lang) or pd.isna(target_lang) or pd.isna(manufacturer) or pd.isna(sector):
            continue
        
        # Create compound key for lookups
        compound_key = f"{translator}{JsonKeys.KEY_DELIMITER}{source_lang}{JsonKeys.KEY_DELIMITER}{target_lang}"
        
        # Create language pair key
        lang_pair_key = f"{source_lang}{JsonKeys.KEY_DELIMITER}{target_lang}"
        
        # Update translator_data
        translators_dict = translator_data[JsonKeys.TRANSLATORS]
        if translator not in translators_dict:
            translators_dict[translator] = {JsonKeys.LANGUAGE_PAIRS: {}}
        
        language_pairs_dict = translators_dict[translator][JsonKeys.LANGUAGE_PAIRS]
        if lang_pair_key not in language_pairs_dict:
            language_pairs_dict[lang_pair_key] = {
                JsonKeys.MANUFACTURERS: {},
                JsonKeys.SECTORS: {},
                JsonKeys.QUALITY: {JsonKeys.SUM: 0, JsonKeys.COUNT: 0, JsonKeys.AVERAGE: 0}
            }
        
        # Update manufacturer counts
        lang_data = language_pairs_dict[lang_pair_key]
        manufacturers_dict = lang_data[JsonKeys.MANUFACTURERS]
        if manufacturer not in manufacturers_dict:
            manufacturers_dict[manufacturer] = 0
        manufacturers_dict[manufacturer] += 1
        
        # Update sector counts
        sectors_dict = lang_data[JsonKeys.SECTORS]
        if sector not in sectors_dict:
            sectors_dict[sector] = 0
        sectors_dict[sector] += 1
        
        # Update quality metrics if available
        if not pd.isna(quality):
            quality_dict = lang_data[JsonKeys.QUALITY]
            quality_dict[JsonKeys.SUM] += quality
            quality_dict[JsonKeys.COUNT] += 1
            quality_dict[JsonKeys.AVERAGE] = quality_dict[JsonKeys.SUM] / quality_dict[JsonKeys.COUNT]
        
        # Update lookup indexes
        manufacturers_index = lookup_indexes[JsonKeys.MANUFACTURERS]
        if manufacturer not in manufacturers_index:
            manufacturers_index[manufacturer] = []
        if compound_key not in manufacturers_index[manufacturer]:
            manufacturers_index[manufacturer].append(compound_key)
        
        sectors_index = lookup_indexes[JsonKeys.SECTORS]
        if sector not in sectors_index:
            sectors_index[sector] = []
        if compound_key not in sectors_index[sector]:
            sectors_index[sector].append(compound_key)
    
    # Remove temporary sum field from quality data
    for translator in translator_data[JsonKeys.TRANSLATORS]:
        translator_data_dict = translator_data[JsonKeys.TRANSLATORS][translator][JsonKeys.LANGUAGE_PAIRS]
        for lang_pair in translator_data_dict:
            quality_dict = translator_data_dict[lang_pair][JsonKeys.QUALITY]
            if JsonKeys.SUM in quality_dict:
                del quality_dict[JsonKeys.SUM]
    
    # Write JSON files
    logger.info("Writing JSON files...")
    
    with open(OUTPUT_DIR / Files.DEPRECATED_TRANSLATOR_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(translator_data, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / Files.DEPRECATED_LOOKUP_INDEXES_JSON, "w", encoding="utf-8") as f:
        json.dump(lookup_indexes, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Successfully created JSON files in {OUTPUT_DIR}")
    
    # Print stats
    translator_count = len(translator_data[JsonKeys.TRANSLATORS])
    manufacturer_count = len(lookup_indexes[JsonKeys.MANUFACTURERS])
    sector_count = len(lookup_indexes[JsonKeys.SECTORS])
    
    logger.info(f"Processed {translator_count} translators, {manufacturer_count} manufacturers, {sector_count} sectors")

if __name__ == "__main__":
    generate_json_files()