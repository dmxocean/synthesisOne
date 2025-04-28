"""
Constants used throughout the translation data processing system.

This module defines constants for:
- CSV column names
- JSON structure keys
- File paths and names

Import this module in other files to avoid using "magic strings".
"""

# CSV Column Names
class Columns:
    TRANSLATOR = "TRANSLATOR"
    SOURCE_LANG = "SOURCE_LANG"
    TARGET_LANG = "TARGET_LANG"
    MANUFACTURER = "MANUFACTURER"
    MANUFACTURER_SECTOR = "MANUFACTURER_SECTOR"
    QUALITY_EVALUATION = "QUALITY_EVALUATION"
    HOURLY_RATE = "HOURLY_RATE"
    TASK_TYPE = "TASK_TYPE"
    PROJECT_ID = "PROJECT_ID"
    TASK_ID = "TASK_ID"
    START = "START"
    END = "END"
    PM = "PM"
    ASSIGNED = "ASSIGNED"
    READY = "READY"
    WORKING = "WORKING"
    DELIVERED = "DELIVERED"
    RECEIVED = "RECEIVED"
    CLOSE = "CLOSE"
    FORECAST = "FORECAST"
    COST = "COST"
    MANUFACTURER_INDUSTRY_GROUP = "MANUFACTURER_INDUSTRY_GROUP"
    MANUFACTURER_INDUSTRY = "MANUFACTURER_INDUSTRY"
    MANUFACTURER_SUBINDUSTRY = "MANUFACTURER_SUBINDUSTRY"

# JSON Structure Keys
class JsonKeys:
    # Top level keys
    TRANSLATORS = "translators"
    MANUFACTURERS = "manufacturers"
    SECTORS = "sectors"
    
    # Translator data keys
    LANGUAGE_PAIRS = "language_pairs"
    QUALITY = "quality"
    AVERAGE = "average"
    COUNT = "count"
    SUM = "sum"
    
    # Compound key delimiter
    KEY_DELIMITER = "|"
    
    # Additional keys for summaries
    NAME = "name"
    SOURCE = "source"
    TARGET = "target"
    JOB_COUNT = "job_count"
    AVERAGE_QUALITY = "average_quality"

# File Names
class Files:
    DATA_CSV = "data.csv"
    TRANSLATOR_DATA_JSON = "translator_specialties.json"
    LOOKUP_INDEXES_JSON = "specialties_lookup_indexes.json"

