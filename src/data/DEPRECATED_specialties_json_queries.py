"""
Module providing query functions for translation project data.

This module provides three main query functions:
1. get_translators_for_manufacturer - Find translators with experience in a specific manufacturer
2. get_translators_for_sector - Find translators with experience in a specific sector
3. get_manufacturer_data_for_translator - Find manufacturers/sectors a specific translator has worked with
4. get_quality_for_translator - Get average quality evaluation for a specific translator-language pair

Usage:
    from query_functions import QueryFunctions
    
    # Initialize the query functions
    query = QueryFunctions()
    
    # Example usage
    translators = query.get_translators_for_manufacturer("Appcelerate")
    manufacturer_data = query.get_manufacturer_data_for_translator("Priscila", "English", "Spanish (LA)")
    quality = query.get_quality_for_translator("Priscila", "English", "Spanish (LA)")
"""

import os
import json
from pathlib import Path
import logging
from .data_constants import JsonKeys, Files

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryFunctions:
    """Class for querying translator data from JSON files."""
    
    def __init__(self, data_dir=None):
        """
        Initialize QueryFunctions with data directory.
        
        Args:
            data_dir: Path to the directory containing JSON files.
                      If None, uses ../data/interim/ relative to this file.
        """
        if data_dir is None:
            # Default to ../data/interim/ relative to this file
            self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "interim"
        else:
            self.data_dir = Path(data_dir)
        
        # Load JSON data
        self._load_data()
    
    def _load_data(self):
        """Load JSON data from files."""
        try:
            # Load translator data
            translator_data_path = self.data_dir / Files.DEPRECATED_TRANSLATOR_DATA_JSON
            with open(translator_data_path, "r", encoding="utf-8") as f:
                self.translator_data = json.load(f)
            
            # Load lookup indexes
            lookup_indexes_path = self.data_dir / Files.DEPRECATED_LOOKUP_INDEXES_JSON
            with open(lookup_indexes_path, "r", encoding="utf-8") as f:
                self.lookup_indexes = json.load(f)
            
            logger.info(f"Successfully loaded data from {self.data_dir}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            # Initialize empty data structures to avoid errors
            self.translator_data = {JsonKeys.TRANSLATORS: {}}
            self.lookup_indexes = {JsonKeys.MANUFACTURERS: {}, JsonKeys.SECTORS: {}}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            # Initialize empty data structures to avoid errors
            self.translator_data = {JsonKeys.TRANSLATORS: {}}
            self.lookup_indexes = {JsonKeys.MANUFACTURERS: {}, JsonKeys.SECTORS: {}}
    
    def get_translators_for_manufacturer(self, manufacturer):
        """
        Find translators with experience in a specific manufacturer.
        
        Args:
            manufacturer: Name of the manufacturer to query
            
        Returns:
            List of translators with their language pairs in format:
            [{"translator": "Name", "source_lang": "Lang1", "target_lang": "Lang2"}, ...]
        """
        result = []
        compound_keys = self.lookup_indexes.get(JsonKeys.MANUFACTURERS, {}).get(manufacturer, [])
        
        for key in compound_keys:
            parts = key.split(JsonKeys.KEY_DELIMITER)
            if len(parts) == 3:
                result.append({
                    JsonKeys.NAME: parts[0],
                    JsonKeys.SOURCE: parts[1],
                    JsonKeys.TARGET: parts[2]
                })
        
        return result
    
    def get_translators_for_sector(self, sector):
        """
        Find translators with experience in a specific sector.
        
        Args:
            sector: Name of the sector to query
            
        Returns:
            List of translators with their language pairs in format:
            [{"translator": "Name", "source_lang": "Lang1", "target_lang": "Lang2"}, ...]
        """
        result = []
        compound_keys = self.lookup_indexes.get(JsonKeys.SECTORS, {}).get(sector, [])
        
        for key in compound_keys:
            parts = key.split(JsonKeys.KEY_DELIMITER)
            if len(parts) == 3:
                result.append({
                    JsonKeys.NAME: parts[0],
                    JsonKeys.SOURCE: parts[1],
                    JsonKeys.TARGET: parts[2]
                })
        
        return result
    
    def get_manufacturer_data_for_translator(self, translator, source_lang, target_lang)-> dict:
        """
        Find manufacturers/sectors a specific translator has worked with.
        
        Args:
            translator: Name of the translator
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Dictionary containing manufacturers and sectors with their counts:
            {"manufacturers": {"ManufacturerName": count, ...},
             "sectors": {"SectorName": count, ...}}
        """
        try:
            
            lang_pair_key = f"{source_lang}{JsonKeys.KEY_DELIMITER}{target_lang}"
            data = self.translator_data[JsonKeys.TRANSLATORS][translator][JsonKeys.LANGUAGE_PAIRS][lang_pair_key]
            return {
                JsonKeys.MANUFACTURERS: data[JsonKeys.MANUFACTURERS],
                JsonKeys.SECTORS: data[JsonKeys.SECTORS]
            }
        except KeyError:
            # Return empty dictionaries if data not found
            return {JsonKeys.MANUFACTURERS: {}, JsonKeys.SECTORS: {}}
    
    def get_quality_for_translator(self, translator, source_lang, target_lang):
        """
        Get average quality evaluation for a specific translator-language pair.
        
        Args:
            translator: Name of the translator
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Float representing average quality evaluation, or None if not found
        """
        try:
            lang_pair_key = f"{source_lang}{JsonKeys.KEY_DELIMITER}{target_lang}"
            quality_data = self.translator_data[JsonKeys.TRANSLATORS][translator][JsonKeys.LANGUAGE_PAIRS][lang_pair_key][JsonKeys.QUALITY]
            return quality_data[JsonKeys.AVERAGE]
        except KeyError:
            return None
    
    def reload_data(self):
        """Reload JSON data from files."""
        self._load_data()
        
    def get_all_manufacturers(self):
        """Get a list of all manufacturers."""
        return list(self.lookup_indexes.get(JsonKeys.MANUFACTURERS, {}).keys())
    
    def get_all_sectors(self):
        """Get a list of all sectors."""
        return list(self.lookup_indexes.get(JsonKeys.SECTORS, {}).keys())
    
    def get_all_translators(self):
        """Get a list of all translators."""
        return list(self.translator_data.get(JsonKeys.TRANSLATORS, {}).keys())
    
    def get_translator_summary(self, translator):
        """
        Get a summary of a translator's experience and quality.
        
        Args:
            translator: Name of the translator
            
        Returns:
            Dictionary with summary information
        """
        result = {
            JsonKeys.NAME: translator,
            JsonKeys.LANGUAGE_PAIRS: [],
            JsonKeys.MANUFACTURERS: set(),
            JsonKeys.SECTORS: set(),
            JsonKeys.AVERAGE_QUALITY: 0,
            JsonKeys.JOB_COUNT: 0
        }
        
        try:
            translator_data = self.translator_data[JsonKeys.TRANSLATORS][translator]
            
            quality_sum = 0
            quality_count = 0
            
            # Iterate through language pairs
            for lang_pair, data in translator_data[JsonKeys.LANGUAGE_PAIRS].items():
                # Parse source and target languages from the key
                source_lang, target_lang = lang_pair.split(JsonKeys.KEY_DELIMITER)
                
                # Add language pair
                pair_info = {
                    JsonKeys.SOURCE: source_lang,
                    JsonKeys.TARGET: target_lang,
                    JsonKeys.JOB_COUNT: sum(data[JsonKeys.MANUFACTURERS].values()),
                    JsonKeys.QUALITY: data[JsonKeys.QUALITY][JsonKeys.AVERAGE] if data[JsonKeys.QUALITY][JsonKeys.COUNT] > 0 else None
                }
                result[JsonKeys.LANGUAGE_PAIRS].append(pair_info)
                
                # Collect manufacturers and sectors
                result[JsonKeys.MANUFACTURERS].update(data[JsonKeys.MANUFACTURERS].keys())
                result[JsonKeys.SECTORS].update(data[JsonKeys.SECTORS].keys())
                
                # Add to job count
                result[JsonKeys.JOB_COUNT] += pair_info[JsonKeys.JOB_COUNT]
                
                # Add to quality calculations
                if data[JsonKeys.QUALITY][JsonKeys.COUNT] > 0:
                    quality_sum += data[JsonKeys.QUALITY][JsonKeys.AVERAGE] * data[JsonKeys.QUALITY][JsonKeys.COUNT]
                    quality_count += data[JsonKeys.QUALITY][JsonKeys.COUNT]
            
            # Calculate overall quality average
            if quality_count > 0:
                result[JsonKeys.AVERAGE_QUALITY] = quality_sum / quality_count
            else:
                result[JsonKeys.AVERAGE_QUALITY] = None
                
            # Convert sets to lists for JSON serialization
            result[JsonKeys.MANUFACTURERS] = list(result[JsonKeys.MANUFACTURERS])
            result[JsonKeys.SECTORS] = list(result[JsonKeys.SECTORS])
                
        except KeyError:
            pass
        
        return result


if __name__ == "__main__":
    # Example usage
    query = QueryFunctions()
    
    # Print some example queries
    print("\nExample Query 1: Get translators for manufacturer 'Appcelerate'")
    print(query.get_translators_for_manufacturer("Appcelerate"))
    
    print("\nExample Query 2: Get manufacturer data for translator 'Priscila', English to Spanish (LA)")
    print(query.get_manufacturer_data_for_translator("Priscila", "English", "Spanish (LA)"))
    
    print("\nExample Query 3: Get quality for translator 'Priscila', English to Spanish (LA)")
    print(query.get_quality_for_translator("Priscila", "English", "Spanish (LA)"))