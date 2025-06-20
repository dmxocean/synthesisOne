"""
JSON Queries Module for Translation Project Data
This module provides functions to query all JSON files in ./data/processed/base/artifacts/

There are too many functions, just everything I could come up with, add more if you need anything else
"""

import json
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Any
from datetime import datetime

# File name constants
METADATA_FILE = 'metadata.json'
TRANSLATOR_MAPPING_FILE = 'translator_mapping.json'
TRANSLATOR_METRICS_FILE = 'translator_metrics.json'
TRANSLATOR_EFFICIENCY_FILE = 'translator_efficiency_metrics.json'
TRANSLATOR_RATES_FILE = 'translator_hourly_rates.json'
CLIENTS_DATA_FILE = 'clients_data.json'
TRANSLATOR_CAPABILITIES_FILE = 'translator_capabilities.json'
TRANSLATOR_SCHEDULE_FILE = 'translator_schedule_metrics.json'
LANGUAGE_PAIR_METRICS_FILE = 'language_pair_metrics.json'

# JSON key constants
PREPROCESSING_DATE_KEY = 'preprocessing_date'
ARTIFACTS_KEY = 'artifacts'
TRANSLATOR_TO_ID_KEY = 'translator_to_id'
ID_TO_TRANSLATOR_KEY = 'id_to_translator'
GLOBAL_AVERAGE_KEY = '__global_average__'
ONTIME_RATE_KEY = 'ontime_rate'
AVG_QUALITY_KEY = 'avg_quality'
AVG_COST_KEY = 'avg_cost'
AVG_FORECAST_KEY = 'avg_forecast'
TASK_COUNT_KEY = 'task_count'
CLIENT_HISTORY_KEY = 'client_history'
SECTOR_HISTORY_KEY = 'sector_history'
TASK_TYPE_HISTORY_KEY = 'task_type_history'
SELLING_HOURLY_PRICE_KEY = 'selling_hourly_price'
MIN_QUALITY_KEY = 'min_quality'
WILDCARD_KEY = 'wildcard'
WEEKLY_HOURS_KEY = 'weekly_hours'
AVAILABILITY_KEY = 'availability'
TRANSLATOR_COUNT_KEY = 'translator_count'

# Sort criteria constants
SORT_BY_QUALITY = 'avg_quality'
SORT_BY_COST = 'avg_cost'
SORT_BY_FORECAST = 'avg_forecast'

# Default directory constant
DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "processed", "base", "artifacts")


class JSONQueryHandler:
    """Main class for handling JSON queries across all artifact files"""
    
    def __init__(self, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self._cache = {}
    
    def _load_json(self, filename: str) -> dict:
        """Load JSON file with caching"""
        if filename not in self._cache:
            file_path = os.path.join(self.artifacts_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                self._cache[filename] = json.load(f)
        return self._cache[filename]
    
    def clear_cache(self):
        """Clear the JSON cache"""
        self._cache = {}


class MetadataQueries:
    """Queries for metadata.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_preprocessing_date(self) -> str:
        """Get the preprocessing date"""
        data = self.handler._load_json(METADATA_FILE)
        return data.get(PREPROCESSING_DATE_KEY)
    
    def get_artifacts_list(self) -> pd.DataFrame:
        """Get list of available artifacts as DataFrame"""
        data = self.handler._load_json(METADATA_FILE)
        artifacts = data.get(ARTIFACTS_KEY, [])
        return pd.DataFrame({
            'artifact_name': artifacts,
            'order': range(len(artifacts))
        })


class TranslatorMappingQueries:
    """Queries for translator_mapping.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_id_from_name(self, translator_name: str) -> Optional[int]:
        """Get translator ID from name"""
        data = self.handler._load_json(TRANSLATOR_MAPPING_FILE)
        return data.get(TRANSLATOR_TO_ID_KEY, {}).get(translator_name)
    
    def get_name_from_id(self, translator_id: int) -> Optional[str]:
        """Get translator name from ID"""
        data = self.handler._load_json(TRANSLATOR_MAPPING_FILE)
        return data.get(ID_TO_TRANSLATOR_KEY, {}).get(str(translator_id))
    
    def get_mapping_df(self) -> pd.DataFrame:
        """Get translator ID to name mapping as DataFrame"""
        data = self.handler._load_json(TRANSLATOR_MAPPING_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {'translator_name': name, 'translator_id': id_val}
            for name, id_val in data.get(TRANSLATOR_TO_ID_KEY, {}).items()
        ]
        
        return pd.DataFrame.from_records(records)


class TranslatorMetricsQueries:
    """Queries for translator_metrics.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_translator_metrics(self, translator_name: str) -> pd.DataFrame:
        """Get all metrics for a specific translator"""
        data = self.handler._load_json(TRANSLATOR_METRICS_FILE)
        translator_data = data.get(translator_name)
        if not translator_data:
            return pd.DataFrame()
        
        # Create single row DataFrame
        df = pd.DataFrame([translator_data])
        df['translator_name'] = translator_name
        return df
    
    def get_global_average_metrics(self) -> pd.DataFrame:
        """Get global average metrics"""
        data = self.handler._load_json(TRANSLATOR_METRICS_FILE)
        global_avg = data.get(GLOBAL_AVERAGE_KEY, {})
        df = pd.DataFrame([global_avg])
        df['metric_type'] = 'global_average'
        return df
    
    def get_all_translator_metrics(self) -> pd.DataFrame:
        """Get metrics for all translators"""
        data = self.handler._load_json(TRANSLATOR_METRICS_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {**metrics, 'translator_name': translator}
            for translator, metrics in data.items()
            if translator != GLOBAL_AVERAGE_KEY
        ]
        
        return pd.DataFrame.from_records(records)
    
    def get_translators_by_ontime_rate(self, min_rate: float = 0.0, max_rate: float = 1.0) -> pd.DataFrame:
        """Get translators within specified ontime rate range"""
        df = self.get_all_translator_metrics()
        return df[
            (df[ONTIME_RATE_KEY] >= min_rate) & 
            (df[ONTIME_RATE_KEY] <= max_rate)
        ].sort_values(ONTIME_RATE_KEY, ascending=False)
    
    def get_translators_by_quality(self, min_quality: float = 0.0, max_quality: float = 10.0) -> pd.DataFrame:
        """Get translators within specified quality range"""
        df = self.get_all_translator_metrics()
        return df[
            (df[AVG_QUALITY_KEY] >= min_quality) & 
            (df[AVG_QUALITY_KEY] <= max_quality)
        ].sort_values(AVG_QUALITY_KEY, ascending=False)
    
    def get_translator_client_history(self, translator_name: str) -> pd.DataFrame:
        """Get client history for a translator"""
        data = self.handler._load_json(TRANSLATOR_METRICS_FILE)
        translator_data = data.get(translator_name, {})
        client_history = translator_data.get(CLIENT_HISTORY_KEY, {})
        
        # Use dictionary comprehension for efficiency
        records = [
            {
                'translator_name': translator_name,
                'client_name': client,
                'task_count': count
            }
            for client, count in client_history.items()
        ]
        
        return pd.DataFrame.from_records(records).sort_values('task_count', ascending=False)
    
    def get_translator_sector_history(self, translator_name: str) -> pd.DataFrame:
        """Get sector history for a translator"""
        data = self.handler._load_json(TRANSLATOR_METRICS_FILE)
        translator_data = data.get(translator_name, {})
        sector_history = translator_data.get(SECTOR_HISTORY_KEY, {})
        
        # Use dictionary comprehension for efficiency
        records = [
            {
                'translator_name': translator_name,
                'sector': sector,
                'task_count': count
            }
            for sector, count in sector_history.items()
        ]
        
        return pd.DataFrame.from_records(records).sort_values('task_count', ascending=False)
    

class TranslatorEfficiencyQueries:
    """Queries for translator_efficiency_metrics.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_efficiency_by_language_pair(self, translator_name: str, source_lang: str, 
                                      target_lang: str) -> pd.DataFrame:
        """Get efficiency metrics for specific translator and language pair"""
        data = self.handler._load_json(TRANSLATOR_EFFICIENCY_FILE)
        translator_data = data.get(translator_name, {})
        source_data = translator_data.get(source_lang, {})
        target_data = source_data.get(target_lang)
        
        if not target_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([target_data])
        df['translator_name'] = translator_name
        df['source_language'] = source_lang
        df['target_language'] = target_lang
        return df
    
    def get_all_translator_language_pairs(self) -> pd.DataFrame:
        """Get all language pairs for all translators with their efficiency metrics"""
        data = self.handler._load_json(TRANSLATOR_EFFICIENCY_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {**metrics, 'translator_name': translator, 'source_language': source_lang, 'target_language': target_lang}
            for translator, languages in data.items()
            for source_lang, targets in languages.items()
            for target_lang, metrics in targets.items()
        ]
        
        return pd.DataFrame.from_records(records)

    def get_translators_for_language_pair(self, source_lang: str, target_lang: str, 
                                     sort_by: str = None) -> pd.DataFrame:
        """Find best translators for a language pair"""
        df = self.get_all_translator_language_pairs()
        filtered_df = df[
            (df['source_language'] == source_lang) & 
            (df['target_language'] == target_lang)
        ]
        
        if sort_by in [SORT_BY_QUALITY, SORT_BY_COST, SORT_BY_FORECAST]:
            ascending = sort_by != SORT_BY_QUALITY
            return filtered_df.sort_values(sort_by, ascending=ascending)
        
        return filtered_df
    def get_translator_avg_quality(self,translator_name: str, source_lang: str, target_lang:str)-> float:
        """Get a given translator & source/target language pair's average quality"""
        data = self.handler._load_json(TRANSLATOR_EFFICIENCY_FILE)
        translator=data.get(translator_name,{})
        average_quality=translator.get(source_lang,{}).get(target_lang,{})[AVG_QUALITY_KEY]

        return average_quality

class TranslatorRatesQueries:
    """Queries for translator_hourly_rates.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_translator_rate_for_lang_pair(self, translator_name: str, source_lang: str, 
                         target_lang: str) -> Optional[float]:
        """Get hourly rate for specific translator and language pair"""
        data = self.handler._load_json(TRANSLATOR_RATES_FILE)
        translator_data = data.get(translator_name, {})
        source_data = translator_data.get(source_lang, {})
        return source_data.get(target_lang)
    
    def get_all_translator_rates(self) -> pd.DataFrame:
        """Get all rates for all translators"""
        data = self.handler._load_json(TRANSLATOR_RATES_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {
                'translator_name': translator,
                'source_language': source_lang,
                'target_language': target_lang,
                'hourly_rate': rate
            }
            for translator, languages in data.items()
            for source_lang, targets in languages.items()
            for target_lang, rate in targets.items()
        ]
        
        return pd.DataFrame.from_records(records)
    

class ClientsQueries:
    """Queries for clients_data.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_all_clients(self) -> pd.DataFrame:
        """Get information for all clients"""
        data = self.handler._load_json(CLIENTS_DATA_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {**info, 'client_name': client_name}
            for client_name, info in data.items()
        ]
        
        return pd.DataFrame.from_records(records)
    
    def get_client_info(self, client_name: str) -> pd.DataFrame:
        """Get information for a specific client"""
        data = self.handler._load_json(CLIENTS_DATA_FILE)
        client_data = data.get(client_name)
        if not client_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([client_data])
        df['client_name'] = client_name
        return df
    
    def get_clients_by_price_range(self, min_price: float, max_price: float) -> pd.DataFrame:
        """Get clients within specified price range"""
        df = self.get_all_clients()
        return df[
            (df[SELLING_HOURLY_PRICE_KEY] >= min_price) & 
            (df[SELLING_HOURLY_PRICE_KEY] <= max_price)
        ].sort_values(SELLING_HOURLY_PRICE_KEY, ascending=False)
    
    def get_clients_by_min_quality(self, min_quality: float) -> pd.DataFrame:
        """Get clients with minimum quality requirement"""
        df = self.get_all_clients()
        return df[df[MIN_QUALITY_KEY] >= min_quality].sort_values(MIN_QUALITY_KEY, ascending=False)
    
    def get_clients_by_wildcard(self, wildcard: str) -> pd.DataFrame:
        """Get clients with specific wildcard"""
        df = self.get_all_clients()
        return df[df[WILDCARD_KEY] == wildcard]


class TranslatorCapabilitiesQueries:
    """Queries for translator_capabilities.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_all_translator_capabilities(self) -> pd.DataFrame:
        """Get language capabilities for all translators"""
        data = self.handler._load_json(TRANSLATOR_CAPABILITIES_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {
                'translator_name': translator,
                'source_language': source_lang,
                'target_language': target_lang
            }
            for translator, languages in data.items()
            for source_lang, targets in languages.items()
            for target_lang in targets
        ]
        
        return pd.DataFrame.from_records(records)
    
    def find_translators_for_pair(self, source_lang: str, target_lang: str) -> pd.DataFrame:
        """Find translators who can translate a specific language pair"""
        df = self.get_all_translator_capabilities()
        return df[
            (df['source_language'] == source_lang) & 
            (df['target_language'] == target_lang)
        ]['translator_name'].drop_duplicates().to_frame()
    
    def get_all_language_pairs(self) -> pd.DataFrame:
        """Get all unique language pairs across all translators"""
        df = self.get_all_translator_capabilities()
        return df[['source_language', 'target_language']].drop_duplicates().sort_values(['source_language', 'target_language'])
    
    def get_translator_languages(self, translator_name: str) -> pd.DataFrame:
        """Get all language pairs for a specific translator"""
        df = self.get_all_translator_capabilities()
        return df[df['translator_name'] == translator_name][['source_language', 'target_language']]


class ScheduleQueries:
    """Queries for translator_schedule_metrics.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_translator_weekly_hours(self, translator_name: str) -> Optional[float]:
        """Get weekly hours for a translator"""
        data = self.handler._load_json(TRANSLATOR_SCHEDULE_FILE)
        translator_data = data.get(translator_name)
        if translator_data:
            return translator_data.get(WEEKLY_HOURS_KEY)
        return None
    
    def get_all_weekly_hours(self) -> pd.DataFrame:
        """Get weekly hours for all translators"""
        data = self.handler._load_json(TRANSLATOR_SCHEDULE_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {
                'translator_name': translator,
                WEEKLY_HOURS_KEY: schedule.get(WEEKLY_HOURS_KEY, 0)
            }
            for translator, schedule in data.items()
        ]
        
        return pd.DataFrame.from_records(records).sort_values(WEEKLY_HOURS_KEY, ascending=False)
    
    def get_availability_matrix(self, translator_name: str) -> pd.DataFrame:
        """Get availability matrix for a translator"""
        data = self.handler._load_json(TRANSLATOR_SCHEDULE_FILE)
        translator_data = data.get(translator_name)
        if not translator_data:
            return pd.DataFrame()
        
        availability = translator_data.get(AVAILABILITY_KEY, {})
        
        # Create DataFrame from availability
        days = list(availability.keys())
        hours = set()
        for day_availability in availability.values():
            hours.update(day_availability.keys())
        hours = sorted(list(hours))
        
        # Create matrix
        matrix = []
        for day in days:
            row = {'day': day}
            for hour in hours:
                row[f'hour_{hour}'] = availability[day].get(hour, 0)
            matrix.append(row)
        
        return pd.DataFrame(matrix)
    
    def find_available_translators(self, day: str, hour: str) -> pd.DataFrame:
        """Find all translators available at specific day and hour"""
        data = self.handler._load_json(TRANSLATOR_SCHEDULE_FILE)
        
        # Use dictionary comprehension with filtering for efficiency
        records = [
            {
                'translator_name': translator,
                'day': day,
                'hour': hour,
                WEEKLY_HOURS_KEY: schedule.get(WEEKLY_HOURS_KEY, 0)
            }
            for translator, schedule in data.items()
            if schedule.get(AVAILABILITY_KEY, {}).get(day, {}).get(hour, 0) == 1
        ]
        
        return pd.DataFrame.from_records(records)


class LanguagePairQueries:
    """Queries for language_pair_metrics.json"""
    
    def __init__(self, handler: JSONQueryHandler):
        self.handler = handler
    
    def get_all_language_pair_metrics(self) -> pd.DataFrame:
        """Get metrics for all language pairs"""
        data = self.handler._load_json(LANGUAGE_PAIR_METRICS_FILE)
        
        # Use dictionary comprehension for efficiency
        records = [
            {**metrics, 'source_language': source_lang, 'target_language': target_lang}
            for source_lang, targets in data.items()
            if source_lang != GLOBAL_AVERAGE_KEY
            for target_lang, metrics in targets.items()
        ]
        
        return pd.DataFrame.from_records(records)
    
    def get_pair_metrics(self, source_lang: str, target_lang: str) -> pd.DataFrame:
        """Get metrics for a specific language pair"""
        data = self.handler._load_json(LANGUAGE_PAIR_METRICS_FILE)
        source_data = data.get(source_lang, {})
        pair_data = source_data.get(target_lang)
        
        if not pair_data:
            return pd.DataFrame()
        
        df = pd.DataFrame([pair_data])
        df['source_language'] = source_lang
        df['target_language'] = target_lang
        return df
    
    def get_global_average_metrics(self) -> pd.DataFrame:
        """Get global average metrics for all language pairs"""
        data = self.handler._load_json(LANGUAGE_PAIR_METRICS_FILE)
        global_avg = data.get(GLOBAL_AVERAGE_KEY, {})
        df = pd.DataFrame([global_avg])
        df['metric_type'] = 'global_average'
        return df
    
    def find_most_common_pairs(self, top_n: int = 10) -> pd.DataFrame:
        """Find most common language pairs by task count"""
        df = self.get_all_language_pair_metrics()
        return df.nlargest(top_n, TASK_COUNT_KEY)
    
    def find_highest_quality_pairs(self, top_n: int = 10) -> pd.DataFrame:
        """Find language pairs with highest average quality"""
        df = self.get_all_language_pair_metrics()
        return df.nlargest(top_n, AVG_QUALITY_KEY)


class UnifiedQueryHandler:
    """Unified interface for all query types"""
    
    def __init__(self, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR):
        self.handler = JSONQueryHandler(artifacts_dir)
        
        # Initialize all query modules
        self.metadata = MetadataQueries(self.handler)
        self.translator_mapping = TranslatorMappingQueries(self.handler)
        self.translator_metrics = TranslatorMetricsQueries(self.handler)
        self.translator_efficiency = TranslatorEfficiencyQueries(self.handler)
        self.translator_rates = TranslatorRatesQueries(self.handler)
        self.clients = ClientsQueries(self.handler)
        self.translator_capabilities = TranslatorCapabilitiesQueries(self.handler)
        self.schedule = ScheduleQueries(self.handler)
        self.language_pairs = LanguagePairQueries(self.handler)
    
    def clear_cache(self):
        """Clear all cached JSON data"""
        self.handler.clear_cache()


# Example usage and helper functions
def example_usage():
    """Example usage of the query system"""
    # Initialize the unified query handler
    query = UnifiedQueryHandler()
    
    # Example 1: Find best translators for English to Spanish (Iberian)
    best_translators = query.translator_efficiency.get_translators_for_language_pair(
        "English", "Spanish (Iberian)", sort_by=SORT_BY_QUALITY
    )
    
    # Example 2: Get client information
    client_info = query.clients.get_client_info("MotorForge")
    
    # Example 3: Find available translators
    available = query.schedule.find_available_translators("MON", "14")
    
    # Example 4: Find translators with high ontime rate
    reliable_translators = query.translator_metrics.get_translators_by_ontime_rate(0.8, 1.0)
    
    # Example 5: Get all language pairs
    all_pairs = query.language_pairs.get_all_language_pair_metrics()
    
    return {
        'best_translators': best_translators.head(5).to_dict(),
        'client_info': client_info.to_dict(),
        'available': available.to_dict(),
        'reliable_translators': reliable_translators.head(10).to_dict(),
        'total_language_pairs': len(all_pairs)
    }


if __name__ == "__main__":
    # Run example usage
    results = example_usage()
    print(json.dumps(results, indent=2, default=str))