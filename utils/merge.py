import logging


def log_dataframe_info(logger, df, name):
    """
    Log basic information about a dataframe
    
    Parameters:
        logger: The logger to use
        df: pandas DataFrame
        name (str): Name to identify this dataframe in logs
    """
    logger.info(f"DataFrame: {name}")
    logger.info(f"  - Rows: {len(df)}")
    logger.info(f"  - Columns: {list(df.columns)}")
    
    # Log more details at debug level
    logger.debug(f"  - First 5 rows:\n{df.head().to_string()}")

def log_merge_stats(logger, df_left, df_right, df_result, key_left, key_right):
    """
    Log statistics about a dataframe merge operation
    
    Parameters:
        logger: The logger to use
        df_left: Left DataFrame in the merge
        df_right: Right DataFrame in the merge  
        df_result: Resulting merged DataFrame
        key_left (str): Column name used as key in left dataframe
        key_right (str): Column name used as key in right dataframe
    """
    # Get unique keys from each side
    keys_left = set(df_left[key_left].dropna().unique())
    keys_right = set(df_right[key_right].dropna().unique())
    
    # Calculate missing keys
    missing_keys = keys_left - keys_right
    
    # Log the statistics
    logger.info(f"Merge Statistics:")
    logger.info(f"  - Left rows: {len(df_left)}")
    logger.info(f"  - Right rows: {len(df_right)}")
    logger.info(f"  - Result rows: {len(df_result)}")
    logger.info(f"  - Keys only in left: {len(missing_keys)}")
    
    # Show some examples of missing keys
    if missing_keys:
        logger.info(f"  - Examples of missing keys: {list(missing_keys)[:5]}")