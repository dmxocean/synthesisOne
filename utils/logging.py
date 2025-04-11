import os
import logging
from datetime import datetime


class BracketAlignedFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style="%", level_width=7):
        """
        Custom formatter that keeps brackets tight against the level name
        but adds padding after the closing bracket to ensure alignment
        
        Params:
            fmt (str, optional): The format string for log messages
            datefmt (str, optional): The format string for dates
            style (str, optional): The style of format string
            level_width (int, optional): The width to allocate for the longest log level name "WARNING"
        
        Returns:
            str: Formatted log message with consistent alignment
        """
        super().__init__(fmt, datefmt, style)
        self.level_width = level_width
    
    def format(self, record):
        original_levelname = record.levelname # Store the original levelname

        result = super().format(record) # Format the message using the standard formatter
        
        padding_needed = self.level_width - len(original_levelname) # Calculate padding needed after the level bracket
        padding = " " * padding_needed if padding_needed > 0 else ""
        
        # Find and replace the level bracket part with padded version
        level_part = f"[{original_levelname}]"
        padded_level_part = f"[{original_levelname}]{padding}"
        
        result = result.replace(level_part, padded_level_part, 1) # Replace only the level name part
        
        return result

def logger_setup(name, log_dir, log_file=None, mode="w"):
    """
    Set up a logger with both file and console output
    
    Parameters:
        name (str): Name of the logger (used in log messages)
        log_dir (str): Directory to save log files
        log_file (str, optional): Specific filename for log. If None, uses name + timestamp
        mode (str, optional): File mode for writing logs ("a" for append, "w" for overwrite)
        
    Returns:
        logger: Configured logging.Logger object
    """
    logger = logging.getLogger(name)  # Create logger
    logger.setLevel(logging.DEBUG)  # Capture all levels
    
    if logger.hasHandlers():  # Clear any existing handlers if logger exists
        logger.handlers.clear()
        
    os.makedirs(log_dir, exist_ok=True)  # Create log directory if it doesn't exist
    
    # Base format without fixed-width specifiers
    base_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Use our custom formatter for alignment
    formatter = BracketAlignedFormatter(
        fmt=base_format,
        datefmt=date_format,
        level_width=7  # For standard log levels, "WARNING" is the longest
    )
    
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    
    # Create file handler with specified mode
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if mode == "w":
        logger.info(f"Log file {log_file} created in overwrite mode")
    else:
        logger.info(f"Log file {log_file} opened in append mode")
        
    return logger

def log_section(logger, title):
    """
    Log a section header with clear separation
    
    Parameters:
        logger: The logger to use
        title (str): Section title
    """
    separator = " "
    logger.info(separator)
    logger.info(title)
    logger.info(separator)