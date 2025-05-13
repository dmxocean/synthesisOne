# src/utils/time.py

"""
Utility functions for time calculations
"""

import datetime

def time_difference(
    start_time,
    end_time
):
    """
    Calculate the duration between two times, properly handling overnight shifts

    Parameters:
        start_time (str): Start time in format "HH:MM:SS"
        end_time (str): End time in format "HH:MM:SS"

    Returns:
        float: Duration in hours

    Examples:
    >>> df["DIFFERENCE"] = df.apply(
    ...     lambda row: calculate_time_difference_hours(row["START"], row["END"]),
    ...     axis=1
    ... )
    """

    # Parse the time strings
    start = datetime.datetime.strptime(start_time, "%H:%M:%S").time()
    end = datetime.datetime.strptime(end_time, "%H:%M:%S").time()

    # Convert to total seconds since start of day
    start_seconds = start.hour * 3600 + start.minute * 60 + start.second
    end_seconds = end.hour * 3600 + end.minute * 60 + end.second

    # Check if shift crosses midnight
    if end < start:  # Overnight shift
        seconds_before_midnight = 24 * 3600 - start_seconds
        total_seconds = seconds_before_midnight + end_seconds
    else:  # Regular shift
        total_seconds = end_seconds - start_seconds

    # Convert to hours (float)
    hours = total_seconds / 3600

    return hours