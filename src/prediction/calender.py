# src/prediction/calender.py

"""
Calendar Management for Translator Assignment System

Manages translator availability schedules and task assignments
Checks conflicts and validates task allocation against working hours

IMPORTANT: Central component for schedule integrity and assignment validation
"""

import pandas as pd
from datetime import datetime, timedelta, time
import json
import os
from typing import Dict, List, Tuple, Optional

# Path configuration
PATH_PREDICTION = os.path.dirname(os.path.abspath(__file__))
PATH_SRC = os.path.dirname(PATH_PREDICTION)
PATH_ROOT = os.path.dirname(PATH_SRC)

# Configure paths
PATH_DATA_INTERIM = os.path.join(PATH_ROOT, "data", "interim")
PATH_DATA_PREDICTION = os.path.join(PATH_ROOT, "data", "prediction")
PATH_SCHEDULES_CSV = os.path.join(PATH_DATA_INTERIM, "schedules.csv")
PATH_TRANSLATOR_SCHEDULE = os.path.join(PATH_DATA_PREDICTION, "translator_schedule.json")


def check_calendar_status(translator_calendar, task_start=None, task_deadline=None):
    """
    Check if calendar is loaded and has relevant tasks
    
    Args:
        translator_calendar: The loaded calendar dictionary
        task_start: Optional start datetime to check for relevant conflicts
        task_deadline: Optional deadline datetime to check for relevant conflicts
    
    Returns:
        dict: {
            'loaded': bool,
            'has_tasks': bool,
            'total_translators': int,
            'total_assignments': int,
            'relevant_conflicts': int (if time range provided)
        }
    """
    status = {
        'loaded': False,
        'has_tasks': False,
        'total_translators': 0,
        'total_assignments': 0
    }
    
    # Check if calendar loaded successfully
    if not isinstance(translator_calendar, dict):
        return status
    
    status['loaded'] = True
    status['total_translators'] = len(translator_calendar)
    
    # Count total assignments
    for translator, assignments in translator_calendar.items():
        if isinstance(assignments, list):
            status['total_assignments'] += len(assignments)
    
    status['has_tasks'] = status['total_assignments'] > 0
    
    # Check for relevant conflicts if time range provided
    if task_start and task_deadline and status['has_tasks']:
        relevant_conflicts = 0
        for translator, assignments in translator_calendar.items():
            for assignment in assignments:
                if len(assignment) >= 3:
                    _, start_time, end_time = assignment[0], assignment[1], assignment[2]
                    
                    # Convert strings to datetime if needed
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    
                    # Check if assignment overlaps with our time range
                    if not (end_time <= task_start or start_time >= task_deadline):
                        relevant_conflicts += 1
        
        status['relevant_conflicts'] = relevant_conflicts
    
    return status

def load_translator_calendar(path=None):
    """Load translator calendar from JSON file"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            calendar = json.load(f)
            
        # Convert the new format to the expected tuple format
        converted_calendar = {}
        for translator, assignments in calendar.items():
            converted_calendar[translator] = []
            
            for i, assignment in enumerate(assignments):
                # Handle both dictionary format and tuple format
                if isinstance(assignment, dict):
                    # Dictionary format: {"task_id": ..., "start": ..., "end": ...}
                    # May also contain additional fields like "alternatives" which we'll ignore
                    task_id = assignment.get('task_id')
                    start_time = assignment.get('start')
                    end_time = assignment.get('end')
                    
                    if task_id is not None and start_time and end_time:
                        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                        converted_calendar[translator].append((task_id, start_dt, end_dt))
                        
                elif isinstance(assignment, (tuple, list)) and len(assignment) >= 3:
                    # Tuple format: (task_id, start_datetime, end_datetime)
                    task_id, start_time, end_time = assignment[0], assignment[1], assignment[2]
                    
                    # Convert datetime strings if needed
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    
                    converted_calendar[translator].append((task_id, start_time, end_time))
            
        return converted_calendar
    else:
        return {}

def load_translator_calendar_from_data(calendar_data):
    """Load translator calendar from provided data (dict or JSON string)"""
    if isinstance(calendar_data, str):
        calendar = json.loads(calendar_data)
    elif isinstance(calendar_data, dict):
        calendar = calendar_data
    else:
        return {}
    
    # Convert the new format to the expected tuple format
    converted_calendar = {}
    for translator, assignments in calendar.items():
        converted_calendar[translator] = []
        for assignment in assignments:
            # Handle both dictionary format and tuple format
            if isinstance(assignment, dict):
                # Dictionary format: {"task_id": ..., "start": ..., "end": ...}
                # May also contain additional fields like "alternatives" which we'll ignore
                task_id = assignment.get('task_id')
                start_time = assignment.get('start')
                end_time = assignment.get('end')
                
                if task_id is not None and start_time and end_time:
                    converted_calendar[translator].append(
                        (task_id, 
                         datetime.fromisoformat(start_time.replace('Z', '+00:00')), 
                         datetime.fromisoformat(end_time.replace('Z', '+00:00')))
                    )
            elif isinstance(assignment, (tuple, list)) and len(assignment) >= 3:
                # Tuple format: (task_id, start_datetime, end_datetime)
                task_id, start_time, end_time = assignment[0], assignment[1], assignment[2]
                
                # Convert datetime strings if needed
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                if isinstance(end_time, str):
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
                converted_calendar[translator].append((task_id, start_time, end_time))
    
    return converted_calendar
    
def save_assignments_to_csv(assignments, path='final_assignments.csv'):
    """Save final assignments to CSV file"""
    df = pd.DataFrame(assignments)
    df.to_csv(path, index=False)

def save_schedule_to_json(assignments, path='temp_schedule.json'):
    """Save schedule to temporary JSON file"""
    with open(path, 'w') as f:
        json.dump(assignments, f, indent=4, default=str)

def export_translator_calendar(translator_calendar):
    """Export translator calendar to serializable format (for interface to save)"""
    return {
        k: [(entry[0], entry[1].isoformat(), entry[2].isoformat()) for entry in v]
        for k, v in translator_calendar.items()
    }

def load_translator_schedules(schedules_path=None) -> Dict[str, Dict]:
    """Load translator weekly schedules from CSV file"""
    if schedules_path is None:
        schedules_path = PATH_SCHEDULES_CSV
    
    schedules_df = pd.read_csv(schedules_path)
    translator_schedules = {}
    
    for _, row in schedules_df.iterrows():
        translator = row['NAME']
        translator_schedules[translator] = {
            'work_start': row['START'],
            'work_end': row['END'],
            'mon': bool(row['MON']),
            'tue': bool(row['TUES']),  # Note: CSV has TUES, not TUE
            'wed': bool(row['WED']),
            'thu': bool(row['THURS']),  # Note: CSV has THURS, not THU
            'fri': bool(row['FRI']),
            'sat': bool(row['SAT']),
            'sun': bool(row['SUN'])
        }
    
    return translator_schedules

def is_translator_available(task_row, translator_calendar, translator_schedules=None, modify_calendar=True):
    """Check if translator is available for the given task with debug messages"""
    from datetime import datetime, time, timedelta

    # Extract task information
    task_start = task_row['start']
    task_deadline = task_row['deadline']
    duration = task_row['forecast']
    translator = task_row['translator']
    task_id = task_row.get('task_id', 'unknown')

    # Convert strings to datetime if needed
    if isinstance(task_start, str):
        task_start = datetime.fromisoformat(task_start.replace('Z', '+00:00'))
    if isinstance(task_deadline, str):
        task_deadline = datetime.fromisoformat(task_deadline.replace('Z', '+00:00'))

    # Translator schedule lookup
    if translator_schedules and translator in translator_schedules:
        schedule = translator_schedules[translator]
        work_start_str = schedule['work_start']
        work_end_str = schedule['work_end']
        
        if len(work_start_str.split(':')) == 3:
            work_start = datetime.strptime(work_start_str, "%H:%M:%S").time()
            work_end = datetime.strptime(work_end_str, "%H:%M:%S").time()
        else:
            work_start = datetime.strptime(work_start_str, "%H:%M").time()
            work_end = datetime.strptime(work_end_str, "%H:%M").time()
            
        weekday_avail = {i: schedule.get(day, False) for i, day in enumerate(['mon','tue','wed','thu','fri','sat','sun'])}
    else:
        work_start = time(9, 0)
        work_end = time(17, 0)
        weekday_avail = {i: (i < 5) for i in range(7)}  # Mon-Fri

    # Initialize translator calendar if not exists
    if translator not in translator_calendar:
        translator_calendar[translator] = []
    scheduled = translator_calendar[translator].copy()
    new_assignments = []

    current_time = task_start
    hours_remaining = duration

    while current_time.date() <= task_deadline.date() and hours_remaining > 0:
        current_date = current_time.date()
        weekday = current_time.weekday()
        if weekday_avail.get(weekday, False):
            start_dt = datetime.combine(current_date, work_start)
            end_dt = datetime.combine(current_date, work_end)

            busy = [(s, e) for (_, s, e) in scheduled if s.date() == current_date]
            busy.sort()

            # find free slots
            free_slots = []
            last_end = start_dt
            for b_start, b_end in busy:
                if b_start > last_end:
                    free_slots.append((last_end, b_start))
                last_end = max(last_end, b_end)
            if last_end < end_dt:
                free_slots.append((last_end, end_dt))

            for slot_start, slot_end in free_slots:
                free_hours = (slot_end - slot_start).total_seconds() / 3600
                if free_hours <= 0:
                    continue
                hours_to_assign = min(hours_remaining, free_hours)
                assignment_end = slot_start + timedelta(hours=hours_to_assign)
                new_assignments.append((task_id, slot_start, assignment_end))
                scheduled.append((task_id, slot_start, assignment_end))
                hours_remaining -= hours_to_assign
                if hours_remaining <= 0:
                    break

        current_time += timedelta(days=1)

    # Outcome
    if hours_remaining <= 0:
        if modify_calendar:
            translator_calendar[translator].extend(new_assignments)
        return True
    else:
        return False

def assign_tasks_from_model_output(task_rankings, existing_calendar_data=None, translator_schedules=None):
    """Assign tasks to translators based on model rankings and availability
    
    Args:
        task_rankings: Dictionary of task rankings from models
        existing_calendar_data: Existing translator assignments (dict, JSON string, or file path)
        translator_schedules: Weekly schedules for translators
    
    Returns:
        tuple: (assignments, final_translator_calendar)
    """
    # Load existing calendar
    if existing_calendar_data is None:
        existing_calendar_data = PATH_TRANSLATOR_SCHEDULE
    
    if isinstance(existing_calendar_data, str) and not os.path.exists(existing_calendar_data):
        translator_calendar = {}
    elif isinstance(existing_calendar_data, str):
        if os.path.exists(existing_calendar_data):
            # It's a file path
            translator_calendar = load_translator_calendar(existing_calendar_data)
        else:
            # It's JSON string
            translator_calendar = load_translator_calendar_from_data(existing_calendar_data)
    elif isinstance(existing_calendar_data, dict):
        translator_calendar = load_translator_calendar_from_data(existing_calendar_data)
    else:
        translator_calendar = {}
    
    assignments = []
    successful_assignments = 0
    failed_assignments = 0
    
    for task_id, task_df in task_rankings.items():
        task_assigned = False
        
        for rank, (_, row) in enumerate(task_df.iterrows(), 1):
            # Create enhanced row with task_id for tracking
            enhanced_row = row.copy() if hasattr(row, 'copy') else dict(row)
            if isinstance(enhanced_row, dict):
                enhanced_row['task_id'] = task_id
            else:
                enhanced_row = enhanced_row.to_dict()
                enhanced_row['task_id'] = task_id
                
            if is_translator_available(enhanced_row, translator_calendar, translator_schedules, modify_calendar=True):
                assignment = {
                    'task_id': task_id,
                    'translator': enhanced_row['translator'],
                    'start': enhanced_row['start'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(enhanced_row['start'], datetime) else str(enhanced_row['start']),
                    'deadline': enhanced_row['deadline'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(enhanced_row['deadline'], datetime) else str(enhanced_row['deadline']),
                    'source_lang': enhanced_row.get('source_lang'),
                    'target_lang': enhanced_row.get('target_lang'),
                    'forecast': enhanced_row.get('forecast'),
                    'industry': enhanced_row.get('industry', enhanced_row.get('manufacturer_sector'))
                }
                assignments.append(assignment)
                task_assigned = True
                successful_assignments += 1
                break  # Stop after assigning to one available translator
        
        if not task_assigned:
            failed_assignments += 1
            print(f"Warning: Could not assign task {task_id} to any available translator")
    
    return assignments, translator_calendar

def check_translator_availability_only(
    task_row,
    existing_calendar_data=None,
    translator_schedules=None
):
    """
    Check if translator is available without modifying calendar, using debug prints
    """
    # Load existing calendar
    if existing_calendar_data is None:
        existing_calendar_data = PATH_TRANSLATOR_SCHEDULE
        translator_calendar = {}
    elif isinstance(existing_calendar_data, str) and os.path.exists(existing_calendar_data):
        translator_calendar = load_translator_calendar(existing_calendar_data)
    elif isinstance(existing_calendar_data, (str, dict)):
        translator_calendar = load_translator_calendar_from_data(existing_calendar_data)
    else:
        translator_calendar = {}

    # Load schedules if not provided
    if translator_schedules is None:
        translator_schedules = load_translator_schedules()

    # Always call the debug-enabled availability checker
    available = is_translator_available(
        task_row,
        translator_calendar,
        translator_schedules,
        modify_calendar=False
    )
    print(f"[DEBUG] Availability for {task_row['translator']}: {available}")
    return available