import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional

def load_translator_calendar(path='translator_calendar.json'):
    """Load translator calendar from JSON file"""
    if os.path.exists(path):
        with open(path, 'r') as f:
            calendar = json.load(f)
            return {
                k: [(entry[0], datetime.fromisoformat(entry[1]), datetime.fromisoformat(entry[2])) for entry in v]
                for k, v in calendar.items()
            }
    return {}

def load_translator_calendar_from_data(calendar_data):
    """Load translator calendar from provided data (dict or JSON string)"""
    if isinstance(calendar_data, str):
        calendar = json.loads(calendar_data)
    elif isinstance(calendar_data, dict):
        calendar = calendar_data
    else:
        return {}
    
    return {
        k: [(entry[0], datetime.fromisoformat(entry[1]), datetime.fromisoformat(entry[2])) for entry in v]
        for k, v in calendar.items()
    }

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

def load_translator_schedules(schedules_path='data/interim/schedules.csv') -> Dict[str, Dict]:
    """Load translator weekly schedules from CSV file"""
    try:
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
    except Exception as e:
        print(f"Error loading translator schedules: {e}")
        return {}

def is_translator_available(task_row, translator_calendar, translator_schedules=None, modify_calendar=True):
    """Check if translator is available for the given task
    
    Args:
        task_row: Task information
        translator_calendar: Current translator calendar (will be modified if modify_calendar=True)
        translator_schedules: Weekly schedules for translators
        modify_calendar: If True, adds assignment to calendar; if False, only checks availability
    
    Returns:
        bool: True if translator is available and task can be scheduled
    """
    # Extract task information
    if isinstance(task_row, dict):
        task_start = task_row['start']
        task_deadline = task_row['deadline'] 
        duration = task_row['forecast']
        translator = task_row['translator']
    else:
        # Assume it's a pandas Series
        task_start = task_row['start']
        task_deadline = task_row['deadline']
        duration = task_row['forecast']
        translator = task_row['translator']
    
    # Convert strings to datetime if needed
    if isinstance(task_start, str):
        task_start = datetime.fromisoformat(task_start.replace('Z', '+00:00'))
    if isinstance(task_deadline, str):
        task_deadline = datetime.fromisoformat(task_deadline.replace('Z', '+00:00'))
    
    # Get translator's weekly schedule
    if translator_schedules and translator in translator_schedules:
        schedule = translator_schedules[translator]
        
        # Handle different time formats (HH:MM or HH:MM:SS)
        work_start_str = schedule['work_start']
        work_end_str = schedule['work_end']
        
        # Parse time strings - handle both HH:MM and HH:MM:SS formats
        try:
            if len(work_start_str.split(':')) == 3:  # HH:MM:SS format
                work_start = datetime.strptime(work_start_str, "%H:%M:%S").time()
                work_end = datetime.strptime(work_end_str, "%H:%M:%S").time()
            else:  # HH:MM format
                work_start = datetime.strptime(work_start_str, "%H:%M").time()
                work_end = datetime.strptime(work_end_str, "%H:%M").time()
        except ValueError as e:
            print(f"Warning: Error parsing time for translator {translator}: {e}")
            print(f"  work_start: '{work_start_str}', work_end: '{work_end_str}'")
            # Use default times if parsing fails
            work_start = datetime.strptime("09:00", "%H:%M").time()
            work_end = datetime.strptime("17:00", "%H:%M").time()
        
        weekday_avail = {
            0: schedule['mon'], 1: schedule['tue'], 2: schedule['wed'],
            3: schedule['thu'], 4: schedule['fri'], 5: schedule['sat'], 6: schedule['sun']
        }
    else:
        # Fallback to task_row data if available
        try:
            work_start_str = task_row['work_start']
            work_end_str = task_row['work_end']
            
            # Handle different time formats
            if len(work_start_str.split(':')) == 3:  # HH:MM:SS format
                work_start = datetime.strptime(work_start_str, "%H:%M:%S").time()
                work_end = datetime.strptime(work_end_str, "%H:%M:%S").time()
            else:  # HH:MM format
                work_start = datetime.strptime(work_start_str, "%H:%M").time()
                work_end = datetime.strptime(work_end_str, "%H:%M").time()
                
            weekday_avail = {
                0: task_row['mon'], 1: task_row['tue'], 2: task_row['wed'],
                3: task_row['thu'], 4: task_row['fri'], 5: task_row['sat'], 6: task_row['sun']
            }
        except (KeyError, TypeError, ValueError) as e:
            print(f"Warning: No schedule data found for translator {translator}: {e}")
            return False

    # Initialize translator calendar if not exists
    if translator not in translator_calendar:
        translator_calendar[translator] = []
    
    # Get current scheduled tasks (read-only copy for checking)
    scheduled = translator_calendar[translator].copy()
    
    # Track new assignments to add (only if modify_calendar=True)
    new_assignments = []

    current_time = task_start
    hours_remaining = duration

    while current_time <= task_deadline and hours_remaining > 0:
        # Check if translator is available on this weekday
        if weekday_avail.get(current_time.weekday(), 0):
            start_dt = datetime.combine(current_time.date(), work_start)
            end_dt = datetime.combine(current_time.date(), work_end)

            # Get busy periods for this day (from existing schedule + potential new assignments)
            busy = [(s, e) for (_, s, e) in scheduled if s.date() == current_time.date()]
            busy.sort()

            # Find free slots
            free_slots = []
            last_end = start_dt
            for b_start, b_end in busy:
                if b_start > last_end:
                    free_slots.append((last_end, b_start))
                last_end = max(last_end, b_end)
            if last_end < end_dt:
                free_slots.append((last_end, end_dt))

            # Try to assign work in free slots
            for slot_start, slot_end in free_slots:
                free_hours = (slot_end - slot_start).total_seconds() / 3600
                if free_hours <= 0:
                    continue
                    
                hours_to_assign = min(hours_remaining, free_hours)
                assignment_end = slot_start + timedelta(hours=hours_to_assign)
                
                # Create the assignment
                new_assignment = (task_row.get('task_id', 'unknown'), slot_start, assignment_end)
                new_assignments.append(new_assignment)
                
                # Add to scheduled list for further availability checking in this loop
                scheduled.append(new_assignment)
                
                hours_remaining -= hours_to_assign
                if hours_remaining <= 0:
                    break
        
        current_time += timedelta(days=1)

    # Check if task was fully scheduled
    if hours_remaining <= 0:
        # Only modify the calendar if requested
        if modify_calendar:
            translator_calendar[translator].extend(new_assignments)
        return True
    else:
        # Task couldn't be completed - don't add any assignments
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
    
    for task_id, task_df in task_rankings.items():
        task_assigned = False
        
        for _, row in task_df.iterrows():
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
                break  # Stop after assigning to one available translator
        
        if not task_assigned:
            print(f"Warning: Could not assign task {task_id} to any available translator")
    
    return assignments, translator_calendar

def check_translator_availability_only(task_row, existing_calendar_data=None, translator_schedules=None):
    """Check if translator is available without modifying calendar
    
    Args:
        task_row: Task information
        existing_calendar_data: Existing translator assignments
        translator_schedules: Weekly schedules for translators
    
    Returns:
        bool: True if translator is available
    """
    # Load existing calendar
    if existing_calendar_data is None:
        translator_calendar = {}
    elif isinstance(existing_calendar_data, str):
        if os.path.exists(existing_calendar_data):
            translator_calendar = load_translator_calendar(existing_calendar_data)
        else:
            translator_calendar = load_translator_calendar_from_data(existing_calendar_data)
    elif isinstance(existing_calendar_data, dict):
        translator_calendar = load_translator_calendar_from_data(existing_calendar_data)
    else:
        translator_calendar = {}
    
    return is_translator_available(task_row, translator_calendar, translator_schedules, modify_calendar=False)