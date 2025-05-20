import pandas as pd
from datetime import datetime, timedelta
import json
import os

def load_translator_calendar(path='translator_calendar.json'):
    if os.path.exists(path):
        with open(path, 'r') as f:
            calendar = json.load(f)
            return {
                k: [(entry[0], datetime.fromisoformat(entry[1]), datetime.fromisoformat(entry[2])) for entry in v]
                for k, v in calendar.items()
            }
    return {}

def save_translator_calendar(calendar, path='translator_calendar.json'):
    serializable_calendar = {
        k: [(entry[0], entry[1].isoformat(), entry[2].isoformat()) for entry in v]
        for k, v in calendar.items()
    }
    with open(path, 'w') as f:
        json.dump(serializable_calendar, f, indent=4)

def save_assignments_to_csv(assignments, path='final_assignments.csv'):
    df = pd.DataFrame(assignments)
    df.to_csv(path, index=False)

def save_schedule_to_json(assignments, path='temp_schedule.json'):
    with open(path, 'w') as f:
        json.dump(assignments, f, indent=4, default=str)

def is_translator_available(task_row, translator_calendar):
    task_start = task_row['start']
    task_deadline = task_row['deadline']
    duration = task_row['forecast']

    work_start = datetime.strptime(task_row['work_start'], "%H:%M").time()
    work_end = datetime.strptime(task_row['work_end'], "%H:%M").time()

    weekday_avail = {
        0: task_row['mon'], 1: task_row['tue'], 2: task_row['wed'],
        3: task_row['thu'], 4: task_row['fri'], 5: task_row['sat'], 6: task_row['sun']
    }

    translator = task_row['translator']
    if translator not in translator_calendar:
        translator_calendar[translator] = []
    scheduled = translator_calendar[translator]

    current_time = task_start
    hours_remaining = duration

    while current_time <= task_deadline and hours_remaining > 0:
        if weekday_avail.get(current_time.weekday(), 0):
            start_dt = datetime.combine(current_time.date(), work_start)
            end_dt = datetime.combine(current_time.date(), work_end)

            busy = [(s, e) for (_, s, e) in scheduled if s.date() == current_time.date()]
            busy.sort()

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
                scheduled.append((task_row['task_id'], slot_start, slot_start + timedelta(hours=hours_to_assign)))
                hours_remaining -= hours_to_assign
                if hours_remaining <= 0:
                    break
        current_time += timedelta(days=1)

    if hours_remaining <= 0:
        translator_calendar[translator] = scheduled
        return True
    return False

def assign_tasks_from_model_output(task_rankings, translator_calendar):
    assignments = []
    for task_id, task_df in task_rankings.items():
        for _, row in task_df.iterrows():
            if is_translator_available(row, translator_calendar):
                assignments.append({
                    'task_id': task_id,
                    'translator': row['translator'],
                    'start': row['start'].strftime('%Y-%m-%d %H:%M:%S'),
                    'deadline': row['deadline'].strftime('%Y-%m-%d %H:%M:%S'),
                    'source_lang': row.get('source_lang'),
                    'target_lang': row.get('target_lang'),
                    'forecast': row.get('forecast'),
                    'industry': row.get('industry')
                })
                break  # stop after assigning to one available translator
    return assignments
