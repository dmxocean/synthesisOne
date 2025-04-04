import csv
import os

CSV_FOLDER=os.path.join(os.path.abspath(__file__),"../../../data/raw/")
CSV_CLIENTS="clients.csv"                       # Client data
CSV_SAMPLE="sample.csv"                         # Historical data
CSV_SCHEDULES="schedules.csv"                   # Translator Schedules
CSV_TRANSLATORS_COST="translatorsCostPairs.csv" # Translator Data

USEFUL_COLUMNS=["TASK_ID","SOURCE_LANG","TARGET_LANG","missing"]


def readTasks(csv_tasks_file):
    """
    Reads the given tasks file and returns a list of dictionaries with only "useful" information

    Args:
        csv_tasks_file (string): The address of the tasks file

    Returns:  
        list: A list of dictionaries containing all tasks in the input 
    """
    new_tasks=[]
    
    with open(os.path.abspath(CSV_FOLDER+csv_tasks_file)) as tasks_file:
        tasks_csv = csv.DictReader(tasks_file)
        for task in tasks_csv:
            print(task["TASK_ID"])
            new_tasks.append(task[USEFUL_COLUMNS])
    return new_tasks
readTasks(CSV_SAMPLE)
