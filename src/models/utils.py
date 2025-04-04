import csv
import os


#TODO Make it resilient to inexistent columns in "clumns_to_extract"
def read_csv(
                csv_folder=os.path.join(os.path.abspath(__file__),"../../../data/raw/"),
                csv_to_read="sample.csv",                     
                columns_to_extract=["TASK_ID","SOURCE_LANG","TARGET_LANG"]):
    """
    Reads the given tasks file and returns a list of dictionaries with only "useful" information
    Args:
        csv_folder (string): the address of the folder holding the csv
        csv_to_read (string): the name of the file to read
        columns_to_extract (list): list with the names of the columns to be extracted
    Returns:  
        list: A list of dictionaries containing all tasks in the input 
    """
    new_tasks=[]
    
    with open(os.path.abspath(csv_folder+csv_to_read)) as tasks_file:
        tasks_csv = csv.DictReader(tasks_file)
        for task in tasks_csv:
            temp={task[k] for k in columns_to_extract}
            new_tasks.append(temp)
    return new_tasks
print(read_csv())
