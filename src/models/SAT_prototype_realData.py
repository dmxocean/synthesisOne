from ortools.sat.python import cp_model
import csv
import os


TASK_ID="TASK_ID"
SOURCE_LANG="SOURCE_LANG"
TARGET_LANG="TARGET_LANG"
MANUFACTURER="MANUFACTURER"
CLIENT_NAME="CLIENT_NAME"
SELLING_HOURLY_PRICE="SELLING_HOURLY_PRICE"
TRANSLATOR="TRANSLATOR"
HOURLY_RATE="HOURLY_RATE"


#TODO Make it resilient to inexistent columns in "clumns_to_extract"?
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
            temp={k: task[k] for k in columns_to_extract}
            new_tasks.append(temp)
    return new_tasks



def main():
    # Define multiple tasks.
    # Each task requires a translator with a matching source language, target language,
    # and an hourly rate no higher than the task's max_rate.
    tasks = read_csv(csv_to_read="sample.csv",columns_to_extract=[TASK_ID,SOURCE_LANG,TARGET_LANG,MANUFACTURER]) # Missing max rate
    tasks=tasks[:100]
    # This modification is only necessary if we use multiple csvs, so it's temporal
    temp = read_csv(csv_to_read="clients.csv",columns_to_extract=[CLIENT_NAME,SELLING_HOURLY_PRICE])
    for task in tasks:
        for client in temp:
            if client[CLIENT_NAME]==task[MANUFACTURER]:
                task.update({SELLING_HOURLY_PRICE: float(client[SELLING_HOURLY_PRICE])})

    # List of translators.
    translators = read_csv(csv_to_read="translatorsCostPairs.csv",columns_to_extract=[TRANSLATOR,SOURCE_LANG,TARGET_LANG,HOURLY_RATE]) 
    [translator.update({HOURLY_RATE:float(translator[HOURLY_RATE])}) for translator in translators]

    model = cp_model.CpModel()
    num_tasks = len(tasks)
    num_translators = len(translators)

    # Decision variables: x[i,j] is 1 if translator j is assigned to task i.
    x = {}
    for i in range(num_tasks):
        for j in range(num_translators):
            var = model.NewBoolVar(f"x_{i}_{j}")
            x[(i, j)] = var
            # Enforce eligibility: if translator j doesn't meet the task i's requirements,
            # force x[i,j] to 0.
            if (translators[j][SOURCE_LANG] != tasks[i][SOURCE_LANG] or 
                translators[j][TARGET_LANG] != tasks[i][TARGET_LANG] or 
                translators[j][HOURLY_RATE] > tasks[i][SELLING_HOURLY_PRICE]):
                model.Add(var == 0)

    # Each task must be assigned exactly one translator.
    for i in range(num_tasks):
        model.Add(sum(x[(i, j)] for j in range(num_translators)) == 1)

    # Each translator can only be assigned to at most one task.
    for j in range(num_translators):
        model.Add(sum(x[(i, j)] for i in range(num_tasks)) <= 1)

    # Objective: Minimize the total cost (sum of hourly rates for assigned translators).
    total_cost = sum(x[(i, j)] * translators[j][HOURLY_RATE] 
                     for i in range(num_tasks) for j in range(num_translators))
    model.Minimize(total_cost)

    # Solve the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Assignment of Translators to Tasks:")
        for i in range(num_tasks):
            for j in range(num_translators):
                if solver.BooleanValue(x[(i, j)]):
                    print(f"  Task '{tasks[i][TASK_ID]}' is assigned to translator '{translators[j][TRANSLATOR]}' "
                          f"with cost {translators[j][HOURLY_RATE]}")
        print(f"Total Cost: {solver.ObjectiveValue()}\n")

        """
        # Print a decision tree for each task.
        print("Decision Tree for each task and translator:")
        for i, task in enumerate(tasks):
            print(f"\nTask: {task[TASK_ID]} (Source: {task[SOURCE_LANG]}, Target: {task[TARGET_LANG]}, Max Rate: {task[SELLING_HOURLY_PRICE]})")
            for j, translator in enumerate(translators):
                print(f"  Translator: {translator[TRANSLATOR]}")
                # Evaluate each condition.
                source_match = translator[SOURCE_LANG] == task[SOURCE_LANG]
                target_match = translator[TARGET_LANG] == task[TARGET_LANG]
                rate_match = translator[HOURLY_RATE] <= task[SELLING_HOURLY_PRICE]
                print(f"    ├─ Source Match: {translator[SOURCE_LANG]} == {task[SOURCE_LANG]} -> {source_match}")
                print(f"    ├─ Target Match: {translator[TARGET_LANG]} == {task[TARGET_LANG]} -> {target_match}")
                print(f"    └─ Rate Check: {translator[HOURLY_RATE]} <= {task[SELLING_HOURLY_PRICE]} -> {rate_match}")
                if solver.BooleanValue(x[(i, j)]):
                    print("       => Selected for task")
        """
    else:
        print("No feasible assignment found.")

if __name__ == '__main__':
    main()
