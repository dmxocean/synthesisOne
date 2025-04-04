from ortools.sat.python import cp_model

def main():
    # Define multiple tasks.
    # Each task requires a translator with a matching source language, target language,
    # and an hourly rate no higher than the task's max_rate.
    tasks = [
        {"name": "Task1", "source": "English", "target": "Basque", "max_rate": 30},
        {"name": "Task2", "source": "English", "target": "French", "max_rate": 28},
        # Add more tasks as needed.
    ]

    # List of translators.
    translators = [
        {"name": "Aaron",   "source": "English", "target": "Basque",  "rate": 27},
        {"name": "Betty",   "source": "English", "target": "Basque",  "rate": 25},
        {"name": "Charlie", "source": "English", "target": "French",  "rate": 25},
        {"name": "David",   "source": "Spanish", "target": "Basque",  "rate": 20},
    ]

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
            if (translators[j]["source"] != tasks[i]["source"] or 
                translators[j]["target"] != tasks[i]["target"] or 
                translators[j]["rate"] > tasks[i]["max_rate"]):
                model.Add(var == 0)

    # Each task must be assigned exactly one translator.
    for i in range(num_tasks):
        model.Add(sum(x[(i, j)] for j in range(num_translators)) == 1)

    # Each translator can only be assigned to at most one task.
    for j in range(num_translators):
        model.Add(sum(x[(i, j)] for i in range(num_tasks)) <= 1)

    # Objective: Minimize the total cost (sum of hourly rates for assigned translators).
    total_cost = sum(x[(i, j)] * translators[j]["rate"] 
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
                    print(f"  Task '{tasks[i]['name']}' is assigned to translator '{translators[j]['name']}' "
                          f"with cost {translators[j]['rate']}")
        print(f"Total Cost: {solver.ObjectiveValue()}\n")

        # Print a decision tree for each task.
        print("Decision Tree for each task and translator:")
        for i, task in enumerate(tasks):
            print(f"\nTask: {task['name']} (Source: {task['source']}, Target: {task['target']}, Max Rate: {task['max_rate']})")
            for j, translator in enumerate(translators):
                print(f"  Translator: {translator['name']}")
                # Evaluate each condition.
                source_match = translator["source"] == task["source"]
                target_match = translator["target"] == task["target"]
                rate_match = translator["rate"] <= task["max_rate"]
                print(f"    ├─ Source Match: {translator['source']} == {task['source']} -> {source_match}")
                print(f"    ├─ Target Match: {translator['target']} == {task['target']} -> {target_match}")
                print(f"    └─ Rate Check: {translator['rate']} <= {task['max_rate']} -> {rate_match}")
                if solver.BooleanValue(x[(i, j)]):
                    print("       => Selected for task")
    else:
        print("No feasible assignment found.")

if __name__ == '__main__':
    main()