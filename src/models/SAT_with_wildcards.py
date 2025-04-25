from ortools.sat.python import cp_model

# Optionally import graphviz for generating decision trees.
try:
    from graphviz import Digraph
    graphviz_available = True
except ImportError:
    graphviz_available = False

def main():
    # === Define Tasks ===
    tasks = [
        {
            "name": "Task1",
            "client_name": "Accesstra",
            "source": "English",
            "target": "Basque",
            "selling_hourly_price": 25,
            "min_quality": 7,
            "wildcard": ["Quality"],
            "MANUFACTURER_SECTOR": "SectorA",
            "MANUFACTURER_INDUSTRY_GROUP": "GroupA",
            "MANUFACTURER_INDUSTRY": "IndustryA",
            "MANUFACTURER_SUBINDUSTRY": "SubindustryA",
            "TASK_TYPE": "Translation"
        },
        {
            "name": "Task2",
            "client_name": "OtherClient",
            "source": "English",
            "target": "French",
            "selling_hourly_price": 28,
            "min_quality": 7,
            "wildcard": [],
            "MANUFACTURER_SECTOR": "SectorB",
            "MANUFACTURER_INDUSTRY_GROUP": "GroupB",
            "MANUFACTURER_INDUSTRY": "IndustryB",
            "MANUFACTURER_SUBINDUSTRY": "SubindustryB",
            "TASK_TYPE": "ProofReading"
        },
    ]

    # === Define Translators ===
    translators = [
        {
            "name": "Aaron",
            "source": "English",
            "target": "Basque",
            "rate": 24,
            "quality": 9,
            "manufacturer_specialties": {"SectorA": 8, "GroupA": 10, "IndustryA": 6, "SubindustryA": 4},
            "task_types": {"Translation": 5, "ProofReading": 1, "DTP": 0}
        },
        {
            "name": "Betty",
            "source": "English",
            "target": "Basque",
            "rate": 25,
            "quality": 8,
            "manufacturer_specialties": {"SectorA": 10, "GroupA": 8, "IndustryA": 7, "SubindustryA": 5},
            "task_types": {"Translation": 12, "ProofReading": 1, "DTP": 0}
        },
        {
            "name": "Charlie",
            "source": "English",
            "target": "French",
            "rate": 25,
            "quality": 7,
            "manufacturer_specialties": {"SectorB": 7, "GroupB": 6, "IndustryB": 4, "SubindustryB": 3},
            "task_types": {"ProofReading": 8, "Translation": 5, "Engineering": 4}
        },
        {
            "name": "David",
            "source": "English",
            "target": "French",
            "rate": 23,
            "quality": 7,
            "manufacturer_specialties": {"SectorB": 5, "GroupB": 4, "IndustryB": 3, "SubindustryB": 2},
            "task_types": {"ProofReading": 10, "Translation": 6, "Engineering": 3}
        },
    ]

    # Create a CP-SAT Model.
    model = cp_model.CpModel()
    num_tasks = len(tasks)
    num_translators = len(translators)
    
    # === Create Decision Variables ===
    x = {}
    for i in range(num_tasks):
        task = tasks[i]
        wildcards = task.get("wildcard", [])
        if isinstance(wildcards, str):
            wildcards = [wildcards] if wildcards != "" else []
        for j in range(num_translators):
            var = model.NewBoolVar(f"x_{i}_{j}")
            x[(i, j)] = var

            translator = translators[j]
            # Mandatory: Language Matching (always enforced)
            if translator["source"] != task["source"]:
                model.Add(var == 0)
            if translator["target"] != task["target"]:
                model.Add(var == 0)

            # Rate Constraint
            if "Rate" not in wildcards and translator["rate"] > task["selling_hourly_price"]:
                model.Add(var == 0)
            # Quality Constraint
            if "Quality" not in wildcards and translator["quality"] < task["min_quality"]:
                model.Add(var == 0)

    # Each task gets at most one translator (modified to allow unassigned tasks).
    for i in range(num_tasks):
        model.Add(sum(x[(i, j)] for j in range(num_translators)) <= 1)
    # Each translator is assigned to at most one task.
    for j in range(num_translators):
        model.Add(sum(x[(i, j)] for i in range(num_tasks)) <= 1)

    # Weights for scoring
    W_manufacturer = 1
    W_task_type = 2
    W_rate = 10
    W_rate_wild = 2

    # Build objective with a scores dict for fallback
    objective_terms = []
    scores = {}
    for i in range(num_tasks):
        task = tasks[i]
        wildcards = task.get("wildcard", [])
        if isinstance(wildcards, str):
            wildcards = [wildcards] if wildcards != "" else []
        for j in range(num_translators):
            translator = translators[j]
            # Manufacturer
            mspec = translator.get("manufacturer_specialties", {})
            total_manufacturer = sum(mspec.values())
            match_manufacturer = sum(mspec.get(dim, 0) for dim in [task["MANUFACTURER_SECTOR"],
                                                                    task["MANUFACTURER_INDUSTRY_GROUP"],
                                                                    task["MANUFACTURER_INDUSTRY"],
                                                                    task["MANUFACTURER_SUBINDUSTRY"]])
            norm_manufacturer = 0 if "Manufacturer" in wildcards else (int(100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0)
            # TaskType
            ttypes = translator.get("task_types", {})
            total_task_type = sum(ttypes.values())
            match_task_type = ttypes.get(task["TASK_TYPE"], 0)
            norm_task_type = 0 if "TaskType" in wildcards else (int(100 * match_task_type / total_task_type) if total_task_type > 0 else 0)
            # Cost
            normalized_rate = 100 * (task["selling_hourly_price"] - translator["rate"]) / task["selling_hourly_price"]
            cost_component = normalized_rate * (W_rate_wild if "Rate" in wildcards else W_rate)

            score_value = W_manufacturer * norm_manufacturer + W_task_type * norm_task_type + cost_component
            scores[(i, j)] = score_value
            objective_terms.append(x[(i, j)] * score_value)

    model.Maximize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    print("Assignment of Translators to Tasks:")
    for i in range(num_tasks):
        task = tasks[i]
        wildcards = task.get("wildcard", [])
        if isinstance(wildcards, str):
            wildcards = [wildcards] if wildcards != "" else []
        print(f"\nTask '{task['name']}' [Wildcards: {wildcards}] (Client: {task['client_name']})")
        # Check assignment
        assigned = None
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for j in range(num_translators):
                if solver.BooleanValue(x[(i, j)]):
                    assigned = j
                    break
        if assigned is not None:
            # Print selected translator details
            translator = translators[assigned]
            norm_rate = 100 * (task["selling_hourly_price"] - translator["rate"]) / task["selling_hourly_price"]
            print(f"  Selected Translator: {translator['name']}")
            print(f"    Rate: {translator['rate']} (Max: {task['selling_hourly_price']}, Normalized Cost: {norm_rate:.2f}%)")
            print(f"    Quality: {translator['quality']}")
            print(f"    Score: {scores[(i, assigned)]:.2f}")
        else:
            # Fallback: best-scoring candidate by language match
            lang_candidates = [j for j in range(num_translators)
                               if translators[j]["source"] == task["source"] and translators[j]["target"] == task["target"]]
            if lang_candidates:
                best_j = max(lang_candidates, key=lambda j: scores[(i, j)])
                t = translators[best_j]
                print("  No valid translator satisfies all hard constraints for this task.")
                print(f"  Next best candidate: {t['name']}")
                print(f"    Score: {scores[(i, best_j)]:.2f}")
                print(f"    Rate: {t['rate']} (Normalized Cost: {100 * (task['selling_hourly_price'] - t['rate'])/task['selling_hourly_price']:.2f}%)")
                print(f"    Quality: {t['quality']}")
            else:
                print("  No translator found for this task.")

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\nTotal Objective Value: {solver.ObjectiveValue():.2f}\n")

if __name__ == '__main__':
    main()
