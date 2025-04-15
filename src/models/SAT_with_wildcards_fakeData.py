from ortools.sat.python import cp_model

# Optionally import graphviz for generating decision trees.
try:
    from graphviz import Digraph
    graphviz_available = True
except ImportError:
    graphviz_available = False

def main():
    # === Define Tasks ===
    #
    # Each task must always specify source and target languages.
    # For every other attribute (rate, quality, manufacturer specialties, task type),
    # a task may specify a list of wildcards (the attribute names) to relax that requirement.
    # Here, for example, Task1 relaxes the quality requirement.
    tasks = [
        {
            "name": "Task1",
            "client_name": "Accesstra",
            "source": "English",         # Mandatory; cannot be wildcarded
            "target": "Basque",          # Mandatory; cannot be wildcarded
            "selling_hourly_price": 25,  # Maximum permitted rate for this task
            "min_quality": 7,
            # Wildcards: these attributes will not be enforced as hard constraints.
            # They still contribute to the final score—but with lower weight if needed.
            "wildcard": ["Quality"],     # Could also include "Rate", "Manufacturer", "TaskType"
            # Manufacturer specialty dimensions:
            "MANUFACTURER_SECTOR": "SectorA",
            "MANUFACTURER_INDUSTRY_GROUP": "GroupA",
            "MANUFACTURER_INDUSTRY": "IndustryA",
            "MANUFACTURER_SUBINDUSTRY": "SubindustryA",
            # Task type:
            "TASK_TYPE": "Translation"
        },
        {
            "name": "Task2",
            "client_name": "OtherClient",
            "source": "English",
            "target": "French",
            "selling_hourly_price": 28,
            "min_quality": 7,
            # No wildcards: all requirements (including Quality and Rate) must be met
            "wildcard": [],
            # Manufacturer specialty dimensions:
            "MANUFACTURER_SECTOR": "SectorB",
            "MANUFACTURER_INDUSTRY_GROUP": "GroupB",
            "MANUFACTURER_INDUSTRY": "IndustryB",
            "MANUFACTURER_SUBINDUSTRY": "SubindustryB",
            # Task type:
            "TASK_TYPE": "ProofReading"
        },
    ]

    # === Define Translators ===
    #
    # Each translator specifies:
    # - Source and target languages.
    # - Their hourly rate and quality.
    # - Historical counts in manufacturer specialties (a dictionary for each dimension).
    # - Historical counts in task types.
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
    # x[i,j] is a boolean variable equal to 1 if translator j is assigned to task i.
    x = {}
    for i in range(num_tasks):
        task = tasks[i]
        # Normalize wildcards: ensure it's a list.
        wildcards = task.get("wildcard", [])
        if isinstance(wildcards, str):
            wildcards = [wildcards] if wildcards != "" else []
        for j in range(num_translators):
            var = model.NewBoolVar(f"x_{i}_{j}")
            x[(i, j)] = var

            translator = translators[j]
            # --- Mandatory: Language Matching (always enforced) ---
            if translator["source"] != task["source"]:
                model.Add(var == 0)
            if translator["target"] != task["target"]:
                model.Add(var == 0)

            # --- Rate Constraint ---
            # Enforce rate only if "Rate" is NOT wildcarded.
            if "Rate" not in wildcards:
                if translator["rate"] > task["selling_hourly_price"]:
                    model.Add(var == 0)
            # --- Quality Constraint ---
            # Enforce quality only if "Quality" is NOT wildcarded.
            if "Quality" not in wildcards:
                if translator["quality"] < task["min_quality"]:
                    model.Add(var == 0)
            # Note: Manufacturer specialties and TaskType are used for scoring only.

    # === Assignment Constraints ===
    # Each task gets exactly one translator.
    for i in range(num_tasks):
        model.Add(sum(x[(i, j)] for j in range(num_translators)) == 1)
    # Each translator is assigned to at most one task.
    for j in range(num_translators):
        model.Add(sum(x[(i, j)] for i in range(num_tasks)) <= 1)

    # === Define Weights for Scoring Factors ===
    #
    # We want all three factors (manufacturer proficiency, task type proficiency, and cost) to have roughly equal impact on the final score.
    # For manufacturer and task type, we are already working with percentages (0–100).
    #
    # For the cost factor, we first compute a normalized cost percentage: normalized_rate = 100 * (selling_hourly_price - translator_rate) / selling_hourly_price so that a translator charging the full price gets 0 and a lower rate gives a positive value.
    #
    # We then multiply this percentage by a weight. Here we use:
    #
    #   - W_rate when the cost factor is enforced normally.
    #   - W_rate_wild when "Rate" is wildcarded (thus cost matters less).
    #
    W_manufacturer = 1   # Manufacturer proficiency weight
    W_task_type = 2      # Task type proficiency weight (we really want people that have the correct proficiency)
    W_rate = 10          # Weight for cost factor when RATE is enforced
    W_rate_wild = 2      # Lower weight when RATE is wildcarded

    # === Build the Objective Function (to be maximized) ===
    #
    # For each translator-task candidate, we compute:
    #
    # 1. **Normalized Manufacturer Proficiency (norm_manufacturer):**
    #      = 100 * (sum of matching manufacturer counts) / (total manufacturer counts)
    #      If "Manufacturer" is wildcarded, we set it to 0.
    #
    # 2. **Normalized Task Type Proficiency (norm_task_type):**
    #      = 100 * (count for the task's TASK_TYPE) / (total task type counts)
    #      If "TaskType" is wildcarded, we set it to 0.
    #
    # 3. **Normalized Cost (normalized_rate):**
    #      = 100 * (selling_hourly_price - translator_rate) / selling_hourly_price
    #      This gives a higher percentage for lower rates.
    #      Then, we multiply by W_rate if "Rate" is not wildcarded,
    #      or by W_rate_wild if "Rate" is wildcarded.
    #
    # The overall score is:
    #
    #    overall_score = (W_manufacturer * norm_manufacturer) +
    #                    (W_task_type * norm_task_type) +
    #                    (effective_cost_component)
    #
    # A higher overall score means the translator is more attractive (higher proficiency and lower cost).
    objective_terms = []
    for i in range(num_tasks):
        task = tasks[i]
        wildcards = task.get("wildcard", [])
        if isinstance(wildcards, str):
            wildcards = [wildcards] if wildcards != "" else []
        for j in range(num_translators):
            translator = translators[j]

            # ---- Manufacturer Proficiency Calculation ----
            mspec = translator.get("manufacturer_specialties", {})
            total_manufacturer = sum(mspec.values())
            match_manufacturer = (mspec.get(task["MANUFACTURER_SECTOR"], 0) +
                                  mspec.get(task["MANUFACTURER_INDUSTRY_GROUP"], 0) +
                                  mspec.get(task["MANUFACTURER_INDUSTRY"], 0) +
                                  mspec.get(task["MANUFACTURER_SUBINDUSTRY"], 0))
            if "Manufacturer" in wildcards:
                norm_manufacturer = 0
            else:
                norm_manufacturer = int(100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0

            # ---- Task Type Proficiency Calculation ----
            ttypes = translator.get("task_types", {})
            total_task_type = sum(ttypes.values())
            match_task_type = ttypes.get(task["TASK_TYPE"], 0)
            if "TaskType" in wildcards:
                norm_task_type = 0
            else:
                norm_task_type = int(100 * match_task_type / total_task_type) if total_task_type > 0 else 0

            # ---- Normalized Cost (Rate) Calculation ----
            # This formula produces a percentage that is higher when the translator's rate is
            # much lower than the permitted maximum.
            normalized_rate = 100 * (task["selling_hourly_price"] - translator["rate"]) / task["selling_hourly_price"]
            if "Rate" in wildcards:
                # When rate is wildcarded, cost still matters, but with a lower weight.
                cost_component = normalized_rate * W_rate_wild
            else:
                cost_component = normalized_rate * W_rate

            # ---- Overall Score for This Candidate ----
            score_value = (W_manufacturer * norm_manufacturer +
                           W_task_type * norm_task_type +
                           cost_component)
            objective_terms.append(x[(i, j)] * score_value)

    # Maximize the total score for all assigned translator–task pairs.
    model.Maximize(sum(objective_terms))

    # === Solve the Model ===
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("Assignment of Translators to Tasks:")
        for i in range(num_tasks):
            task = tasks[i]
            wildcards = task.get("wildcard", [])
            if isinstance(wildcards, str):
                wildcards = [wildcards] if wildcards != "" else []
            print(f"\nTask '{task['name']}' [Wildcards: {wildcards}] (Client: {task['client_name']})")
            for j in range(num_translators):
                if solver.BooleanValue(x[(i, j)]):
                    translator = translators[j]
                    # Recompute components for reporting:
                    mspec = translator.get("manufacturer_specialties", {})
                    total_manufacturer = sum(mspec.values())
                    match_manufacturer = (mspec.get(task["MANUFACTURER_SECTOR"], 0) +
                                          mspec.get(task["MANUFACTURER_INDUSTRY_GROUP"], 0) +
                                          mspec.get(task["MANUFACTURER_INDUSTRY"], 0) +
                                          mspec.get(task["MANUFACTURER_SUBINDUSTRY"], 0))
                    if "Manufacturer" in wildcards:
                        norm_manufacturer = 0
                    else:
                        norm_manufacturer = int(100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0

                    ttypes = translator.get("task_types", {})
                    total_task_type = sum(ttypes.values())
                    match_task_type = ttypes.get(task["TASK_TYPE"], 0)
                    if "TaskType" in wildcards:
                        norm_task_type = 0
                    else:
                        norm_task_type = int(100 * match_task_type / total_task_type) if total_task_type > 0 else 0

                    normalized_rate = 100 * (task["selling_hourly_price"] - translator["rate"]) / task["selling_hourly_price"]
                    if "Rate" in wildcards:
                        cost_component = normalized_rate * W_rate_wild
                    else:
                        cost_component = normalized_rate * W_rate

                    overall_score = (W_manufacturer * norm_manufacturer +
                                     W_task_type * norm_task_type +
                                     cost_component)
                    
                    print(f"  Selected Translator: {translator['name']}")
                    print(f"    Rate: {translator['rate']} (Max Permitted: {task['selling_hourly_price']}, Normalized Cost: {normalized_rate:.2f}%, Weighted Component: {cost_component:.2f})")
                    print(f"    Quality: {translator['quality']} (Enforced: {'Quality' not in wildcards})")
                    print(f"    Normalized Manufacturer Proficiency: {norm_manufacturer}%")
                    print(f"    Normalized Task Type Proficiency: {norm_task_type}%")
                    print(f"    Overall Score: {overall_score}")
        print(f"\nTotal Objective Value: {solver.ObjectiveValue()}\n")
    else:
        print("No feasible assignment found.")

if __name__ == '__main__':
    main()
