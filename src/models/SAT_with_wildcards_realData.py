#WIP Still doesn't take translators from the data (completely)
#TODO Fully finalize taking translators from the data
#TODO Make it so the SAT assigns the tasks it can, even if some can't be asigned, right now if 1 can't be assigned it won't assing any
from ortools.sat.python import cp_model
import pandas as pd
import os
from abc import ABC
# Optionally import graphviz for generating decision trees.
try:
    from graphviz import Digraph
    graphviz_available = True
except ImportError:
    graphviz_available = False

class Wildcards(ABC):
    HOURLY_RATE="Price"
    MIN_QUALITY="Quality"
    DEADLINE="Deadline"
class Column_Names(ABC):
    #TASKS
    PROJECT_ID="PROJECT_ID"
    TASK_ID="TASK_ID"
    SOURCE_LANG="SOURCE_LANG"
    TARGET_LANG="TARGET_LANG"
    CLIENT_HOURLY_PRICE="CLIENT_HOURLY_PRICE"
    CLIENT_MIN_QUALITY="CLIENT_MIN_QUALITY"
    CLIENT_WILDCARD="CLIENT_WILDCARD"
    MANUFACTURER="MANUFACTURER"
    MANUFACTURER_SECTOR="MANUFACTURER_SECTOR"
    MANUFACTURER_INDUSTRY_GROUP="MANUFACTURER_INDUSTRY_GROUP"
    MANUFACTURER_INDUSTRY="MANUFACTURER_INDUSTRY"
    MANUFACTURER_SUBINDUSTRY="MANUFACTURER_SUBINDUSTRY"
    TASK_TYPE="TASK_TYPE"

    #TRANSLATORS
    TRANSLATOR="TRANSLATOR" 
    PAIR_KEY="PAIR_KEY"
    TRANSLATOR_HOURLY_RATE_LATEST="TRANSLATOR_HOURLY_RATE_LATEST"

class Globals(ABC):
    PATH_BASE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PATH_MERGE_CSV=os.path.join(PATH_BASE,"data","interim","mergedALL.csv")
    DF_MERGED_DATA=pd.read_csv(PATH_MERGE_CSV,low_memory=False)
    

    TASKS_DATASTRUCTURE=[
                Column_Names.PROJECT_ID,Column_Names.TASK_ID,   # Both "needed" to identify each row
                Column_Names.SOURCE_LANG,                       # Mandatory; cannot be wildcarded
                Column_Names.TARGET_LANG,                       # Mandatory; cannot be wildcarded
                Column_Names.CLIENT_HOURLY_PRICE,               # Maximum permitted rate for this task
                Column_Names.CLIENT_MIN_QUALITY,
                Column_Names.CLIENT_WILDCARD,                   # Wildcard: this attribute will not be enforced as hard constraints.
                                                                # It will still contribute to the final score—but with lower weight if needed.
                Column_Names.MANUFACTURER,
                Column_Names.MANUFACTURER_SECTOR,
                Column_Names.MANUFACTURER_INDUSTRY_GROUP,
                Column_Names.MANUFACTURER_INDUSTRY,
                Column_Names.MANUFACTURER_SUBINDUSTRY,
                Column_Names.TASK_TYPE]
    
    TRANSLATORS_DATASTRUCTURE=[
                Column_Names.TRANSLATOR,                   # Name
                Column_Names.PAIR_KEY,                     # Translator Language Pairs
                Column_Names.TRANSLATOR_HOURLY_RATE_LATEST # Current Translator Hourly Rate
                #TODO No average quality column?            
                #TODO Specialities columns            
                #TODO Task_Type Columns            
                             ]
    
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
    W_MANUFACTURER = 1   # Manufacturer proficiency weight
    W_TASK_TYPE = 2      # Task type proficiency weight (we really want people that have the correct proficiency)
    W_RATE = 10          # Weight for cost factor when RATE is enforced
    W_RATE_WILD = 2      # Lower weight when RATE is wildcarded




def get_tasks():
    """
    Get the historic list of tasks with only the useful information
    Simple method, exists only to standardize procedure
    returns(dataframe): DF of tasks with the columns listed on Globals.TASKS_DATASTRUCTURE 
    """ 
    return Globals.DF_MERGED_DATA[Globals.TASKS_DATASTRUCTURE].iloc[0:2]#TODO Remove iloc, used for testing

def get_translators():
    df_translators=Globals.DF_MERGED_DATA[Globals.TRANSLATORS_DATASTRUCTURE]

    #Remove duplicates using "PAIR_KEY" as the identifyer
    df_translators = df_translators.drop_duplicates(subset=[Column_Names.PAIR_KEY], keep="first")

    return df_translators

def main():

    tasks=get_tasks()
    # === Define Translators ===
    #
    # Each translator specifies:
    # - Source and target languages.
    # - Their hourly rate and quality.
    # - Historical counts in manufacturer specialties (a dictionary for each dimension).
    # - Historical counts in task types.
    translators = [
        {
            Column_Names.TRANSLATOR: "Aaron",
            Column_Names.PAIR_KEY:("Aaron","English","Spanish (LA)"),
            Column_Names.TRANSLATOR_HOURLY_RATE_LATEST: 24,
            "quality": 9,
            "manufacturer_specialties": {"SectorA": 8, "GroupA": 10, "IndustryA": 6, "SubindustryA": 4},
            "task_types": {"Translation": 5, "ProofReading": 1, "DTP": 0}
        },
        {
            Column_Names.TRANSLATOR: "Betty",
            Column_Names.PAIR_KEY:("Betty","English","Spanish (LA)"),
            Column_Names.TRANSLATOR_HOURLY_RATE_LATEST: 25,
            "quality": 8,
            "manufacturer_specialties": {"SectorA": 10, "GroupA": 8, "IndustryA": 7, "SubindustryA": 5},
            "task_types": {"Translation": 12, "ProofReading": 1, "DTP": 0}
        },
        {
            Column_Names.TRANSLATOR: "Charlie",
            Column_Names.PAIR_KEY:("Charlie","English","French"),
            Column_Names.TRANSLATOR_HOURLY_RATE_LATEST: 25,
            "quality": 7,
            "manufacturer_specialties": {"SectorB": 7, "GroupB": 6, "IndustryB": 4, "SubindustryB": 3},
            "task_types": {"ProofReading": 8, "Translation": 5, "Engineering": 4}
        },
        {
            Column_Names.TRANSLATOR: "David",
            Column_Names.PAIR_KEY:("David","English","French"),
            Column_Names.TRANSLATOR_HOURLY_RATE_LATEST: 23,
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
        task = tasks.iloc[i]
        task.drop(columns=[0]) #Drop newly created index column
        wildcard = task[Column_Names.CLIENT_WILDCARD]
        for j in range(num_translators):
            var = model.NewBoolVar(f"x_{i}_{j}")
            x[(i, j)] = var

            translator = translators[j]

            # --- Mandatory: Language Matching (always enforced) ---
            #Index 0 of PAIR_KEY is the translator, 1 is the source, 2 is the target
            if translator[Column_Names.PAIR_KEY][1] != task[Column_Names.SOURCE_LANG]:
                model.Add(var == 0)
            if translator[Column_Names.PAIR_KEY][2] != task[Column_Names.TARGET_LANG]:
                model.Add(var == 0)

            # --- Rate Constraint ---
            # Enforce rate only if Hourly Rate is NOT the wildcard.
            if wildcard is not Wildcards.HOURLY_RATE:
                if translator[Column_Names.TRANSLATOR_HOURLY_RATE_LATEST] > task[Column_Names.CLIENT_HOURLY_PRICE]:
                    model.Add(var == 0)
            # --- Quality Constraint ---
            # Enforce rate only if Mininum Quality is NOT the wildcard.
            elif wildcard is not Wildcards.MIN_QUALITY:
                #TODO Translator quality
                if translator["quality"] < task[Column_Names.CLIENT_MIN_QUALITY]:
                    model.Add(var == 0)
            # Note: Manufacturer specialties and TaskType are used for scoring only.

    # === Assignment Constraints ===
    # Each task gets exactly one translator.
    for i in range(num_tasks):
        model.Add(sum(x[(i, j)] for j in range(num_translators)) == 1)
    # Each translator is assigned to at most one task. FOR NOW
    for j in range(num_translators):
        model.Add(sum(x[(i, j)] for i in range(num_tasks)) <= 1)

   

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
    #    overall_score = (W_MANUFACTURER * norm_manufacturer) +
    #                    (W_TASK_TYPE * norm_task_type) +
    #                    (effective_cost_component)
    #
    # A higher overall score means the translator is more attractive (higher proficiency and lower cost).
    objective_terms = []
    for i in range(num_tasks):
        task = tasks.iloc[i]
        wildcard = task[Column_Names.CLIENT_WILDCARD]
        for j in range(num_translators):
            translator = translators[j]

            # ---- Manufacturer Proficiency Calculation ----
            mspec = translator.get("manufacturer_specialties", {}) #TODO Remember to get specialties from data
            total_manufacturer = sum(mspec.values())
            match_manufacturer = (mspec.get(task[Column_Names.MANUFACTURER_SECTOR], 0) +
                                  mspec.get(task[Column_Names.MANUFACTURER_INDUSTRY_GROUP], 0) +
                                  mspec.get(task[Column_Names.MANUFACTURER_INDUSTRY], 0) +
                                  mspec.get(task[Column_Names.MANUFACTURER_SUBINDUSTRY], 0))
            if wildcard == "Manufacturer":  #TODO Check if this is a possible wildcard What does this mean? Is this even possible?
                norm_manufacturer = 0
            else:
                norm_manufacturer = int(100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0

            # ---- Task Type Proficiency Calculation ----
            ttypes = translator.get("task_types", {}) #TODO Remember to get task_type specialties from data
            total_task_type = sum(ttypes.values())
            match_task_type = ttypes.get(task[Column_Names.TASK_TYPE], 0)
            if wildcard == "TaskType":
                norm_task_type = 0
            else:
                norm_task_type = int(100 * match_task_type / total_task_type) if total_task_type > 0 else 0

            # ---- Normalized Cost (Rate) Calculation ----
            # This formula produces a percentage that is higher when the translator's rate is
            # much lower than the permitted maximum.
            normalized_rate = 100 * (task[Column_Names.CLIENT_HOURLY_PRICE] - translator[Column_Names.TRANSLATOR_HOURLY_RATE_LATEST]) / task[Column_Names.CLIENT_HOURLY_PRICE]
            if wildcard == Wildcards.HOURLY_RATE:
                # When rate is wildcarded, cost still matters, but with a lower weight.
                cost_component = normalized_rate * Globals.W_RATE_WILD
            else:
                cost_component = normalized_rate * Globals.W_RATE

            # ---- Overall Score for This Candidate ----
            score_value = (Globals.W_MANUFACTURER * norm_manufacturer +
                           Globals.W_TASK_TYPE * norm_task_type +
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
            task = tasks.iloc[i]
            wildcard = task[Column_Names.CLIENT_WILDCARD]
            print(f"\nTask '{task[Column_Names.PROJECT_ID]}', '{task[Column_Names.TASK_ID]}' [Wildcard: {wildcard}] (Client: {task[Column_Names.MANUFACTURER]})")
            for j in range(num_translators):
                if solver.BooleanValue(x[(i, j)]):
                    translator = translators[j]
                    # Recompute components for reporting:
                    mspec = translator.get("manufacturer_specialties", {}) #TODO change this when you get specialties from the data
                    total_manufacturer = sum(mspec.values())
                    match_manufacturer = (mspec.get(task[Column_Names.MANUFACTURER_SECTOR], 0) +
                                          mspec.get(task[Column_Names.MANUFACTURER_INDUSTRY_GROUP], 0) +
                                          mspec.get(task[Column_Names.MANUFACTURER_INDUSTRY], 0) +
                                          mspec.get(task[Column_Names.MANUFACTURER_SUBINDUSTRY], 0))
                    if wildcard == "Manufacturer":
                        norm_manufacturer = 0
                    else:
                        norm_manufacturer = int(100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0

                    ttypes = translator.get("task_types", {})
                    total_task_type = sum(ttypes.values())
                    match_task_type = ttypes.get(task[Column_Names.TASK_TYPE], 0)
                    if wildcard == "TaskType":
                        norm_task_type = 0
                    else:
                        norm_task_type = int(100 * match_task_type / total_task_type) if total_task_type > 0 else 0

                    normalized_rate = 100 * (task[Column_Names.CLIENT_HOURLY_PRICE] - translator[Column_Names.TRANSLATOR_HOURLY_RATE_LATEST]) / task[Column_Names.CLIENT_HOURLY_PRICE]
                    if wildcard == Wildcards.HOURLY_RATE:
                        cost_component = normalized_rate * Globals.W_RATE_WILD
                    else:
                        cost_component = normalized_rate * Globals.W_RATE

                    overall_score = (Globals.W_MANUFACTURER * norm_manufacturer +
                                     Globals.W_TASK_TYPE * norm_task_type +
                                     cost_component)
                    
                    print(f"  Selected Translator: {translator[Column_Names.TRANSLATOR]}")
                    print(f"    Rate: {translator[Column_Names.TRANSLATOR_HOURLY_RATE_LATEST]} (Max Permitted: {task[Column_Names.CLIENT_HOURLY_PRICE]}, Normalized Cost: {normalized_rate:.2f}%, Weighted Component: {cost_component:.2f})")
                    print(f"    Quality: {translator['quality']} (Enforced: {wildcard is not Wildcards.MIN_QUALITY})") #TODO change 'quality' when you get it from the data
                    print(f"    Normalized Manufacturer Proficiency: {norm_manufacturer}%")
                    print(f"    Normalized Task Type Proficiency: {norm_task_type}%")
                    print(f"    Overall Score: {overall_score}")
        print(f"\nTotal Objective Value: {solver.ObjectiveValue()}\n")
    else:
        print("No feasible assignment found.")

if __name__ == '__main__':
    main()
