from typing import Dict, List, Optional, Tuple
import pandas as pd
import sys
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).resolve().parents[2]  # Go up two levels from current file
sys.path.insert(0, str(project_root))
from src.data.json_queries import UnifiedQueryHandler  

#TODO CENTRALISE THE CONSTANTS PROJECT-WISE
WC_DEADLINE="Deadline"
WC_PRICE="Price"
WC_QUALITY="Quality"

COL_PROJECT_ID="PROJECT_ID"
COL_TASK_ID="TASK_ID"
COL_TASK_TYPE="TASK_TYPE"
COL_SOURCE_LANG="SOURCE_LANG"
COL_TARGET_LANG="TARGET_LANG"
COL_TRANSLATOR="TRANSLATOR"
COL_HOURLY_RATE="HOURLY_RATE"
COL_MANUFACTURER="MANUFACTURER"
COL_MANUFACTURER_SECTOR="MANUFACTURER_SECTOR"
COL_WILDCARD="WILDCARD"

MODEL_FLAG                  = "SAT"
COL_MODEL_FLAG              = "MODEL_FLAG"
COL_TOTAL_SCORE             = MODEL_FLAG+"_TOTAL_SCORE"
COL_SPECIALISATION_SCORE    = MODEL_FLAG+"_SPECIALISATION_SCORE"
COL_PRICE_SCORE             = MODEL_FLAG+"_PRICE_SCORE"
COL_QUALITY_SCORE           = MODEL_FLAG+"_QUALITY_SCORE"
COL_IS_WILDCARD_COMPATIBLE  = MODEL_FLAG+"_IS_WILDCARD_COMPATIBLE"
COL_IS_PRICE_COMPATIBLE     = MODEL_FLAG+"_IS_PRICE_COMPATIBLE"
COL_IS_QUALITY_COMPATIBLE   = MODEL_FLAG+"_IS_QUALITY_COMPATIBLE"


SORTING_TRANSLATOR_PRIORITIES=[COL_TOTAL_SCORE,
                               COL_SPECIALISATION_SCORE,
                               COL_IS_PRICE_COMPATIBLE,
                               COL_IS_QUALITY_COMPATIBLE,
                               COL_IS_WILDCARD_COMPATIBLE,
                               COL_HOURLY_RATE]
# Scoring Weights (Adjust if needed)
W_MANUFACTURER = 60
W_TASK_TYPE = 2 #UNUSED FOR NOW
W_RATE = 190
W_RATE_WILD = 1 # Lower weight when RATE is wildcarded
W_QUALITY = 1
W_QUALITY_WILD = 1

QUERY=UnifiedQueryHandler()
MAX_TRANSLATORS=1000
class ErrorArgs():
    ID_NON_EXISTANT_CLIENT=1
    MSG_NON_EXISTANT_CLIENT=" manufacturer is not in the database. Please input them before running any of their tasks. " \
    "Skipping to next task"
    VALUE_NON_EXISTANT_CLIENT=pd.DataFrame()

    ID_MISSING_TREANSLATOR_INFO=2
    MSG_MISSING_TREANSLATOR_INFO=" translator's information is not in the database. Please input it before running any of their tasks. "\
    "Skipping to next Translator"

class Task():
    def __init__(self, original_task: pd.Series,  df_translators: Optional[pd.DataFrame]=None):
        self.unique_id=original_task[COL_PROJECT_ID]+ " "+  original_task[COL_TASK_ID]

        self.original_task = original_task
        self.project_id=original_task[COL_PROJECT_ID]
        self.task_id = original_task[COL_TASK_ID]
        #self.task_type = original_task[COL_TASK_TYPE]
        self.source_lang = original_task[COL_SOURCE_LANG]
        self.target_lang = original_task[COL_TARGET_LANG]
        self.manufacturer = original_task[COL_MANUFACTURER]
        self.manufacturer_sector = original_task[COL_MANUFACTURER_SECTOR]

        if df_translators is None:
            self.df_translators = pd.DataFrame(columns=original_task.index.to_list()+SORTING_TRANSLATOR_PRIORITIES)
        else:
            self.df_translators = df_translators


    def assign_shared_columns(self):
        self.df_translators=self.df_translators.assign(
                                                    PROJECT_ID=self.project_id,
                                                    TASK_ID=self.task_id,
                                                    #TASK_TYPE=self.task_type,
                                                    SOURCE_LANG=self.source_lang, 
                                                    TARGET_LANG=self.target_lang,
                                                    MANUFACTURER=self.manufacturer,
                                                    MANUFACTURER_SECTOR=self.manufacturer_sector,
                                                    MODEL_FLAG=MODEL_FLAG)
        
def run_sat_inference(df_tasks: pd.DataFrame,top_k:Optional[int]) -> Dict[str, List[Tuple[str, float]]]:
    df_tasks_safe = df_tasks.__deepcopy__()  # Defensive copy

    top_k=top_k if top_k else MAX_TRANSLATORS 
    dirty_results=ranking_Rules_Based(df_tasks_safe,top_k)
    clean_results=format_results_into_dict(dirty_results)
    return clean_results

def format_results_into_dict(list_tasks: List[Task]) -> Dict[str, List[Tuple[str, float]]]:
    # Create a dictionary to hold the results
    results = {}

    # Iterate through each row in the DataFrame
    for task in list_tasks:
        # Get the task ID and translator name
        results[task.unique_id] = []

        for _, row in task.df_translators.iterrows():
            translator_name = row[COL_TRANSLATOR]
            score = row[COL_TOTAL_SCORE]
            # Append the translator name and score to the list for this task ID
            results[task.unique_id].append((translator_name, score))

    return results


def ranking_Rules_Based(df_tasks: pd.DataFrame,top_k:int) -> list[Task]:
    ranked_outputs = []    
    for _, row in df_tasks.iterrows():
        task=Task(row)
        single_task_processing(task,top_k)
        ranked_outputs.append(task)

    
    return ranked_outputs

def single_task_processing(task: Task,top_k:int)-> None:

    language_matching_translators = QUERY.translator_efficiency.get_translators_for_language_pair(task.source_lang,task.target_lang)["translator_name"].tolist() #TODO remove magic string
    # Create an empty dataframe of the desired size pre-emtively 
    task.df_translators[COL_TRANSLATOR]=language_matching_translators
    task.assign_shared_columns()

    try:
        task.df_translators=task.df_translators.apply(single_translator_processing,axis=1)

        #nlargest would be more efficient but can't handle list of "ascending"
        task.df_translators=task.df_translators.sort_values(SORTING_TRANSLATOR_PRIORITIES,ascending=[False,False,False,False,False,True],na_position='last').head(top_k)
    except ValueError as e:
        match e.args[0]: # Match ValueError ID
            case ErrorArgs.ID_NON_EXISTANT_CLIENT:
                task.df_translators=pd.DataFrame(columns=task.df_translators.columns) #If client doesn't exist, return an empty dataframe
            case _:
                raise e
        print(e.args[1]) # Print Error Message

    
def single_translator_processing(ranked_translator:pd.Series)->pd.Series:

    (ranked_translator[COL_TOTAL_SCORE], 
    ranked_translator[COL_SPECIALISATION_SCORE],
    ranked_translator[COL_PRICE_SCORE],
    ranked_translator[COL_QUALITY_SCORE]
    ) = compute_translator_score(ranked_translator)

    add_translator_availability_inline(ranked_translator)
    add_metadata_inline(ranked_translator)
    return ranked_translator



def compute_translator_score(ranked_translator: pd.Series)-> tuple[float,float,float,float]:
    translator = ranked_translator[COL_TRANSLATOR]
    manufacturer=ranked_translator[COL_MANUFACTURER]
    source_lang=ranked_translator[COL_SOURCE_LANG]
    target_lang=ranked_translator[COL_TARGET_LANG]

    # Manufacturer Proficiency Score
    client_history=QUERY.translator_metrics.get_translator_client_history(translator) 
    client_info=QUERY.clients.get_client_info(manufacturer)
    
    if any([client_history.empty,client_info.empty]):
        raise ValueError(ErrorArgs.ID_NON_EXISTANT_CLIENT,str(manufacturer) + ErrorArgs.MSG_NON_EXISTANT_CLIENT)
    
    translator_hourly_rate = QUERY.translator_rates.get_translator_rate_for_lang_pair(translator,source_lang,target_lang) # type: ignore
    translator_quality =  QUERY.translator_efficiency.get_translator_avg_quality(translator,source_lang,target_lang) # type: ignore

    if translator_hourly_rate is None or translator_quality is None: #TODO handle propperly 
        raise ValueError(ErrorArgs.ID_MISSING_TREANSLATOR_INFO,str(translator) + ErrorArgs.MSG_MISSING_TREANSLATOR_INFO) 
    
    # I don't think there's any case where an hourly rates is passed but, just in case
    ranked_translator[COL_HOURLY_RATE]=translator_hourly_rate
    
    #TODO remove magic strings
    total_manufacturer = float(client_history["task_count"].sum())
    temp2=client_history['client_name'] ==manufacturer
    temp1=client_history[temp2]['task_count'].iloc[0] if not client_history[temp2].empty else 0
    match_manufacturer = float(temp1) if manufacturer in client_history["client_name"].tolist()  else 0.0
 
    norm_manufacturer = (100 * match_manufacturer / total_manufacturer) if total_manufacturer > 0 else 0
 
    
    
 
    # Normalized Cost (Rate) Score Component
    normalized_rate=-float('inf')
    client_rate=client_info["selling_hourly_price"][0]
    if  client_rate > 0 :
        normalized_rate = 100 * (client_rate - translator_hourly_rate) / client_rate

    # Quality Score Component
    normalized_quality=-float('inf')
    client_min_quality=client_info["min_quality"][0]
    if client_min_quality>0 :
        #1000 because quality is <=10 while price is almost always 10+
        normalized_quality= 1000 * (translator_quality - client_min_quality) / client_min_quality

    # Apply wildcard weights
    cost_component=-float('inf')
    quality_component=-float('inf')

    wildcard=client_info["wildcard"][0]
    effective_rate_weight = W_RATE_WILD if wildcard == WC_PRICE else W_RATE
    effective_quality_weight= W_QUALITY_WILD if wildcard ==WC_QUALITY else W_QUALITY

    if normalized_rate > -float('inf'):
        cost_component = normalized_rate * effective_rate_weight
    if normalized_quality > -float('inf'):
        quality_component = normalized_quality * effective_quality_weight

    # Final Score Calculation
    score_value=-float('inf')
    if cost_component > -float('inf'):
        score_value = (W_MANUFACTURER * norm_manufacturer +
                       # W_TASK_TYPE * norm_task_type + #Check if useful
                       quality_component+
                        cost_component)

    return (score_value,norm_manufacturer,cost_component,quality_component)


def add_translator_availability_inline(ranked_translator:pd.Series)->None:
    translator=ranked_translator[COL_TRANSLATOR]
    pass

def add_metadata_inline(ranked_translator:pd.Series)-> None:
    ranked_translator[COL_MODEL_FLAG]=MODEL_FLAG
    add_explainability_inline(ranked_translator)

def add_explainability_inline(ranked_translator:pd.Series)-> None:
    translator=ranked_translator[COL_TRANSLATOR]

    source_lang=ranked_translator[COL_SOURCE_LANG]
    target_lang=ranked_translator[COL_TARGET_LANG]

    translator_hourly_rate= QUERY.translator_rates.get_translator_rate_for_lang_pair(translator,source_lang,target_lang) #type: ignore
    translator_quality= QUERY.translator_efficiency.get_translator_avg_quality(translator,source_lang,target_lang)#type: ignore

    client_info=QUERY.clients.get_client_info(ranked_translator[COL_MANUFACTURER])
    #TODO Centralize magic strings
    client_max_price=      client_info["selling_hourly_price"][0]
    client_min_quality=    client_info["min_quality"][0]
    client_wildcard=       client_info["wildcard"][0]

    match_price=translator_hourly_rate<=client_max_price
    match_quality=translator_quality>=client_min_quality
    match_wildcard=None
    
    if client_wildcard == WC_DEADLINE: # Agreggator handles deadline, assume it's possible
        match_wildcard = True
    elif client_wildcard == WC_PRICE:
        match_wildcard = match_price
    elif client_wildcard == WC_QUALITY:
        match_wildcard = match_quality


    ranked_translator[COL_IS_PRICE_COMPATIBLE]    = match_price
    ranked_translator[COL_IS_QUALITY_COMPATIBLE]  = match_quality
    ranked_translator[COL_IS_WILDCARD_COMPATIBLE] = match_wildcard

def create_dummy_tasks()->pd.DataFrame:
    """Dummy tasks, First 5 tasks of the data, although last 4 are bascially copies of eachother"""
    dummy_tasks = {
        "PROJECT_ID":           ['P1', 'P2', 'P2', 'P2', 'P2'],
        "TASK_ID":              ['T1', 'T1', 'T2', 'T3', 'T4'],
        "TASK_TYPE":            ["ProofReading","Engineering","Engineering","Engineering","Engineering"],
        "SOURCE_LANG":          ['English', 'English', 'English', 'English', 'English'],
        "TARGET_LANG":          ['Spanish (LA)', 'Spanish (Iberian)', 'Spanish (Iberian)', 'Spanish (Iberian)', 'Spanish (Iberian)'],
        "MANUFACTURER":         ['Appcelerate', 'InnovateWorks', 'InnovateWorks', 'InnovateWorks', 'InnovateWorks'], 
        "MANUFACTURER_SECTOR":  ['Information Technology', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary']
    }
    
    return pd.DataFrame.from_dict(dummy_tasks)

def fake_sweep():
    """Debugging tool to asses fit. I am become gradient descent. 
    I basically use this to guess at what might be the best values for our weights, yes, that's the point we're at."""
    dummy_tasks=create_dummy_tasks()
    dummy_tasks=dummy_tasks.iloc[1].to_frame().T

    # Try different combinations of weights for W_MANUFACTURER, W_RATE, W_QUALITY
    best_so_far=[]
    lowest_diff=3

    for w_manufacturer in [60]:
        for w_rate in [190]:
            for w_rate_wild in range(1,10,1):
                for w_quality in range(1,10,1):
                        for w_quality_wild in [1]:
                            global W_MANUFACTURER, W_RATE, W_QUALITY,W_RATE_WILD, W_QUALITY_WILD
                            W_MANUFACTURER = w_manufacturer
                            W_RATE = w_rate
                            W_RATE_WILD=w_rate_wild
                            W_QUALITY = w_quality
                            W_QUALITY_WILD=w_quality_wild

                            print(f"W_MANUFACTURER={W_MANUFACTURER}, W_RATE={W_RATE}, W_RATE_WILD={W_RATE_WILD}, W_QUALITY={W_QUALITY}, W_QUALITY_WILD={W_QUALITY_WILD}")
                            infered=run_sat_inference(dummy_tasks,None).get('P2 T1')
                            cheng_score = next((score for name, score in infered if name == "Cheng"), None)
                            top5=infered[0:5]
                            difference=(float(top5[4][1])-cheng_score)/(float(top5[4][1]))
                            print("\n"+str(top5) + "\nCheng: " + str(cheng_score) +"\n Difference: " + str(difference)) 
                            if difference<lowest_diff:
                                lowest_diff=difference
                                best_so_far.append((lowest_diff,W_MANUFACTURER,W_RATE,W_RATE_WILD,W_QUALITY,W_QUALITY_WILD))
                            best_so_far.sort()
                            print("Best So Far: " +str(best_so_far[0]))

    best_so_far.sort()
    print("Best: " +str(best_so_far))

def main():
    # fake_sweep()
    print("DEBUGGING: Wrong function call / wrong file has been ran. " \
    "Running sample execution.")
    dummy_tasks=create_dummy_tasks()

    print(run_sat_inference(dummy_tasks,None))


if __name__ == '__main__':
    main()