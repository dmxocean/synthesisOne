# app/app.py

"""
Streamlit Web Application for Translator Assignment System

Interactive interface for task assignment using all models
Manages translator schedules and provides real-time availability checking

IMPORTANT: Integrates with prediction pipeline and schedule management
"""

import streamlit as st
import pandas as pd
import altair as alt
import json
import os
from statistics import mean
from datetime import datetime, timedelta
import importlib.util

# Path configuration
PATH_APP = os.path.dirname(os.path.abspath(__file__))
PATH_ROOT = os.path.dirname(PATH_APP)

# Configure paths
PATH_SRC_PREDICTION = os.path.join(PATH_ROOT, "src", "prediction")
PATH_DATA_PREDICTION = os.path.join(PATH_ROOT, "data", "prediction")
PATH_ARTIFACTS = os.path.join(PATH_ROOT, "data", "processed", "base", "artifacts")

# Load constraints.py from prediction directory
constraints_path = os.path.join(PATH_SRC_PREDICTION, "constraints.py")
spec = importlib.util.spec_from_file_location("constraints", constraints_path)
constraints = importlib.util.module_from_spec(spec)
spec.loader.exec_module(constraints)

# Import prediction and availability functions
run_prediction_pipeline = constraints.run_prediction_pipeline
get_available_translators = constraints.get_available_translators
convert_to_simple_format = constraints.convert_to_simple_format

# Configure file paths
SCHEDULE_FILE = os.path.join(PATH_DATA_PREDICTION, "translator_schedule.json")
PATH_TRANSLATOR_METRICS = os.path.join(PATH_ARTIFACTS, "translator_metrics.json")
PATH_TRANSLATOR_RATES = os.path.join(PATH_ARTIFACTS, "translator_hourly_rates.json")

# Load JSON files
with open(PATH_TRANSLATOR_RATES, encoding="utf-8") as f:
    rates = json.load(f)
with open(PATH_TRANSLATOR_METRICS, encoding="utf-8") as f:
    metrics = json.load(f)

# -- PAGE CONFIG AND CSS STYLING --
st.set_page_config(page_title="Translator Assignment", layout="wide")
st.markdown(
    """
    <style>
    .css-1v3fvcr.e1fqkh3o4 { font-size: 2rem !important; font-weight: bold !important; }
    .translator-card { background-color: #f8f9fa; border-radius: 10px; padding: 10px; margin-bottom: 8px; border: 1px solid #e3e6ea; transition: background-color 0.3s; color: #000 !important; }
    .translator-card:hover { background-color: #e9ecef; }
    .assigned-card { background-color: #FFD480 !important; border: 2px solid #FFA500 !important; color: #000 !important; }
    .task-card { background-color: #ffffff; border-radius: 8px; padding: 4px 8px; margin-bottom: 6px; border: 1px solid #d3d3d3; color: #000 !important; }
    .task-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .streamlit-expanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)


def load_schedule():
    """
    Load translator schedule from JSON file
    
    Reads the existing translator schedule from the configured JSON file,
    creating the directory structure if it doesn't exist
    
    Returns:
        dict: Dictionary mapping translator names to their assigned tasks
    """
    os.makedirs(PATH_DATA_PREDICTION, exist_ok=True)
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_schedule(schedule: dict):
    """
    Save translator schedule to JSON file
    
    Persists the current translator schedule to disk with proper formatting
    and file synchronization to ensure data integrity
    
    Args:
        schedule: Dictionary mapping translator names to their task assignments
    """
    os.makedirs(PATH_DATA_PREDICTION, exist_ok=True)
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def recompute_suggestions():
    """
    Recompute suggestions with availability filtering for unassigned tasks
    
    Updates translator recommendations by filtering out already assigned tasks
    and checking translator availability against current schedule
    """
    # Ensure we have models and an uploaded file
    if 'uploaded_file' not in st.session_state or not st.session_state.models:
        return

    # Load schedule and get assigned task IDs as strings
    sched = load_schedule()
    assigned_ids = {str(t['task_id']) for tasks in sched.values() for t in tasks}

    # Prune new_tasks DataFrame by TASK_ID strings
    st.session_state.new_tasks = (
        st.session_state.new_tasks
        .loc[~st.session_state.new_tasks['TASK_ID'].isin(assigned_ids)]
        .reset_index(drop=True)
    )

    # Remaining task IDs as strings
    remaining_ids = st.session_state.new_tasks['TASK_ID'].tolist()

    # Filter previous suggestions to only these tasks
    filtered_recs = {
        tid: recs
        for tid, recs in st.session_state.suggestions.items()
        if tid in remaining_ids
    }

    # Reload raw CSV and filter to remaining tasks
    csv_file = st.session_state.uploaded_file
    if hasattr(csv_file, 'seek'):
        csv_file.seek(0)
    # Read TASK_ID as string to maintain consistency
    df_all = pd.read_csv(csv_file, dtype={'TASK_ID': int})
    df_batch = df_all[df_all['TASK_ID'].isin(remaining_ids)]

    # Use availability checker on existing recommendations
    st.session_state.suggestions = get_available_translators(
        df_batch,
        filtered_recs,
        calendar_path=str(SCHEDULE_FILE)
    )


def assign_task(task_id, name):
    """
    Assign a specific task to a translator
    
    Creates a new task assignment by updating both the persistent schedule
    and the UI session state, then triggers recomputation of remaining suggestions
    
    Args:
        task_id: Unique identifier for the task to assign
        name: Name of the translator receiving the assignment
    """
    row = st.session_state.raw_tasks.loc[
        st.session_state.raw_tasks['TASK_ID'].astype(int) == int(task_id)
    ].iloc[0]
    start_dt = datetime.fromisoformat(row['START'])
    end_dt   = start_dt + timedelta(hours=float(row['FORECAST']))

    suggestion_entries = st.session_state.suggestions.get(task_id, [])
    max_raw = max((e['score'] for e in suggestion_entries), default=1)
    remaining_opts = [
        {
            'translator': e['translator'],
            'raw_score':  e['score'],
            'norm_score': round((e['score']/max_raw*10)*2)/2
        }
        for e in suggestion_entries if e['translator'] != name
    ]

    sched = load_schedule()
    existing = sched.get(name, [])
    new_entry = {
        'task_id': int(task_id),
        'start':   start_dt.isoformat(),
        'end':     end_dt.isoformat(),
        'alternatives': remaining_opts
    }
    if not any(t['task_id'] == new_entry['task_id'] for t in existing):
        sched.setdefault(name, []).append(new_entry)
    else:
        for t in existing:
            if t['task_id'] == new_entry['task_id']:
                t.update(new_entry)
    save_schedule(sched)

    st.session_state.assigned.append({
        'Task Name': row.get('Task Name', str(task_id)),
        'Due':       row['Due'],
        'translator': name,
        'suggested': remaining_opts
    })

    # Remove from new_tasks and suggestions
    st.session_state.new_tasks = st.session_state.new_tasks[
        st.session_state.new_tasks['TASK_ID'] != task_id
    ].reset_index(drop=True)
    st.session_state.suggestions.pop(task_id, None)

    # Recompute for remaining tasks
    recompute_suggestions()

def reassign_task(index, name):
    """
    Reassign an already assigned task to a different translator
    
    Transfers a task assignment from one translator to another by removing
    it from the original schedule and adding to the new translator's schedule
    
    Args:
        index: Index of the task in the assigned tasks list
        name: Name of the new translator to receive the assignment
    """
    entry = st.session_state.assigned[index]
    old = entry['translator']
    task_id = entry['Task Name']

    # Remove from old schedule
    sched = load_schedule()
    old_list = sched.get(old, [])
    for i, t in enumerate(old_list):
        if t['task_id'] == int(task_id):
            removed = old_list.pop(i)
            break
    sched[old] = old_list

    # Prepare alternatives, adding the old translator back
    new_alts = removed.get('alternatives', [])
    new_alts.append({'translator': old, 'raw_score': None, 'norm_score': None})

    # Time bounds
    row = st.session_state.raw_tasks.loc[
        st.session_state.raw_tasks['TASK_ID'].astype(int) == int(task_id)
    ].iloc[0]
    start_dt = datetime.fromisoformat(row['START'])
    end_dt   = start_dt + timedelta(hours=float(row['FORECAST']))

    # Assign to new translator
    existing = sched.get(name, [])
    new_entry = {
        'task_id':     int(task_id),
        'start':       start_dt.isoformat(),
        'end':         end_dt.isoformat(),
        'alternatives': new_alts
    }
    if not any(t['task_id'] == new_entry['task_id'] for t in existing):
        sched.setdefault(name, []).append(new_entry)
    else:
        for t in existing:
            if t['task_id'] == new_entry['task_id']:
                t.update(new_entry)
    save_schedule(sched)

    # Update UI state
    entry['translator'] = name
    entry['suggested']  = [opt for opt in new_alts if opt['translator'] != name]

    # Keep removed from new_tasks and suggestions
    st.session_state.new_tasks = st.session_state.new_tasks[
        st.session_state.new_tasks['TASK_ID'].astype(int) != int(task_id)
    ].reset_index(drop=True)
    st.session_state.suggestions.pop(task_id, None)

    # Recompute for remaining tasks
    recompute_suggestions()

# -- INITIALIZE SESSION STATE --
for key, default in [
    ('new_tasks', pd.DataFrame()),
    ('raw_tasks', pd.DataFrame()),
    ('suggestions', {}),
    ('assigned', []),
    ('models', []),
    ('filters', {'name':'','quality':(0.0,10.0),'rate':(0.0,100.0),'languages':[],'sectors':[],'task_types':[]}),
    ('page_idx', 0),
    ('page_cache', {}),
    ('sort_option','Alphabetical'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_data
def load_translators(rates, metrics):
    """
    Load and process translator data from rates and metrics artifacts
    
    Combines translator hourly rates and performance metrics into a unified
    data structure for UI display and filtering with computed aggregates
    
    Args:
        rates: Dictionary of translator hourly rates by language pair
        metrics: Dictionary of translator performance metrics and history
        
    Returns:
        dict: Processed translator data with computed aggregates
    """
    data = {}
    for name, pairs in rates.items():
        if name == '__global_average__': continue
        langs = {src: list(tgts.keys()) for src, tgts in pairs.items()}
        all_rates = [r for tgts in pairs.values() for r in tgts.values()]
        mean_rate = round(mean(all_rates) * 2) / 2 if all_rates else None
        m_entry = metrics.get(name, {})
        qv = m_entry.get('avg_quality')
        if qv is None:
            vals = []
            for sm in m_entry.values():
                if isinstance(sm, dict):
                    for tm in sm.values():
                        if isinstance(tm, dict) and 'avg_quality' in tm:
                            vals.append(tm['avg_quality'])
            qv = mean(vals) if vals else None
        quality = round(qv * 2) / 2 if qv is not None else None
        data[name] = {
            'languages': langs,
            'mean_rate': mean_rate,
            'quality': quality,
            'sector_history': m_entry.get('sector_history', {}),
            'task_type_history': m_entry.get('task_type_history', {}),
            'lang_pairs': pairs
        }
    return data

translator_data = load_translators(rates, metrics)
translator_df = pd.DataFrame([
    {
        'name': n,
        'quality': translator_data[n]['quality'] or 0,
        'mean_rate': translator_data[n]['mean_rate'] or 0,
        'sectors': list(translator_data[n]['sector_history'].keys()),
        'task_types': list(translator_data[n]['task_type_history'].keys()),
        'languages': [lang for tgts in translator_data[n]['languages'].values() for lang in tgts]
    }
    for n in translator_data
])
d_max = float(translator_df['mean_rate'].max() or 100)
st.session_state.filters['rate'] = (0.0, d_max)

def run_scheduler(csv_file, model_names):
    """
    Generate translator suggestions using prediction models and availability checking
    
    Orchestrates the complete suggestion pipeline by running selected prediction models
    on uploaded task data, then filtering results by translator availability
    
    Args:
        csv_file: Uploaded CSV file containing task data
        model_names: List of model names to use for prediction
        
    Returns:
        dict: Available translators by task ID with confidence scores
    """
    if hasattr(csv_file, "seek"):
        csv_file.seek(0)
    df = pd.read_csv(csv_file)
    #df['TASK_ID'] = df['TASK_ID'].astype(str)

    raw_suggestions = run_prediction_pipeline(
            df_tasks=df,
            top_k=5,
            models_to_use=[m.lower() for m in model_names],
        )

    predictions_with_scores = convert_to_simple_format(raw_suggestions)

    available_translators = get_available_translators(
                df,
                predictions_with_scores
            )
        
    print(available_translators)
    return available_translators

# -- TABS SETUP --
tab1, tab2, tab3 = st.tabs(["New Tasks", "Assigned Tasks", "Translators"])

# -- NEW TASKS --
with tab1:
    st.header("New Tasks")

    # CSV uploader (always visible)
    uploaded = st.file_uploader(
        "Import historical tasks CSV",
        type=["csv"]
    )
    # Reset state when a new CSV is uploaded
    if uploaded is not None:
        if st.session_state.get('uploaded_file_name') != uploaded.name:
            # New file detected: clear previous data
            st.session_state.uploaded_file_name = uploaded.name
            st.session_state.raw_tasks = pd.DataFrame()
            st.session_state.new_tasks = pd.DataFrame()
            st.session_state.suggestions = {}
            st.session_state.assigned = []
            st.session_state.models = []
        # process uploaded file
        df_raw = pd.read_csv(uploaded)
        required = [
            "PROJECT_ID","PM","TASK_ID","START","END","TASK_TYPE",
            "SOURCE_LANG","TARGET_LANG","FORECAST","COST",
            "MANUFACTURER","MANUFACTURER_SECTOR","WILDCARD"
        ]
        if not all(col in df_raw.columns for col in required):
            st.error(f"CSV must include columns: {', '.join(required)}")
        else:
            df_raw['Due'] = pd.to_datetime(df_raw['END'], errors='coerce')
            df_raw['TASK_ID'] = df_raw['TASK_ID'].astype(int)
            st.session_state.raw_tasks = df_raw
            st.session_state.new_tasks = df_raw[['TASK_ID', 'Due']]


    # Model selection
    LABELS = {
    "sat": "Rule Based",
    "ranking": "Machine Learning"
    }

    options = ["sat", "ranking"]
    models = st.multiselect(
        "Select Suggestion Models:",
        options,
        default=st.session_state.get("models", options),
        format_func=lambda opt: LABELS.get(opt, opt)
    )

    st.session_state.models = models

    # Suggest button
    if st.button("Suggest"):
        if uploaded is None:
            st.warning("Please upload a CSV before generating suggestions.")
        elif not st.session_state.models:
            st.warning("Select at least one model to generate suggestions.")
        else:
            assignments = run_scheduler(
                uploaded,
                model_names=[m.lower() for m in st.session_state.models]
            )
            st.session_state.suggestions = assignments
            # remember for later re‐runs
            st.session_state.uploaded_file = uploaded

    # Display tasks and their suggestions
    if not st.session_state.new_tasks.empty:
        for _, task in st.session_state.new_tasks.iterrows():
            tid, due = task['TASK_ID'], task['Due']
            suggestion_entries = st.session_state.suggestions.get(tid, [])
            if not suggestion_entries:
                continue

            # Task header
            st.markdown(
                f"<div class='task-card'><strong>{tid}</strong> — due "
                f"{due.strftime('%Y-%m-%d %H:%M') if pd.notna(due) else 'N/A'}</div>",
                unsafe_allow_html=True
            )

            # Normalize scores (highest→10), round to .5
            # Filter out NaN scores and get max
            valid_scores = [e["score"] for e in suggestion_entries if not pd.isna(e["score"])]
            max_raw = max(valid_scores) if valid_scores else 1.0
            
            cols = st.columns(2)
            for i, entry in enumerate(suggestion_entries):
                name = entry["translator"]
                raw  = entry["score"]
                
                # Handle NaN scores
                if pd.isna(raw) or max_raw == 0:
                    norm = 0.0
                else:
                    norm = round((raw / max_raw * 10) * 2) / 2
                score_str = f"{norm:.1f}"
                info = translator_data[name]
                rate_str = f"{info['mean_rate']:.1f}" if info['mean_rate'] is not None else "N/A"
                qual_str = f"{info['quality']:.1f}" if info['quality'] is not None else "N/A"

                with cols[i % 2]:
                    st.markdown(
                        f"<div class='translator-card'>"
                          f"<div style='font-weight:600;color:#000;'>{name}</div>"
                          f"<div>Score: {score_str}/10<br>"
                          f"Quality: {qual_str}/10<br>"
                          f"Rate: €{rate_str}/hr</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    with st.expander(f"More info about {name}"):
                        df_sec = pd.DataFrame({
                            'Sector': list(info['sector_history'].keys()),
                            'Count':  list(info['sector_history'].values())
                        })
                        df_tt = pd.DataFrame({
                            'Task Type': list(info['task_type_history'].keys()),
                            'Count':     list(info['task_type_history'].values())
                        })
                        c1, c2 = st.columns(2)
                        if not df_sec.empty:
                            c1.altair_chart(
                                alt.Chart(df_sec)
                                   .mark_arc()
                                   .encode(theta='Count:Q', color='Sector:N'),
                                use_container_width=True
                            )
                        if not df_tt.empty:
                            c2.altair_chart(
                                alt.Chart(df_tt)
                                   .mark_arc()
                                   .encode(theta='Count:Q', color='Task Type:N'),
                                use_container_width=True
                            )
                    st.button(
                        f"Assign {name}",
                        key=f"assign_{tid}_{name}",
                        on_click=assign_task,
                        args=(tid, name)
                    )
    else:
        st.info("No new tasks loaded. Upload a CSV to begin.")


# -- ASSIGNED TASKS --
with tab2:
    st.header("Assigned Tasks")
    if st.session_state.assigned:
        for i, entry in enumerate(st.session_state.assigned):
            tn, due, tr = entry['Task Name'], entry['Due'], entry['translator']
            st.markdown(
                f"<div class='task-card assigned-card'><strong>{tn}</strong> — due {due.strftime('%Y-%m-%d %H:%M') if pd.notna(due) else 'N/A'}<br>"
                f"<em>Assigned to: {tr}</em></div>", unsafe_allow_html=True
            )
            with st.expander("View & reassign suggestions"):
                cols = st.columns(2)
                for j, opt in enumerate(entry['suggested']):
                    name = opt['translator']
                    score = opt.get('norm_score', 'N/A')
                    info = translator_data[name]
                    rate_str = f"{info['mean_rate']:.1f}" if info['mean_rate'] is not None else "N/A"
                    qual_str = f"{info['quality']:.1f}" if info['quality'] is not None else "N/A"
                    with cols[j % 2]:
                        st.markdown(
                            f"<div class='translator-card'><div style='font-weight:600;color:#000;'>{name}</div>"
                            f"<div>Score: {score}/10<br>Quality: {qual_str}/10<br>Rate: €{rate_str}/hr</div></div>",
                            unsafe_allow_html=True
                        )
                        st.button(f"Reassign to {name}", key=f"reassign_{i}_{name}", on_click=reassign_task, args=(i,name))
    else:
        st.info("No tasks assigned yet.")


# -- TRANSLATORS --
with tab3:
    st.header("Translators")

    # Search / Filters expander
    with st.expander("Search / Filters", expanded=False):
        with st.form("filter_form"):
            f = st.session_state.filters
            name_input = st.text_input("Name:", value=f['name'])
            quality_range = st.slider("Quality:", 0.0, 10.0, f['quality'], step=0.5)
            rate_range = st.slider("Rate (€):", 0.0, d_max, f['rate'])
            langs = st.multiselect("Languages:", sorted({l for lst in translator_df['languages'] for l in lst}), default=f['languages'])
            sectors = st.multiselect("Sectors:", sorted({s for lst in translator_df['sectors'] for s in lst}), default=f['sectors'])
            ttypes = st.multiselect("Task Types:", sorted({t for lst in translator_df['task_types'] for t in lst}), default=f['task_types'])
            c1, c2 = st.columns(2)
            sb = c1.form_submit_button("Search")
            cb = c2.form_submit_button("Clear")
            if sb:
                st.session_state.filters = {'name': name_input, 'quality': quality_range, 'rate': rate_range, 'languages': langs, 'sectors': sectors, 'task_types': ttypes}
                st.session_state.page_idx = 0
                st.session_state.page_cache.clear()
            if cb:
                st.session_state.filters = {'name': '', 'quality': (0.0, 10.0), 'rate': (0.0, d_max), 'languages': [], 'sectors': [], 'task_types': []}
                st.session_state.page_idx = 0
                st.session_state.page_cache.clear()
                st.session_state.filters = {'name': '', 'quality': (0.0, 10.0), 'rate': (0.0, d_max), 'languages': [], 'sectors': [], 'task_types': []}
                st.session_state.page_idx = 0
                st.session_state.page_cache.clear()

    # Determine assigned translators to prioritize
    assigned_names = [a['translator'] for a in st.session_state.assigned]

    # Base DataFrame
    df = translator_df.copy()
    # Apply filters
    f = st.session_state.filters
    df = df[df['name'].str.contains(f['name'], case=False)]
    df = df[df['quality'].between(*f['quality'])]
    df = df[df['mean_rate'].between(*f['rate'])]
    if f['languages']:
        df = df[df['languages'].apply(lambda lst: any(l in lst for l in f['languages']))]
    if f['sectors']:
        df = df[df['sectors'].apply(lambda lst: any(s in lst for s in f['sectors']))]
    if f['task_types']:
        df = df[df['task_types'].apply(lambda lst: any(t in lst for t in f['task_types']))]

    # Sort within assigned and unassigned
    df_assigned = df[df['name'].isin(assigned_names)]
    df_unassigned = df[~df['name'].isin(assigned_names)]
    # Apply overall sort_option within each group
    def sort_group(d):
        if st.session_state.sort_option == 'Alphabetical': return d.sort_values('name')
        if st.session_state.sort_option == 'Rate': return d.sort_values('mean_rate')
        return d.sort_values('quality', ascending=False)
    df_assigned = sort_group(df_assigned)
    df_unassigned = sort_group(df_unassigned)
    df = pd.concat([df_assigned, df_unassigned])

    # Pagination
    page_size = 10
    total = len(df)
    idx = st.session_state.page_idx
    start = idx * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    cols = st.columns(2)
    assigned_map = {a['translator']: [] for a in st.session_state.assigned}
    for a in st.session_state.assigned:
        assigned_map.setdefault(a['translator'], []).append(a['Task Name'])

    for i, row in page_df.iterrows():
        info = translator_data[row['name']]
        is_assigned = row['name'] in assigned_map
        cls = 'translator-card assigned-card' if is_assigned else 'translator-card'
        with cols[(i - start) % 2]:
            alist = assigned_map.get(row['name'], [])
            asign_html = '<br><em>Assigned: ' + ', '.join(alist) + '</em>' if alist else ''
            rate_str = f"{info['mean_rate']:.1f}" if info['mean_rate'] is not None else "N/A"
            qual_str = f"{info['quality']:.1f}" if info['quality'] is not None else "N/A"
            st.markdown(
                f"<div class='{cls}'><div style='font-size:1.25rem;font-weight:600;color:#000;'>{row['name']}</div>"
                f"<div style='font-size:0.9rem;color:#495057;'>Quality: {qual_str}/10<br>Rate: €{rate_str}/hr<br>{asign_html}<br>"
                + '<br>'.join(f"<strong>{s}:</strong> {', '.join(tgts)}" for s, tgts in info['languages'].items())
                + "</div></div>", unsafe_allow_html=True
            )
            with st.expander(f"Details for {row['name']}"):
                dfp = pd.DataFrame([{'Language Pair': f"{s}→{t}", 'Rate (€)': r}
                                    for s, tmap in info['lang_pairs'].items() for t, r in tmap.items()])
                st.table(dfp)
                if info['quality'] is not None:
                    st.markdown(f"**Quality:** {qual_str}/10")
                sec_df = pd.DataFrame({'Sector': list(info['sector_history'].keys()), 'Count': list(info['sector_history'].values())})
                tt_df = pd.DataFrame({'Task Type': list(info['task_type_history'].keys()), 'Count': list(info['task_type_history'].values())})
                c1, c2 = st.columns(2)
                if not sec_df.empty:
                    c1.altair_chart(alt.Chart(sec_df).mark_arc().encode(theta='Count:Q', color='Sector:N'), use_container_width=True)
                if not tt_df.empty:
                    c2.altair_chart(alt.Chart(tt_df).mark_arc().encode(theta='Count:Q', color='Task Type:N'), use_container_width=True)

    # Navigation buttons
    nav1, nav2 = st.columns(2)
    if nav1.button("Previous") and st.session_state.page_idx > 0:
        st.session_state.page_idx -= 1
    if nav2.button("Next") and end < total:
        st.session_state.page_idx += 1
    st.write(f"Showing {start+1}-{min(end, total)} of {total}")