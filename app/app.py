import streamlit as st
import pandas as pd
import altair as alt
import json
import random
from statistics import mean

# -- Page config and CSS styling --
st.set_page_config(page_title="Translator Assignment", layout="wide")
st.markdown(
    """
    <style>
    .css-1v3fvcr.e1fqkh3o4 { font-size: 2rem !important; font-weight: bold !important; }
    .translator-card { background-color: #f8f9fa; border-radius: 10px; padding: 10px; margin-bottom: 8px; border: 1px solid #e3e6ea; transition: background-color 0.3s; color: #000 !important; }
    .translator-card:hover { background-color: #e9ecef; }
    .assigned-card { background-color: #e2f0d9 !important; color: #000 !important; }
    .task-card { background-color: #ffffff; border-radius: 8px; padding: 4px 8px; margin-bottom: 6px; border: 1px solid #d3d3d3; color: #000 !important; }
    .task-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    .streamlit-expanderHeader { font-size: 1.1rem !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

# -- Callbacks for button actions --
def assign_task(task_name, name):
    # find and remove the task by name
    df = st.session_state.new_tasks
    mask = df['Task Name'] == task_name
    if not mask.any():
        return
    task_row = df[mask].iloc[0]
    # record assignment
    st.session_state.assigned.append({
        'Task Name': task_name,
        'Due': task_row['Due'],
        'translator': name,
        'suggested': [t for t in st.session_state.suggestions.get(task_name, []) if t != name]
    })
    # drop the task
    st.session_state.new_tasks = df[~mask]
    # remove suggestions for this task
    st.session_state.suggestions.pop(task_name, None)

# reassign remains unchanged
def reassign_task(index, name):
    assigned = st.session_state.assigned
    entry = assigned[index]
    old = entry['translator']
    entry['translator'] = name
    entry['suggested'].append(old)
    entry['suggested'].remove(name)

# -- Initialize session state --
for key, default in [('new_tasks', pd.DataFrame()), ('suggestions', {}), ('assigned', []), ('filters', {'name':'','quality':(0.0,10.0),'rate':(0.0,100.0),'languages':[],'sectors':[],'task_types':[]}), ('page_idx', 0), ('page_cache', {}), ('sort_option','Alphabetical')]:
    if key not in st.session_state:
        st.session_state[key] = default

# -- Load translators --
@st.cache_data
def load_translators(rates_file='translator_hourly_rates.json', metrics_file='translator_metrics.json'):
    rates = json.load(open(rates_file))
    metrics = json.load(open(metrics_file))
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
        data[name] = {'languages':langs,'mean_rate':mean_rate,'quality':quality,'sector_history':m_entry.get('sector_history',{}),'task_type_history':m_entry.get('task_type_history',{}),'lang_pairs':pairs}
    return data

translator_data = load_translators()
translator_df = pd.DataFrame([
    {'name':n,'quality':translator_data[n]['quality'] or 0,'mean_rate':translator_data[n]['mean_rate'] or 0,'sectors':list(translator_data[n]['sector_history'].keys()),'task_types':list(translator_data[n]['task_type_history'].keys()),'languages':[lang for tgts in translator_data[n]['languages'].values() for lang in tgts]}
    for n in translator_data
])
d_max = float(translator_df['mean_rate'].max() or 100)
st.session_state.filters['rate'] = (0.0, d_max)

# -- Tabs setup --
tab1, tab2, tab3 = st.tabs(["New Tasks", "Assigned Tasks", "Translators"])

# --- New Tasks ---
with tab1:
    st.header("New Tasks")
    # only show uploader and suggest if no suggestions
    if not st.session_state.suggestions:
        uploaded = st.file_uploader("Import tasks (CSV with Task Name, Due Day, Due Hour)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            cols = {c.lower():c for c in df.columns}
            name_col = cols.get('task name') or next(iter(cols.values()))
            day_col, hour_col = cols.get('due day'), cols.get('due hour')
            if day_col and hour_col:
                df['Due'] = pd.to_datetime(df[day_col].astype(str) + ' ' + df[hour_col].astype(str))
            else:
                df['Due'] = pd.NaT
                st.error("CSV must include Due Day and Due Hour columns")
            df['Task Name'] = df[name_col]
            st.session_state.new_tasks = df[['Task Name','Due']]
        if not st.session_state.new_tasks.empty:
            st.markdown("**Select Suggestion Models:**")
            models = [m for m in ['SAT','ML','DL'] if st.checkbox(m)]
            if st.button("Suggest"):
                if not models:
                    st.warning("Select at least one model")
                else:
                    names = list(translator_data.keys())
                    for _, row in st.session_state.new_tasks.iterrows():
                        st.session_state.suggestions[row['Task Name']] = random.sample(names,5)
    # display tasks
    for _, task in st.session_state.new_tasks.iterrows():
        tn = task['Task Name']
        due = task['Due']
        st.markdown(
            f"<div class='task-card'><strong>{tn}</strong> — due {due.strftime('%Y-%m-%d %H:%M') if pd.notna(due) else 'N/A'}</div>",
            unsafe_allow_html=True
        )
        sugg = st.session_state.suggestions.get(tn, [])
        if sugg:
            cols = st.columns(2)
            for i, name in enumerate(sugg):
                with cols[i%2]:
                    info = translator_data[name]
                    rate_str = f"{info['mean_rate']:.1f}" if info['mean_rate'] is not None else "N/A"
                    qual_str = f"{info['quality']:.1f}" if info['quality'] is not None else "N/A"
                    st.markdown(
                        f"<div class='translator-card'><div style='font-weight:600;color:#000;'>{name}</div>"
                        f"<div>Quality: {qual_str}/10<br>Rate: €{rate_str}/hr</div></div>",
                        unsafe_allow_html=True
                    )
                    with st.expander(f"More info about {name}"):
                        df_sec = pd.DataFrame({'Sector': list(info['sector_history'].keys()), 'Count': list(info['sector_history'].values())})
                        df_tt = pd.DataFrame({'Task Type': list(info['task_type_history'].keys()), 'Count': list(info['task_type_history'].values())})
                        c1, c2 = st.columns(2)
                        if not df_sec.empty:
                            c1.altair_chart(
                                alt.Chart(df_sec).mark_arc().encode(theta='Count:Q', color='Sector:N'),
                                use_container_width=True
                            )
                        if not df_tt.empty:
                            c2.altair_chart(
                                alt.Chart(df_tt).mark_arc().encode(theta='Count:Q', color='Task Type:N'),
                                use_container_width=True
                            )
                    st.button(
                        f"Assign {name}",
                        key=f"assign_{tn}_{name}",
                        on_click=assign_task,
                        args=(tn, name)
                    )
    # save button to clear new tasks and suggestions
    if st.session_state.suggestions:
        if st.button("Save Assignments"):
            st.session_state.new_tasks = pd.DataFrame()
            st.session_state.suggestions = {}

# --- Assigned Tasks ---
with tab2:
    st.header("Assigned Tasks")
    ats = st.session_state.assigned
    if ats:
        # visual cards
        for i, entry in enumerate(sorted(ats, key=lambda x: x['Due'])):
            st.markdown(
                f"<div class='task-card assigned-card'><strong>{entry['Task Name']}</strong> — due {entry['Due'].strftime('%Y-%m-%d %H:%M') if pd.notna(entry['Due']) else 'N/A'}<br>"
                f"<em>Assigned to: {entry['translator']}</em></div>", unsafe_allow_html=True)
            with st.expander("View & reassign suggestions"):
                cols = st.columns(2)
                for j, name in enumerate(entry['suggested']):
                    with cols[j%2]:
                        info = translator_data[name]
                        st.markdown(
                            f"<div class='translator-card'><div style='font-weight:600;color:#000;'>{name}</div>"
                            f"<div>Quality: {info['quality']:.1f}/10<br>Rate: €{info['mean_rate']}/hr</div></div>", unsafe_allow_html=True)
                        st.button(f"Reassign to {name}", key=f"reassign_{i}_{name}", on_click=reassign_task, args=(i,name))
    else:
        st.info("No tasks assigned yet.")

# --- Translators ---
with tab3:
    st.header("Translators")
    c1, c2, c3 = st.columns(3)
    if c1.button("Alphabetical"): st.session_state.sort_option='Alphabetical'; st.session_state.page_idx=0; st.session_state.page_cache.clear()
    if c2.button("By Rate"): st.session_state.sort_option='Rate'; st.session_state.page_idx=0; st.session_state.page_cache.clear()
    if c3.button("By Quality"): st.session_state.sort_option='Quality'; st.session_state.page_idx=0; st.session_state.page_cache.clear()
    with st.expander("Search / Filters", expanded=False):
        with st.form("filter_form"):
            f=st.session_state.filters
            name_input=st.text_input("Name:", value=f['name'])
            quality_range=st.slider("Quality:",0.0,10.0,f['quality'],step=0.5)
            rate_range=st.slider("Rate (€):",0.0,d_max,f['rate'])
            langs=st.multiselect("Languages:",sorted({l for lst in translator_df['languages'] for l in lst}),default=f['languages'])
            sectors=st.multiselect("Sectors:",sorted({s for lst in translator_df['sectors'] for s in lst}),default=f['sectors'])
            ttypes=st.multiselect("Task Types:",sorted({t for lst in translator_df['task_types'] for t in lst}),default=f['task_types'])
            sb=st.form_submit_button("Search"); cb=st.form_submit_button("Clear")
            if sb: st.session_state.filters={'name':name_input,'quality':quality_range,'rate':rate_range,'languages':langs,'sectors':sectors,'task_types':ttypes}; st.session_state.page_idx=0; st.session_state.page_cache.clear()
            if cb: st.session_state.filters={'name':'','quality':(0.0,10.0),'rate':(0.0,d_max),'languages':[],'sectors':[],'task_types':[]}; st.session_state.page_idx=0; st.session_state.page_cache.clear()
    df=translator_df.copy(); f=st.session_state.filters
    df=df[df['name'].str.contains(f['name'],case=False)]
    df=df[df['quality'].between(*f['quality'])]
    df=df[df['mean_rate'].between(*f['rate'])]
    if f['languages']: df=df[df['languages'].apply(lambda lst:any(l in lst for l in f['languages']))]
    if f['sectors']: df=df[df['sectors'].apply(lambda lst:any(s in lst for s in f['sectors']))]
    if f['task_types']: df=df[df['task_types'].apply(lambda lst:any(t in lst for t in f['task_types']))]
    so=st.session_state.sort_option
    if so=='Alphabetical': df=df.sort_values('name')
    elif so=='Rate': df=df.sort_values('mean_rate')
    else: df=df.sort_values('quality',ascending=False)
    ps=10; tot=len(df); idx=st.session_state.page_idx
    if idx in st.session_state.page_cache: page_df=st.session_state.page_cache[idx]
    else: page_df=df.iloc[idx*ps:(idx+1)*ps]; st.session_state.page_cache[idx]=page_df
    cols=st.columns(2)
    assigned_map={}
    for a in st.session_state.assigned: assigned_map.setdefault(a['translator'],[]).append(a['Task Name'])
    for i,row in page_df.iterrows():
        info=translator_data[row['name']]
        is_assigned=row['name'] in assigned_map
        cls='translator-card assigned-card' if is_assigned else 'translator-card'
        with cols[(i-idx*ps)%2]:
            alist=assigned_map.get(row['name'],[])
            asign_html='<br><em>Assigned: '+', '.join(alist)+'</em>' if alist else ''
            st.markdown(f"<div class='{cls}'><div style='font-size:1.25rem;font-weight:600;color:#000;'>{row['name']}</div><div style='font-size:0.9rem;color:#495057;'>Quality: {info['quality']:.1f}/10<br>Rate: €{info['mean_rate']}/hr<br>{asign_html}<br>"+ '<br>'.join(f"<strong>{s}:</strong> {', '.join(tgts)}" for s,tgts in info['languages'].items())+"</div></div>",unsafe_allow_html=True)
            with st.expander(f"Details for {row['name']}"):
                dfp=pd.DataFrame([{'Language Pair':f"{s}→{t}",'Rate (€)':r} for s,tmap in info['lang_pairs'].items() for t,r in tmap.items()])
                st.table(dfp)
                if info['quality'] is not None: st.markdown(f"**Quality:** {info['quality']:.1f}/10")
                c1,c2=st.columns(2)
                if info['sector_history']: st.altair_chart(alt.Chart(pd.DataFrame({'Sector':list(info['sector_history'].keys()),'Count':list(info['sector_history'].values())})).mark_arc().encode(theta='Count:Q',color='Sector:N'),use_container_width=True)
                if info['task_type_history']: st.altair_chart(alt.Chart(pd.DataFrame({'Task Type':list(info['task_type_history'].keys()),'Count':list(info['task_type_history'].values())})).mark_arc().encode(theta='Count:Q',color='Task Type:N'),use_container_width=True)
    n1,n2=st.columns(2)
    if n1.button("Previous") and st.session_state.page_idx>0: st.session_state.page_idx-=1
    if n2.button("Next") and st.session_state.page_idx<(tot-1)//ps: st.session_state.page_idx+=1
    st.write(f"Showing {st.session_state.page_idx*ps+1}-{min((st.session_state.page_idx+1)*ps,tot)} of {tot}")
# -- End of app --
