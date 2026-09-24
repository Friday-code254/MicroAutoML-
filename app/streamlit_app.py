"""
MicroAutoML-Agent — Streamlit Dashboard
Provides a visual interface to monitor active runs and inspect the database.
"""

import streamlit as st

st.set_page_config(page_title="MicroAutoML Dashboard", layout="wide")

st.title("MicroAutoML Dashboard")
st.write("Welcome to the MicroAutoML autonomous search monitor.")

# Sidebar
st.sidebar.header("Configuration")
run_id = st.sidebar.text_input("Run ID to inspect", value="RUN_001")

if st.sidebar.button("Load Data"):
    st.info(f"Loading data for {run_id} from SQLite database...")
    # Mock data loading
    st.success("Loaded successfully.")

st.header("Search Progress")
st.progress(50, text="50% Budget Exhausted")

col1, col2, col3 = st.columns(3)
col1.metric("Best CV Score", "0.9412", "+0.012")
col2.metric("Experiments", "42", "")
col3.metric("Search State", "SEARCHING")

st.header("Leaderboard")
st.dataframe({
    "Rank": [1, 2, 3],
    "ID": ["EXP_041", "EXP_022", "EXP_015"],
    "Operator": ["HPO", "FEATURE_ENG", "BASELINE"],
    "CV Score": [0.9412, 0.9292, 0.8841],
    "Runtime (s)": [14.2, 5.1, 1.2]
})

st.header("Recent Failures")
st.warning("EXP_039 failed with MEMORY (Out of memory). Repaired via reducing n_estimators.")
