import streamlit as st
import pandas as pd
from utils_ui import apply_custom_css, get_db_connection

# Page Configuration
st.set_page_config(
    page_title="Healthcare Decision Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium CSS
apply_custom_css()

# Dashboard Header
st.title("🏥 Healthcare Decision Intelligence Agent")
st.markdown("""
Welcome to the Capstone Decision Intelligence platform. This application leverages the MIMIC-IV 
clinical dataset to predict **30-day Heart Failure Readmission Risk** using Logistic Regression, 
explains individual patient risks using **SHAP**, and generates highly personalized clinical 
precautions and alerts using **Google Gemini 2.5 Flash**.
""")

# Load Database Metrics
st.divider()
st.subheader("📊 Central Database Metrics (DuckDB)")

try:
    con = get_db_connection()
    if con:
        total_patients = con.execute("SELECT COUNT(DISTINCT subject_id) FROM model_features").fetchone()[0]
        total_admissions = con.execute("SELECT COUNT(hadm_id) FROM model_features").fetchone()[0]
        readmission_rate = con.execute("SELECT AVG(readmitted_30d) * 100 FROM model_features").fetchone()[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unique Patients", f"{total_patients:,}")
        with col2:
            st.metric("Analyzed Admissions", f"{total_admissions:,}")
        with col3:
            st.metric("30-Day Readmission Rate", f"{readmission_rate:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Sample Clinical Data Snapshot")
        sample_df = con.execute("SELECT hadm_id, age_at_admission, gender, los_days, readmitted_30d FROM model_features LIMIT 5").fetchdf()
        st.dataframe(sample_df, use_container_width=True)
        
except Exception as e:
    st.error(f"Error connecting to DuckDB: {e}")

st.divider()

# Navigation helper (Since using Streamlit multipage, users will use the sidebar)
st.info("👈 **Use the sidebar on the left to navigate:**\n\n"
        "1. **Existing Patient Lookup:** Search historical patients and view their pre-computed LLM clinical summaries.\n"
        "2. **New Admission Triage:** Type in live lab values and doctor notes to calculate real-time ML risk and stream live Gemini alerts.")
