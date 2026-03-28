import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_ui import apply_custom_css, get_db_connection, load_model, load_precomputed_llm_outputs, load_rag_summaries, predict_patient_risk

# Configuration
st.set_page_config(page_title="Historical Patient Lookup", page_icon="🏥", layout="wide")
apply_custom_css()

st.title("🏥 Historical Patient Lookup")
st.markdown("Instantly retrieve historical predictions, SHAP explanations, and pre-computed Gemini AI clinical insights for existing patients in the MIMIC-IV dataset.")
st.divider()

# Load Data
artifacts = load_model()
llm_outputs = load_precomputed_llm_outputs()
rag_summaries = load_rag_summaries()

if not llm_outputs:
    st.warning("No pre-computed LLM outputs found. Please run the Gemini generation notebook first.")
    st.stop()

# 1. UI: Select Patient ID
hadm_ids = list(llm_outputs.keys())
selected_id = st.selectbox("Search Historical Patient Admissions (hadm_id)", hadm_ids)

if selected_id:
    # 2. Extract Data
    llm_data = llm_outputs[selected_id]
    risk_level = llm_data.get('risk_level', 'UNKNOWN')
    risk_pct = llm_data.get('predicted_risk_pct', 'N/A')
    
    # Render Output Panels
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.subheader("🤖 Gemini 2.5 Clinical Insights")
        
        # Risk Badge
        alert_class = "high-risk" if "HIGH" in risk_level.upper() else "medium-risk" if "MEDIUM" in risk_level.upper() else "low-risk"
        st.markdown(f'<div class="{alert_class}"><h4>Current Alert Level: {risk_level} ({risk_pct}%)</h4></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        doctor_alert = llm_data.get('doctor_alert', {})
        st.markdown("##### 🩺 Doctor's Summary")
        st.info(doctor_alert.get('risk_summary', 'No summary available.'))
        
        st.markdown("##### 📋 Actionable Patient Precautions")
        for idx, prec in enumerate(llm_data.get('patient_precautions', [])):
            st.write(f"{idx+1}. {prec}")
            
        st.markdown("##### 🗓️ Follow-Up Route")
        st.success(llm_data.get('follow_up_recommendations', 'No follow-up given.'))
        
        with st.expander("View Original Clinical History"):
            st.write(rag_summaries.get(selected_id, "No narrative available."))
            
    with col_right:
        st.subheader("📉 SHAP Risk Drivers")
        st.markdown("Local feature importance explaining **why** this specific patient received this score.")
        
        # Recalculate SHAP dynamically for the UI plot
        con = get_db_connection()
        query = f"SELECT * FROM model_features WHERE hadm_id = {selected_id}"
        df_patient = con.execute(query).fetchdf()
        
        if not df_patient.empty:
            # Drop identifiers before prediction
            drop_cols = ['subject_id', 'hadm_id', 'readmitted_30d']
            X_feats = df_patient.drop(columns=[c for c in drop_cols if c in df_patient.columns])
            
            # Use our util function
            r_prob, t_drivers, X_final, sv = predict_patient_risk(artifacts, X_feats)
            
            # Plot SHAP Waterfall
            fig, ax = plt.subplots(figsize=(6, 4))
            # Create Explanation object for plotting
            shap_obj = shap.Explanation(
                values=sv.values[0], 
                base_values=sv.base_values[0], 
                data=X_final.iloc[0], 
                feature_names=X_final.columns
            )
            shap.plots.waterfall(shap_obj, max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        else:
            st.error("Could not locate patient tabular features in database.")
