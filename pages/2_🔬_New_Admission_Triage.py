import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_ui import apply_custom_css, get_db_connection, load_model, predict_patient_risk, generate_live_clinical_insights

st.set_page_config(page_title="Live Admission Triage", page_icon="🔬", layout="wide")
apply_custom_css()

st.title("🔬 Live AI Admission Triage")
st.markdown("Simulate a live patient triage. Adjust clinical features live, type an admitting note, and instantly generate a predicted risk and Gemini AI clinical plan.")
st.divider()

artifacts = load_model()
con = get_db_connection()

# Load a random baseline patient to populate the editor
@st.cache_data
def load_baseline():
    return con.execute("SELECT * FROM model_features LIMIT 10").fetchdf().sample(1)

df_baseline = load_baseline()
drop_cols = ['subject_id', 'hadm_id', 'readmitted_30d']
df_edit = df_baseline.drop(columns=[c for c in drop_cols if c in df_baseline.columns]).reset_index(drop=True)

col_data, col_ai = st.columns([1, 1.2])

with col_data:
    st.subheader("1. Patient Laboratory Data")
    st.markdown("Modify the baseline tabular features below to test the model.")
    
    # We transpose for a taller, narrower editor, but Streamlit data_editor works well
    edited_df = st.data_editor(df_edit, use_container_width=True, num_rows="fixed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("2. Admitting Physician's Note")
    doctors_note = st.text_area("Type unstructured history & physical context:", 
                 "Patient presents with shortness of breath and elevated heart rate. Has a history of type 2 diabetes and hypertension.",
                 height=150)
    
    analyze_btn = st.button("🚀 Analyze Live Patient", type="primary", use_container_width=True)

with col_ai:
    st.subheader("3. Live Model & Generative AI Results")
    
    if analyze_btn:
        with st.spinner("Processing tabular features through Logistic Regression..."):
            r_prob, t_drivers, X_final, sv = predict_patient_risk(artifacts, edited_df)
            
            st.markdown(f"**Calculated Live Risk Score:** <span style='font-size:24px; color:#ff4b4b;'>{r_prob:.1%}</span>", unsafe_allow_html=True)
            
            # Plot SHAP
            fig, ax = plt.subplots(figsize=(6, 4))
            shap_obj = shap.Explanation(values=sv.values[0], base_values=sv.base_values[0], 
                                        data=X_final.iloc[0], feature_names=X_final.columns)
            shap.plots.waterfall(shap_obj, max_display=7, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
        with st.spinner("Connecting to Google Gemini 2.5 Flash API..."):
            try:
                # In production, pull API key from st.secrets. For now, hardcoded from our notebook logic:
                # If the key provided earlier is invalid here, user must provide it. We'll use the one we injected:
                API_KEY = "AIzaSyBB-OGlKuvOGW7WzDdKDmsZmi44BlvlHbM"
                
                llm_data = generate_live_clinical_insights(r_prob, t_drivers, doctors_note, API_KEY)
                
                st.markdown("##### 🩺 Doctor's Live Alert")
                st.info(llm_data.get('doctor_alert', {}).get('risk_summary', 'N/A'))
                
                st.markdown("##### 📋 Actionable Patient Precautions")
                for idx, prec in enumerate(llm_data.get('patient_precautions', [])):
                    st.write(f"{idx+1}. {prec}")
                
                st.success(llm_data.get('follow_up_recommendations', ''))
                
            except Exception as e:
                st.error(f"Gemini API Error: {str(e)}")
    else:
        st.info("👈 Edit patient data and click 'Analyze Live Patient' to start the inference engine.")
