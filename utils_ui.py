import streamlit as st
import duckdb
import pickle
import pandas as pd
import numpy as np
import os
import json
import shap
import google.generativeai as genai

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = '/Users/sameer/Desktop/AIheathAgent/Capstone_Healthcare_Decision_Intelligence_Agent'
ORIGINAL_DATA_DIR = '/Users/sameer/Documents/DataScience_Capstone_Project/Capstone_Healthcare_Decision_Intelligence_Agent/dataset/'
DB_PATH = os.path.join(ORIGINAL_DATA_DIR, 'hf_project.duckdb')
MODEL_PATH = os.path.join(BASE_DIR, 'model_artifacts.pkl')
LLM_PATH = os.path.join(BASE_DIR, 'llm_outputs_gemini.json')
RAG_SUMMARY_PATH = os.path.join(BASE_DIR, 'dataset', 'RAG_data_summary.json')

# ==========================================
# UI THEMING (Premium Dark Mode)
# ==========================================
def apply_custom_css():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #58a6ff !important;
            font-family: 'Inter', -apple-system, sans-serif;
        }

        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            color: #2ea043;
            font-size: 2.5rem;
            font-weight: 800;
        }
        
        /* Containers */
        .glass-card {
            background: rgba(48, 54, 61, 0.4);
            border: 1px solid rgba(240, 246, 252, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }

        /* Alert styling */
        .high-risk {
            border-left: 5px solid #f85149;
            background-color: rgba(248, 81, 73, 0.1);
            padding: 15px;
            border-radius: 5px;
            color: #ff7b72;
        }
        .medium-risk {
            border-left: 5px solid #d29922;
            background-color: rgba(210, 153, 34, 0.1);
            padding: 15px;
            border-radius: 5px;
            color: #e3b341;
        }
        .low-risk {
            border-left: 5px solid #2ea043;
            background-color: rgba(46, 160, 67, 0.1);
            padding: 15px;
            border-radius: 5px;
            color: #56d364;
        }
        </style>
    """, unsafe_allow_html=True)


# ==========================================
# DATA & MODEL LOADING (Cached)
# ==========================================
@st.cache_resource
def get_db_connection(db_path=DB_PATH):
    if not os.path.exists(db_path):
        st.error(f"Database not found at {db_path}")
        return None
    return duckdb.connect(db_path, read_only=True)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model artifacts not found at {MODEL_PATH}")
        return None
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_precomputed_llm_outputs():
    outputs = {}
    if os.path.exists(LLM_PATH):
        with open(LLM_PATH, 'r') as f:
            data = json.load(f)
            for row in data:
                if 'hadm_id' in row and 'error' not in row:
                    outputs[str(row['hadm_id'])] = row
    return outputs

@st.cache_data
def load_rag_summaries():
    summaries = {}
    if os.path.exists(RAG_SUMMARY_PATH):
        with open(RAG_SUMMARY_PATH, 'r') as f:
            data = json.load(f)
            for item in data:
                for k, v in item.items():
                    summaries[str(k)] = str(v)
    return summaries


# ==========================================
# INFERENCE LOGIC
# ==========================================
def predict_patient_risk(artifacts, patient_features_df):
    """Passes raw features through the scikit-learn pipeline."""
    scaler = artifacts['scaler']
    selector = artifacts['selector']
    model = artifacts['model']
    selected_features = artifacts['selected_features']
    
    # 1. Transform features
    X_sel = selector.transform(patient_features_df)
    X_sc = scaler.transform(X_sel)
    X_final = pd.DataFrame(X_sc, columns=selected_features)
    
    # 2. Predict Probability
    risk_prob = model.predict_proba(X_final)[0, 1]
    
    # 3. Compute SHAP Values
    explainer = shap.LinearExplainer(model, X_final)
    shap_values = explainer(X_final)
    
    # 4. Extract Top 3 increasing drivers
    patient_shap = shap_values.values[0]
    feature_names = shap_values.feature_names
    sorted_idx = np.argsort(patient_shap)[::-1]
    top_drivers = [(feature_names[i], patient_shap[i]) for i in sorted_idx[:3] if patient_shap[i] > 0]
    
    return risk_prob, top_drivers, X_final, shap_values

# ==========================================
# GEMINI API WRAPPER
# ==========================================
def generate_live_clinical_insights(risk_prob, top_drivers, doctors_note, api_key):
    """Calls Gemini 2.5 Flash for a live patient."""
    genai.configure(api_key=api_key)
    
    drivers_str = "\\n".join([f"- {feat} (Impact score: {val:.4f})" for feat, val in top_drivers])
    if not drivers_str:
        drivers_str = "No significant risk-increasing factors found."
    
    prompt_context = f"""--- PATIENT CLINICAL HISTORY (Doctor's Admission Note) ---
{doctors_note}

--- AI MODEL PREDICTION ---
Predicted Readmission Risk: {risk_prob:.1%}

--- KEY RISK DRIVERS (SHAP) ---
The model identified the following clinical factors driving this readmission risk:
{drivers_str}
"""
    
    SYSTEM_INSTRUCTION = (
        'You are an expert clinical decision support AI embedded in a hospital readmission prevention system. '
        'You will receive a patient clinical summary, predicted 30-day readmission risk, and SHAP risk drivers. '
        'Respond ONLY with valid JSON using this exact structure: '
        '{"doctor_alert": {"risk_level": "HIGH / MEDIUM / LOW", '
        '"risk_summary": "2-3 sentence summary for the attending physician linking risk drivers to patient history."}, '
        '"patient_precautions": ['
        '"Precaution 1 written for the patient to understand.", '
        '"Precaution 2 written for the patient to understand.", '
        '"Precaution 3 written for the patient to understand.", '
        '"Precaution 4 written for the patient to understand."], '
        '"follow_up_recommendations": "1-2 sentence follow-up care recommendation for the care team."}. '
    )
    
    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={"response_mime_type": "application/json", "temperature": 0.3}
    )
    
    response = model.generate_content(prompt_context)
    return json.loads(response.text)
