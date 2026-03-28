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
# IN-MEMORY CACHE
# ==========================================
_DB_CONN = None
_MODEL_ARTIFACTS = None
_LLM_OUTPUTS = None
_RAG_SUMMARIES = None

def get_db_connection():
    # Use a new connection per request to avoid asyncio threadpool concurrency collision
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return duckdb.connect(DB_PATH, read_only=True)

def load_model():
    global _MODEL_ARTIFACTS
    if _MODEL_ARTIFACTS is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifacts not found at {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            _MODEL_ARTIFACTS = pickle.load(f)
            
        # Hot-patch for scikit-learn version mismatch (adds missing attribute)
        if not hasattr(_MODEL_ARTIFACTS['model'], 'multi_class'):
            _MODEL_ARTIFACTS['model'].multi_class = 'auto'
            
    return _MODEL_ARTIFACTS

def load_precomputed_llm_outputs():
    global _LLM_OUTPUTS
    if _LLM_OUTPUTS is None:
        outputs = {}
        if os.path.exists(LLM_PATH):
            with open(LLM_PATH, 'r') as f:
                data = json.load(f)
                for row in data:
                    if 'hadm_id' in row and 'error' not in row:
                        outputs[str(row['hadm_id'])] = row
        _LLM_OUTPUTS = outputs
    return _LLM_OUTPUTS

def load_rag_summaries():
    global _RAG_SUMMARIES
    if _RAG_SUMMARIES is None:
        summaries = {}
        if os.path.exists(RAG_SUMMARY_PATH):
            with open(RAG_SUMMARY_PATH, 'r') as f:
                data = json.load(f)
                for item in data:
                    for k, v in item.items():
                        summaries[str(k)] = str(v)
        _RAG_SUMMARIES = summaries
    return _RAG_SUMMARIES

# ==========================================
# INFERENCE LOGIC (REST API FORMAT)
# ==========================================
def predict_patient_risk(artifacts, patient_features_df):
    """Passes raw features through scikit-learn pipeline & returns API-friendly SHAP data."""
    scaler = artifacts['scaler']
    selector = artifacts['selector']
    model = artifacts['model']
    selected_features = artifacts['selected_features']
    expected_cols = selector.feature_names_in_
    
    # Add lab change features
    LABS = [
        'creatinine', 'urea_nitrogen', 'sodium', 'potassium', 'glucose',
        'hemoglobin', 'white_blood_cells', 'platelet_count', 'bicarbonate',
        'calcium_total', 'inrpt', 'ptt', 'troponin_t', 'creatine_kinase_mb_isoenzyme'
    ]
    for lab in LABS:
        if f'{lab}_last' in patient_features_df.columns and f'{lab}_first' in patient_features_df.columns:
            patient_features_df[f'{lab}_change'] = patient_features_df[f'{lab}_last'] - patient_features_df[f'{lab}_first']
            
    # Dummy encoding for categoricals
    cat_cols = ['gender', 'admission_type', 'insurance', 'marital_status', 'race', 'discharge_location']
    for col in cat_cols:
        if col in patient_features_df.columns:
            patient_features_df[col] = patient_features_df[col].fillna('UNKNOWN')
            
    # Fill remaining numerical NaNs
    patient_features_df = patient_features_df.fillna(0)
            
    X_encoded = pd.get_dummies(patient_features_df, columns=[c for c in cat_cols if c in patient_features_df.columns])
    
    # Align to expected training columns (pad missing categories with 0)
    for col in expected_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[list(expected_cols)]
    
    # 1. Transform features
    X_sel = selector.transform(X_encoded)
    X_sc = scaler.transform(X_sel)
    X_final = pd.DataFrame(X_sc, columns=selected_features)
    
    # 2. Predict Probability
    risk_prob = float(model.predict_proba(X_final)[0, 1])
    
    # 3. Compute Linear Feature Contributions (Native SHAP Equivalent)
    # Since X_train_summary isn't in the artifacts, and X_sc is StandardScaled (mean ~ 0),
    # the additive feature contribution is precisely the logistic weight * scaled_value.
    coefs = model.coef_[0]
    patient_shap = coefs * X_final.iloc[0].values
    
    base_value = float(model.intercept_[0])
    feature_names = list(selected_features)
    
    # Extract ALL shap data cleanly for React chart rendering
    shap_data = []
    for i in range(len(feature_names)):
        shap_data.append({
            "feature": feature_names[i],
            "value": float(patient_shap[i]),
            "raw_value": float(X_final.iloc[0, i])  # Scaled value
        })
    # Sort by absolute impact for the frontend waterfall
    shap_data = sorted(shap_data, key=lambda x: abs(x['value']), reverse=True)
    
    # Extract Top 3 increasing drivers for the LLM
    sorted_idx = np.argsort(patient_shap)[::-1]
    top_drivers = [(feature_names[i], patient_shap[i]) for i in sorted_idx[:3] if patient_shap[i] > 0]
    
    return risk_prob, top_drivers, shap_data, base_value

# ==========================================
# GEMINI API WRAPPER
# ==========================================
def generate_live_clinical_insights(risk_prob, top_drivers, doctors_note, api_key):
    """Calls Gemini 2.5 Flash for a live patient via API context."""
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
