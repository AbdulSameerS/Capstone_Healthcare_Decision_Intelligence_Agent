import duckdb
import pickle
import pandas as pd
import numpy as np
import os
import json
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, 'dataset')
DB_PATH           = os.path.join(ORIGINAL_DATA_DIR, 'hf_project.duckdb')
MODEL_PATH        = os.path.join(BASE_DIR, 'model_artifacts.pkl')
LLM_PATH          = os.path.join(BASE_DIR, 'llm_outputs_semantic.json')   # semantic outputs (4,508 patients)
RAG_SUMMARY_PATH  = os.path.join(BASE_DIR, 'dataset', 'RAG_data_summary.json')

# Gemini API key — set GEMINI_API_KEY environment variable before starting the server
# e.g.  export GEMINI_API_KEY="AIza..."
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# ==========================================
# IN-MEMORY CACHE
# ==========================================
_MODEL_ARTIFACTS = None
_LLM_OUTPUTS     = None
_RAG_SUMMARIES   = None

def get_db_connection():
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
        # Hot-patch for scikit-learn version mismatch
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
        print(f"Loaded {len(_LLM_OUTPUTS)} precomputed LLM outputs from {os.path.basename(LLM_PATH)}")
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
# INFERENCE LOGIC
# ==========================================
def predict_patient_risk(artifacts, patient_features_df):
    """Passes raw features through scikit-learn pipeline & returns API-friendly SHAP data."""
    scaler           = artifacts['scaler']
    selector         = artifacts['selector']
    model            = artifacts['model']
    selected_features = artifacts['selected_features']
    expected_cols    = selector.feature_names_in_

    # Add lab change features
    LABS = [
        'creatinine', 'urea_nitrogen', 'sodium', 'potassium', 'glucose',
        'hemoglobin', 'white_blood_cells', 'platelet_count', 'bicarbonate',
        'calcium_total', 'inrpt', 'ptt', 'troponin_t', 'creatine_kinase_mb_isoenzyme'
    ]
    for lab in LABS:
        if f'{lab}_last' in patient_features_df.columns and f'{lab}_first' in patient_features_df.columns:
            patient_features_df[f'{lab}_change'] = (
                patient_features_df[f'{lab}_last'] - patient_features_df[f'{lab}_first']
            )

    # Encode categoricals
    cat_cols = ['gender', 'admission_type', 'insurance', 'marital_status', 'race', 'discharge_location']
    for col in cat_cols:
        if col in patient_features_df.columns:
            patient_features_df[col] = patient_features_df[col].fillna('UNKNOWN')

    patient_features_df = patient_features_df.fillna(0)
    X_encoded = pd.get_dummies(patient_features_df, columns=[c for c in cat_cols if c in patient_features_df.columns])

    # Align to training columns
    for col in expected_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[list(expected_cols)]

    X_sel   = selector.transform(X_encoded)
    X_sc    = scaler.transform(X_sel)
    X_final = pd.DataFrame(X_sc, columns=selected_features)

    risk_prob = float(model.predict_proba(X_final)[0, 1])

    # Linear feature contributions (SHAP equivalent for logistic regression)
    coefs        = model.coef_[0]
    patient_shap = coefs * X_final.iloc[0].values
    base_value   = float(model.intercept_[0])
    feature_names = list(selected_features)

    shap_data = sorted([
        {"feature": feature_names[i], "value": float(patient_shap[i]), "raw_value": float(X_final.iloc[0, i])}
        for i in range(len(feature_names))
    ], key=lambda x: abs(x['value']), reverse=True)

    sorted_idx  = np.argsort(patient_shap)[::-1]
    top_drivers = [(feature_names[i], patient_shap[i]) for i in sorted_idx[:3] if patient_shap[i] > 0]

    return risk_prob, top_drivers, shap_data, base_value

# ==========================================
# GROQ LIVE CLINICAL INSIGHTS
# ==========================================
def generate_live_clinical_insights(risk_prob, top_drivers, doctors_note, api_key=None):
    """Calls Gemini 2.5 Flash for a live patient triage request."""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment variable before starting the server."
        )

    drivers_str = "\n".join([f"- {feat} (Impact score: {val:.4f})" for feat, val in top_drivers])
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
        '"risk_summary": "2-3 sentence clinical summary for the attending physician explaining why this patient is at risk, referencing the SHAP drivers and clinical history."}, '
        '"doctor_precautions": ['
        '"Clinical precaution 1: specific medical action the physician should take, referencing labs or risk factors.", '
        '"Clinical precaution 2: another specific clinical intervention or monitoring step.", '
        '"Clinical precaution 3: medication or treatment consideration for this patient.", '
        '"Clinical precaution 4: discharge planning or specialist referral recommendation."], '
        '"patient_precautions": ['
        '"Simple step 1 written in plain language the patient can understand — no medical jargon.", '
        '"Simple step 2 written in plain language — practical daily action for the patient.", '
        '"Simple step 3 written in plain language — warning signs the patient should watch for.", '
        '"Simple step 4 written in plain language — lifestyle or medication adherence tip."], '
        '"follow_up_recommendations": "2-3 sentences written FOR THE DOCTOR AND CARE TEAM about hospital-side logistics: e.g. ensure bed availability, alert cardiology team, prepare medication stock, schedule specialist review, flag in EHR for rapid readmission protocol. Focus on what the hospital must prepare before the patient potentially returns.", '
        '"patient_follow_up": "2-3 sentences written FOR THE PATIENT in simple friendly language about what they personally need to do: e.g. book a follow-up appointment within X days, arrange transport, pick up prescriptions, keep emergency contact ready, call the helpline if symptoms appear. No medical jargon — practical daily actions only."}. '
        'No markdown. No extra text. Only strictly valid JSON.'
    )

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt_context,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type='application/json',
            temperature=0.3
        )
    )
    return json.loads(response.text)


def generate_chat_reply(message: str, patient_context: dict, api_key=None) -> str:
    """Calls Gemini 2.5 Flash to answer a patient question in plain language."""
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("No Gemini API key found.")

    risk_level  = patient_context.get("risk_level", "UNKNOWN")
    risk_pct    = float(patient_context.get("risk_pct", 0))
    precautions = patient_context.get("precautions", [])
    shap_top3   = patient_context.get("shap_top3", [])
    follow_up   = patient_context.get("follow_up", "")

    prec_text = "\n".join(f"- {p}" for p in precautions) or "None provided."
    shap_text = ", ".join(shap_top3) if shap_top3 else "Not available."

    SYSTEM_INSTRUCTION = (
        f"You are a friendly, caring health assistant helping a patient understand their "
        f"heart failure readmission risk. "
        f"The patient's 30-day readmission risk is {risk_level} ({risk_pct:.1f}%). "
        f"Their top clinical risk factors are: {shap_text}. "
        f"Their care instructions are:\n{prec_text}\n"
        f"Their follow-up plan: {follow_up}\n\n"
        f"Rules: Answer in simple, friendly language a non-medical person can understand. "
        f"Never diagnose or prescribe. Never tell the patient to change their medications. "
        f"Always encourage them to consult their doctor for medical decisions. "
        f"Keep answers under 100 words. Be warm and reassuring."
    )

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.5
        )
    )
    return response.text.strip()


def calculate_mental_health_burden(patient_row):
    """Compute mental health burden proxy from clinical features."""
    import math
    def safe_float(key, default=0.0):
        val = patient_row.get(key, default)
        if val is None: return default
        try:
            f = float(val)
            return default if math.isnan(f) else f
        except: return default

    score = 0
    signals = []
    num_prior = safe_float('num_prior_admissions')
    num_drugs  = safe_float('num_unique_drugs')
    los        = safe_float('los_days', safe_float('length_of_stay_days', safe_float('los')))

    if num_prior >= 3:
        score += 1
        signals.append(f"{int(num_prior)} prior admissions — chronic readmission pattern")
    elif num_prior >= 1:
        signals.append(f"{int(num_prior)} prior admission(s) noted")
    if num_drugs >= 10:
        score += 1
        signals.append(f"Polypharmacy: {int(num_drugs)} medications — high complexity")
    elif num_drugs >= 5:
        signals.append(f"{int(num_drugs)} active medications")
    if los >= 7:
        score += 1
        signals.append(f"Extended stay: {int(los)} days — severe episode")
    elif los >= 3:
        signals.append(f"Hospital stay: {int(los)} days")
    if not signals:
        signals = ['No significant burden signals from available proxy features']

    level = 'HIGH' if score >= 2 else 'MEDIUM' if score == 1 else 'LOW'
    return {'level': level, 'score': score, 'max_score': 3, 'signals': signals}
