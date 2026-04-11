from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
import pandas as pd
from pydantic import BaseModel
from utils_api import (
    get_db_connection, load_model, load_precomputed_llm_outputs,
    load_rag_summaries, predict_patient_risk, generate_live_clinical_insights,
    calculate_mental_health_burden, generate_chat_reply
)

app = FastAPI(title="Healthcare Capstone API", version="1.0.0")

# Allow the React frontend to communicate with the FastAPI backend without CORS blocking
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to localhost in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to precache models
@app.on_event("startup")
def startup_event():
    load_model()
    get_db_connection()
    load_precomputed_llm_outputs()
    load_rag_summaries()

# Models
class TriageRequest(BaseModel):
    doctors_note: str
    features: Dict[str, Any]

class PatientContext(BaseModel):
    risk_level: str
    risk_pct: float
    precautions: List[str]
    shap_top3: List[str]
    follow_up: str

class ChatRequest(BaseModel):
    hadm_id: str
    message: str
    patient_context: PatientContext

# Mount the frontend application
app.mount("/app", StaticFiles(directory="../frontend", html=True), name="frontend")

@app.get("/api/metrics")
def get_dashboard_metrics():
    try:
        con = get_db_connection()
        total_patients = con.execute("SELECT COUNT(DISTINCT subject_id) FROM model_features").fetchone()[0]
        total_admissions = con.execute("SELECT COUNT(hadm_id) FROM model_features").fetchone()[0]
        readmission_rate = con.execute("SELECT AVG(readmitted_30d) * 100 FROM model_features").fetchone()[0]
        
        return {
            "total_patients": total_patients,
            "total_admissions": total_admissions,
            "readmission_rate_pct": float(readmission_rate),
            "ai_alerts_generated": len(load_precomputed_llm_outputs())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/patients")
def list_available_historical_patients():
    """Returns a list of hadm_ids directly from the database (up to 50)."""
    con = get_db_connection()
    df_patients = con.execute("SELECT hadm_id FROM model_features LIMIT 50").fetchdf()
    return {"hadm_ids": df_patients['hadm_id'].astype(str).tolist()}

@app.get("/api/patient/{hadm_id}")
def get_historical_patient(hadm_id: str):
    llm_outputs = load_precomputed_llm_outputs()
    rag_summaries = load_rag_summaries()
    
    # Reconstruct SHAP data on the fly from duckdb
    con = get_db_connection()
    query = f"SELECT * FROM model_features WHERE hadm_id = {hadm_id}"
    df_patient = con.execute(query).fetchdf()
    
    if df_patient.empty:
        raise HTTPException(status_code=404, detail="Patient not found in features database.")
    
    drop_cols = ['subject_id', 'hadm_id', 'readmitted_30d']
    X_feats = df_patient.drop(columns=[c for c in drop_cols if c in df_patient.columns])
    artifacts = load_model()
    r_prob, t_drivers, shap_data, base_value = predict_patient_risk(artifacts, X_feats)
    
    # Graceful fallback if no LLM output is precomputed
    if hadm_id in llm_outputs:
        insights = dict(llm_outputs[hadm_id])  # copy so we don't mutate cache
        if "predicted_risk_pct" not in insights:
            insights["predicted_risk_pct"] = float(r_prob * 100)
        # Old records don't have doctor_precautions — generate on-the-fly
        if not insights.get("doctor_precautions"):
            original_note = rag_summaries.get(hadm_id, "No narrative available.")
            try:
                gemini_json = generate_live_clinical_insights(r_prob, t_drivers, original_note)
                insights["doctor_precautions"] = gemini_json.get("doctor_precautions", [])
                # Also enrich follow_up if missing
                if not insights.get("follow_up_recommendations"):
                    insights["follow_up_recommendations"] = gemini_json.get("follow_up_recommendations", "")
            except Exception:
                insights["doctor_precautions"] = []
    else:
        # Dynamically generate missing insights using the RAG summary as the doctor's note
        original_note = rag_summaries.get(hadm_id, "No narrative available. Please evaluate based strictly on the provided objective ML risk drivers.")
        try:
            gemini_json = generate_live_clinical_insights(r_prob, t_drivers, original_note)
            insights = {
                "risk_level": gemini_json.get("doctor_alert", {}).get("risk_level", "UNKNOWN"),
                "predicted_risk_pct": float(r_prob * 100),
                "doctor_alert": gemini_json.get("doctor_alert", {}),
                "doctor_precautions": gemini_json.get("doctor_precautions", gemini_json.get("patient_precautions", [])),
                "patient_precautions": gemini_json.get("patient_precautions", []),
                "follow_up_recommendations": gemini_json.get("follow_up_recommendations", "")
            }
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                user_msg = "Gemini Rate Limit Exceeded. Please wait ~30 seconds before generating more AI Insights."
                precautions = ["Rate Limit Active - Please pause briefly."]
            else:
                user_msg = f"Dynamic Generation Failed: {error_msg[:100]}..."
                precautions = ["API Error - Could not generate."]
                
            insights = {
                "risk_level": "UNKNOWN",
                "predicted_risk_pct": float(r_prob * 100),
                "doctor_alert": {"risk_summary": user_msg},
                "patient_precautions": precautions,
                "follow_up_recommendations": "N/A"
            }
            
    return {
        "hadm_id": hadm_id,
        "llm_insights": insights,
        "original_summary": rag_summaries.get(hadm_id, "No original narrative available in RAG database."),
        "shap_explanation": {
            "shap_waterfall": shap_data,
            "base_value": base_value
        },
        "mental_health_burden": calculate_mental_health_burden(df_patient.iloc[0].to_dict()),
        "rag_source": "retrieved" if (hadm_id in rag_summaries and rag_summaries.get(hadm_id, '') not in ['', 'No clinical summary available.']) else "synthesized"
    }

@app.post("/api/chat")
def patient_chat(request: ChatRequest):
    """Answers a patient question about their readmission risk in plain language."""
    try:
        context_dict = request.patient_context.dict()
        reply = generate_chat_reply(request.message, context_dict)
        return {"reply": reply}
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(status_code=429, detail="AI assistant is temporarily unavailable. Please wait a moment and try again.")
        raise HTTPException(status_code=500, detail="Could not generate a response. Please try again.")

@app.get("/api/model/info")
def get_model_info():
    artifacts = load_model()
    feats = artifacts.get('selected_features', [])
    MED_KW = ['loop','ace_arb','beta','aldo','digoxin','anticoag','unique_drugs','furosemide','gdmt']
    med_feats = [f for f in feats if any(k in f for k in MED_KW)]
    return {
        "model_type": artifacts.get("model_type", "LR + FeatureSelect"),
        "model_class": type(artifacts['model']).__name__,
        "feature_count": len(feats),
        "selected_features": feats,
        "medication_features": med_feats,
        "has_xgb": "xgb_model" in artifacts,
        "pipeline": "SelectKBest(k=50) → StandardScaler → LogisticRegression(class_weight='balanced')"
    }

@app.get("/api/triage/baseline")
def get_triage_baseline_patient():
    """Fetches a random patient's tabular data ordered by model feature importance."""
    con = get_db_connection()
    df_baseline = con.execute("SELECT * FROM model_features LIMIT 50").fetchdf().sample(1)
    drop_cols = ['subject_id', 'hadm_id', 'readmitted_30d']
    df_edit = df_baseline.drop(columns=[c for c in drop_cols if c in df_baseline.columns]).reset_index(drop=True)
    patient = df_edit.fillna(0).to_dict(orient='records')[0]

    # Order features by model importance so the UI shows the most impactful ones first
    try:
        artifacts = load_model()
        selected_features = artifacts.get('selected_features', [])
        coefs = artifacts['model'].coef_[0]
        # Map selected feature → abs coefficient importance
        importance = {feat: abs(coefs[i]) for i, feat in enumerate(selected_features)}
        # Sort patient features: selected ones first (by importance), then the rest
        def sort_key(k):
            return -importance.get(k, 0)  # negated so highest importance comes first
        ordered = dict(sorted(patient.items(), key=lambda x: sort_key(x[0])))
        return ordered
    except Exception:
        return patient

@app.post("/api/triage/live")
def generate_live_triage(request: TriageRequest):
    """Processes a live patient scenario via ML & GenAI simultaneously."""
    # Convert incoming feature dictionary to a pandas dataframe
    df_features = pd.DataFrame([request.features])
    
    artifacts = load_model()
    r_prob, t_drivers, shap_data, base_value = predict_patient_risk(artifacts, df_features)
    
    try:
        gemini_json = generate_live_clinical_insights(r_prob, t_drivers, request.doctors_note)
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            user_msg = "Gemini Rate Limit Exceeded. Please wait 30 seconds before analyzing."
        else:
            user_msg = f"LLM Generation Failed: {error_msg[:100]}..."
            
        gemini_json = {
            "error": user_msg,
            "patient_precautions": ["Rate Limit Active - Please pause briefly."]
        }
        
    return {
        "risk_probability_pct": float(r_prob * 100),
        "shap_waterfall": shap_data,
        "base_value": base_value,
        "gemini_insights": gemini_json
    }
