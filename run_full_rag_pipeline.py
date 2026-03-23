"""
Full RAG + SHAP + LLM Pipeline
Processes all 1,261 patients with clinical summaries.
Generates doctor-facing risk alerts + patient precautions via Groq LLaMA-3.3-70B.
"""

import json, os, sys, time, pickle
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'Notebook')
from utils import get_db_connection

# ============================================================
#  CONFIGURATION
# ============================================================
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
BASE_DIR     = "/Users/sameer/Desktop/AIheathAgent/Capstone_Healthcare_Decision_Intelligence_Agent"
OUTPUT_PATH  = os.path.join(BASE_DIR, "llm_outputs_full.json")
BATCH_DELAY  = 0.4   # seconds between Groq API calls (rate limit safety)
# ============================================================

# ---- STEP 1: Recreate full dataset with hadm_id ----
print("[ 1/5 ] Loading model features from DB...")
con = get_db_connection()
df = con.execute("SELECT * FROM model_features").fetchdf()
con.close()

LABS = [
    'creatinine', 'urea_nitrogen', 'sodium', 'potassium', 'glucose',
    'hemoglobin', 'white_blood_cells', 'platelet_count', 'bicarbonate',
    'calcium_total', 'inrpt', 'ptt', 'troponin_t', 'creatine_kinase_mb_isoenzyme'
]
for lab in LABS:
    df[f'{lab}_change'] = df[f'{lab}_last'] - df[f'{lab}_first']

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for drop_col in ['readmitted_30d', 'hadm_id', 'subject_id']:
    if drop_col in num_cols:
        num_cols.remove(drop_col)
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['gender', 'admission_type', 'insurance', 'marital_status', 'race', 'discharge_location']
for col in cat_cols:
    df[col] = df[col].fillna('UNKNOWN')

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"    Full dataset: {df_encoded.shape[0]} patients, {df_encoded.shape[1]} columns")

# ---- STEP 2: Load model artifacts ----
print("[ 2/5 ] Loading model artifacts (scaler, selector, model)...")
with open(os.path.join(BASE_DIR, 'model_artifacts.pkl'), 'rb') as f:
    artifacts = pickle.load(f)

scaler          = artifacts['scaler']
selector        = artifacts['selector']
model           = artifacts['model']
selected_features = artifacts['selected_features']

# ---- STEP 3: Load RAG data ----
print("[ 3/5 ] Loading RAG clinical summaries...")
with open(os.path.join(BASE_DIR, 'dataset', 'RAG_data_summary.json')) as f:
    rag_list = json.load(f)
rag_data = {list(item.keys())[0]: list(item.values())[0] for item in rag_list}
rag_ids  = set(rag_data.keys())
print(f"    Loaded {len(rag_data)} RAG summaries")

# ---- STEP 4: Filter to RAG patients & run SHAP ----
print("[ 4/5 ] Filtering to RAG patients and running SHAP...")
drop_cols = ['subject_id', 'readmitted_30d']
available_drop = [c for c in drop_cols if c in df_encoded.columns]
X_all      = df_encoded.drop(columns=available_drop)
hadm_ids   = X_all['hadm_id'].values
X_features = X_all.drop(columns=['hadm_id'])

# Filter to only patients with RAG summaries
rag_mask   = np.array([str(hid) in rag_ids for hid in hadm_ids])
X_rag      = X_features[rag_mask]
ids_rag    = hadm_ids[rag_mask]

print(f"    Patients with RAG summaries: {len(ids_rag)}")

# Preprocess
X_rag_selected = selector.transform(X_rag)
X_rag_scaled   = scaler.transform(X_rag_selected)
X_rag_df       = pd.DataFrame(X_rag_scaled, columns=selected_features)

# Get predicted probabilities
y_pred_proba = model.predict_proba(X_rag_df)[:, 1]

# Compute SHAP values
print("    Computing SHAP values (this may take 30-60 seconds)...")
explainer   = shap.LinearExplainer(model, X_rag_df)
shap_values = explainer(X_rag_df)
print(f"    SHAP done. Shape: {shap_values.values.shape}")

# ---- STEP 5: LLM Generation via Groq ----
print(f"[ 5/5 ] Running Groq LLM for {len(ids_rag)} patients...")
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an expert clinical decision support AI embedded in a hospital readmission prevention system.
You will receive a patient's clinical summary, the AI model's predicted 30-day readmission risk, and the key SHAP-identified risk drivers.

Generate a structured response in VALID JSON ONLY:
{
  "doctor_alert": {
    "risk_level": "HIGH / MEDIUM / LOW",
    "risk_summary": "2-3 sentence summary for the attending physician explaining the predicted risk and its key clinical drivers. Be specific and reference the patient's actual conditions."
  },
  "patient_precautions": [
    "Actionable precaution 1 — written clearly for the patient to understand",
    "Actionable precaution 2 — written clearly for the patient to understand",
    "Actionable precaution 3 — written clearly for the patient to understand",
    "Actionable precaution 4 — written clearly for the patient to understand"
  ],
  "follow_up_recommendations": "1-2 sentence follow-up care recommendation for the care team."
}

Output only valid JSON. No markdown fences, no extra text."""

def build_prompt(hadm_id, risk_prob, shap_row, feature_names, rag_summary):
    risk_level = "HIGH" if risk_prob >= 0.70 else ("MEDIUM" if risk_prob >= 0.40 else "LOW")
    sorted_idx  = np.argsort(shap_row)[::-1]
    top_drivers = [(feature_names[i], float(shap_row[i])) for i in sorted_idx[:4] if shap_row[i] > 0]
    drivers_str = "\n".join([f"  - {f} (impact: {v:+.4f})" for f, v in top_drivers]) or "  No significant positive drivers."

    return f"""--- PATIENT CLINICAL HISTORY ---
{rag_summary}

--- AI MODEL PREDICTION ---
Predicted 30-Day Readmission Risk: {risk_prob:.1%}  [{risk_level} RISK]

--- KEY RISK DRIVERS (SHAP Analysis) ---
{drivers_str}"""

def call_groq(prompt):
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
        max_tokens=1000
    )
    return r.choices[0].message.content

results = []
feature_names = shap_values.feature_names

for i, (hid, prob, shap_row) in enumerate(zip(ids_rag, y_pred_proba, shap_values.values)):
    hid_str = str(hid)
    summary = rag_data.get(hid_str, "No clinical summary available.")
    prompt  = build_prompt(hid_str, prob, shap_row, feature_names, summary)

    print(f"  [{i+1:04d}/{len(ids_rag)}] hadm_id={hid} risk={prob:.1%} ... ", end="", flush=True)
    try:
        raw    = call_groq(prompt)
        parsed = json.loads(raw)
        results.append({
            "hadm_id":                int(hid),
            "predicted_risk_pct":     round(float(prob) * 100, 1),
            "risk_level":             parsed.get("doctor_alert", {}).get("risk_level", ""),
            "doctor_alert":           parsed.get("doctor_alert", {}),
            "patient_precautions":    parsed.get("patient_precautions", []),
            "follow_up_recommendations": parsed.get("follow_up_recommendations", "")
        })
        print("OK")
    except Exception as e:
        results.append({"hadm_id": int(hid), "predicted_risk_pct": round(float(prob)*100,1), "error": str(e)})
        print(f"ERR: {e}")
    time.sleep(BATCH_DELAY)

# Save
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

ok_count  = sum(1 for r in results if "error" not in r)
err_count = sum(1 for r in results if "error" in r)
print(f"\n{'='*60}")
print(f"DONE! Processed {len(results)} patients.")
print(f"  Successful: {ok_count}  |  Errors: {err_count}")
print(f"  Results saved to: {OUTPUT_PATH}")

# Print 2 sample outputs
print("\n===== SAMPLE OUTPUTS =====")
for r in [x for x in results if "error" not in x][:2]:
    print(f"\nPatient {r['hadm_id']} | Risk: {r['predicted_risk_pct']}% [{r['risk_level']}]")
    da = r.get("doctor_alert", {})
    print(f"  DOCTOR ALERT: {da.get('risk_summary','')}")
    print("  PATIENT PRECAUTIONS:")
    for p in r.get("patient_precautions", []):
        print(f"    - {p}")
    print(f"  FOLLOW-UP: {r.get('follow_up_recommendations','')}")
