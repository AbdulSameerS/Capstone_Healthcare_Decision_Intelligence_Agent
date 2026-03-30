"""
Resume script for the full RAG + SHAP + LLM pipeline.
Skips already-successful patients and retries only those with errors.
"""

import json, os, sys, time, pickle
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'Notebook')
from utils import get_db_connection
from groq import Groq

# ============================================================
#  CONFIGURATION
# ============================================================
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH  = os.path.join(BASE_DIR, "llm_outputs_full.json")
BATCH_DELAY  = 0.5   # slightly longer delay for paid tier safety
# ============================================================

# Load existing results
with open(OUTPUT_PATH) as f:
    existing = json.load(f)

# Identify which hadm_ids already succeeded
done_ids = {r["hadm_id"] for r in existing if "error" not in r}
print(f"Already successful: {len(done_ids)} patients — will skip these.")
print(f"Need to process:    {sum(1 for r in existing if 'error' in r)} patients with errors.")

# ---- Recreate full dataset with hadm_id ----
print("\n[ 1/4 ] Loading model features from DB...")
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

# ---- Load model artifacts ----
print("[ 2/4 ] Loading model artifacts...")
with open(os.path.join(BASE_DIR, 'model_artifacts.pkl'), 'rb') as f:
    artifacts = pickle.load(f)
scaler            = artifacts['scaler']
selector          = artifacts['selector']
model             = artifacts['model']
selected_features = artifacts['selected_features']

# ---- Load RAG data ----
print("[ 3/4 ] Loading RAG summaries...")
with open(os.path.join(BASE_DIR, 'dataset', 'RAG_data_summary.json')) as f:
    rag_list = json.load(f)
rag_data = {list(item.keys())[0]: list(item.values())[0] for item in rag_list}
rag_ids  = set(rag_data.keys())

# ---- Filter to only FAILED patients ----
drop_cols      = [c for c in ['subject_id', 'readmitted_30d'] if c in df_encoded.columns]
X_all          = df_encoded.drop(columns=drop_cols)
hadm_ids       = X_all['hadm_id'].values
X_features     = X_all.drop(columns=['hadm_id'])

rag_mask       = np.array([str(hid) in rag_ids for hid in hadm_ids])
X_rag          = X_features[rag_mask]
ids_rag        = hadm_ids[rag_mask]

# Only process those that previously failed
retry_mask     = np.array([int(hid) not in done_ids for hid in ids_rag])
X_retry        = X_rag[retry_mask]
ids_retry      = ids_rag[retry_mask]
print(f"    Patients to retry: {len(ids_retry)}")

# ---- SHAP ----
print("[ 4/4 ] Computing SHAP values for retry patients...")
X_sel          = selector.transform(X_retry)
X_scl          = scaler.transform(X_sel)
X_df           = pd.DataFrame(X_scl, columns=selected_features)
y_prob         = model.predict_proba(X_df)[:, 1]
explainer      = shap.LinearExplainer(model, X_df)
shap_vals      = explainer(X_df)
print(f"    SHAP done. Shape: {shap_vals.values.shape}")

# ---- LLM ----
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are an expert clinical decision support AI embedded in a hospital readmission prevention system. "
    "You will receive a patient's clinical summary, the AI model's predicted 30-day readmission risk, and the SHAP-identified risk drivers. "
    "Generate a structured response in VALID JSON ONLY with this exact structure: "
    '{"doctor_alert": {"risk_level": "HIGH / MEDIUM / LOW", '
    '"risk_summary": "2-3 sentence summary for the attending physician."}, '
    '"patient_precautions": ["precaution 1", "precaution 2", "precaution 3", "precaution 4"], '
    '"follow_up_recommendations": "1-2 sentence follow-up recommendation."} '
    "Output only valid JSON. No markdown."
)

def build_prompt(hid, prob, shap_row, feature_names, summary):
    risk_level  = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.40 else "LOW")
    sorted_idx  = np.argsort(shap_row)[::-1]
    top_drivers = [(feature_names[i], float(shap_row[i])) for i in sorted_idx[:4] if shap_row[i] > 0]
    drivers_str = "\n".join([f"  - {f} (impact: {v:+.4f})" for f, v in top_drivers]) or "  No significant positive drivers."
    return (
        f"--- PATIENT CLINICAL HISTORY ---\n{summary}\n\n"
        f"--- AI MODEL PREDICTION ---\nPredicted 30-Day Readmission Risk: {prob:.1%}  [{risk_level} RISK]\n\n"
        f"--- KEY RISK DRIVERS (SHAP Analysis) ---\n{drivers_str}"
    )

# Build lookup for existing (successful) results keyed by hadm_id
results_by_id = {r["hadm_id"]: r for r in existing if "error" not in r}

new_count   = 0
error_count = 0
feature_names = shap_vals.feature_names

for i, (hid, prob, shap_row) in enumerate(zip(ids_retry, y_prob, shap_vals.values)):
    hid_str = str(hid)
    summary = rag_data.get(hid_str, "No clinical summary available.")
    prompt  = build_prompt(hid_str, prob, shap_row, feature_names, summary)

    print(f"  [{i+1:04d}/{len(ids_retry)}] hadm_id={hid} risk={prob:.1%} ... ", end="", flush=True)
    try:
        raw    = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=1000
        ).choices[0].message.content
        parsed = json.loads(raw)
        results_by_id[int(hid)] = {
            "hadm_id":                   int(hid),
            "predicted_risk_pct":        round(float(prob) * 100, 1),
            "risk_level":                parsed.get("doctor_alert", {}).get("risk_level", ""),
            "doctor_alert":              parsed.get("doctor_alert", {}),
            "patient_precautions":       parsed.get("patient_precautions", []),
            "follow_up_recommendations": parsed.get("follow_up_recommendations", "")
        }
        new_count += 1
        print("OK")
    except Exception as e:
        results_by_id[int(hid)] = {"hadm_id": int(hid), "predicted_risk_pct": round(float(prob)*100,1), "error": str(e)}
        error_count += 1
        print(f"ERR: {e}")

    time.sleep(BATCH_DELAY)

    # Auto-save every 50 patients
    if (i + 1) % 50 == 0:
        final = list(results_by_id.values())
        with open(OUTPUT_PATH, "w") as f:
            json.dump(final, f, indent=2)
        print(f"  [Checkpoint saved — {i+1} done]")

# Final save
final = list(results_by_id.values())
with open(OUTPUT_PATH, "w") as f:
    json.dump(final, f, indent=2)

ok_total  = sum(1 for r in final if "error" not in r)
err_total = sum(1 for r in final if "error" in r)
print(f"\n{'='*60}")
print(f"DONE! Total in file: {len(final)}")
print(f"  Successful: {ok_total}  |  Errors: {err_total}")
print(f"  New this run: {new_count}  |  New errors: {error_count}")
print(f"  Saved to: {OUTPUT_PATH}")
