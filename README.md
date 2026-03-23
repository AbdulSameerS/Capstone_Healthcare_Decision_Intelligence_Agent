# Heart Failure 30-Day Readmission Prediction
## Healthcare Decision Intelligence Platform

## Project Overview
This project builds an end-to-end machine learning + AI pipeline to predict **30-day hospital readmission** for heart failure (HF) patients using the **MIMIC-IV** clinical database, explain predictions via **SHAP analysis**, and generate **clinician-ready patient summaries** using a **RAG + LLM pipeline**.

The system helps doctors identify high-risk patients before discharge and provides actionable, patient-specific precautions to reduce readmission rates.

## Current Status
✅ Data loading, cohort construction, feature engineering, preprocessing  
✅ Iterative model training with hyperparameter tuning  
✅ SHAP explainability analysis (global + per-patient risk drivers)  
✅ RAG retrieval pipeline (clinical notes + SHAP-grounded prompts)  
✅ LLM generation — doctor alerts + patient precautions (via Groq LLaMA-3.3-70B)  

## Dataset
- **Source:** MIMIC-IV (Medical Information Mart for Intensive Care, version IV)
- **Cohort:** 4,508 heart failure admissions from 4,074 unique patients (after excluding deaths within 30 days)
- **Label:** `readmitted_30d` — binary (1 = readmitted within 30 days, 0 = not readmitted)
- **Class distribution:** 78.5% not readmitted / 21.5% readmitted
- **RAG data:** 1,261 patients with clinical HPI notes and AI-generated summaries (`HPI.json`, `RAG_data_summary.json`)

## Pipeline Architecture

### Step 0–1: Data Loading (`01_Data_Loading.ipynb`)
- All 11 CSV files loaded into a persistent **DuckDB** database for fast SQL-based analytics

### Step 2: Cohort Construction (`02_Cohort_Construction.ipynb`)
- Identified HF admissions via ICD codes, excluded 253 patients who died within 30 days to prevent data leakage

### Step 3: Feature Engineering (`03_Feature_Engineering.ipynb`)
| Feature Group | Features Created |
|---|---|
| **Lab values** (14 tests) | First, last, min, max per admission (56 features) |
| **Demographics** | Age, gender |
| **Admission info** | Type, insurance, marital status, race, discharge location |
| **Clinical** | Length of stay, comorbidity count, procedure count, infection flag, prior admissions |
| **Temporal** | Lab change features (last − first) for all 14 labs |

**Final feature matrix:** 4,508 rows × 137 columns (after one-hot encoding)

### Step 4: Preprocessing (`04_Data_Preprocessing.ipynb`)
- Median imputation for numerical, 'UNKNOWN' for categorical
- One-hot encoding of 6 categorical variables
- Stratified 80/20 train/test split

### Step 5: Model Training (`05_Model_Training.ipynb`)

| Step | What We Did | Test ROC-AUC | Test F1 |
|------|-------------|-------------|---------|
| 5A | Baseline LR + XGBoost | 0.6305 / 0.6065 | 0.3911 / 0.1545 |
| 5B | + Feature Selection (SelectKBest, top 40) | **0.6559** | 0.4056 |
| 5C | + SMOTE oversampling | 0.6448 | 0.40 |
| 5D | + GridSearchCV tuning | 0.6546 | 0.40 |

**Best model:** Logistic Regression (L1, C=0.05, balanced) — **Test ROC-AUC: 0.6546**  
Artifacts saved to `model_artifacts.pkl` (scaler, selector, model, SHAP explainer, selected features)

### Step 6: Risk Prediction Table (`06_Risk_Prediction_Table.ipynb`)
- Generates a ranked risk table for all test patients

### Step 7: SHAP Explainability (`07_SHAP.ipynb`)
- Applied `shap.LinearExplainer` to the Tuned Logistic Regression model
- Generated global **SHAP Summary Plot** (bar) for feature importance
- Generated per-patient **Waterfall Plots** showing individual risk contribution of each feature
- Top global risk drivers: `bicarbonate_first`, `discharge_location_HOME HEALTH CARE`, `hemoglobin_max`, `marital_status_SINGLE`, `num_prior_admissions`

### Step 8: RAG Retrieval Pipeline (`08_RAG_Retrieval.ipynb`)
- Recreates the test set with `hadm_id` preserved for patient-level mapping
- Loads `RAG_data_summary.json` (1,261 AI clinical summaries) and `HPI.json` (raw clinical notes)
- Matches 236 test patients and 1,192 full-dataset patients to their clinical summaries
- Builds structured prompt contexts combining:
  - Patient clinical history
  - Predicted readmission risk (%)
  - Top 3–4 SHAP-identified risk drivers
- Saves 236 formulated prompts to `rag_prompts.json`

### Step 9: LLM Clinical Generation (`09_RAG_LLM_Generation.ipynb`)
Uses **Groq LLaMA-3.3-70B** (free tier) to generate structured clinical output per patient:

```json
{
  "doctor_alert": {
    "risk_level": "HIGH / MEDIUM / LOW",
    "risk_summary": "Clinical explanation for attending physician..."
  },
  "patient_precautions": [
    "Actionable step 1 for the patient",
    "Actionable step 2 for the patient",
    "Actionable step 3 for the patient",
    "Actionable step 4 for the patient"
  ],
  "follow_up_recommendations": "Care team guidance..."
}
```

Results saved to `llm_outputs_full.json`. To resume generation for remaining patients: `python resume_rag_pipeline.py`

## Key Findings
- Our AUC of **0.61–0.65** is consistent with published MIMIC-based HF readmission studies (0.60–0.68)
- **Logistic Regression consistently outperforms Gradient Boosting** on this moderately-sized, imbalanced dataset
- Feature selection provided the **largest single improvement** in model performance
- SHAP analysis reveals `bicarbonate_first`, discharge location, and hemoglobin levels as the strongest readmission predictors

## Tech Stack
- **Database:** DuckDB
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, imbalanced-learn (SMOTE)
- **Explainability:** SHAP (LinearExplainer)
- **LLM / RAG:** Groq API (LLaMA-3.3-70B), google-generativeai (Gemini)
- **Visualization:** Matplotlib
- **App:** Streamlit (`app.py`)

## Repository Structure
```
├── Notebook/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_Cohort_Construction.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Data_Preprocessing.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Risk_Prediction_Table.ipynb
│   ├── 07_SHAP.ipynb
│   ├── 08_RAG_Retrieval.ipynb
│   └── 09_RAG_LLM_Generation.ipynb
├── dataset/
│   ├── HPI.json                    # Raw clinical notes (1,261 patients)
│   └── RAG_data_summary.json       # AI-generated clinical summaries
├── model_artifacts.pkl             # Trained model, scaler, selector, SHAP explainer
├── rag_prompts.json                # Pre-built RAG prompt contexts (236 patients)
├── llm_outputs_full.json           # LLM-generated doctor alerts + precautions
├── resume_rag_pipeline.py          # Resume LLM generation for remaining patients
├── app.py                          # Streamlit dashboard
├── requirements.txt
└── README.md
```

## How to Run
```bash
# 1. Clone the repo
git clone https://github.com/AbdulSameerS/Capstone_Healthcare_Decision_Intelligence_Agent.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks sequentially (01 → 09)

# 4. For LLM generation, set your Groq API key in 09_RAG_LLM_Generation.ipynb
#    Get a free key at: https://console.groq.com

# 5. Launch the Streamlit app
streamlit run app.py
```

## References
- MIMIC-IV Database: https://physionet.org/content/mimiciv/
- Yu & Son (2024) — Heart failure readmission prediction systematic review
- CMS Hospital Readmissions Reduction Program
- Groq LLaMA-3.3-70B: https://console.groq.com

