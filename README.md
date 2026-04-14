# Healthcare Decision Intelligence Agent
### Predicting 30-Day Heart Failure Readmission with Explainable ML, RAG, and Generative AI

DS 5500 — Capstone: Applications in Data Science  
Team 20 | Abdul Sameer Shaik | Kirubhaharan Joseph Abraham  
Instructor: Dr. Fatema Nafa | Northeastern University | Spring 2026

---

## What This Project Does

Heart failure patients are frequently readmitted to the hospital within 30 days of discharge — and in many cases, that readmission could have been prevented. The problem is that most hospitals either don't have a good way to identify which patients are truly high-risk before they leave, or if they do have a risk score, it's just a number with no explanation attached to it.

This project builds a full system that addresses both sides of the problem. It predicts which heart failure patients are likely to be readmitted within 30 days, explains why using SHAP feature attribution, and then uses a Retrieval-Augmented Generation (RAG) pipeline with Google Gemini to write two different reports from the same patient data — one for the doctor with clinical language, and one for the patient in plain English they can actually understand.

Everything is served through a web application where clinicians can look up any patient, run a live triage for a new admission, and patients can ask follow-up questions through a simple chatbot.

---

## Project Status

- Data loading, cohort construction, feature engineering, and preprocessing are complete (Notebooks 01–04)
- Model training across 5 experimental steps is complete, final model selected (Notebook 05)
- SHAP explainability analysis with global and per-patient visualizations is complete (Notebook 06)
- RAG retrieval pipeline with ChromaDB semantic vector store is complete (Notebooks 07, 07b)
- Batch LLM generation for 2,263 patients using Gemini 2.5 Flash is complete (Notebook 08)
- Multi-dimensional LLM evaluation including BERTScore and Flesch readability is complete (Notebook 09)
- Full-stack web application with FastAPI backend and VanillaJS frontend is complete
- Final technical report and presentation are complete

---

## The Dataset

We used MIMIC-IV, a publicly available and fully de-identified electronic health record database from Beth Israel Deaconess Medical Center in Boston, covering hospital admissions from 2008 to 2019. Access requires completing a data use agreement through PhysioNet — the data is IRB-exempt because it contains no personally identifiable information.

We filtered the full database down to heart failure patients by looking for the relevant ICD diagnosis codes. After removing records with data quality issues, patients who died in hospital, and patients who died within 30 days of discharge (since they can't be readmitted), we ended up with 4,508 admissions from 4,074 unique patients. About 21.5% of them were readmitted within 30 days.

---

## How the Pipeline Works

### Step 1 — Data Loading
All MIMIC-IV tables are loaded into a DuckDB database. This makes SQL-based queries fast and keeps everything in one place without needing a full database server.

### Step 2 — Cohort Construction
We identified heart failure admissions using ICD-9 code 428.x and ICD-10 code I50.x, then carefully removed edge cases that would introduce data leakage into the labels — specifically patients who died during the admission or within 30 days after discharge.

### Step 3 — Feature Engineering
We built 146 features organized into six groups:

| Group | What it includes |
|---|---|
| Lab values | First, last, min, max, and change (last minus first) for 14 lab tests — 56 features total |
| Medications | GDMT drug flags (loop diuretics, beta-blockers, ACE/ARB, etc.), GDMT composite score, furosemide dose, unique drug count |
| Demographics | Age, gender |
| Admission details | Admission type, insurance, marital status, race, discharge location |
| Administrative | Length of stay, number of prior admissions, number of ICD diagnoses |
| Comorbidities | Diabetes, CKD, COPD, atrial fibrillation, hypertension |
| Vital signs | Mean heart rate, blood pressure, oxygen saturation, respiratory rate |

The lab change features (creatinine rising, for example) were particularly important because they capture how a patient's condition was trending during the stay, not just a snapshot.

### Step 4 — Preprocessing
Numerical columns were imputed with medians computed on the training set only. Categorical columns were filled with the string 'UNKNOWN' rather than mode imputation, because the fact that something is unknown is itself a signal. We then applied one-hot encoding and did a stratified 80/20 train/test split.

### Step 5 — Model Training

We ran five rounds of experiments to find the best model:

| Step | Model | AUC-ROC | F1 | What we learned |
|---|---|---|---|---|
| Baseline | Majority class | 0.500 | 0.215 | Reference point |
| 5A | Logistic Regression, all 146 features | 0.6305 | 0.3911 | Decent start |
| 5A | Gradient Boosting, all 146 features | 0.6065 | 0.1545 | GB completely failed on the minority class |
| 5B | LR + SelectKBest (top 50 features) | 0.6559 | 0.4056 | Biggest gain in the whole experiment |
| 5C | LR + SMOTE oversampling | 0.6448 | 0.4000 | SMOTE added noise and made things slightly worse |
| 5D | LR + GridSearchCV tuning | 0.5987 | 0.3614 | Overfitting to cross-validation folds |
| 5E | LR + XGBoost ensemble | ~0.670 | ~0.410 | Small gain, too complex for the benefit |
| **Final** | **LR + SelectKBest (k=50)** | **0.6739** | **0.4142** | Best overall |

The final model uses Logistic Regression with C=0.1, L2 penalty, balanced class weighting, and SelectKBest with k=50. We chose logistic regression over more complex models for three reasons: the dataset is not large enough for deep learning, clinical interpretability is a hard requirement, and logistic regression allows exact SHAP computation without any approximation.

### Step 6 — SHAP Explainability
For logistic regression, the SHAP value for each feature is simply the model coefficient multiplied by the scaled feature value. This is exact — no sampling or approximation needed. We generated summary plots, beeswarm plots, per-patient waterfall charts, and a medication impact chart. The top risk drivers across the cohort were creatinine change, number of prior admissions, length of stay, furosemide dose, and GDMT score.

### Step 7 and 7b — RAG Pipeline
We embedded all 4,508 MIMIC-IV discharge summaries using Sentence Transformers (all-MiniLM-L6-v2), producing 384-dimensional vectors, and stored them in a ChromaDB vector store. When a patient is analyzed, the system retrieves the top 3 most semantically similar past discharge summaries using cosine similarity and injects them into the Gemini prompt as clinical context. This grounds the AI-generated reports in real clinical language rather than letting the model hallucinate.

### Step 8 — LLM Generation
Google Gemini 2.5 Flash receives the patient's risk probability, their top 3 SHAP drivers, and the RAG-retrieved clinical context. It returns a structured JSON with six fields:

- `doctor_alert` — risk level (HIGH/MEDIUM/LOW) and a 2-3 sentence clinical summary for the attending physician
- `doctor_precautions` — four specific clinical interventions the doctor should take
- `patient_precautions` — four plain-language steps the patient can understand and act on
- `follow_up_recommendations` — hospital-side logistics for the care team
- `patient_follow_up` — what the patient personally needs to do after discharge

We pre-computed outputs for 2,263 patients. The remaining patients get their report generated live on first load. We enforce JSON output via `response_mime_type='application/json'` which gave us 100% parseable output across all patients.

### Step 9 — LLM Evaluation
We evaluated the LLM outputs across six dimensions:

| Metric | Value | Threshold | Result |
|---|---|---|---|
| AUC-ROC | 0.6739 | 0.65 | Pass |
| Output completeness | 100% | 100% | Pass |
| BERTScore F1 | 0.8410 | 0.75–0.92 | Pass |
| Average Flesch Reading Ease (patient text) | 60.4 | 60 | Pass |
| Risk level alignment with ground truth | 71.1% | 80% | Fair |
| Records with Flesch score above 60 | 56.5% | 80% | Fail |

The BERTScore of 0.8410 tells us that doctor and patient texts are semantically covering the same clinical situation but are written differently enough to confirm that the plain-language simplification is actually happening. The Flesch failure is our most significant quality gap — medically complex patients tend to get summaries with more clinical jargon bleeding through.

---

## The Web Application

The app runs at `http://localhost:8000/app` and has four views:

- **Dashboard** — shows total patients, admissions, readmission rate, and number of AI alerts generated
- **Patient Analysis** — enter an admission ID to see the SHAP waterfall chart, doctor report, patient report, chatbot, and the RAG-retrieved clinical context
- **Live Triage** — manually enter clinical features for a new patient to get a real-time risk score and Gemini report
- **Model Information** — shows the pipeline configuration and all 50 selected features

The patient chatbot is built into the Patient Analysis view. It sends the patient's full context (risk level, top SHAP drivers, care instructions, follow-up plan) with every message and responds in under 100 words in plain language. It's designed to be stateless — no session data is stored server-side.

There's also a Mental Health Burden score computed from three proxy signals — number of prior admissions, number of unique medications, and length of stay. It's not a diagnosis; it's a flag to alert clinicians when a social work or psychiatry consultation might be warranted.

---

## Testing

We wrote 22 test cases across three levels:

**Data pipeline unit tests (UT-01 to UT-08)** cover cohort size, readmission rate, lab change computation, missing value imputation, one-hot encoding, stratified split balance, feature alignment, and SelectKBest output shape.

**Model unit tests (MT-01 to MT-07)** cover risk probability range, SHAP sum property, SHAP sort order, model reproducibility, artifact loading, zero-input stability, and column alignment with missing features.

**API integration tests (IT-01 to IT-07)** cover patient lookup for pre-computed records, 404 for invalid IDs, live triage endpoint response, chatbot response length and format, Gemini rate limit handling, metrics endpoint, and JSON schema compliance.

---

## Running the Project

```bash
# Clone the repo
git clone https://github.com/AbdulSameerS/Capstone_Healthcare_Decision_Intelligence_Agent.git
cd Capstone_Healthcare_Decision_Intelligence_Agent

# Install dependencies
pip install -r requirements.txt

# Place MIMIC-IV tables in the dataset/ folder, then run the notebooks in order:
# 01 → 02 → 03 → 04 → 05 → 06 → 07 → 07b → 08 → 09

# Set your Gemini API key
export GEMINI_API_KEY="your_key_here"

# Start the backend
uvicorn backend.main:app --reload --port 8000

# Open in browser
# http://localhost:8000/app
```

**Available API endpoints:**

| Endpoint | Method | What it does |
|---|---|---|
| `/api/metrics` | GET | Dashboard population-level metrics |
| `/api/patients` | GET | Lists available admission IDs |
| `/api/patient/{hadm_id}` | GET | Full patient analysis with ML, SHAP, LLM, and RAG |
| `/api/triage/baseline` | GET | Fetches a random patient for the live triage view |
| `/api/triage/live` | POST | Real-time risk prediction and Gemini report |
| `/api/chat` | POST | Patient chatbot |
| `/api/model/info` | GET | Model configuration and selected features |

---

## Repository Structure

```
Capstone_Healthcare_Decision_Intelligence_Agent/
├── Notebook/
│   ├── 01_Data_Loading.ipynb
│   ├── 02_Cohort_Construction.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 03b_Medications_Vitals.ipynb
│   ├── 04_Data_Preprocessing.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_SHAP.ipynb
│   ├── 07_RAG_Retrieval.ipynb
│   ├── 07b_Vector_RAG.ipynb
│   ├── 08_RAG_LLM_Generation.ipynb
│   └── 09_LLM_Evaluation.ipynb
├── backend/
│   ├── main.py
│   └── utils_api.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── dataset/
│   ├── hf_project.duckdb
│   └── RAG_data_summary.json
├── model_artifacts.pkl
├── llm_outputs_semantic.json
├── rag_prompts_semantic.json
├── DS5500_Final_Technical_Report.tex
├── DS5500_Final_Presentation.md
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Database:** DuckDB
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn (Logistic Regression, SelectKBest, StandardScaler)
- **Explainability:** Linear SHAP attribution
- **RAG:** ChromaDB, Sentence Transformers (all-MiniLM-L6-v2)
- **LLM:** Google Gemini 2.5 Flash via google-genai SDK
- **LLM evaluation:** bert-score, textstat
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** VanillaJS, Chart.js
- **Visualization:** Matplotlib, SHAP

---

## Important Notes

MIMIC-IV is fully de-identified and no patient information is stored, transmitted, or displayed by this application. Gemini API calls only receive anonymized clinical feature summaries — no names, dates of birth, or identifiers of any kind.

This system is a research prototype built for the DS 5500 capstone course. It is not intended for direct clinical use and has not been validated for deployment. Clinical adoption would require prospective validation on independent patient populations and regulatory approval.

---

## References

- MIMIC-IV: https://physionet.org/content/mimiciv/
- Frizzell et al. (2017) — Prediction of 30-day all-cause readmissions in heart failure, JAMA Cardiology
- Lundberg & Lee (2017) — A unified approach to interpreting model predictions, NeurIPS
- Lewis et al. (2020) — Retrieval-Augmented Generation for knowledge-intensive NLP tasks, NeurIPS
- Zhang et al. (2019) — BERTScore: Evaluating text generation with BERT, arXiv
- Google Gemini 2.5 Flash: https://ai.google.dev
- CMS Hospital Readmissions Reduction Program: https://www.cms.gov