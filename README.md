# Heart Failure 30-Day Readmission Prediction

## Project Overview
This project builds a machine learning pipeline to predict **30-day hospital readmission** for heart failure (HF) patients using the **MIMIC-IV** clinical database. The goal is to identify patients at high risk of returning to the hospital within 30 days of discharge, enabling targeted interventions to improve outcomes and reduce healthcare costs.

## Current Status
**Completed** — Data loading, cohort construction, feature engineering, preprocessing, and iterative model training with hyperparameter tuning.

**Upcoming** — SHAP explainability analysis, final model evaluation, and clinical interpretation.

## Dataset
- **Source:** MIMIC-IV (Medical Information Mart for Intensive Care, version IV)
- **Cohort:** 4,508 heart failure admissions from 4,074 unique patients (after excluding deaths within 30 days)
- **Label:** `readmitted_30d` — binary (1 = readmitted within 30 days, 0 = not readmitted)
- **Class distribution:** 78.5% not readmitted / 21.5% readmitted

## Pipeline Architecture

### Step 0–1: Data Loading
- All 11 CSV files loaded into a persistent **DuckDB** database on Google Drive for fast SQL-based analytics
- Tables include: `admissions`, `patients`, `diagnoses_icd`, `heart_diagnoses`, `heart_labevents_first_lab`, `heart_labevents_examination_group`, `heart_procedures`, `heart_microbiologyevents`, and others

### Step 2: Cohort Construction (`hf_labeled`)
- Identified HF admissions using ICD codes from `heart_diagnoses` joined with `admissions`
- Removed 175 invalid rows with `dischtime < admittime`
- Computed 30-day readmission label using a self-join on `admissions`
- **Excluded 253 patients who died within 30 days** to prevent data leakage (dead patients cannot be readmitted)

### Step 3: Feature Engineering
| Feature Group | Source Table | Features Created |
|---|---|---|
| **Lab values** (14 tests) | `heart_labevents_examination_group` | First, last, min, max per admission (56 features) |
| **Demographics** | `patients` | Age, gender |
| **Admission info** | `admissions` | Admission type, insurance, marital status, race, discharge location |
| **Clinical** | Multiple | Length of stay, comorbidity count, procedure count, infection flag, prior admission count |
| **Temporal** | Derived | Lab change features (last − first) for all 14 labs |

**Final feature matrix:** 4,508 rows × 137 columns (after one-hot encoding)

### Step 4: Preprocessing
1. **Lab change features** — `last - first` values to capture patient trajectory
2. **Missing value imputation** — Median for numerical, 'UNKNOWN' for categorical
3. **One-hot encoding** — 6 categorical variables → 61 binary columns
4. **Stratified 80/20 train/test split** preserving class ratio

### Step 5: Model Training & Iterative Improvement

| Step | What We Did | Test ROC-AUC | Test F1 | Key Insight |
|------|-------------|-------------|---------|-------------|
| 5A | Baseline LR + XGBoost | 0.6305 / 0.6065 | 0.3911 / 0.1545 | LR with `class_weight='balanced'` outperforms XGBoost |
| 5B | + Feature Selection (SelectKBest, top 40) | **0.6559** / 0.6083 | 0.4056 / 0.3622 | Better features > better algorithms |
| 5C | + SMOTE oversampling | 0.6448 / 0.6091 | 0.40 / 0.13 | SMOTE adds no value; `class_weight` already handles imbalance |
| 5D | + GridSearchCV tuning | 0.6546 / 0.6434 | 0.40 / 0.385 | Best LR: C=0.05, L1 penalty, CV AUC=0.6248 |

**Best model:** Logistic Regression (L1, C=0.05, balanced) — **Test ROC-AUC: 0.6546, CV ROC-AUC: 0.6248**

## Key Findings
- Our AUC of **0.61–0.65** is consistent with published MIMIC-based HF readmission studies (0.60–0.68)
- The moderate performance reflects the **inherent difficulty** of readmission prediction — post-discharge factors (medication adherence, social support, lifestyle) are not captured in structured EHR data
- **Logistic Regression consistently outperforms Gradient Boosting** on this moderately-sized, imbalanced dataset
- Feature selection provided the **largest single improvement** in model performance

## Tech Stack
- **Database:** DuckDB (in-memory analytical SQL engine)
- **Data Processing:** Pandas, NumPy, PyArrow
- **Machine Learning:** Scikit-learn (Logistic Regression, Gradient Boosting, GridSearchCV, SelectKBest)
- **Imbalanced Learning:** imbalanced-learn (SMOTE)
- **Visualization:** Matplotlib
- **Environment:** Google Colab (GPU: A100)

## Repository Structure
```
├── Project_Data_Loading.ipynb    # Main notebook (Steps 0–5D)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── DataScience_Capstone_Project/ # Data folder (on Google Drive)
    ├── admissions.csv
    ├── patients.csv
    ├── diagnoses_icd.csv
    ├── heart_diagnoses.csv
    ├── heart_labevents_first_lab.csv
    ├── heart_labevents_examination_group.csv
    ├── heart_diagnoses_all.csv
    ├── heart_diagnoses_all_true.csv
    ├── heart_procedures.csv
    ├── heart_microbiologyevents.csv
    ├── heart_microbiologyevents_first_micro.csv
    └── hf_project.duckdb
```

## How to Run
1. Clone this repository
2. Upload the data files to your Google Drive under `DataScience_Capstone_Project/`
3. Open `Project_Data_Loading.ipynb` in Google Colab
4. Mount Google Drive and run all cells sequentially
5. Install dependencies: `pip install -r requirements.txt`

## References
- MIMIC-IV Database: https://physionet.org/content/mimiciv/
- Yu & Son (2024) — Heart failure readmission prediction systematic review
- CMS Hospital Readmissions Reduction Program
