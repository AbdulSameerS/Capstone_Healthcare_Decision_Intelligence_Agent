# Heart Diagnoses Dataset

This dataset is designed around **patient cases** as the fundamental unit and is primarily used for medical analysis and machine learning applications involving heart-related diagnoses. The dataset is derived from the **MIMIC-IV** and **MIMIC-IV-Note** databases, and integrates various aspects of a patient's hospital stay, including diagnosis codes, laboratory tests, ECG reports, and procedures.

## 📁 Dataset Structure

Below is an overview of each file in the dataset:

### 1. `heart_diagnoses.csv`
- **Core file** of the dataset.
- Contains identifiers: `note_id`, `subject_id` (from `discharge.csv` in MIMIC-IV-Note), and `hadm_id`.
- Includes patient details such as **history of present illness**, **chief complaint**, and **exam results**.
- The `reports` field contains **ECG machine-generated reports**, separated by `"|"`.

| Name           | Postgres data type    |   Content                                                                 |
|----------------|------------------------|-------------------------------------------------------------------------|
| note_id        | VARCHAR(25) NOT NULL   | A unique identifier for the given note.                                 |
| subject_id     | INTEGER NOT NULL       | `subject_id` is a unique identifier which specifies an individual patient. |
| hadm_id        | INTEGER NOT NULL       | `hadm_id` is an integer identifier which is unique for each patient hospitalization. |
| note_type      | CHAR(2) NOT NULL       | The type of note recorded in the row. 1. DS - discharge summary; 2. AD - discharge summary addendum. |
| note_seq       | INTEGER NOT NULL       | A monotonically increasing integer which chronologically sorts the notes within `note_type` categories. |
| charttime      | TIMESTAMP NOT NULL     | The time at which the note was charted.                                |
| storetime      | TIMESTAMP              | The time at which the note was stored in the database.                 |
| PHI            | VARCHAR OR None        | Present history of illness.                                            |
| physical_exam  | VARCHAR OR None        | Physical examination results.                                          |
| chief_complaint| VARCHAR OR None        | Chief complaint.                                                       |
| invasions      | VARCHAR OR None        | Invasive surgery.                                                      |
| X-ray          | VARCHAR OR None        | X-ray results.                                                         |
| CT             | VARCHAR OR None        | CT results.                                                            |
| Ultrasound     | VARCHAR OR None        | Ultrasound results.                                                   |
| CATH           | VARCHAR OR None        | CATH results.                                                         |
| ECG            | VARCHAR OR None        | ECG results from discharge reports.                                    |
| MRI            | VARCHAR OR None        | MRI results.                                                          |
| reports        | VARCHAR OR None        | ECG machine reports.                                                  |


### 2. `heart_diagnoses_all.csv`
- Derived from `diagnoses_icd.csv`.
- ICD codes are truncated to the **first three characters**, representing broader disease categories.

| Name        | Postgres data type | Content                                                                 |
|-------------|---------------------|-------------------------------------------------------------------------|
| subject_id  | INTEGER NOT NULL    | `subject_id` is a unique identifier which specifies an individual patient. |
| hadm_id     | INTEGER NOT NULL    | `hadm_id` is an integer identifier which is unique for each patient hospitalization. |
| seq_num     | INTEGER NOT NULL    | The priority assigned to the diagnoses.                                |
| icd_code    | VARCHAR(7)          | `icd_code` is the International Coding Definitions (ICD) code.         |
| long_title  | VARCHAR(255)        | The `long_title` provides the meaning of the ICD code.                 |


### 3. `heart_diagnoses_all_true.csv`
- Also derived from `diagnoses_icd.csv`.
- Contains **full ICD codes** and **complete diagnosis descriptions**.

### 4. `heart_labevents_examination_group.csv`
- Extracted from `labevents.csv`.
- Includes a column `examination_group`, representing the group of the test.

| Name              | Postgres data type | Content                                                                 |
|-------------------|--------------------|-------------------------------------------------------------------------|
| hadm_id           | INTEGER             | `hadm_id` is an integer identifier which is unique for each patient hospitalization. |
| specimen_id       | INTEGER NOT NULL    | Uniquely denoted the specimen from which the lab measurement was made. |
| charttime         | TIMESTAMP(0)        | The time at which the laboratory measurement was charted.              |
| storetime         | TIMESTAMP(0)        | The time at which the measurement was made available in the laboratory system. |
| value             | VARCHAR(200)        | The result of the laboratory measurement and, if it is numeric, the value cast as a numeric data type. |
| valuenum          | DOUBLE PRECISION    |                                                                         |
| valueuom          | VARCHAR(20)         | The unit of measurement for the laboratory concept.                    |
| ref_range_lower   | DOUBLE PRECISION    | Upper and lower reference ranges indicating the normal range for the laboratory measurements. Values outside the reference ranges are considered abnormal. |
| ref_range_upper   | DOUBLE PRECISION    |                                                                         |
| flag              | VARCHAR(10)         | A brief string mainly used to indicate if the laboratory measurement is abnormal. |
| comments          | TEXT                | Deidentified free-text comments associated with the laboratory measurement. |
| label             | VARCHAR(50)         | The `label` column describes the concept which is represented by the `itemid`. |
| fluid             | VARCHAR(50)         | `fluid` describes the substance on which the measurement was made.     |
| category          | VARCHAR(50)         | `category` provides higher level information as to the type of measurement. |
| examination_group | VARCHAR(50)         | This field describes which examination group this item belongs to.     |


### 5. `heart_microbiologyevents.csv`
- Another extract from `labevents.csv`.
- Does **not** contain the `examination_group` column.

| Name               | Postgres data type   | Content                                                                 |
|--------------------|----------------------|-------------------------------------------------------------------------|
| microevent_id      | INTEGER NOT NULL     | A unique integer denoting the row.                                     |
| subject_id         | INTEGER NOT NULL     | `subject_id` is a unique identifier which specifies an individual patient. |
| hadm_id            | INTEGER              | `hadm_id` is an integer identifier which is unique for each patient hospitalization. |
| micro_specimen_id  | INTEGER NOT NULL     | Uniquely denoted the specimen from which the microbiology measurement was made. |
| order_provider_id  | VARCHAR(10)          | `order_provider_id` provides an anonymous identifier for the provider who ordered the microbiology test. |
| chartdate          | TIMESTAMP(0) NOT NULL| `chartdate` is the same as `charttime`, except there is no time available. |
| charttime          | TIMESTAMP(0)         | `charttime` records the time at which an observation was charted.      |
| spec_itemid        | INTEGER NOT NULL     | The specimen which is tested for bacterial growth.                      |
| spec_type_desc     | VARCHAR(100) NOT NULL| The specimen which is tested for bacterial growth.                      |
| test_seq           | INTEGER NOT NULL     | If multiple samples are drawn, the `test_seq` will delineate them.     |
| storedate          | TIMESTAMP(0)         | The date or date-time of when the microbiology result was available.    |
| storetime          | TIMESTAMP(0)         | The date or date-time of when the microbiology result was available.    |
| test_itemid        | INTEGER              | The test performed on the given specimen.                              |
| test_name          | VARCHAR(100)         | The test performed on the given specimen.                              |
| org_itemid         | INTEGER              | The organism, if any, which grew when tested.                          |
| org_name           | VARCHAR(100)         | The organism, if any, which grew when tested.                          |
| isolate_num        | SMALLINT             | For testing antibiotics, the isolated colony (starts at 1).            |
| quantity           | VARCHAR(50)          |                                                                         |
| ab_itemid          | INTEGER              | If an antibiotic was tested, the antibiotic item ID is listed here.    |
| ab_name            | VARCHAR(30)          | If an antibiotic was tested, the antibiotic name is listed here.       |
| dilution_text      | VARCHAR(10)          | Dilution values when testing antibiotic sensitivity.                   |
| dilution_comparison| VARCHAR(20)          | Dilution values when testing antibiotic sensitivity.                   |
| dilution_value     | DOUBLE PRECISION     | Dilution values when testing antibiotic sensitivity.                   |
| interpretation     | VARCHAR(5)           | Interpretation of antibiotic sensitivity (S, R, I, or P).              |
| comments           | TEXT                 | Deidentified free-text comments associated with the microbiology measurement. |


### 6. `heart_labevents_first_lab.csv`
- Subset of `heart_labevents_examination_group.csv`.
- Retains **only the first occurrence** of each examination per hadm_id.

### 7. `heart_microbiologyevents_first_micro.csv`
- Subset of `heart_microbiologyevents.csv`.
- Keeps **only the first instance** of each microbiology test for each patient.

### 8. `heart_procedures.csv`
- Derived from `procedures.csv`.
- Lists **procedures performed** during the patient's hospital stay.

| Name        | Postgres data type | Content                                                                 |
|-------------|---------------------|-------------------------------------------------------------------------|
| subject_id  | INTEGER NOT NULL    | `subject_id` is a unique identifier which specifies an individual patient. |
| hadm_id     | INTEGER NOT NULL    | `hadm_id` is an integer identifier which is unique for each patient hospitalization. |
| seq_num     | INTEGER NOT NULL    | An assigned priority for procedures which occurred within the hospital stay. |
| chartdate   | DATE NOT NULL       | The date of the associated procedures. Date does **not** strictly correlate with `seq_num`. |
| icd_code    | VARCHAR(7)          | `icd_code` is the International Coding Definitions (ICD) code.         |
| long_title  | VARCHAR(255)        | The title fields provide a brief definition for the given procedure code. |


### 9. `HPI.json`
- Used as part of a **Retrieval-Augmented Generation (RAG)** system.
- Stores data on **History of Present Illness (HPI)** for RAG applications.

### 10. `RAG_data.json`
- Also supports the **RAG** system.
- Serves as a general database of **retrievable knowledge** for use in machine learning models.

## 🔗 Relationships

- All data files are **linked via `hadm_id`**, enabling relational joins.
- `note_id` and `subject_id` come from MIMIC-IV-Note and are only present in `heart_diagnoses.csv`.

## 📦 Usage Instructions

1. **Setup Environment**:
   - Ensure you have appropriate access to MIMIC-IV and MIMIC-IV-Note data.
   - Recommended environment includes Python 3.8+ with `pandas`, `numpy`, and `json` libraries installed.

2. **Loading the Data**:
   ```python
   import pandas as pd
   df = pd.read_csv("heart_diagnoses.csv")
   ```

3. **Linking Files**:
   Use `hadm_id` to merge files for a comprehensive view:
   ```python
   diagnoses = pd.read_csv("heart_diagnoses_all.csv")
   merged = pd.merge(df, diagnoses, on="hadm_id", how="left")
   ```

## 📄 License & Attribution

This dataset is derived from the **MIMIC-IV** and **MIMIC-IV-Note** databases provided by the **MIT Lab for Computational Physiology**. Ensure compliance with their [data use agreement](https://physionet.org/content/mimiciv/).
