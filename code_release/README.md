# Landmark-based MV Prediction

Shareable analysis code for landmark-based prediction of tracheostomy and short-term mortality in mechanically ventilated patients.

## Purpose

This directory contains a public-facing version of the analysis workflow. The current release focuses on:

- removal of local machine-specific paths
- conversion of the original notebook-style workflow into command-line scripts
- separation of reproducibility notes from manuscript text
- preparation of a repository structure suitable for GitHub sharing

## Current contents

```text
code_release/
├─ scripts/
│  └─ 01_build_landmark_supplement.py
├─ docs/
│  └─ REPRODUCIBILITY_CHECKLIST.md
├─ .gitignore
├─ requirements.txt
└─ run_example.sh
```

## Included script

### `scripts/01_build_landmark_supplement.py`
Builds:
- landmark eligibility summaries by INDEX
- outcome-specific inclusion flags for LM0 and LM3
- feature dictionaries used for supplementary tables

Example:

```bash
python scripts/01_build_landmark_supplement.py \
  --excel-path data/your_dataset.xlsx \
  --out-dir outputs/supplement
```

## Expected input data

The script assumes an Excel workbook containing structured routinely collected data used in the manuscript workflow.
Expected columns include:

- identifiers: `INDEX`
- baseline date: `Base_Date` or equivalent
- duration on mechanical ventilation: `人工呼吸_連続日数` or `All_Day`
- outcome dates: `死亡日`, `気管切開日` or their English alternatives
- feature columns such as `Day0_*`, `Day3_*`, medication variables, laboratory variables, and department labels

Because the raw data include individual-level clinical information, no sample dataset is bundled in this repository.

## Environment

The original analysis environment used Python 3.11 with the following major packages:

- numpy
- pandas
- scipy
- scikit-learn
- statsmodels
- xgboost
- lightgbm
- optuna
- matplotlib
- shap
- openpyxl
- pillow

Install with:

```bash
pip install -r requirements.txt
```

## Notes for public release

Before publishing or updating this repository, confirm the following:

- no patient-level data are included
- no facility-specific internal paths remain
- no temporary local files or exported spreadsheets with raw identifiers are committed
- model artifacts do not contain restricted data
- the manuscript and repository versions are aligned

## Manuscript linkage

This repository accompanies the manuscript:

**Landmark-Based Prediction of Tracheostomy and Short-Term Mortality in Mechanically Ventilated Patients Using Routinely Collected Hospital Data: A Multicenter Retrospective Cohort Study**

## Disclaimer

This code is provided for research use. It was developed for retrospective model development and evaluation, not for direct real-time clinical deployment without further validation, recalibration, governance review, and implementation testing.
