# Landmark-based MV Prediction

Reusable code package for developing and evaluating landmark-based prediction models for tracheostomy and short-term mortality in mechanically ventilated patients.

## Repository purpose

This directory contains a cleaned, shareable version of the analysis code originally developed in Jupyter-style script form. The main changes for public sharing are:

- local machine paths were removed and replaced with command-line arguments
- analysis steps were split into standalone scripts
- outputs are written relative to a user-specified output directory
- a minimal repository structure and execution guide were added

## Repository structure

```text
code_release/
├─ scripts/
│  ├─ 01_build_landmark_supplement.py
│  ├─ 02_plot_event_timing_tiff.py
│  ├─ 03_train_temporal_models.py
│  ├─ 04_make_manuscript_tables.py
│  ├─ 05_plot_event_timing_public.py
│  ├─ 06_make_nonshap_figures.py
│  └─ 07_make_shap_figures.py
├─ docs/
│  └─ REPRODUCIBILITY_CHECKLIST.md
├─ data/
│  └─ .gitkeep
├─ outputs/
│  └─ .gitkeep
├─ artifacts/
│  └─ .gitkeep
├─ .gitignore
├─ requirements.txt
└─ run_example.sh
```

## Expected input data

The scripts assume an Excel workbook containing structured routinely collected data used in the manuscript workflow.
The following columns are expected by different parts of the pipeline:

- identifiers: `INDEX`
- baseline date: `Base_Date` or equivalent
- duration on mechanical ventilation: `人工呼吸_連続日数` or `All_Day`
- outcome dates: `死亡日`, `気管切開日` or their English alternatives
- feature columns such as `Day0_*`, `Day3_*`, medications, laboratory values, and department labels

Because the raw data include individual-level clinical information, no sample dataset is bundled in this repository.

## Recommended execution order

### 1. Build landmark eligibility and feature dictionaries

```bash
python scripts/01_build_landmark_supplement.py \
  --excel-path data/your_dataset.xlsx \
  --out-dir outputs/supplement
```

### 2. Train models and export temporal validation outputs

```bash
python scripts/03_train_temporal_models.py \
  --excel-path data/your_dataset.xlsx \
  --out-dir outputs/project_run \
  --run-mode Final
```

For a lighter test run:

```bash
python scripts/03_train_temporal_models.py \
  --excel-path data/your_dataset.xlsx \
  --out-dir outputs/project_run_light \
  --run-mode Light \
  --skip-logo
```

### 3. Build manuscript tables

```bash
python scripts/04_make_manuscript_tables.py \
  --results-root outputs/project_run/results_temporal_binary_landmark_excel_complete_cv_sigmoid \
  --source-excel-path data/your_dataset.xlsx \
  --table-mode Final
```

### 4. Build figures

Event timing figure:

```bash
python scripts/05_plot_event_timing_public.py \
  --excel-path data/your_dataset.xlsx \
  --out-dir outputs/figures
```

Non-SHAP manuscript figures:

```bash
python scripts/06_make_nonshap_figures.py \
  --results-root outputs/project_run/results_temporal_binary_landmark_excel_complete_cv_sigmoid \
  --fig5-model XGBoost
```

SHAP figures:

```bash
python scripts/07_make_shap_figures.py \
  --excel-path data/your_dataset.xlsx \
  --out-dir outputs/project_run
```

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

Before publishing this repository, confirm the following:

- no patient-level data are included
- no facility-specific internal paths remain
- no temporary local files or exported spreadsheets with raw identifiers are committed
- model artifacts do not contain restricted data
- the manuscript and repository versions are aligned

## Citation / manuscript linkage

This repository accompanies the manuscript:

**Landmark-Based Prediction of Tracheostomy and Short-Term Mortality in Mechanically Ventilated Patients Using Routinely Collected Hospital Data: A Multicenter Retrospective Cohort Study**

If you use or adapt this code, please cite the associated manuscript.

## Disclaimer

This code is provided for research use. It was developed for retrospective model development and evaluation, not for direct real-time clinical deployment without further validation, recalibration, governance review, and implementation testing.
