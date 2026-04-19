#!/usr/bin/env bash
set -euo pipefail

EXCEL_PATH="data/your_dataset.xlsx"
OUT_DIR="outputs/project_run"

python scripts/01_build_landmark_supplement.py --excel-path "$EXCEL_PATH" --out-dir "$OUT_DIR/supplement"
python scripts/03_train_temporal_models.py --excel-path "$EXCEL_PATH" --out-dir "$OUT_DIR" --run-mode Final
python scripts/04_make_manuscript_tables.py --results-root "$OUT_DIR/results_temporal_binary_landmark_excel_complete_cv_sigmoid" --source-excel-path "$EXCEL_PATH" --table-mode Final
python scripts/05_plot_event_timing_public.py --excel-path "$EXCEL_PATH" --out-dir "$OUT_DIR/figures"
python scripts/06_make_nonshap_figures.py --results-root "$OUT_DIR/results_temporal_binary_landmark_excel_complete_cv_sigmoid" --fig5-model XGBoost
python scripts/07_make_shap_figures.py --excel-path "$EXCEL_PATH" --out-dir "$OUT_DIR"
