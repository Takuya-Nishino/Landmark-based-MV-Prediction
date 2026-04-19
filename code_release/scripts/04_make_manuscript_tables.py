# -*- coding: utf-8 -*-
# =========================================================
# Manuscript Tables aligned to CURRENT development outputs
#
# Compatible with:
# - Light / Final development outputs
# - LOGO present / absent
# - Logistic Regression / XGBoost / LightGBM (auto-detected)
#
# Public-sharing version:
# - local paths removed
# - root directories supplied by CLI
# - table generation kept modular for manuscript reproducibility
# =========================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manuscript tables from temporal model outputs.")
    parser.add_argument("--results-root", required=True, help="Root directory of development outputs")
    parser.add_argument("--source-excel-path", required=True, help="Original source Excel workbook")
    parser.add_argument("--out-dir", default=None, help="Optional output directory; defaults to results-root/tables_public")
    parser.add_argument("--table-mode", choices=["Light", "Final"], default="Final")
    return parser.parse_args()


ARGS = parse_args()
RESULTS_ROOT = Path(ARGS.results_root)
SOURCE_EXCEL_PATH = Path(ARGS.source_excel_path)
OUT_DIR = Path(ARGS.out_dir) if ARGS.out_dir else RESULTS_ROOT / "tables_public"
TABLE_MODE = ARGS.table_mode
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIG
# =========================================================
LANDMARKS = ["LM0", "LM3"]
OUTCOMES = ["Event_気管切開", "Event_死亡"]
MODEL_ORDER = ["Logistic Regression", "XGBoost", "LightGBM"]
DISPLAY_OUTCOME = {
    "Event_気管切開": "Tracheostomy",
    "Event_死亡": "Death",
}

# =========================================================
# HELPERS
# =========================================================
def _task_dir(landmark: str, outcome: str) -> Path:
    return RESULTS_ROOT / f"{landmark}_{outcome}"


def load_performance_sheet(landmark: str, outcome: str) -> pd.DataFrame:
    path = _task_dir(landmark, outcome) / "model_performance.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def load_predictions_sheet(landmark: str, outcome: str) -> pd.DataFrame:
    path = _task_dir(landmark, outcome) / "temporal_validation_predictions.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_excel(path)


def format_metric(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def make_main_performance_table() -> pd.DataFrame:
    rows: List[dict] = []
    for landmark in LANDMARKS:
        for outcome in OUTCOMES:
            perf = load_performance_sheet(landmark, outcome)
            if perf.empty:
                continue
            for model in MODEL_ORDER:
                sub = perf.loc[perf["Model"] == model].copy()
                if sub.empty:
                    continue
                r = sub.iloc[0]
                rows.append({
                    "Landmark": landmark,
                    "Outcome": DISPLAY_OUTCOME.get(outcome, outcome),
                    "Model": model,
                    "Selected calibration": r.get("Selected calibration", np.nan),
                    "n": int(r["n"]),
                    "Events": int(r["Events"]),
                    "Event rate": float(r["Event rate"]),
                    "AUROC": float(r["AUROC"]),
                    "AUPRC": float(r["AUPRC"]),
                    "Brier score": float(r["Brier"]),
                    "Calibration intercept": float(r["CalibrationIntercept"]),
                    "Calibration slope": float(r["CalibrationSlope"]),
                })
    return pd.DataFrame(rows)


def make_pretty_main_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Events/n (%)"] = out.apply(lambda r: f"{int(r['Events'])}/{int(r['n'])} ({100*r['Event rate']:.1f}%)", axis=1)
    out["AUROC"] = out["AUROC"].map(lambda x: format_metric(x, 3))
    out["AUPRC"] = out["AUPRC"].map(lambda x: format_metric(x, 3))
    out["Brier score"] = out["Brier score"].map(lambda x: format_metric(x, 3))
    out["Calibration intercept"] = out["Calibration intercept"].map(lambda x: format_metric(x, 3))
    out["Calibration slope"] = out["Calibration slope"].map(lambda x: format_metric(x, 3))
    out = out[[
        "Landmark", "Outcome", "Model", "Selected calibration", "Events/n (%)",
        "AUROC", "AUPRC", "Brier score", "Calibration intercept", "Calibration slope"
    ]]
    return out


def build_quadrant_summary(landmark: str, model_name: str, threshold: float = 0.20) -> pd.DataFrame:
    tr_path = _task_dir(landmark, "Event_気管切開") / "temporal_validation_predictions.xlsx"
    de_path = _task_dir(landmark, "Event_死亡") / "temporal_validation_predictions.xlsx"
    if not tr_path.exists() or not de_path.exists():
        return pd.DataFrame()

    tr = pd.read_excel(tr_path)
    de = pd.read_excel(de_path)
    id_col = [c for c in tr.columns if c not in ["Observed", "Logistic Regression", "XGBoost", "LightGBM"]][0]

    merged = tr[[id_col, "Observed", model_name]].rename(columns={"Observed": "Observed_trach", model_name: "Prob_trach"}).merge(
        de[[id_col, "Observed", model_name]].rename(columns={"Observed": "Observed_death", model_name: "Prob_death"}),
        on=id_col,
        how="inner",
    )

    merged["Trach_high"] = (merged["Prob_trach"] >= threshold).astype(int)
    merged["Death_high"] = (merged["Prob_death"] >= threshold).astype(int)

    def quad(row):
        if row["Trach_high"] == 0 and row["Death_high"] == 0:
            return "Q1"
        if row["Trach_high"] == 1 and row["Death_high"] == 0:
            return "Q2"
        if row["Trach_high"] == 0 and row["Death_high"] == 1:
            return "Q3"
        return "Q4"

    merged["Quadrant"] = merged.apply(quad, axis=1)
    defs = {
        "Q1": "Low tracheostomy / Low death",
        "Q2": "High tracheostomy / Low death",
        "Q3": "Low tracheostomy / High death",
        "Q4": "High tracheostomy / High death",
    }
    total_n = len(merged)
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = merged.loc[merged["Quadrant"] == q].copy()
        if sub.empty:
            rows.append({
                "Landmark": landmark,
                "Model": model_name,
                "Quadrant": q,
                "Definition": defs[q],
                "n": 0,
                "% of cohort": 0.0,
                "No event n": 0,
                "Tracheostomy n": 0,
                "Tracheostomy + Death n": 0,
                "Death n": 0,
                "Threshold": threshold,
            })
            continue
        no_event = ((sub["Observed_trach"] == 0) & (sub["Observed_death"] == 0)).sum()
        trach_only = ((sub["Observed_trach"] == 1) & (sub["Observed_death"] == 0)).sum()
        both = ((sub["Observed_trach"] == 1) & (sub["Observed_death"] == 1)).sum()
        death_only = ((sub["Observed_trach"] == 0) & (sub["Observed_death"] == 1)).sum()
        rows.append({
            "Landmark": landmark,
            "Model": model_name,
            "Quadrant": q,
            "Definition": defs[q],
            "n": int(len(sub)),
            "% of cohort": 100 * len(sub) / total_n,
            "No event n": int(no_event),
            "Tracheostomy n": int(trach_only),
            "Tracheostomy + Death n": int(both),
            "Death n": int(death_only),
            "Threshold": threshold,
        })
    return pd.DataFrame(rows)


def build_all_quadrant_tables() -> pd.DataFrame:
    frames = []
    for landmark in LANDMARKS:
        for model_name in MODEL_ORDER:
            qdf = build_quadrant_summary(landmark, model_name, threshold=0.20)
            if not qdf.empty:
                frames.append(qdf)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def build_feature_dictionary_export() -> pd.DataFrame:
    path = RESULTS_ROOT.parent / "supplement" / "Supplement_Landmark_Eligibility_and_Feature_Dictionary.xlsx"
    if path.exists():
        try:
            return pd.read_excel(path, sheet_name="feature_dictionary")
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_tables() -> None:
    main_df = make_main_performance_table()
    pretty_df = make_pretty_main_table(main_df)
    pretty_df.to_excel(OUT_DIR / "Table_3_temporal_validation_performance.xlsx", index=False)

    raw_perf = main_df.copy()
    raw_perf.to_excel(OUT_DIR / "Table_S8_raw_model_performance.xlsx", index=False)

    quad_df = build_all_quadrant_tables()
    if not quad_df.empty:
        quad_df.to_excel(OUT_DIR / "Table_S9_quadrant_summary.xlsx", index=False)

    feat_df = build_feature_dictionary_export()
    if not feat_df.empty:
        feat_df.to_excel(OUT_DIR / "Supplementary_feature_dictionary.xlsx", index=False)

    with pd.ExcelWriter(OUT_DIR / "All_tables_bundle.xlsx", engine="openpyxl") as writer:
        pretty_df.to_excel(writer, sheet_name="Table3", index=False)
        raw_perf.to_excel(writer, sheet_name="TableS8", index=False)
        if not quad_df.empty:
            quad_df.to_excel(writer, sheet_name="TableS9", index=False)
        if not feat_df.empty:
            feat_df.to_excel(writer, sheet_name="FeatureDictionary", index=False)

    summary = {
        "table_mode": TABLE_MODE,
        "results_root": str(RESULTS_ROOT),
        "output_dir": str(OUT_DIR),
        "n_rows_main": int(len(main_df)),
        "n_rows_quadrant": int(len(quad_df)) if not quad_df.empty else 0,
    }
    with open(OUT_DIR / "table_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    print("=" * 57)
    print("Creating manuscript tables...")
    print(f"RESULTS_ROOT = {RESULTS_ROOT}")
    print(f"OUT_DIR      = {OUT_DIR}")
    save_tables()
    print("Saved manuscript tables.")
    print("=" * 57)


if __name__ == "__main__":
    main()
