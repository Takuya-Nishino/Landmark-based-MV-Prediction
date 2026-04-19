# -*- coding: utf-8 -*-
# =========================================================
# Temporal本線（binary separate outcomes）
# - LM0 / LM3
# - Outcomeごとに別学習（気管切開, 死亡）
# - モデル: LogisticRegression / XGBoost / LightGBM
# - temporal validation: intubation >= 2024-01-01
# - training subsetの中でCV-HPO
# - calibration subsetで sigmoid / isotonic を比較して選択
# - Final時は calibration込みでテーブル/図用成果物を完全出力
#
# Public-sharing adjustments:
# - all local paths replaced by CLI arguments
# - outputs written under a user-specified directory
# - script kept close to original monolithic workflow for reproducibility
# =========================================================

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train temporal landmark prediction models.")
    parser.add_argument("--excel-path", required=True, help="Path to source Excel workbook")
    parser.add_argument("--out-dir", required=True, help="Directory for all outputs")
    parser.add_argument("--run-mode", choices=["Light", "Final"], default="Final")
    parser.add_argument("--sheet-name", default=0, help="Excel sheet name or index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-logo", action="store_true", help="Skip leave-one-group-out style sensitivity outputs")
    return parser.parse_args()


ARGS = parse_args()
EXCEL_PATH = Path(ARGS.excel_path)
OUT_DIR = Path(ARGS.out_dir)
RUN_MODE = ARGS.run_mode
SHEET_NAME = ARGS.sheet_name
SEED = ARGS.seed
SKIP_LOGO = ARGS.skip_logo

OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT = OUT_DIR / "results_temporal_binary_landmark_excel_complete_cv_sigmoid"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# =========================================================
# CONFIG
# =========================================================
RANDOM_STATE = SEED
N_SPLITS = 5
HPO_TRIALS = 10 if RUN_MODE == "Light" else 40
TEMPORAL_CUTOFF = pd.Timestamp("2024-01-01")
CALIBRATION_TEST_SIZE = 0.30
LANDMARK_HORIZON_DAYS = 21
LM3_OFFSET_DAYS = 3

DATE_COL_CANDIDATES = ["Base_Date", "BaseDate", "Date"]
INDEX_COL_CANDIDATES = ["INDEX", "Index", "ID"]
MV_DAYS_COL_CANDIDATES = ["人工呼吸_連続日数", "All_Day"]
DEPARTMENT_COL_CANDIDATES = ["Department", "診療科名", "科名"]

OUTCOME_DATE_CANDIDATES = {
    "Event_死亡": ["死亡日", "Death_Date", "DeathDate"],
    "Event_気管切開": ["気管切開日", "Tracheostomy_Date", "TracheostomyDate"],
}

BASE_FEATURES = [
    "Age", "Male", "BMI", "IntubationDate",
    "Dialysis", "CirculatoryDevice", "Hypothermia", "CPR"
]

LAB_NAMES = [
    "ALT", "AST", "Alb", "APTT", "BUN", "CK", "CRP", "Cl", "Cre", "D-dimer",
    "Hb", "K", "Na", "PLT", "PT-INR", "T-Bil", "TP", "WBC"
]

MED_NAMES = [
    "CoreSed", "Opioid", "NAD", "Adrenaline", "DOA", "DOB",
    "Insulin", "Steroid", "Vasopressin", "Diuretic",
    "AntiMRSA", "Antibiotic", "Carbapenem", "Antifungal",
    "FFP", "Plt", "RBC"
]

# =========================================================
# UTILS
# =========================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_first_existing(columns: List[str], candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise KeyError(f"Missing required column from candidates: {candidates}")
    return None


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_department_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    src_col = None
    for c in DEPARTMENT_COL_CANDIDATES:
        if c in out.columns:
            src_col = c
            break

    if src_col is None:
        out["Department_en"] = "Other"
        return out

    mapping = {
        "救命救急": "Emergency and Critical Care",
        "心臓": "Cardiac",
        "脳疾患": "Neurological",
        "その他": "Other",
    }
    out["Department_en"] = out[src_col].astype(str).str.strip().map(mapping).fillna("Other")
    return out


def add_delta_labs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base in LAB_NAMES:
        c0 = f"Day0_{base}"
        c3 = f"Day3_{base}"
        if c0 in out.columns and c3 in out.columns:
            out[f"Delta_{base}"] = safe_to_numeric(out[c3]) - safe_to_numeric(out[c0])
    return out


def build_feature_columns(df: pd.DataFrame, landmark: str) -> Tuple[List[str], List[str]]:
    df = prepare_department_features(df)
    dept_cols = []
    for d in ["Emergency and Critical Care", "Cardiac", "Neurological", "Other"]:
        col = f"Department_{d}"
        df[col] = (df["Department_en"] == d).astype(np.uint8)
        dept_cols.append(col)

    base_cols = [c for c in BASE_FEATURES if c in df.columns]
    day0_labs = [f"Day0_{x}" for x in LAB_NAMES if f"Day0_{x}" in df.columns]
    day0_meds = [f"Day0_{x}" for x in MED_NAMES if f"Day0_{x}" in df.columns]
    day3_meds = [f"Day3_{x}" for x in MED_NAMES if f"Day3_{x}" in df.columns]
    delta_labs = [f"Delta_{x}" for x in LAB_NAMES if f"Delta_{x}" in df.columns]

    if landmark == "LM0":
        used = base_cols + dept_cols + day0_labs + day0_meds
    elif landmark == "LM3":
        used = base_cols + dept_cols + day0_labs + delta_labs + day3_meds
    else:
        raise ValueError(landmark)

    used = [c for c in used if c in df.columns]
    return used, dept_cols


def build_landmark_dataset(df_raw: pd.DataFrame, landmark: str, target_col: str) -> pd.DataFrame:
    cols = list(df_raw.columns)
    base_date_col = find_first_existing(cols, DATE_COL_CANDIDATES, required=True)
    mv_days_col = find_first_existing(cols, MV_DAYS_COL_CANDIDATES, required=True)
    outcome_date_col = find_first_existing(cols, OUTCOME_DATE_CANDIDATES[target_col], required=True)

    df = df_raw.copy()
    df[base_date_col] = pd.to_datetime(df[base_date_col], errors="coerce")
    df[outcome_date_col] = pd.to_datetime(df[outcome_date_col], errors="coerce")
    df = df.dropna(subset=[base_date_col]).copy()

    if landmark == "LM0":
        df["Landmark_Date"] = df[base_date_col]
    elif landmark == "LM3":
        df[mv_days_col] = safe_to_numeric(df[mv_days_col])
        df = df.loc[df[mv_days_col] >= 4].copy()
        df["Landmark_Date"] = df[base_date_col] + pd.Timedelta(days=LM3_OFFSET_DAYS)
    else:
        raise ValueError(landmark)

    pre_landmark = df[outcome_date_col].notna() & (df[outcome_date_col] < df["Landmark_Date"])
    df = df.loc[~pre_landmark].copy()

    horizon_end = df["Landmark_Date"] + pd.Timedelta(days=LANDMARK_HORIZON_DAYS)
    in_window = (
        df[outcome_date_col].notna()
        & (df[outcome_date_col] > df["Landmark_Date"])
        & (df[outcome_date_col] <= horizon_end)
    )
    df[target_col] = in_window.astype(int)
    return df.reset_index(drop=True)


def calibration_slope_intercept(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    from sklearn.linear_model import LogisticRegression as LR
    X = logit_p.reshape(-1, 1)
    lr = LR(solver="lbfgs")
    lr.fit(X, y_true)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return slope, intercept


def eval_binary(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    auroc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, prob)
    slope, intercept = calibration_slope_intercept(y_true, prob)
    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier),
        "CalibrationSlope": float(slope),
        "CalibrationIntercept": float(intercept),
    }

# =========================================================
# MODEL BUILDERS
# =========================================================
def make_lr_pipeline(feature_names: List[str], dept_cols: List[str]) -> Pipeline:
    numeric_cols = [c for c in feature_names if c not in dept_cols]
    categorical_cols = [c for c in feature_names if c in dept_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), categorical_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_xgb_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
    )


def make_lgbm_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbose=-1,
    )


def fit_calibrated_model(base_model, X_train, y_train, X_cal, y_cal, method: str):
    fitted = clone(base_model)
    fitted.fit(X_train, y_train)
    calib = CalibratedClassifierCV(fitted, method=method, cv="prefit")
    calib.fit(X_cal, y_cal)
    return calib


def choose_calibration(base_model, X_train, y_train, X_cal, y_cal):
    candidates = {}
    for method in ["sigmoid", "isotonic"]:
        try:
            model = fit_calibrated_model(base_model, X_train, y_train, X_cal, y_cal, method)
            prob = model.predict_proba(X_cal)[:, 1]
            score = average_precision_score(y_cal, prob)
            candidates[method] = (score, model)
        except Exception:
            continue
    if not candidates:
        fitted = clone(base_model)
        fitted.fit(X_train, y_train)
        return "none", fitted
    best_method = max(candidates, key=lambda k: candidates[k][0])
    return best_method, candidates[best_method][1]

# =========================================================
# HPO
# =========================================================
def tune_xgb(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": 4,
        }
        scores = []
        for tr_idx, va_idx in cv.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            prob = model.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[va_idx], prob))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=HPO_TRIALS)
    return study.best_params


def tune_lgbm(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    def objective(trial):
        params = {
            "objective": "binary",
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": 4,
            "verbose": -1,
        }
        scores = []
        for tr_idx, va_idx in cv.split(X, y):
            model = lgb.LGBMClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            prob = model.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(average_precision_score(y.iloc[va_idx], prob))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=HPO_TRIALS)
    return study.best_params

# =========================================================
# MAIN WORKFLOW
# =========================================================
@dataclass
class TaskSpec:
    landmark: str
    outcome: str


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_single_task(df_raw: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    df = build_landmark_dataset(df_raw, task.landmark, task.outcome)
    df = add_delta_labs(df)
    df = prepare_department_features(df)
    used_cols, dept_cols = build_feature_columns(df, task.landmark)

    date_col = find_first_existing(list(df.columns), DATE_COL_CANDIDATES)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    temporal_val = df[date_col] >= TEMPORAL_CUTOFF
    dev_df = df.loc[~temporal_val].copy()
    tv_df = df.loc[temporal_val].copy()

    train_df, cal_df = train_test_split(
        dev_df,
        test_size=CALIBRATION_TEST_SIZE,
        stratify=dev_df[task.outcome],
        random_state=RANDOM_STATE,
    )

    X_train = train_df[used_cols].copy()
    y_train = train_df[task.outcome].astype(int)
    X_cal = cal_df[used_cols].copy()
    y_cal = cal_df[task.outcome].astype(int)
    X_tv = tv_df[used_cols].copy()
    y_tv = tv_df[task.outcome].astype(int)

    # numeric coercion
    for frame in [X_train, X_cal, X_tv]:
        for c in frame.columns:
            if c not in dept_cols:
                frame[c] = safe_to_numeric(frame[c])

    results_rows = []
    model_store = {}

    # Logistic regression
    lr_pipe = make_lr_pipeline(used_cols, dept_cols)
    lr_method, lr_model = choose_calibration(lr_pipe, X_train, y_train, X_cal, y_cal)
    lr_prob = lr_model.predict_proba(X_tv)[:, 1]
    lr_metrics = eval_binary(y_tv.to_numpy(), lr_prob)
    results_rows.append({
        "Landmark": task.landmark,
        "Outcome": task.outcome,
        "Model": "Logistic Regression",
        "Selected calibration": lr_method,
        "n": int(len(y_tv)),
        "Events": int(y_tv.sum()),
        "Event rate": float(y_tv.mean()),
        **lr_metrics,
    })
    model_store["Logistic Regression"] = (lr_model, used_cols)

    # XGBoost
    xgb_params = tune_xgb(X_train.fillna(X_train.median(numeric_only=True)), y_train)
    xgb_model_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        **xgb_params,
    )
    xgb_method, xgb_model = choose_calibration(
        xgb_model_base,
        X_train.fillna(X_train.median(numeric_only=True)), y_train,
        X_cal.fillna(X_train.median(numeric_only=True)), y_cal,
    )
    xgb_prob = xgb_model.predict_proba(X_tv.fillna(X_train.median(numeric_only=True)))[:, 1]
    xgb_metrics = eval_binary(y_tv.to_numpy(), xgb_prob)
    results_rows.append({
        "Landmark": task.landmark,
        "Outcome": task.outcome,
        "Model": "XGBoost",
        "Selected calibration": xgb_method,
        "n": int(len(y_tv)),
        "Events": int(y_tv.sum()),
        "Event rate": float(y_tv.mean()),
        **xgb_metrics,
    })
    model_store["XGBoost"] = (xgb_model, used_cols)

    # LightGBM
    lgb_params = tune_lgbm(X_train.fillna(X_train.median(numeric_only=True)), y_train)
    lgb_model_base = lgb.LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=4, verbose=-1, **lgb_params)
    lgb_method, lgb_model = choose_calibration(
        lgb_model_base,
        X_train.fillna(X_train.median(numeric_only=True)), y_train,
        X_cal.fillna(X_train.median(numeric_only=True)), y_cal,
    )
    lgb_prob = lgb_model.predict_proba(X_tv.fillna(X_train.median(numeric_only=True)))[:, 1]
    lgb_metrics = eval_binary(y_tv.to_numpy(), lgb_prob)
    results_rows.append({
        "Landmark": task.landmark,
        "Outcome": task.outcome,
        "Model": "LightGBM",
        "Selected calibration": lgb_method,
        "n": int(len(y_tv)),
        "Events": int(y_tv.sum()),
        "Event rate": float(y_tv.mean()),
        **lgb_metrics,
    })
    model_store["LightGBM"] = (lgb_model, used_cols)

    task_dir = RESULTS_ROOT / f"{task.landmark}_{task.outcome}"
    task_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results_rows).to_excel(task_dir / "model_performance.xlsx", index=False)
    save_json(task_dir / "feature_columns.json", {"used_cols": used_cols, "dept_cols": dept_cols})
    tv_export = tv_df[[find_first_existing(list(tv_df.columns), INDEX_COL_CANDIDATES), task.outcome]].copy()
    tv_export.rename(columns={task.outcome: "Observed"}, inplace=True)
    for model_name, (_, _) in model_store.items():
        if model_name == "Logistic Regression":
            tv_export[model_name] = lr_prob
        elif model_name == "XGBoost":
            tv_export[model_name] = xgb_prob
        elif model_name == "LightGBM":
            tv_export[model_name] = lgb_prob
    tv_export.to_excel(task_dir / "temporal_validation_predictions.xlsx", index=False)

    # save models
    for model_name, (model_obj, feature_cols) in model_store.items():
        safe_name = model_name.replace(" ", "_")
        joblib.dump({"model": model_obj, "feature_cols": feature_cols}, task_dir / f"{safe_name}_bundle.pkl")

    # lightweight shap exports for tree models
    if RUN_MODE == "Final":
        for model_name, model_obj, X_used in [
            ("XGBoost", xgb_model, X_tv.fillna(X_train.median(numeric_only=True))),
            ("LightGBM", lgb_model, X_tv.fillna(X_train.median(numeric_only=True))),
        ]:
            try:
                base_est = model_obj.calibrated_classifiers_[0].estimator if hasattr(model_obj, "calibrated_classifiers_") else model_obj
                explainer = shap.TreeExplainer(base_est)
                sample_X = X_used.iloc[: min(500, len(X_used))].copy()
                shap_values = explainer.shap_values(sample_X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                abs_mean = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({"feature": sample_X.columns, "mean_abs_shap": abs_mean}).sort_values("mean_abs_shap", ascending=False)
                shap_df.to_excel(task_dir / f"{model_name.replace(' ', '_')}_shap_importance.xlsx", index=False)
            except Exception:
                pass

    return pd.DataFrame(results_rows)


def main() -> None:
    set_global_seed(RANDOM_STATE)
    df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME).copy()

    summary_frames = []
    for landmark in ["LM0", "LM3"]:
        for outcome in ["Event_気管切開", "Event_死亡"]:
            print("=" * 57)
            print(f"{landmark} | {outcome}")
            perf = run_single_task(df_raw, TaskSpec(landmark=landmark, outcome=outcome))
            print(perf[["Model", "n", "Events", "Event rate", "AUROC", "AUPRC"]].to_string(index=False))
            summary_frames.append(perf)
            gc.collect()

    final_summary = pd.concat(summary_frames, axis=0, ignore_index=True)
    final_summary.to_excel(RESULTS_ROOT / "all_model_performance_summary.xlsx", index=False)
    print("=" * 57)
    print(f"Saved summary: {RESULTS_ROOT / 'all_model_performance_summary.xlsx'}")


if __name__ == "__main__":
    main()
