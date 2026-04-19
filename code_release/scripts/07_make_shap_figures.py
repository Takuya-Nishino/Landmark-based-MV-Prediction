# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gc
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")

# =========================================================
# USER SETTINGS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SHAP summary figures from saved model bundles.")
    parser.add_argument("--excel-path", required=True, help="Path to source Excel workbook")
    parser.add_argument("--out-dir", required=True, help="Project output directory containing results folder")
    parser.add_argument("--artifact-dir", default=None, help="Optional directory for intermediate SHAP artifacts")
    parser.add_argument("--max-samples", type=int, default=500)
    return parser.parse_args()


ARGS = parse_args()
EXCEL_PATH = Path(ARGS.excel_path)
OUT_DIR = Path(ARGS.out_dir)
RESULTS_ROOT = OUT_DIR / "results_temporal_binary_landmark_excel_complete_cv_sigmoid"
ARTIFACT_DIR = Path(ARGS.artifact_dir) if ARGS.artifact_dir else OUT_DIR / "artifacts_shap"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = ARGS.max_samples

DISPLAY_OUTCOME = {"Event_気管切開": "Tracheostomy", "Event_死亡": "Death"}
LANDMARKS = ["LM0", "LM3"]
TARGETS = ["Event_気管切開", "Event_死亡"]
TREE_MODELS = ["XGBoost", "LightGBM"]

# =========================================================
# HELPERS
# =========================================================
def save_jpeg_from_current_figure(out_path: Path, dpi: int = 600, quality: int = 92) -> None:
    tmp_png = out_path.with_suffix(".tmp.png")
    plt.savefig(tmp_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    img = Image.open(tmp_png).convert("RGB")
    img.save(out_path, format="JPEG", quality=quality, subsampling=0, dpi=(dpi, dpi), optimize=True)
    img.close()
    tmp_png.unlink(missing_ok=True)


def find_first_existing(columns: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def prepare_department_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    src_col = find_first_existing(list(out.columns), ["Department", "診療科名", "科名"])
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
    lab_names = [
        "ALT", "AST", "Alb", "APTT", "BUN", "CK", "CRP", "Cl", "Cre", "D-dimer",
        "Hb", "K", "Na", "PLT", "PT-INR", "T-Bil", "TP", "WBC"
    ]
    out = df.copy()
    for base in lab_names:
        c0 = f"Day0_{base}"
        c3 = f"Day3_{base}"
        if c0 in out.columns and c3 in out.columns:
            out[f"Delta_{base}"] = safe_to_numeric(out[c3]) - safe_to_numeric(out[c0])
    return out


def build_landmark_dataset(df_raw: pd.DataFrame, landmark: str, target_col: str) -> pd.DataFrame:
    date_col = find_first_existing(list(df_raw.columns), ["Base_Date", "BaseDate", "Date"])
    mv_days_col = find_first_existing(list(df_raw.columns), ["人工呼吸_連続日数", "All_Day"])
    outcome_col = find_first_existing(list(df_raw.columns), ["気管切開日", "Tracheostomy_Date", "TracheostomyDate"] if target_col == "Event_気管切開" else ["死亡日", "Death_Date", "DeathDate"])
    if date_col is None or mv_days_col is None or outcome_col is None:
        raise KeyError("Required columns are missing for landmark dataset construction.")

    df = df_raw.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[outcome_col] = pd.to_datetime(df[outcome_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    if landmark == "LM0":
        df["Landmark_Date"] = df[date_col]
    elif landmark == "LM3":
        df[mv_days_col] = safe_to_numeric(df[mv_days_col])
        df = df.loc[df[mv_days_col] >= 4].copy()
        df["Landmark_Date"] = df[date_col] + pd.Timedelta(days=3)
    else:
        raise ValueError(landmark)

    pre_landmark = df[outcome_col].notna() & (df[outcome_col] < df["Landmark_Date"])
    df = df.loc[~pre_landmark].copy()
    horizon_end = df["Landmark_Date"] + pd.Timedelta(days=21)
    in_window = df[outcome_col].notna() & (df[outcome_col] > df["Landmark_Date"]) & (df[outcome_col] <= horizon_end)
    df[target_col] = in_window.astype(int)
    return df.reset_index(drop=True)


def load_model_bundle(task_dir: Path, model_name: str):
    safe_name = model_name.replace(" ", "_")
    path = task_dir / f"{safe_name}_bundle.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_X_for_model(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in X.columns:
        if c.startswith("Department_"):
            X[c] = X[c].fillna(0)
        else:
            X[c] = safe_to_numeric(X[c])
    return X


def get_tree_estimator(model_obj):
    if hasattr(model_obj, "calibrated_classifiers_"):
        return model_obj.calibrated_classifiers_[0].estimator
    return model_obj


def save_single_shap_summary(estimator, X_sample: pd.DataFrame, title: str, out_path: Path) -> None:
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    plt.figure(figsize=(8.5, 5.8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title(title, fontsize=12)
    save_jpeg_from_current_figure(out_path, dpi=600, quality=92)
    plt.close()


def combine_four_panels(image_paths: List[Path], out_path: Path, panel_titles: List[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, img_path, ttl in zip(axes.flatten(), image_paths, panel_titles):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    save_jpeg_from_current_figure(out_path, dpi=600, quality=92)
    plt.close(fig)

# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("=" * 57)
    print("Creating SHAP figures...")
    print(f"EXCEL_PATH   = {EXCEL_PATH}")
    print(f"RESULTS_ROOT = {RESULTS_ROOT}")
    print(f"ARTIFACT_DIR = {ARTIFACT_DIR}")

    df_raw = pd.read_excel(EXCEL_PATH).copy()
    df_raw = add_delta_labs(prepare_department_features(df_raw))

    combined_targets = []

    for target in TARGETS:
        single_paths: List[Path] = []
        panel_titles: List[str] = []
        for landmark in LANDMARKS:
            df_lm = build_landmark_dataset(df_raw, landmark, target)
            # deterministic subsample for plotting
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=min(MAX_SAMPLES, len(df_lm)) / len(df_lm), random_state=42)
            idx_all = np.arange(len(df_lm))
            try:
                _, sample_idx = next(splitter.split(idx_all.reshape(-1, 1), df_lm[target]))
            except Exception:
                sample_idx = idx_all[: min(MAX_SAMPLES, len(df_lm))]
            df_sample = df_lm.iloc[sample_idx].copy()

            task_dir = RESULTS_ROOT / f"{landmark}_{target}"
            for model_name in TREE_MODELS:
                bundle = load_model_bundle(task_dir, model_name)
                feature_cols = bundle["feature_cols"]
                model_obj = bundle["model"]
                X_sample = prepare_X_for_model(df_sample, feature_cols)
                X_sample = X_sample.fillna(X_sample.median(numeric_only=True))
                est = get_tree_estimator(model_obj)
                out_single = ARTIFACT_DIR / f"{landmark}_{target}_{model_name.replace(' ', '_')}_shap.jpg"
                save_single_shap_summary(
                    estimator=est,
                    X_sample=X_sample,
                    title=f"{landmark} - {DISPLAY_OUTCOME[target]} - {model_name}",
                    out_path=out_single,
                )
                single_paths.append(out_single)
                panel_titles.append(f"{landmark} | {DISPLAY_OUTCOME[target]} | {model_name}")
                gc.collect()

        # target-specific combined figure (first four panels are XGBoost/LightGBM × LM0/LM3)
        target_paths = [p for p, ttl in zip(single_paths, panel_titles) if DISPLAY_OUTCOME[target] in ttl]
        target_titles = [ttl for ttl in panel_titles if DISPLAY_OUTCOME[target] in ttl]
        if len(target_paths) >= 4:
            out_target = RESULTS_ROOT / f"Figure_SHAP_{DISPLAY_OUTCOME[target]}.jpg"
            combine_four_panels(target_paths[:4], out_target, target_titles[:4])
            combined_targets.append(out_target)

    # export lightweight manifest
    manifest = {
        "excel_path": str(EXCEL_PATH),
        "results_root": str(RESULTS_ROOT),
        "artifact_dir": str(ARTIFACT_DIR),
        "combined_figures": [str(p) for p in combined_targets],
    }
    with open(RESULTS_ROOT / "shap_figure_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Saved SHAP figures.")
    print("=" * 57)


if __name__ == "__main__":
    main()
