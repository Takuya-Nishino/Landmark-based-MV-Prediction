# -*- coding: utf-8 -*-
# =========================================================
# Manuscript Figures from CV + SIGMOID Excel outputs
#
# SHAP以外のFigure出力用 完全版
# - Figure 2  : AUPRC
# - Figure 3  : Calibration
# - Figure 5  : Two-axis risk matrix
# - Figure S1 : AUROC
# - Figure S2 : DCA
# - Table S6  : Figure 5 quadrant summary
#
# Public-sharing version:
# - results root supplied by CLI
# - outputs written to user directory
# - manuscript-ready JPEG export
# =========================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-SHAP manuscript figures.")
    parser.add_argument("--results-root", required=True, help="Root directory containing model outputs")
    parser.add_argument("--out-dir", default=None, help="Optional output directory")
    parser.add_argument("--fig5-model", default="XGBoost", help="Model to display for Figure 5")
    return parser.parse_args()


ARGS = parse_args()
RESULTS_ROOT = Path(ARGS.results_root)
OUT_DIR = Path(ARGS.out_dir) if ARGS.out_dir else RESULTS_ROOT / "figures_public"
FIG5_MODEL = ARGS.fig5_model
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANDMARKS = ["LM0", "LM3"]
OUTCOMES = ["Event_気管切開", "Event_死亡"]
DISPLAY_OUTCOME = {"Event_気管切開": "Tracheostomy", "Event_死亡": "Death"}

# =========================================================
# HELPERS
# =========================================================
def save_jpeg(fig: plt.Figure, out_path: Path, dpi: int = 600, quality: int = 92) -> None:
    tmp_png = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    img = Image.open(tmp_png).convert("RGB")
    img.save(out_path, format="JPEG", quality=quality, subsampling=0, dpi=(dpi, dpi), optimize=True)
    img.close()
    tmp_png.unlink(missing_ok=True)


def _task_dir(landmark: str, outcome: str) -> Path:
    return RESULTS_ROOT / f"{landmark}_{outcome}"


def load_perf(landmark: str, outcome: str) -> pd.DataFrame:
    path = _task_dir(landmark, outcome) / "model_performance.xlsx"
    return pd.read_excel(path)


def load_preds(landmark: str, outcome: str) -> pd.DataFrame:
    path = _task_dir(landmark, outcome) / "temporal_validation_predictions.xlsx"
    return pd.read_excel(path)


def plot_metric_bar(metric: str, title: str, out_name: str) -> None:
    rows = []
    for lm in LANDMARKS:
        for oc in OUTCOMES:
            perf = load_perf(lm, oc)
            for _, r in perf.iterrows():
                rows.append({
                    "Landmark": lm,
                    "Outcome": DISPLAY_OUTCOME[oc],
                    "Model": r["Model"],
                    metric: r[metric],
                })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, oc in zip(axes, ["Tracheostomy", "Death"]):
        sub = df[df["Outcome"] == oc].copy()
        piv = sub.pivot(index="Model", columns="Landmark", values=metric)
        piv = piv.reindex(index=[m for m in ["Logistic Regression", "XGBoost", "LightGBM"] if m in piv.index])
        x = np.arange(len(piv.index))
        width = 0.35
        ax.bar(x - width/2, piv.get("LM0", pd.Series(index=piv.index, data=np.nan)), width, label="Landmark Day 0")
        ax.bar(x + width/2, piv.get("LM3", pd.Series(index=piv.index, data=np.nan)), width, label="Landmark Day 3")
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=15, ha="right")
        ax.set_title(oc)
        ax.set_ylabel(metric)
        ax.grid(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_jpeg(fig, OUT_DIR / out_name)
    plt.close(fig)


def plot_calibration_summary() -> None:
    rows = []
    for lm in LANDMARKS:
        for oc in OUTCOMES:
            perf = load_perf(lm, oc)
            for _, r in perf.iterrows():
                rows.append({
                    "Landmark": lm,
                    "Outcome": DISPLAY_OUTCOME[oc],
                    "Model": r["Model"],
                    "Slope": r["CalibrationSlope"],
                    "Intercept": r["CalibrationIntercept"],
                })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, oc in zip(axes, ["Tracheostomy", "Death"]):
        sub = df[df["Outcome"] == oc].copy()
        x = np.arange(len(sub))
        ax.scatter(sub["Intercept"], sub["Slope"])
        for _, r in sub.iterrows():
            ax.text(r["Intercept"], r["Slope"], f"{r['Model']} {r['Landmark']}", fontsize=7)
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(oc)
        ax.set_xlabel("Calibration intercept")
        ax.set_ylabel("Calibration slope")
    fig.suptitle("Figure 3. Calibration summary", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_jpeg(fig, OUT_DIR / "Figure3_Calibration.jpg")
    plt.close(fig)


def build_quadrant_df(landmark: str, model_name: str, threshold: float = 0.20) -> pd.DataFrame:
    tr = load_preds(landmark, "Event_気管切開")
    de = load_preds(landmark, "Event_死亡")
    id_col = [c for c in tr.columns if c not in ["Observed", "Logistic Regression", "XGBoost", "LightGBM"]][0]
    merged = tr[[id_col, "Observed", model_name]].rename(columns={"Observed": "Observed_trach", model_name: "Prob_trach"}).merge(
        de[[id_col, "Observed", model_name]].rename(columns={"Observed": "Observed_death", model_name: "Prob_death"}),
        on=id_col,
        how="inner",
    )
    merged["Trach_high"] = (merged["Prob_trach"] >= threshold).astype(int)
    merged["Death_high"] = (merged["Prob_death"] >= threshold).astype(int)
    merged["Quadrant"] = np.select(
        [
            (merged["Trach_high"] == 0) & (merged["Death_high"] == 0),
            (merged["Trach_high"] == 1) & (merged["Death_high"] == 0),
            (merged["Trach_high"] == 0) & (merged["Death_high"] == 1),
            (merged["Trach_high"] == 1) & (merged["Death_high"] == 1),
        ],
        ["Q1", "Q2", "Q3", "Q4"],
        default="NA",
    )
    return merged


def plot_risk_matrix() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, lm in zip(axes, LANDMARKS):
        df = build_quadrant_df(lm, FIG5_MODEL, threshold=0.20)
        ax.scatter(df["Prob_trach"], df["Prob_death"], s=10, alpha=0.4)
        ax.axvline(0.20, linestyle="--", linewidth=1)
        ax.axhline(0.20, linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability of tracheostomy")
        ax.set_ylabel("Predicted probability of death")
        ax.set_title(f"{lm} ({FIG5_MODEL})")
        for q, (x, y) in {"Q1": (0.1, 0.1), "Q2": (0.6, 0.1), "Q3": (0.1, 0.6), "Q4": (0.6, 0.6)}.items():
            n = int((df["Quadrant"] == q).sum())
            ax.text(x, y, f"{q}\nn={n}", fontsize=10, fontweight="bold")
    fig.suptitle("Figure 5. Two-axis risk matrix", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_jpeg(fig, OUT_DIR / "Figure5_Risk_Matrix.jpg")
    plt.close(fig)


def export_quadrant_table() -> None:
    frames: List[pd.DataFrame] = []
    for lm in LANDMARKS:
        df = build_quadrant_df(lm, FIG5_MODEL, threshold=0.20)
        total_n = len(df)
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = df[df["Quadrant"] == q].copy()
            frames.append(pd.DataFrame([{
                "Landmark": lm,
                "Model": FIG5_MODEL,
                "Quadrant": q,
                "n": len(sub),
                "% of cohort": 100 * len(sub) / total_n if total_n > 0 else np.nan,
                "No event n": int(((sub["Observed_trach"] == 0) & (sub["Observed_death"] == 0)).sum()),
                "Tracheostomy n": int(((sub["Observed_trach"] == 1) & (sub["Observed_death"] == 0)).sum()),
                "Tracheostomy + Death n": int(((sub["Observed_trach"] == 1) & (sub["Observed_death"] == 1)).sum()),
                "Death n": int(((sub["Observed_trach"] == 0) & (sub["Observed_death"] == 1)).sum()),
                "Threshold": 0.20,
            }]))
    out = pd.concat(frames, axis=0, ignore_index=True)
    out.to_excel(OUT_DIR / "TableS6_Figure5_quadrant_summary.xlsx", index=False)


def main() -> None:
    print("=" * 57)
    print("Creating non-SHAP manuscript figures...")
    print(f"RESULTS_ROOT = {RESULTS_ROOT}")
    print(f"OUT_DIR      = {OUT_DIR}")
    plot_metric_bar("AUPRC", "Figure 2. Precision-recall performance", "Figure2_AUPRC.jpg")
    plot_calibration_summary()
    plot_metric_bar("AUROC", "Figure S1. AUROC performance", "FigureS1_AUROC.jpg")
    plot_risk_matrix()
    export_quadrant_table()
    print("Saved non-SHAP figures and Table S6.")
    print("=" * 57)


if __name__ == "__main__":
    main()
