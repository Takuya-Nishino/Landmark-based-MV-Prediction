# -*- coding: utf-8 -*-
"""
Figure:
(A) Landmark Day 0
(B) Landmark Day 3

Development-code aligned event construction:
- Outcome dates:
    - 気管切開日
    - 死亡日
- Landmark datasets:
    - LM0: Landmark_Date = Base_Date
    - LM3: 人工呼吸_連続日数 >= 4, Landmark_Date = Base_Date + 3 days
- Exclude pre-landmark events
- Exclude same-day events
- Event window: within 21 days after landmark

Important revision in this version:
- LM3 uses a COMMON risk set
  = alive at Day 3, no tracheostomy before Day 3, and still on MV at Day 3
- Percentages are shown as n (%) using the denominator of each landmark cohort
- Output folder is explicitly specified

Output:
- TIFF
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# =========================================================
# PATHS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Plot event timing histograms for LM0 and LM3.")
    parser.add_argument("--excel-path", required=True, help="Path to source Excel file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--out-file",
        default="Figure_Event_Timing_LM0_LM3_commonriskset.tiff",
        help="Output TIFF filename",
    )
    return parser.parse_args()


args = parse_args()
EXCEL_PATH = args.excel_path
OUT_DIR = args.out_dir
OUT_FILE = args.out_file

# =========================================================
# SETTINGS
# =========================================================
BASE_DATE_COL = "Base_Date"
MV_DAYS_COL = "人工呼吸_連続日数"
TRACH_DATE_COL = "気管切開日"
DEATH_DATE_COL = "死亡日"

LANDMARK_HORIZON_DAYS = 21
LM3_OFFSET_DAYS = 3
EXCLUDE_SAME_DAY_EVENT = True

DPI = 600
FIGSIZE = (14, 6)

# =========================================================
# LOAD
# =========================================================
df_raw = pd.read_excel(EXCEL_PATH).copy()

required_cols = [BASE_DATE_COL, MV_DAYS_COL, TRACH_DATE_COL, DEATH_DATE_COL]
missing_cols = [c for c in required_cols if c not in df_raw.columns]
if missing_cols:
    raise ValueError(f"必要な列が見つかりません: {missing_cols}")

df_raw[BASE_DATE_COL] = pd.to_datetime(df_raw[BASE_DATE_COL], errors="coerce")
df_raw[TRACH_DATE_COL] = pd.to_datetime(df_raw[TRACH_DATE_COL], errors="coerce")
df_raw[DEATH_DATE_COL] = pd.to_datetime(df_raw[DEATH_DATE_COL], errors="coerce")
df_raw[MV_DAYS_COL] = pd.to_numeric(df_raw[MV_DAYS_COL], errors="coerce")

df_raw = df_raw.dropna(subset=[BASE_DATE_COL]).reset_index(drop=True).copy()

# =========================================================
# HELPERS
# =========================================================
def median_iqr_str(values: np.ndarray) -> str:
    if len(values) == 0:
        return "median NA [IQR NA–NA] days"
    med = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    return f"median {med:.0f} [IQR {q25:.0f}–{q75:.0f}] days"


def safe_plot_kde(ax, values: np.ndarray, color: str):
    if len(values) <= 1:
        return
    if np.allclose(values, values[0]):
        return
    kde = gaussian_kde(values)
    x_grid = np.linspace(1, 21, 400)
    y_kde = kde(x_grid) * len(values)
    ax.plot(x_grid, y_kde, color=color, lw=2.0)


def build_common_landmark_riskset(df: pd.DataFrame, landmark: str) -> pd.DataFrame:
    """
    Build common risk set for each landmark.
    LM0:
        - Landmark_Date = Base_Date
    LM3:
        - still on MV at Day 3: 人工呼吸_連続日数 >= 4
        - alive at Day 3
        - no tracheostomy before Day 3
    """
    out = df.copy()

    if landmark == "LM0":
        out["Landmark_Date"] = out[BASE_DATE_COL]

    elif landmark == "LM3":
        out = out.loc[out[MV_DAYS_COL] >= 4].copy()
        out["Landmark_Date"] = out[BASE_DATE_COL] + pd.Timedelta(days=LM3_OFFSET_DAYS)

        # alive at Day 3
        out = out.loc[
            out[DEATH_DATE_COL].isna() | (out[DEATH_DATE_COL] >= out["Landmark_Date"])
        ].copy()

        # no tracheostomy before Day 3
        out = out.loc[
            out[TRACH_DATE_COL].isna() | (out[TRACH_DATE_COL] >= out["Landmark_Date"])
        ].copy()

    else:
        raise ValueError(f"Unsupported landmark: {landmark}")

    out = out.reset_index(drop=True).copy()
    return out


def add_event_info(df: pd.DataFrame, outcome_date_col: str) -> pd.DataFrame:
    """
    Add event flag and days from landmark within 21 days.
    Same common cohort is used; outcome is changed here only.
    """
    out = df.copy()
    horizon_end = out["Landmark_Date"] + pd.Timedelta(days=LANDMARK_HORIZON_DAYS)

    has_event_date = out[outcome_date_col].notna()

    if EXCLUDE_SAME_DAY_EVENT:
        lower_condition = out[outcome_date_col] > out["Landmark_Date"]
    else:
        lower_condition = out[outcome_date_col] >= out["Landmark_Date"]

    in_window = (
        has_event_date
        & lower_condition
        & (out[outcome_date_col] <= horizon_end)
    )

    out["event_flag"] = in_window.astype(int)
    out["days_from_landmark"] = np.where(
        has_event_date,
        (out[outcome_date_col] - out["Landmark_Date"]).dt.days,
        np.nan
    )
    return out


def extract_event_days_from_common_riskset(df: pd.DataFrame, landmark: str):
    """
    Returns:
        riskset_n, trach_days, death_days
    restricted to 1-21 days after landmark
    """
    riskset = build_common_landmark_riskset(df, landmark=landmark)

    trach_df = add_event_info(riskset, outcome_date_col=TRACH_DATE_COL)
    death_df = add_event_info(riskset, outcome_date_col=DEATH_DATE_COL)

    trach_days = (
        trach_df.loc[trach_df["event_flag"] == 1, "days_from_landmark"]
        .dropna()
        .astype(float)
        .to_numpy()
    )
    death_days = (
        death_df.loc[death_df["event_flag"] == 1, "days_from_landmark"]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    trach_days = trach_days[(trach_days >= 1) & (trach_days <= 21)]
    death_days = death_days[(death_days >= 1) & (death_days <= 21)]

    return len(riskset), trach_days, death_days


def build_legend_label(event_name: str, values: np.ndarray, denom: int) -> str:
    n = len(values)
    pct = 100.0 * n / denom if denom > 0 else np.nan
    return f"{event_name}: {n} ({pct:.1f}%) | {median_iqr_str(values)}"


def plot_panel(
    ax,
    trach_days: np.ndarray,
    death_days: np.ndarray,
    denom: int,
    title: str,
    panel_label: str,
):
    bins = np.arange(1, 23, 1)

    trach_label = build_legend_label("Tracheostomy", trach_days, denom)
    death_label = build_legend_label("Death", death_days, denom)

    n_t, _, _ = ax.hist(
        trach_days,
        bins=bins,
        alpha=0.55,
        edgecolor="black",
        color="steelblue",
        label=trach_label
    )
    n_d, _, _ = ax.hist(
        death_days,
        bins=bins,
        alpha=0.55,
        edgecolor="black",
        color="sandybrown",
        label=death_label
    )

    safe_plot_kde(ax, trach_days, color="blue")
    safe_plot_kde(ax, death_days, color="red")

    ymax = max(
        np.max(n_t) if len(n_t) > 0 else 0,
        np.max(n_d) if len(n_d) > 0 else 0,
        1
    )

    ax.set_xlim(1, 21)
    ax.set_xticks(np.arange(1, 22, 1))
    ax.set_ylim(0, ymax * 1.20)

    ax.set_xlabel("Days after landmark", fontsize=12)
    ax.set_ylabel("Number of events (n)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(False)

    ax.text(
        -0.12, 1.05, panel_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )

    ax.text(
        0.02, 0.98,
        f"Cohort n = {denom}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left"
    )

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.6
    )


# =========================================================
# BUILD DATA FOR PANELS
# =========================================================
lm0_n, lm0_trach_days, lm0_death_days = extract_event_days_from_common_riskset(df_raw, landmark="LM0")
lm3_n, lm3_trach_days, lm3_death_days = extract_event_days_from_common_riskset(df_raw, landmark="LM3")

print("===== Cohort counts used for plotting =====")
print(f"LM0 common risk set n : {lm0_n}")
print(f"LM3 common risk set n : {lm3_n}")

print("===== Event counts used for plotting =====")
print(f"LM0 Tracheostomy: {len(lm0_trach_days)} ({100 * len(lm0_trach_days) / lm0_n:.1f}%)")
print(f"LM0 Death       : {len(lm0_death_days)} ({100 * len(lm0_death_days) / lm0_n:.1f}%)")
print(f"LM3 Tracheostomy: {len(lm3_trach_days)} ({100 * len(lm3_trach_days) / lm3_n:.1f}%)")
print(f"LM3 Death       : {len(lm3_death_days)} ({100 * len(lm3_death_days) / lm3_n:.1f}%)")

if lm3_n != 4101:
    print(f"[WARN] LM3 common risk set is {lm3_n}, not 4101. Check source data/version or event-date definitions.")

# =========================================================
# PLOT
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

plot_panel(
    ax=axes[0],
    trach_days=lm0_trach_days,
    death_days=lm0_death_days,
    denom=lm0_n,
    title="Landmark Day 0",
    panel_label="(A)"
)

plot_panel(
    ax=axes[1],
    trach_days=lm3_trach_days,
    death_days=lm3_death_days,
    denom=lm3_n,
    title="Landmark Day 3",
    panel_label="(B)"
)

plt.tight_layout()

# =========================================================
# SAVE
# =========================================================
os.makedirs(OUT_DIR, exist_ok=True)
outpath = os.path.join(OUT_DIR, OUT_FILE)

plt.savefig(
    outpath,
    dpi=DPI,
    format="tiff",
    bbox_inches="tight"
)
plt.close()

print(f"Saved: {outpath}")
print(f"Output folder: {OUT_DIR}")
