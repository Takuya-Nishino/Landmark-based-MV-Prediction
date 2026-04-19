# =========================================================
# 別セル用:
# 1) INDEXベースのランドマーク母集団 / outcome別解析対象フラグ
# 2) サプリ用 使用特徴量辞書
#
# 仕様:
# - Height, Weight は使わない
# - IntubationDate を使う（整数列としてそのまま）
# - 診療科はワンホット後の列名で辞書化
# - 薬剤/輸血:
#     LM0 = Day0 を使用
#     LM3 = Day3 を使用
#     Delta は使わない
# - 検査:
#     LM0 = Day0
#     LM3 = Day0 + Delta(Day3 - Day0)
# =========================================================

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build landmark eligibility summaries and feature dictionaries."
    )
    parser.add_argument("--excel-path", required=True, help="Path to source Excel file")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--out-file",
        default="Supplement_Landmark_Eligibility_and_Feature_Dictionary.xlsx",
        help="Output Excel filename",
    )
    parser.add_argument("--sheet-name", default=0, help="Excel sheet name or index")
    return parser.parse_args()


args = parse_args()
EXCEL_PATH = args.excel_path
OUT_DIR = args.out_dir
OUT_FILE = args.out_file
SHEET_NAME = args.sheet_name

DATE_COL_CANDIDATES = ["Base_Date", "BaseDate", "Date"]
INDEX_COL_CANDIDATES = ["INDEX", "Index", "ID"]
DAYS_ON_MV_COL_CANDIDATES = ["人工呼吸_連続日数", "All_Day"]

OUTCOME_DATE_CANDIDATES = {
    "Event_死亡": ["死亡日", "Death_Date", "DeathDate"],
    "Event_気管切開": ["気管切開日", "Tracheostomy_Date", "TracheostomyDate"],
}

LANDMARK_HORIZON_DAYS = 21
LM3_OFFSET_DAYS = 3
EXCLUDE_SAME_DAY_EVENT = True

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

# Height, Weight は除外
BASE_CANDIDATES = [
    "Age", "Male", "BMI", "IntubationDate",
    "Dialysis", "CirculatoryDevice", "Hypothermia", "CPR"
]

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def find_first_existing(columns, candidates, required=True):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise KeyError(f"必要な列が見つかりません: {candidates}")
    return None

def ensure_datetime(df, col):
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

def safe_to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def write_excel_book(path, sheets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)

# ---------------------------------------------------------
# DEPARTMENT
# ---------------------------------------------------------
def prepare_department_features(df):
    src_col = None
    for c in ["Department", "診療科名", "科名"]:
        if c in df.columns:
            src_col = c
            break

    out = df.copy()

    if src_col is None:
        out["Department_en"] = "Other"
        return out

    mapping = {
        "救命救急": "Emergency and Critical Care",
        "心臓": "Cardiac",
        "脳疾患": "Neurological",
        "その他": "Other",
    }

    s = out[src_col].astype(str).fillna("その他").str.strip()
    out["Department_en"] = s.map(mapping).fillna("Other")
    return out

def build_department_onehot(df):
    out = prepare_department_features(df.copy())
    dept = out["Department_en"].astype(str).fillna("Other")

    categories = [
        "Emergency and Critical Care",
        "Cardiac",
        "Neurological",
        "Other"
    ]
    dept = pd.Categorical(dept, categories=categories)

    dummies = pd.get_dummies(dept, prefix="Department", prefix_sep="_", dtype=np.uint8)
    dummies = dummies.reindex(
        columns=[
            "Department_Emergency and Critical Care",
            "Department_Cardiac",
            "Department_Neurological",
            "Department_Other",
        ],
        fill_value=0
    )

    out = pd.concat([out, dummies], axis=1)
    return out, list(dummies.columns)

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def select_lab_columns_by_day(df, day):
    return [f"{day}_{x}" for x in LAB_NAMES if f"{day}_{x}" in df.columns]

def select_med_columns_by_day(df, day):
    return [f"{day}_{x}" for x in MED_NAMES if f"{day}_{x}" in df.columns]

def add_delta_labs(df):
    out = df.copy()
    delta_cols = []

    day0_labs = select_lab_columns_by_day(out, "Day0")
    day3_labs = select_lab_columns_by_day(out, "Day3")

    day0_bases = [c.replace("Day0_", "") for c in day0_labs]
    day3_bases = [c.replace("Day3_", "") for c in day3_labs]

    common_bases = sorted(set(day0_bases).intersection(set(day3_bases)))

    for base in common_bases:
        c0 = f"Day0_{base}"
        c3 = f"Day3_{base}"
        if c0 in out.columns and c3 in out.columns:
            new_c = f"Delta_{base}"
            out[new_c] = safe_to_numeric(out[c3]) - safe_to_numeric(out[c0])
            delta_cols.append(new_c)

    return out, delta_cols

def build_feature_matrix(df, landmark):
    out, dept_onehot_cols = build_department_onehot(df.copy())

    base_cols = [c for c in BASE_CANDIDATES if c in out.columns]

    # 検査
    day0_labs = select_lab_columns_by_day(out, "Day0")

    # 薬剤/輸血
    day0_meds = select_med_columns_by_day(out, "Day0")
    day3_meds = select_med_columns_by_day(out, "Day3")

    delta_lab_cols = []

    if landmark == "LM0":
        used = base_cols + dept_onehot_cols + day0_labs + day0_meds

    elif landmark == "LM3":
        out, delta_lab_cols = add_delta_labs(out)
        # LM3では薬剤はDay3のみ、Day0薬剤は使わない
        used = base_cols + dept_onehot_cols + day0_labs + delta_lab_cols + day3_meds

    else:
        raise ValueError(f"未対応のlandmarkです: {landmark}")

    used = [c for c in used if c in out.columns]
    X = out[used].copy()

    for c in X.columns:
        X[c] = safe_to_numeric(X[c]).astype("float32")

    meta = {
        "dept_onehot_cols": dept_onehot_cols,
        "base_cols": base_cols,
        "day0_lab_cols": day0_labs,
        "day0_med_cols": day0_meds,
        "day3_med_cols": day3_meds,
        "delta_lab_cols": delta_lab_cols,
        "used_cols": list(X.columns),
        "lm0_strategy": "Baseline + Department one-hot + Day0 labs + Day0 medications/transfusions",
        "lm3_strategy": "Baseline + Department one-hot + Day0 labs + Delta labs + Day3 medications/transfusions",
    }
    return X, meta

# ---------------------------------------------------------
# LANDMARK DATASET BUILDER
# ---------------------------------------------------------
def build_landmark_dataset(df_raw, landmark, target_col):
    if target_col not in OUTCOME_DATE_CANDIDATES:
        raise ValueError(f"未対応のtarget_colです: {target_col}")

    cols = list(df_raw.columns)

    base_date_col = find_first_existing(cols, DATE_COL_CANDIDATES, required=True)
    days_col = find_first_existing(cols, DAYS_ON_MV_COL_CANDIDATES, required=True)
    outcome_date_col = find_first_existing(cols, OUTCOME_DATE_CANDIDATES[target_col], required=True)

    df = df_raw.copy()
    df = ensure_datetime(df, base_date_col)
    df = ensure_datetime(df, outcome_date_col)
    df = df.dropna(subset=[base_date_col]).copy()

    if landmark == "LM0":
        df["Landmark_Date"] = df[base_date_col]
    elif landmark == "LM3":
        df = df[safe_to_numeric(df[days_col]) >= 4].copy()
        df["Landmark_Date"] = df[base_date_col] + pd.Timedelta(days=LM3_OFFSET_DAYS)
    else:
        raise ValueError(f"未対応のlandmarkです: {landmark}")

    df = df.reset_index(drop=True).copy()

    lm_date_col = "Landmark_Date"
    horizon_end = df[lm_date_col] + pd.Timedelta(days=LANDMARK_HORIZON_DAYS)

    pre_landmark_event = (
        df[outcome_date_col].notna() &
        (df[outcome_date_col] < df[lm_date_col])
    )
    df = df.loc[~pre_landmark_event].reset_index(drop=True).copy()

    horizon_end = df[lm_date_col] + pd.Timedelta(days=LANDMARK_HORIZON_DAYS)
    has_event_date = df[outcome_date_col].notna()

    if EXCLUDE_SAME_DAY_EVENT:
        lower_condition = df[outcome_date_col] > df[lm_date_col]
    else:
        lower_condition = df[outcome_date_col] >= df[lm_date_col]

    in_window = (
        has_event_date &
        lower_condition &
        (df[outcome_date_col] <= horizon_end)
    )

    df[target_col] = in_window.astype(int)
    df[f"{target_col}_days_from_t0"] = np.where(
        has_event_date,
        (df[outcome_date_col] - df[lm_date_col]).dt.days,
        np.nan,
    )
    df["time0"] = df[lm_date_col]
    df["timeH"] = horizon_end

    return df, lm_date_col

# ---------------------------------------------------------
# DESCRIPTION HELPERS
# ---------------------------------------------------------
def classify_feature_origin(feature_name):
    if feature_name.startswith("Department_"):
        return "Department_onehot"
    if feature_name.startswith("Day0_") and feature_name.replace("Day0_", "") in LAB_NAMES:
        return "Day0_Laboratory"
    if feature_name.startswith("Day0_") and feature_name.replace("Day0_", "") in MED_NAMES:
        return "Day0_Medication_or_Transfusion"
    if feature_name.startswith("Day3_") and feature_name.replace("Day3_", "") in MED_NAMES:
        return "Day3_Medication_or_Transfusion"
    if feature_name.startswith("Delta_"):
        return "Delta_Laboratory_Day3_minus_Day0"
    if feature_name in BASE_CANDIDATES:
        return "Baseline"
    return "Other"

def make_feature_description(feature_name):
    if feature_name.startswith("Department_"):
        dept_name = feature_name.replace("Department_", "")
        return f"One-hot encoded admitting department: {dept_name}"
    if feature_name.startswith("Day0_") and feature_name.replace("Day0_", "") in LAB_NAMES:
        return f"{feature_name.replace('Day0_', '')} measured on Day 0"
    if feature_name.startswith("Day0_") and feature_name.replace("Day0_", "") in MED_NAMES:
        return f"{feature_name.replace('Day0_', '')} administered on Day 0"
    if feature_name.startswith("Day3_") and feature_name.replace("Day3_", "") in MED_NAMES:
        return f"{feature_name.replace('Day3_', '')} administered on Day 3"
    if feature_name.startswith("Delta_"):
        return f"Change from Day 0 to Day 3 in {feature_name.replace('Delta_', '')}"
    if feature_name == "IntubationDate":
        return "Days from admission to intubation"
    return feature_name

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
df_raw = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME).copy()
all_cols = list(df_raw.columns)

index_col = find_first_existing(all_cols, INDEX_COL_CANDIDATES, required=True)
base_date_col = find_first_existing(all_cols, DATE_COL_CANDIDATES, required=True)
mv_days_col = find_first_existing(all_cols, DAYS_ON_MV_COL_CANDIDATES, required=True)

df_raw[index_col] = df_raw[index_col].astype(str)
df_raw[base_date_col] = pd.to_datetime(df_raw[base_date_col], errors="coerce")
df_raw[mv_days_col] = safe_to_numeric(df_raw[mv_days_col])

if "IntubationDate" in df_raw.columns:
    df_raw["IntubationDate"] = safe_to_numeric(df_raw["IntubationDate"])

# ---------------------------------------------------------
# 1) INDEXベースのランドマーク母集団 + outcome別解析対象
# ---------------------------------------------------------
eligibility_df = df_raw[[index_col]].copy()

lm0_pool_idx = df_raw.loc[df_raw[base_date_col].notna(), index_col]
lm3_pool_idx = df_raw.loc[
    df_raw[base_date_col].notna() & (df_raw[mv_days_col] >= 4),
    index_col
]

eligibility_df["LM0_landmark_pool"] = eligibility_df[index_col].isin(lm0_pool_idx).astype(int)
eligibility_df["LM3_landmark_pool"] = eligibility_df[index_col].isin(lm3_pool_idx).astype(int)

df_lm0_trach, _ = build_landmark_dataset(df_raw, landmark="LM0", target_col="Event_気管切開")
df_lm0_death, _ = build_landmark_dataset(df_raw, landmark="LM0", target_col="Event_死亡")
df_lm3_trach, _ = build_landmark_dataset(df_raw, landmark="LM3", target_col="Event_気管切開")
df_lm3_death, _ = build_landmark_dataset(df_raw, landmark="LM3", target_col="Event_死亡")

eligibility_df["LM0_trach_analysis_included"] = eligibility_df[index_col].isin(df_lm0_trach[index_col].astype(str)).astype(int)
eligibility_df["LM0_death_analysis_included"] = eligibility_df[index_col].isin(df_lm0_death[index_col].astype(str)).astype(int)
eligibility_df["LM3_trach_analysis_included"] = eligibility_df[index_col].isin(df_lm3_trach[index_col].astype(str)).astype(int)
eligibility_df["LM3_death_analysis_included"] = eligibility_df[index_col].isin(df_lm3_death[index_col].astype(str)).astype(int)

eligibility_df = eligibility_df.sort_values(index_col).reset_index(drop=True)

eligibility_summary_df = pd.DataFrame({
    "flag": [
        "LM0_landmark_pool",
        "LM3_landmark_pool",
        "LM0_trach_analysis_included",
        "LM0_death_analysis_included",
        "LM3_trach_analysis_included",
        "LM3_death_analysis_included",
    ],
    "n_included": [
        int(eligibility_df["LM0_landmark_pool"].sum()),
        int(eligibility_df["LM3_landmark_pool"].sum()),
        int(eligibility_df["LM0_trach_analysis_included"].sum()),
        int(eligibility_df["LM0_death_analysis_included"].sum()),
        int(eligibility_df["LM3_trach_analysis_included"].sum()),
        int(eligibility_df["LM3_death_analysis_included"].sum()),
    ]
})

# ---------------------------------------------------------
# 2) 使用特徴量辞書
# ---------------------------------------------------------
X_lm0, meta_lm0 = build_feature_matrix(df_lm0_trach, landmark="LM0")
X_lm3, meta_lm3 = build_feature_matrix(df_lm3_trach, landmark="LM3")

all_used_features = sorted(set(meta_lm0["used_cols"]).union(set(meta_lm3["used_cols"])))

feature_dict_df = pd.DataFrame({"feature": all_used_features})
feature_dict_df["included_in_LM0"] = feature_dict_df["feature"].isin(meta_lm0["used_cols"]).astype(int)
feature_dict_df["included_in_LM3"] = feature_dict_df["feature"].isin(meta_lm3["used_cols"]).astype(int)
feature_dict_df["is_department_onehot"] = feature_dict_df["feature"].isin(set(meta_lm0["dept_onehot_cols"]).union(set(meta_lm3["dept_onehot_cols"]))).astype(int)
feature_dict_df["is_delta_lab"] = feature_dict_df["feature"].isin(set(meta_lm3["delta_lab_cols"])).astype(int)
feature_dict_df["feature_origin"] = feature_dict_df["feature"].map(classify_feature_origin)
feature_dict_df["description"] = feature_dict_df["feature"].map(make_feature_description)
feature_dict_df["LM0_strategy"] = meta_lm0["lm0_strategy"]
feature_dict_df["LM3_strategy"] = meta_lm3["lm3_strategy"]
feature_dict_df = feature_dict_df.sort_values("feature").reset_index(drop=True)

candidate_master_df = pd.concat([
    pd.DataFrame({"feature_base_name": BASE_CANDIDATES, "feature_group": "Baseline"}),
    pd.DataFrame({"feature_base_name": [
        "Department_Emergency and Critical Care",
        "Department_Cardiac",
        "Department_Neurological",
        "Department_Other"
    ], "feature_group": "Department_onehot"}),
    pd.DataFrame({"feature_base_name": LAB_NAMES, "feature_group": "Laboratory_base_name"}),
    pd.DataFrame({"feature_base_name": MED_NAMES, "feature_group": "Medication_or_Transfusion_base_name"}),
], axis=0, ignore_index=True)

# ---------------------------------------------------------
# 3) 保存
# ---------------------------------------------------------
out_path = Path(OUT_DIR) / OUT_FILE

write_excel_book(
    out_path,
    sheets={
        "LM_eligibility_by_INDEX": eligibility_df,
        "LM_eligibility_summary": eligibility_summary_df,
        "feature_dictionary": feature_dict_df,
        "candidate_feature_master": candidate_master_df,
        "LM0_used_features": pd.DataFrame({"feature": meta_lm0["used_cols"]}),
        "LM3_used_features": pd.DataFrame({"feature": meta_lm3["used_cols"]}),
    }
)

print(f"✅ Saved: {out_path}")
print("")
print("===== Landmark pool =====")
print(f"LM0 landmark pool: {eligibility_df['LM0_landmark_pool'].sum()}")
print(f"LM3 landmark pool: {eligibility_df['LM3_landmark_pool'].sum()}")
print("")
print("===== Outcome-specific analysis sets =====")
print(f"LM0 trach analysis included: {eligibility_df['LM0_trach_analysis_included'].sum()}")
print(f"LM0 death analysis included: {eligibility_df['LM0_death_analysis_included'].sum()}")
print(f"LM3 trach analysis included: {eligibility_df['LM3_trach_analysis_included'].sum()}")
print(f"LM3 death analysis included: {eligibility_df['LM3_death_analysis_included'].sum()}")
print("")
print("===== Feature counts =====")
print(f"LM0 used features: {len(meta_lm0['used_cols'])}")
print(f"LM3 used features: {len(meta_lm3['used_cols'])}")
