# Input Data Schema

This document describes the expected input structure for the public code release.

## Required identifier and time columns

At least one of the following identifier columns must be present:
- `INDEX`
- `Index`
- `ID`

At least one of the following baseline date columns must be present:
- `Base_Date`
- `BaseDate`
- `Date`

At least one of the following ventilation duration columns must be present:
- `人工呼吸_連続日数`
- `All_Day`

## Required outcome date columns

For tracheostomy prediction, at least one of the following must be present:
- `気管切開日`
- `Tracheostomy_Date`
- `TracheostomyDate`

For mortality prediction, at least one of the following must be present:
- `死亡日`
- `Death_Date`
- `DeathDate`

## Optional department columns

One of the following may be used to derive department one-hot variables:
- `Department`
- `診療科名`
- `科名`

If none of these are available, department is treated as `Other`.

## Expected feature groups

### Baseline features
- `Age`
- `Male`
- `BMI`
- `IntubationDate`
- `Dialysis`
- `CirculatoryDevice`
- `Hypothermia`
- `CPR`

### Day 0 laboratory variables
- `Day0_ALT`
- `Day0_AST`
- `Day0_Alb`
- `Day0_APTT`
- `Day0_BUN`
- `Day0_CK`
- `Day0_CRP`
- `Day0_Cl`
- `Day0_Cre`
- `Day0_D-dimer`
- `Day0_Hb`
- `Day0_K`
- `Day0_Na`
- `Day0_PLT`
- `Day0_PT-INR`
- `Day0_T-Bil`
- `Day0_TP`
- `Day0_WBC`

### Day 3 laboratory variables
- `Day3_ALT`
- `Day3_AST`
- `Day3_Alb`
- `Day3_APTT`
- `Day3_BUN`
- `Day3_CK`
- `Day3_CRP`
- `Day3_Cl`
- `Day3_Cre`
- `Day3_D-dimer`
- `Day3_Hb`
- `Day3_K`
- `Day3_Na`
- `Day3_PLT`
- `Day3_PT-INR`
- `Day3_T-Bil`
- `Day3_TP`
- `Day3_WBC`

### Day 0 medication or transfusion variables
- `Day0_CoreSed`
- `Day0_Opioid`
- `Day0_NAD`
- `Day0_Adrenaline`
- `Day0_DOA`
- `Day0_DOB`
- `Day0_Insulin`
- `Day0_Steroid`
- `Day0_Vasopressin`
- `Day0_Diuretic`
- `Day0_AntiMRSA`
- `Day0_Antibiotic`
- `Day0_Carbapenem`
- `Day0_Antifungal`
- `Day0_FFP`
- `Day0_Plt`
- `Day0_RBC`

### Day 3 medication or transfusion variables
- `Day3_CoreSed`
- `Day3_Opioid`
- `Day3_NAD`
- `Day3_Adrenaline`
- `Day3_DOA`
- `Day3_DOB`
- `Day3_Insulin`
- `Day3_Steroid`
- `Day3_Vasopressin`
- `Day3_Diuretic`
- `Day3_AntiMRSA`
- `Day3_Antibiotic`
- `Day3_Carbapenem`
- `Day3_Antifungal`
- `Day3_FFP`
- `Day3_Plt`
- `Day3_RBC`

## Derived variables created inside the pipeline

The pipeline can derive the following variables internally when both Day 0 and Day 3 laboratory variables are available:
- `Delta_ALT`
- `Delta_AST`
- `Delta_Alb`
- `Delta_APTT`
- `Delta_BUN`
- `Delta_CK`
- `Delta_CRP`
- `Delta_Cl`
- `Delta_Cre`
- `Delta_D-dimer`
- `Delta_Hb`
- `Delta_K`
- `Delta_Na`
- `Delta_PLT`
- `Delta_PT-INR`
- `Delta_T-Bil`
- `Delta_TP`
- `Delta_WBC`

## Notes

- The code is tolerant to partial feature availability, but the manuscript-aligned workflow assumes a structured dataset consistent with the original study design.
- Department one-hot variables are generated automatically.
- For LM0, Day 0 variables are primarily used.
- For LM3, Day 0 laboratory variables, Day 3 medication variables, and Day 3 minus Day 0 laboratory changes are used.
