Dynamic Prediction of 21-Day Outcomes in Mechanically Ventilated Patients

This repository contains the official implementation and analysis pipeline for the study: "Dynamic Prediction of 21-Day Outcomes in Mechanically Ventilated Patients Using Routinely Collected EHR–DPC Data: A Multicenter Machine Learning Study."

Overview
This study develops and validates a dynamic machine-learning framework (Landmark Day 0 and Day 3) to predict tracheostomy or mortality within 21 days after intubation. The models are "EHR-native," utilizing only routinely collected electronic health record (EHR) and Diagnosis Procedure Combination (DPC) administrative data.

Key Features
• Landmark Design: Dynamic risk updating at intubation (LM0) and 72 hours (LM3).

• Algorithms: Multinomial Logistic Regression, XGBoost, and LightGBM.

• Interpretability: SHAP-based feature importance analysis.

• Validation: Temporal validation (cutoff: Jan 1, 2024) and internal–external validation across four hospitals.

Repository Structure

• /preprocessing: Scripts for DPC and EHR laboratory data cleaning and feature engineering.

• /modeling: Hyperparameter optimization (Optuna), model training, and probability calibration.


• /evaluation: Code for AUROC/AUPRC calculation, Calibration plots, and Decision Curve Analysis (DCA).

• /visualization: Scripts for generating figures (e.g., Figure 2, 3, and 4).

Requirements

• Python: 3.11.11 (libraries: xgboost, lightgbm, optuna, shap, scikit-learn).

• R: 4.4.1 (for clinical characteristic analysis and specific visualizations).

Data Availability
The clinical data used in this study (DPC claims and EHR-derived laboratory data) are not publicly available due to institutional policies and legal restrictions. This repository provides the full source code to ensure the transparency of our analytical methods and to allow for implementation in other healthcare settings.

Reporting Compliance
This study follows the TRIPOD-AI guidelines
