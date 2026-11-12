# Stage 4: Feature Engineering, Transformation & Selection

This module handles domain feature generation, categorical encoding (One-Hot, Target Encoding), numerical transformations (Box-Cox, Scaling), dimensionality reduction, multistage feature selection, and strict data leakage prevention.

# Stage 4: Feature Engineering, Transformation & Selection

## 📌 Architectural Overview
Stage 4 bridges raw, cleaned telemetry data from Stage 3 and production-ready feature matrices for machine learning models. To prevent data leakage and ensure reproducibility, this module enforces strict **fit-transform separation** (fitting transformers exclusively on training folds and applying them to validation/test splits).

---

## 📌 Module Structure & Chronological Execution

```text
stage-04-feature-engineering-selection/
│
├── config/
│   └── O01_feature_rules.yaml               # Centralized feature engineering rules & hyperparameters
│
├── data/
│   ├── raw/                                 # Symlinked or staged raw inputs
│   └── processed/                           # Output train/test feature matrices (.csv)
│
├── models/                                  # Serialized transformer artifacts (.pkl via joblib)
│   ├── fitted_categorical_encoder.pkl
│   ├── fitted_numerical_transformer.pkl
│   ├── fitted_cardinality_reducer.pkl
│   └── fitted_feature_selector.pkl
│
├── src/
│   ├── O01_domain_feature_builder.py        # Domain-specific financial/usage ratios & tenure buckets
│   ├── O02_categorical_encoder.py           # One-Hot and smoothed Target Encoding
│   ├── O03_numerical_transformer.py         # Yeo-Johnson power transforms & Robust/Standard scaling
│   ├── O04_cardinality_dimensionality_reducer.py # Rare label grouping ('Other') & PCA reduction
│   ├── O05_feature_selector.py              # Variance thresholding, correlation filtering, & Mutual Info
│   ├── O06_data_leakage_validator.py        # Stratified train/test splitting & index overlap auditing
│   └── O07_orchestrator.py                  # Master pipeline execution runner
│
└── README.md                                # This documentation file