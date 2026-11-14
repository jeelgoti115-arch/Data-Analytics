# Stage 5: Model Training, Experimentation & Validation
# Stage 5: Model Training, Experimentation & Validation Strategies

## 📋 Architectural Overview
Stage 5 transitions our engineered feature matrices into high-performance, robust machine learning models. To ensure generalizability and prevent data leakage, this stage enforces **Stratified K-Fold cross-validation**, fold-isolated class imbalance mitigation (SMOTE & balanced weights), Bayesian hyperparameter optimization (Optuna), probability calibration (Isotonic Regression), and local/global model explainability via SHAP.

---

## 📂 Module Structure & Chronological Execution

```text
stage-05-model-training-experimentation/
│
├── config/
│   └── O01_model_rules.yaml                 # Centralized hyperparameters, Optuna search spaces, and CV rules
│
├── data/
│   ├── processed/                           # Input train/test features from Stage 4
│   └── experiments/                         # Cross-validation fold splits & OOF predictions
│
├── models/                                  # Serialized champion models & metadata artifacts
│   ├── champion_model.pkl
│   └── model_metadata.json
│
├── mlruns/                                  # Local experiment tracking store
│
├── src/
│   ├── O01_baseline_models.py               # Logistic Regression & Decision Tree baselines
│   ├── O02_advanced_ensembles.py            # XGBoost, LightGBM, and CatBoost gradient boosting engines
│   ├── O03_imbalance_handler.py             # Class imbalance mitigation (SMOTE & balanced class weights)
│   ├── O04_cross_validation_engine.py       # Stratified K-Fold partitioning engine
│   ├── O05_hyperparameter_optimizer.py      # Optuna-driven Bayesian optimization
│   ├── O06_threshold_calibrator.py          # Probability calibration & F1/Profit threshold tuning
│   ├── O07_model_interpretability.py        # SHAP global and local explanation extractor
│   └── O08_orchestrator.py                  # Master pipeline execution runner
│
└── README.md                                # This documentation file