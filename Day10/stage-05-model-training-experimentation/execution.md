```markdown
# Execution Guide: Stage 5 - Model Training & Validation

##  Cross-Stage Prerequisites & Inputs
* **Upstream Source:** Processed feature matrices from Stage 4 (`stage-04-feature-engineering-selection/data/processed/train_features.csv` and `test_features.csv`).

---

##  Configuration Rules (`config/O01_model_rules.yaml`)
Governed centrally by YAML configuration rules containing:
* **Validation Strategy:** Stratified K-Fold cross-validation split counts and random seed settings.
* **Imbalance Mitigation:** SMOTE sampling ratios and balanced class weight configurations.
* **Model Hyperparameters:** Baseline (Logistic Regression, Decision Tree) and advanced ensemble (XGBoost, LightGBM, CatBoost) search boundaries.
* **Optimization Rules:** Optuna trial counts and evaluation metrics (ROC-AUC).

---

## Execution Instructions

1. **Verify Prerequisites:** Ensure Stage 4 feature matrices (`train_features.csv` and `test_features.csv`) are present in `data/processed/`.
2. **Execute Master Training Orchestrator:** Run the Stage 5 training pipeline:
   ```bash
   python src/O08_orchestrator.py
   ```
   Validate Outputs:

Confirm the trained champion model is serialized at models/champion_model.pkl.

Inspect model metadata, validation metrics, and top SHAP features in models/model_metadata.json.