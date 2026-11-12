
```markdown
# Execution Guide: Stage 4 - Feature Engineering & Selection

## 📌 Cross-Stage Prerequisites & Inputs
* **Upstream Source:** Clean telemetry output from Stage 3 (`stage-03-data-preprocessing-cleaning-eda/data/processed/clean_telco_telemetry.csv`).

---

## 📌 Configuration Rules (`config/O01_feature_rules.yaml`)
Governed centrally by YAML configuration rules containing:
* **Domain Feature Rules:** Custom financial and usage ratio calculations.
* **Categorical Encoding:** Low-cardinality One-Hot Encoding and high-cardinality smoothed Target Encoding parameters.
* **Numerical Transformations:** Yeo-Johnson power transform settings and Robust/Standard scaling rules.
* **Cardinality & Selection:** Rare label frequency thresholds (`Other` group) and Mutual Information top-$K$ feature selection limits.

---

## 📌 Execution Instructions

1. **Verify Prerequisites:** Ensure Stage 3 clean telemetry data is accessible.
2. **Execute Master Feature Orchestrator:** Run the Stage 4 pipeline:
   ```bash
   python src/O07_orchestrator.py
   ```
   Validate Outputs:

Confirm processed feature matrices are saved at data/processed/train_features.csv and data/processed/test_features.csv.

Verify fitted transformer artifacts are serialized in models/ (e.g., fitted_categorical_encoder.pkl, fitted_feature_selector.pkl).