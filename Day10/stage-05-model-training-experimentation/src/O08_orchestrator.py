# Stage 5 Master Orchestrator Module Stub
"""
================================================================================
MODULE: Stage 5 Master Pipeline Orchestrator
FILE: src/O08_orchestrator.py
PURPOSE:
  Serves as the main chronological execution runner for Stage 5. Integrates 
  cross-validation partitioning, advanced ensemble model training, hyperparameter 
  optimization, probability calibration, threshold tuning, SHAP model 
  interpretability, and champions model artifact serialization to 'models/'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import joblib
import json

# Import modular components from Stage 5
from O01_baseline_models import BaselineModelTrainer
from O02_advanced_ensembles import AdvancedEnsembleTrainer
from O03_imbalance_handler import ImbalanceHandler
from O04_cross_validation_engine import CrossValidationEngine
from O05_hyperparameter_optimizer import HyperparameterOptimizer
from O06_threshold_calibrator import ThresholdCalibrator
from O07_model_interpretability import ModelInterpreter

# Configure pipeline logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("Stage5Orchestrator")


def run_stage5_pipeline():
    """Executes the complete end-to-end Model Training, Tuning & Validation pipeline."""
    logger.info("==================================================")
    logger.info(" STARTING STAGE 5: MODEL TRAINING & EXPERIMENTATION")
    logger.info("==================================================")

    # 1. Resolve Paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    config_path = project_root / "config/O01_model_rules.yaml"
    models_dir = project_root / "models"
    processed_dir = project_root / "data/processed"
    
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load processed feature matrices from Stage 4 output
    train_file = processed_dir / "train_features.csv"
    test_file = processed_dir / "test_features.csv"

    if not train_file.exists() or not test_file.exists():
        logger.info("Stage 4 processed feature matrices not found. Generating sample data for Stage 5 testing...")
        np.random.seed(42)
        X_dummy = pd.DataFrame(np.random.randn(200, 10), columns=[f"feat_{i}" for i in range(10)])
        y_dummy = pd.Series(np.random.choice([0, 1], size=200, p=[0.75, 0.25]), name="churn_status")
        
        train_df = pd.concat([X_dummy[:160], y_dummy[:160]], axis=1)
        test_df = pd.concat([X_dummy[160:], y_dummy[160:]], axis=1)
    else:
        logger.info(f"Loading train/test features from Stage 4 path...")
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

    target_col = "churn_status"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from training dataset.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col], errors="ignore")
    y_test = test_df[target_col] if target_col in test_df.columns else None

    # 2. STEP 1: Baseline Model Benchmarking
    logger.info("\n--- STEP 1: Baseline Model Training ---")
    baseline_trainer = BaselineModelTrainer(config_path=config_path)
    # Split a small validation set from training for quick scoring demo
    val_split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_tr, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]
    
    baseline_results = baseline_trainer.train_and_evaluate(X_tr, y_tr, X_val, y_val)

    # 3. STEP 2: Advanced Ensemble Training
    logger.info("\n--- STEP 2: Advanced Ensemble Model Training ---")
    ensemble_trainer = AdvancedEnsembleTrainer(config_path=config_path)
    ensemble_results = ensemble_trainer.train_and_evaluate(X_tr, y_tr, X_val, y_val)

    # Identify best performing champion model based on validation ROC-AUC
    all_results = {**baseline_results, **ensemble_results}
    best_model_name = max(all_results, key=lambda k: all_results[k]["roc_auc"])
    champion_model = all_results[best_model_name]["model"]
    logger.info(f"\n🏆 CHAMPION MODEL SELECTED: {best_model_name} (ROC-AUC: {all_results[best_model_name]['roc_auc']:.4f})")

    # 4. STEP 3: Probability Calibration & Threshold Tuning
    logger.info("\n--- STEP 3: Probability Calibration & Threshold Tuning ---")
    calibrator = ThresholdCalibrator(method="isotonic")
    calibrated_champion = calibrator.calibrate_probabilities(champion_model, X_val, y_val)
    
    val_probs = calibrated_champion.predict_proba(X_val)[:, 1]
    optimal_threshold = calibrator.find_optimal_threshold(y_val.values, val_probs)

    # 5. STEP 4: SHAP Model Interpretability
    logger.info("\n--- STEP 4: Generating SHAP Model Interpretability ---")
    # Use base estimator from CalibratedClassifierCV if wrapped
    base_estimator = champion_model
    interpreter = ModelInterpreter(model=base_estimator, X_sample=X_val.head(50))
    global_importance = interpreter.get_global_importance_df()
    logger.info(f"Top global feature: {global_importance.iloc[0]['feature']}")

    # 6. STEP 5: Artifact Serialization
    logger.info("\n--- STEP 5: Serializing Champion Artifacts ---")
    joblib.dump(calibrated_champion, models_dir / "champion_model.pkl")
    
    metadata = {
        "champion_model_name": best_model_name,
        "validation_roc_auc": float(all_results[best_model_name]["roc_auc"]),
        "validation_f1": float(all_results[best_model_name]["f1_score"]),
        "optimal_decision_threshold": float(optimal_threshold),
        "top_features": global_importance["feature"].head(5).tolist()
    }
    
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Champion model serialized successfully to: {models_dir / 'champion_model.pkl'}")
    logger.info("==================================================")
    logger.info(" STAGE 5 COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    run_stage5_pipeline()