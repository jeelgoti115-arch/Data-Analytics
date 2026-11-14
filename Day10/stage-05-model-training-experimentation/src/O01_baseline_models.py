# Baseline Models Module Stub
"""
================================================================================
MODULE: Baseline Models Engine
FILE: src/O01_baseline_models.py
PURPOSE:
  Trains and evaluates simple linear and non-linear baseline models (Logistic 
  Regression and Decision Tree) to establish performance benchmarks for the 
  customer churn prediction pipeline, parameterized by 'config/O01_model_rules.yaml'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("BaselineModels")


class BaselineModelTrainer:
    """Trains and evaluates baseline classification models."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.trained_models = {}

    def _load_config(self) -> dict:
        """Loads model rules from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def train_and_evaluate(self, X_train: pd.DataFrame, y_series: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """
        Trains baseline models and computes evaluation metrics on validation data.
        """
        random_state = self.config.get("global", {}).get("random_state", 42)
        baseline_cfg = self.config.get("baseline_models", {})
        
        lr_cfg = baseline_cfg.get("logistic_regression", {})
        dt_cfg = baseline_cfg.get("decision_tree", {})

        models = {
            "LogisticRegression": LogisticRegression(
                C=lr_cfg.get("C", 1.0),
                max_iter=lr_cfg.get("max_iter", 1000),
                solver=lr_cfg.get("solver", "lbfgs"),
                class_weight="balanced",
                random_state=random_state
            ),
            "DecisionTree": DecisionTreeClassifier(
                max_depth=dt_cfg.get("max_depth", 5),
                min_samples_split=dt_cfg.get("min_samples_split", 20),
                class_weight="balanced",
                random_state=random_state
            )
        }

        evaluation_results = {}

        for name, model in models.items():
            logger.info(f"Training baseline model: {name}...")
            model.fit(X_train, y_series)
            
            # Predict probabilities and classes
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

            # Calculate metrics
            auc = roc_auc_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)

            evaluation_results[name] = {
                "model": model,
                "roc_auc": auc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }

            logger.info(f"[{name}] Validation Metrics -> ROC-AUC: {auc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            self.trained_models[name] = model

        return evaluation_results


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_model_rules.yaml")
    
    np.random.seed(42)
    X_tr = pd.DataFrame(np.random.randn(100, 10), columns=[f"feat_{i}" for i in range(10)])
    y_tr = pd.Series(np.random.choice([0, 1], size=100, p=[0.8, 0.2]))
    
    X_v = pd.DataFrame(np.random.randn(30, 10), columns=[f"feat_{i}" for i in range(10)])
    y_v = pd.Series(np.random.choice([0, 1], size=30, p=[0.8, 0.2]))

    if config_file.exists():
        trainer = BaselineModelTrainer(config_path=config_file)
        results = trainer.train_and_evaluate(X_tr, y_tr, X_v, y_v)
        print("\n[RESULT] Baseline training executed successfully.")